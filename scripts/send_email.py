"""Email the daily forecast report.

Contents: forecast table (all models, all horizons), chart (last 60 days + forecasts),
last-7-days actual vs forecast, realised accuracy scorecard, and the last 7 days of the four
WTI/Brent futures and spot series. Models are read from the forecast log, so any model added to the
registry shows up automatically.

Credentials come from environment variables (never put them in the repo):
    GMAIL_USER          sender Gmail address
    GMAIL_APP_PASSWORD  a Gmail app password (needs 2-step verification)
Recipients: see forecast/mailer.py (EMAIL_TO / EMAIL_CC override the defaults)

Usage:  python scripts/send_email.py             send the email
        python scripts/send_email.py --dry-run   write email_preview/ (html, png, eml) and send nothing
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forecast import mailer, reporting as rp  # noqa: E402
from forecast.config import OUTPUT_DIR, RAW_DIR, Config  # noqa: E402
from forecast.data import load_inputs  # noqa: E402

N_CHART_DAYS = 60
N_TABLE_DAYS = 7

SERIES = {"WTI futures": ("yahoo_daily.csv", "wti_fut"), "WTI spot": ("fred_daily.csv", "wti_spot"),
          "Brent futures": ("yahoo_daily.csv", "brent_fut"), "Brent spot": ("fred_daily.csv", "brent_spot")}


def market_table(n=N_TABLE_DAYS):
    files = {f: pd.read_csv(RAW_DIR / f, index_col=0, parse_dates=True) for f in {v[0] for v in SERIES.values()}}
    df = pd.concat([files[f][c].rename(k) for k, (f, c) in SERIES.items()], axis=1, sort=True)
    df = df.reindex(files["yahoo_daily.csv"]["wti_fut"].dropna().index).tail(n)
    rows = [[d.strftime("%Y-%m-%d")] + ["<span style='color:#aaa'>n/a</span>" if pd.isna(v) else f"{v:,.2f}" for v in r]
            for d, r in df.iloc[::-1].iterrows()]
    return rp._html_table(["Date"] + list(df.columns), rows)


def _read_json(name):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else {}


def build(cfg):
    inputs = load_inputs(cfg)
    price = inputs.prices[cfg.target].dropna()
    latest = pd.read_csv(OUTPUT_DIR / "latest_forecasts.csv", parse_dates=["asof_date", "target_date_approx"])
    log = pd.read_csv(OUTPUT_DIR / "forecast_log.csv", parse_dates=["asof_date", "target_date_approx"])
    upd, run = _read_json("update_status.json"), _read_json("run_status.json")

    ftab, asof, p0 = rp.forecast_table(latest)
    chart = rp.make_chart(price, latest, N_CHART_DAYS)
    last7 = rp.last_days_table(log, price, N_TABLE_DAYS, horizon=1)
    sc = rp.scorecard(log, np.log(price))
    sc_html = rp.scorecard_html(sc)
    mkt = market_table()

    notes = [f"Target series: <b>{cfg.target}</b> (continuous front-month futures; contract rolls appear as small jumps). "
             f"Forecasts are expected log-returns turned into prices; \"No change\" is the random-walk benchmark."]
    if upd.get("failed_series"):
        notes.append(f"<b style='color:#b00'>Data warning:</b> these series failed to download today (older values kept): "
                     f"{', '.join(upd['failed_series'])}")
    for f in run.get("failed_plugins", []):
        notes.append(f"<b style='color:#b00'>Model failed:</b> {f['plugin']}: {f['error']}")
    for s in run.get("stale_plugins", []):
        notes.append(f"<b style='color:#b00'>Model not retrained for {s['age_days']} days:</b> {s['plugin']}")
    trained = latest.groupby("plugin")["model_train_end"].first().to_dict()
    notes.append("Models last trained on data through: " + ", ".join(f"{k} ({v})" for k, v in trained.items()) + ". "
                 "They are retrained quarterly; each day only the new data is fed through.")
    if upd.get("last_dates", {}).get("eia_wti_futures.csv", "9999") < "2025-01-01":
        notes.append("EIA's futures-curve files stopped in April 2024, so the curve-slope feature is unavailable.")
    note_html = "".join(f"<p style='font-size:12px;color:#555;margin:4px 0'>{n}</p>" for n in notes)

    html = f"""<html><body style="font-family:Arial,sans-serif;max-width:900px">
<h2 style="margin-bottom:2px">Oil forecast: {asof.date()}</h2>
<p style="margin-top:0;color:#555">{cfg.target} = <b>{p0:,.2f}</b> USD/bbl. Cells show forecast price (change vs no-change).</p>
<h3>1. Forecasts from today's data</h3>{ftab}
<h3>2. Last {N_CHART_DAYS} trading days and forecasts</h3><img src="CHART_SRC" style="max-width:100%">
<h3>3. Last {N_TABLE_DAYS} days: actual vs the 1-day-ahead forecast made the day before</h3>{last7}
<p style="font-size:11px;color:#777">Cell = forecast made the previous trading day (actual / forecast - 1 in brackets).</p>
<h3>4. Track record: RMSE vs no-change, by horizon (below 1.000 = better than no-change)</h3>{sc_html}
<h3>5. Market data, last {N_TABLE_DAYS} trading days (USD/bbl)</h3>{mkt}
<p style="font-size:11px;color:#777">n/a = FRED/EIA spot prices are published with a delay of several days.</p>
<hr>{note_html}</body></html>"""
    plain = (f"Oil forecast {asof.date()}: {cfg.target} = {p0:,.2f}\n\n"
             + latest[latest["asof_date"] == asof].pivot_table(index="horizon", columns="model",
                                                                values="forecast_price").round(2).to_string())
    c = latest[latest["asof_date"] == asof]
    combo = c[(c["model"] == "Combo_EW") & (c["horizon"] == c["horizon"].max())]
    tail = f" | Combo +{int(combo['horizon'].iloc[0])}d {combo['forecast_price'].iloc[0]:,.1f}" if len(combo) else ""
    subject = f"Oil forecast {asof.date()}: {cfg.target} {p0:,.2f}{tail}"
    return subject, plain, html, chart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    subject, plain, html, chart = build(cfg)
    msg = mailer.build_message(subject, plain, html.replace("CHART_SRC", "cid:chart"), chart)
    if args.dry_run:
        out = ROOT / "email_preview"
        out.mkdir(exist_ok=True)
        (out / "chart.png").write_bytes(chart)
        (out / "preview.html").write_text(html.replace("CHART_SRC", "chart.png"))
        (out / "message.eml").write_bytes(bytes(msg))
        print(f"Dry run: wrote {out}/ (preview.html, chart.png, message.eml). Nothing sent.\n"
              f"To: {msg['To']}  Cc: {msg.get('Cc')}\nSubject: {subject}")
        return 0
    try:
        to, cc = mailer.send(msg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Sent to {', '.join(to)} (cc {', '.join(cc)}): {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
