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

from forecast import frozen_report, mailer, reporting as rp, signals  # noqa: E402
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
    """Returns subject, plain text, html (images referenced as cid:chart_<target>), {cid: png}."""
    inputs = load_inputs(cfg)
    latest = pd.read_csv(OUTPUT_DIR / "latest_forecasts.csv", parse_dates=["asof_date", "target_date_approx"])
    log = pd.read_csv(OUTPUT_DIR / "forecast_log.csv", parse_dates=["asof_date", "target_date_approx"])
    for df in (latest, log):
        if "target" not in df:
            df["target"] = cfg.target
        if "sigma_kind" not in df:
            df["sigma_kind"] = "rv20"
    upd, run = _read_json("update_status.json"), _read_json("run_status.json")

    try:
        frozen_html, frozen_images = frozen_report.section(cfg)
    except Exception as exc:                                     # the optional section must never break the daily e-mail
        frozen_html, frozen_images = "", {}
        notes_extra = f"Frozen-model section skipped: {exc}"
    else:
        notes_extra = None

    targets = [cfg.target] + [t for t in latest["target"].unique() if t != cfg.target]
    targets = [t for t in targets if t in set(latest["target"])]
    try:
        signal_html = signals.block(cfg, latest, targets)
    except Exception as exc:                                     # optional block: never break the e-mail
        signal_html = ""
        notes_extra = ((notes_extra + "; ") if notes_extra else "") + f"Signal summary skipped: {exc}"
    sections, images, plain_parts, headline = [], {}, [], []
    for t in targets:
        price = inputs.prices[t].dropna()
        price = price.where(price > 0).dropna()
        cur = rp.baseline_log(latest[latest["target"] == t], price)
        asof = cur["asof_date"].max()
        cur = cur[cur["asof_date"] == asof]
        p0 = float(cur["price_now"].iloc[0])
        cur_tab, folded = rp.fold_no_change(cur)                 # models that forecast exactly "no change" go in that column
        live_models = set(cur["model"])                           # models that produced today's forecast; retired ones stay in the log only
        tlog = rp.baseline_log(log[(log["target"] == t) & log["model"].isin(live_models)], price)
        tlog_tab = tlog[~tlog["model"].isin(folded)]
        cid = f"chart_{t}"
        images[cid] = rp.make_chart(price, cur, N_CHART_DAYS, rp.target_label(t))
        unc = rp.uncertainty_table(cur)
        sc = rp.scorecard(tlog, np.log(price))
        if sc is not None:
            sc = sc[~sc["model"].isin(folded)]
        sc_html = rp.scorecard_html(sc)
        fold_note = ("" if not folded else
                     f"<p style='font-size:11.5px;color:#777;margin:4px 0 0'>{', '.join(rp.label(x) for x in folded)} keep no variables "
                     f"today, so their forecast is exactly no change (see the No-change column); they are left out of the other tables below.</p>")
        sections.append(f"""<h2 style="margin-bottom:2px;border-top:2px solid #333;padding-top:10px">{rp.target_label(t)}
<span style="font-weight:normal;font-size:14px;color:#555">({t}) as of {asof.date()}: <b>{p0:,.2f}</b> USD/bbl</span></h2>
<h3>Forecasts from today's data</h3>
<p style="font-size:11.5px;color:#777;margin:0 0 4px">Price (change vs no change). Grey = same as no change.</p>{rp.forecast_table(cur_tab, folded)}{fold_note}
<h3>Uncertainty: +-1 sd range of the price</h3>{unc}
<h3>Last {N_CHART_DAYS} trading days and forecasts</h3><img src="cid:{cid}" style="max-width:100%">
<h3>Last {N_TABLE_DAYS} days: actual vs the 1-day-ahead forecast made the day before</h3>
{rp.last_days_table(tlog_tab, price, N_TABLE_DAYS, horizon=1)}
<p style="font-size:11px;color:#777">Cell = forecast made the previous trading day (actual / forecast - 1 in brackets).</p>
<h3>Track record: RMSE vs no change (below 1.000 = better than no change)</h3>{sc_html}""")
        c = cur.pivot_table(index="horizon", columns="model", values="forecast_price").round(2)
        plain_parts.append(f"{t} as of {asof.date()}: {p0:,.2f}\n{c.to_string()}")
        if t == cfg.target:
            hmax = int(cur["horizon"].max())
            xg = cur[(cur["model"] == "XGB") & (cur["horizon"] == hmax)]
            headline = [f"{t} {p0:,.2f}"] + ([f"XGBoost +{hmax}d {xg['forecast_price'].iloc[0]:,.1f}"] if len(xg) else [])
            asof_main = asof

    notes = [f"Forecasts are expected log-returns turned into prices. \"No change\" is the random-walk benchmark; "
             f"futures series are continuous front-month (contract rolls appear as small jumps). "
             f"The ARMA-GARCH model forecasts almost no change; its value is the volatility band."]
    if upd.get("failed_series"):
        notes.append(f"<b style='color:#b00'>Data warning:</b> these series failed to download today (older values kept): "
                     f"{', '.join(upd['failed_series'])}")
    for f in run.get("failed_plugins", []):
        notes.append(f"<b style='color:#b00'>Model failed:</b> {f['plugin']}: {f['error']}")
    for s_ in run.get("stale_plugins", []):
        notes.append(f"<b style='color:#b00'>Model not retrained for {s_['age_days']} days:</b> {s_['plugin']}")
    trained = latest.groupby("plugin")["model_train_end"].first().to_dict()
    notes.append("Models last trained on data through: " + ", ".join(f"{k} ({v})" for k, v in trained.items()) + ". "
                 "They are retrained quarterly; each day only the new data is fed through.")
    if upd.get("last_dates", {}).get("eia_wti_futures.csv", "9999") < "2025-01-01":
        notes.append("EIA's futures-curve files stopped in April 2024, so the curve-slope feature is unavailable.")
    if notes_extra:
        notes.append(notes_extra)
    note_html = "".join(f"<p style='font-size:12px;color:#555;margin:4px 0'>{n}</p>" for n in notes)

    images.update(frozen_images)
    html = f"""<html><head>{rp.CSS}</head><body style="font-family:Arial,sans-serif;max-width:920px">
<h1 style="margin-bottom:0">Oil forecast report</h1>
{signal_html}
{''.join(sections)}
{frozen_html}
<h2 style="border-top:2px solid #333;padding-top:10px">Market data, last {N_TABLE_DAYS} trading days (USD/bbl)</h2>{market_table()}
<p style="font-size:11px;color:#777">n/a = FRED/EIA spot prices are published with a delay of several days.</p>
<hr>{note_html}</body></html>"""
    plain = "\n\n".join(plain_parts)
    return f"Oil forecast {asof_main.date()}: " + " | ".join(headline), plain, html, images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    subject, plain, html, images = build(cfg)
    msg = mailer.build_message(subject, plain, html, images)
    if args.dry_run:
        out = ROOT / "email_preview"
        out.mkdir(exist_ok=True)
        preview = html
        for cid, png in images.items():
            (out / f"{cid}.png").write_bytes(png)
            preview = preview.replace(f"cid:{cid}", f"{cid}.png")
        (out / "preview.html").write_text(preview)
        (out / "message.eml").write_bytes(bytes(msg))
        print(f"Dry run: wrote {out}/ (preview.html, {len(images)} charts, message.eml). Nothing sent.\n"
              f"To: {msg['To']}  Cc: {msg.get('Cc')}\nSubject: {msg['Subject']}")
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
