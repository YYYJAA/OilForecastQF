"""Tables and charts for the daily e-mail, built from Outputs/forecast_log.csv and the price data.

Design: every model reports, for one target series, the expected log return over h trading days and a
1-sd uncertainty. Models are compared on the same series, so the e-mail has one panel per target series.
  * Forecast table  : price per model and horizon, with the change vs "no change" (grey when ~0).
  * Uncertainty     : the no-change band (recent volatility) next to model-based bands (GARCH).
  * Chart           : actual prices + forecasts; models equal to no-change are drawn as ONE labelled line.
  * Track record    : RMSE vs no-change, and how often outcomes fell inside the +-1 sd band.
A "No change" benchmark is always available (added from the price series if no plugin supplies it).
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["RW", "LASSO", "LASSO_1se", "PCA_reg", "TPCA", "LASSO_RS", "Combo_EW", "ARMA-GARCH"]
LABELS = {"RW": "No change", "LASSO": "LASSO", "LASSO_1se": "LASSO 1-SE", "PCA_reg": "PCA reg.",
          "TPCA": "Targeted PCA", "LASSO_RS": "LASSO regime", "Combo_EW": "Combo (equal wt)",
          "ARMA-GARCH": "ARMA-GARCH"}
TARGET_LABELS = {"brent_fut": "Brent futures", "wti_fut": "WTI futures"}
CSS_TH = "padding:4px 8px;border-bottom:2px solid #444;text-align:right;font-weight:600"
CSS_TD = "padding:4px 8px;border-bottom:1px solid #ddd;text-align:right"
SAME = 0.001            # table: |forecast / no-change - 1| below this is greyed out as "same as no change"
CHART_SAME = 0.0025     # chart: models within this of no change are merged into the no-change line


def order_models(models):
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def label(m):
    return LABELS.get(m, m)


def target_label(t):
    return TARGET_LABELS.get(t, t)


def _html_table(header, rows, first_left=True):
    th = "".join(f"<th style='{CSS_TH}{';text-align:left' if (i == 0 and first_left) else ''}'>{h}</th>"
                 for i, h in enumerate(header))
    body = ""
    for r in rows:
        tds = "".join(f"<td style='{CSS_TD}{';text-align:left' if (i == 0 and first_left) else ''}'>{c}</td>"
                      for i, c in enumerate(r))
        body += f"<tr>{tds}</tr>"
    return (f"<table style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:12.5px'>"
            f"<tr>{th}</tr>{body}</table>")


def _pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{100 * x:+.{d}f}%"


def _grey(txt):
    return f"<span style='color:#999'>{txt}</span>"


# --------------------------------------------------------------------------- baseline
def _naive_sigma(price, h):
    r = np.log(price.dropna().where(lambda s: s > 0)).diff().dropna()
    return float(r.tail(20).std() * np.sqrt(h))


def baseline_log(log, price):
    """Add 'No change' rows (forecast return 0, recent-volatility sigma) where a plugin did not supply them."""
    have = set(zip(log["asof_date"], log["horizon"])) if "RW" in log["model"].values else set()
    log = log.copy()
    keys = log[["asof_date", "horizon", "target", "price_now", "target_date_approx"]].drop_duplicates(["asof_date", "horizon"])
    add = keys[[(a, h) not in have for a, h in zip(keys["asof_date"], keys["horizon"])]]
    if add.empty:
        return log
    px = price.dropna()
    ret = np.log(px.where(px > 0)).diff()
    vol = ret.rolling(20).std()
    rows = pd.DataFrame({"asof_date": add["asof_date"], "target": add["target"], "model": "RW", "plugin": "baseline",
                         "horizon": add["horizon"], "forecast_return": 0.0,
                         "sigma": add["asof_date"].map(vol).to_numpy() * np.sqrt(add["horizon"].to_numpy()),
                         "sigma_kind": "rv20", "price_now": add["price_now"], "forecast_price": add["price_now"],
                         "target_date_approx": add["target_date_approx"], "source": "live"})
    return pd.concat([log, rows], ignore_index=True)


# --------------------------------------------------------------------------- tables
def forecast_table(cur):
    models = order_models(cur["model"].unique())
    rows = []
    for h, g in cur.groupby("horizon"):
        g = g.set_index("model")
        cells = [f"+{h}d", g["target_date_approx"].iloc[0].strftime("%Y-%m-%d")]
        rw = g.loc["RW", "forecast_price"] if "RW" in g.index else np.nan
        for m in models:
            if m not in g.index:
                cells.append(_grey("n/a")); continue
            r = g.loc[m]
            d = r["forecast_price"] / rw - 1 if pd.notna(rw) else np.nan
            txt = f"{r['forecast_price']:.2f} ({_pct(d)})" if m != "RW" else f"{r['forecast_price']:.2f}"
            cells.append(_grey(txt) if (m != "RW" and abs(d) < SAME) else txt)
        rows.append(cells)
    return _html_table(["Horizon", "Target date"] + [label(m) for m in models], rows)


def uncertainty_table(cur):
    """+-1 sd range of the price at each horizon: recent-volatility band vs model-based bands."""
    models = [m for m in order_models(cur["model"].unique())
              if m != "RW" and (cur[cur["model"] == m]["sigma_kind"] != "rv20").any()]
    rows = []
    for h, g in cur.groupby("horizon"):
        g = g.set_index("model")
        if "RW" not in g.index:
            continue
        p0, s0 = g.loc["RW", "price_now"], g.loc["RW", "sigma"]
        cells = [f"+{h}d", f"{p0 * np.exp(-s0):.1f} - {p0 * np.exp(s0):.1f} ({100 * s0:.1f}%)"]
        for m in models:
            if m in g.index and pd.notna(g.loc[m, "sigma"]):
                s, f = g.loc[m, "sigma"], g.loc[m, "forecast_price"]
                cells.append(f"{f * np.exp(-s):.1f} - {f * np.exp(s):.1f} ({100 * s:.1f}%, x{s / s0:.2f})")
            else:
                cells.append(_grey("n/a"))
        rows.append(cells)
    if not rows:
        return ""
    return _html_table(["Horizon", "Recent volatility (20d)"] + [f"{label(m)} (model)" for m in models], rows)


def last_days_table(log, price, n=7, horizon=1):
    """For the last n trading days: actual price and each model's forecast made `horizon` days earlier."""
    days = price.dropna().index
    pos = {d: i for i, d in enumerate(days)}
    models = order_models(log["model"].unique())
    sub = log[log["horizon"] == horizon]
    rows = []
    for d in days[-n:][::-1]:
        i = pos[d]
        origin = days[i - horizon] if i - horizon >= 0 else None
        cells = [d.strftime("%Y-%m-%d"), f"{price.loc[d]:.2f}"]
        for m in models:
            r = sub[(sub["asof_date"] == origin) & (sub["model"] == m)] if origin is not None else sub.iloc[0:0]
            if len(r):
                f = float(r["forecast_price"].iloc[0])
                cells.append(f"{f:.2f} {_grey('(' + _pct(price.loc[d] / f - 1) + ')')}")
            else:
                cells.append(_grey("n/a"))
        rows.append(cells)
    return _html_table(["Date", "Actual"] + [label(m) for m in models], rows)


def scorecard(log, logp):
    """Realised accuracy of logged forecasts: RMSE vs no-change, hit-rate and +-1 sd band coverage."""
    days = logp.dropna().index
    pos = pd.Series(np.arange(len(days)), index=days)
    d = log.copy()
    i = d["asof_date"].map(pos)
    j = i + d["horizon"]
    ok = i.notna() & (j < len(days))
    d = d[ok].copy()
    if d.empty:
        return None
    d["actual"] = logp.dropna().to_numpy()[j[ok].astype(int)] - logp.dropna().to_numpy()[i[ok].astype(int)]
    d["err"] = d["actual"] - d["forecast_return"]
    d["inside"] = d["err"].abs() <= d["sigma"]
    rows = []
    for (h, m), g in d.groupby(["horizon", "model"]):
        rw = d[(d["horizon"] == h) & (d["model"] == "RW")]
        rmse = np.sqrt((g["err"] ** 2).mean())
        rmse_rw = np.sqrt((rw["actual"] ** 2).mean()) if len(rw) else np.nan
        rows.append({"horizon": h, "model": m, "n": len(g), "RMSE": rmse,
                     "ratio_vs_RW": rmse / rmse_rw if rmse_rw else np.nan,
                     "coverage": float(g["inside"].mean()) if g["sigma"].notna().any() else np.nan,
                     "sigma_kind": g["sigma_kind"].iloc[0] if "sigma_kind" in g else "rv20"})
    return pd.DataFrame(rows)


def scorecard_html(sc, min_n=10):
    if sc is None or sc.empty:
        return "<p style='font-size:12px;color:#777'>Track record: no forecast has a realised outcome yet.</p>"
    models = [m for m in order_models(sc["model"].unique()) if m != "RW"]
    rows = []
    for h in sorted(sc["horizon"].unique()):
        cells = [f"+{h}d"]
        for m in models:
            r = sc[(sc["horizon"] == h) & (sc["model"] == m)]
            cells.append(_grey("n/a") if r.empty or r["n"].iloc[0] < min_n else f"{r['ratio_vs_RW'].iloc[0]:.3f}")
        cells.append(str(int(sc[sc["horizon"] == h]["n"].max())))
        rows.append(cells)
    out = _html_table(["Horizon"] + [label(m) for m in models] + ["n"], rows)
    # band calibration: recent-volatility band (No change) vs model-based bands
    cov_models = ["RW"] + [m for m in models if (sc[sc["model"] == m]["sigma_kind"] != "rv20").any()]
    if len(cov_models) > 1:
        crow = []
        for h in sorted(sc["horizon"].unique()):
            cells = [f"+{h}d"]
            for m in cov_models:
                r = sc[(sc["horizon"] == h) & (sc["model"] == m)]
                cells.append(_grey("n/a") if r.empty or r["n"].iloc[0] < min_n or pd.isna(r["coverage"].iloc[0])
                             else f"{100 * r['coverage'].iloc[0]:.0f}%")
            crow.append(cells)
        out += ("<p style='font-size:12px;color:#555;margin:10px 0 4px'>Share of outcomes inside the +-1 sd band "
                "(about 68% if the band is well calibrated):</p>"
                + _html_table(["Horizon"] + [("Recent volatility" if m == "RW" else label(m)) for m in cov_models], crow))
    return out


# --------------------------------------------------------------------------- chart
def make_chart(price, cur, n_days=60, title_name=""):
    """Last n_days of actual prices, then forecasts out to the longest horizon with +-1 sd bands.
    Models whose forecast equals no-change (within CHART_SAME) at every horizon are drawn as ONE labelled line."""
    asof = cur["asof_date"].max()
    cur = cur[cur["asof_date"] == asof]
    hist = price.dropna().loc[:asof].iloc[-n_days:]
    p0 = float(hist.iloc[-1])
    rw = cur[cur["model"] == "RW"].sort_values("horizon")
    rw_price = rw.set_index("horizon")["forecast_price"]
    models = [m for m in order_models(cur["model"].unique()) if m != "RW"]
    same = [m for m in models
            if (np.abs(cur[cur["model"] == m].set_index("horizon")["forecast_price"] / rw_price - 1).max() < CHART_SAME)]
    distinct = [m for m in models if m not in same]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(hist.index, hist.values, color="black", lw=2, label="Actual")
    xs = [asof] + list(rw["target_date_approx"])
    ax.axvspan(asof, xs[-1], color="#f5f5f5", zorder=0)
    ax.text(xs[-1], ax.get_ylim()[0], "forecast  ", ha="right", va="bottom", fontsize=8, color="#888")
    s0 = rw["sigma"].to_numpy()
    ax.plot(xs, [p0] + list(p0 * np.exp(s0)), color="grey", lw=0.9, ls=":", label="Recent-volatility +-1 sd")
    ax.plot(xs, [p0] + list(p0 * np.exp(-s0)), color="grey", lw=0.9, ls=":")
    cmap, ci = plt.get_cmap("tab10"), 0
    for m in models:                                             # model-based uncertainty bands (e.g. GARCH)
        g = cur[cur["model"] == m].sort_values("horizon")
        if (g["sigma_kind"] != "rv20").any() and g["sigma"].notna().any():
            f, sg = g["forecast_price"].to_numpy(), g["sigma"].to_numpy()
            ax.fill_between([asof] + list(g["target_date_approx"]), [p0] + list(f * np.exp(-sg)), [p0] + list(f * np.exp(sg)),
                            color="tab:blue", alpha=0.13, label=f"{label(m)} +-1 sd")
    ax.plot(xs, [p0] * len(xs), color="dimgrey", ls="--", lw=1.2, label="No change (current price)")

    def draw(prices, name, color, lw=1.7):
        ax.plot(xs, [p0] + list(prices), marker="o", ms=4.5, lw=lw, color=color, label=name, zorder=3)
        ax.annotate(f"{prices[-1]:.1f}", (xs[-1], prices[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")

    if same:                                                      # models that say "no change": still drawn, as ONE line
        px = np.mean([cur[cur["model"] == m].sort_values("horizon")["forecast_price"].to_numpy() for m in same], axis=0)
        draw(px, f"{', '.join(label(m) for m in same)} (about no change)", "#008080", lw=2.4)
    for m in distinct:
        g = cur[cur["model"] == m].sort_values("horizon")
        draw(g["forecast_price"].to_numpy(), label(m), cmap(ci % 10)); ci += 1
    ax.axvline(asof, color="k", lw=0.8, ls=":")
    ax.set_ylabel("Price (USD/bbl)")
    ax.set_title(f"{title_name}: last {n_days} trading days and forecasts to +{int(cur['horizon'].max())} days "
                 f"(as of {asof.date()})", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(hist.index[0], xs[-1] + pd.Timedelta(days=10))
    ax.legend(fontsize=7.5, ncol=2, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    return buf.getvalue()
