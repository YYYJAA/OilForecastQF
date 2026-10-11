"""Tables and chart for the daily e-mail, built from Outputs/forecast_log.csv and the price data."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["RW", "LASSO", "LASSO_1se", "PCA_reg", "TPCA", "LASSO_RS", "Combo_EW"]
LABELS = {"RW": "No change", "LASSO": "LASSO", "LASSO_1se": "LASSO 1-SE", "PCA_reg": "PCA reg.",
          "TPCA": "Targeted PCA", "LASSO_RS": "LASSO regime", "Combo_EW": "Combo (equal wt)"}
CSS_TH = "padding:4px 8px;border-bottom:2px solid #444;text-align:right;font-weight:600"
CSS_TD = "padding:4px 8px;border-bottom:1px solid #ddd;text-align:right"


def order_models(models):
    known = [m for m in MODEL_ORDER if m in models]
    return known + sorted(m for m in models if m not in MODEL_ORDER)


def label(m):
    return LABELS.get(m, m)


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


def _pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:+.1f}%"


def forecast_table(latest):
    """Rows = horizon; columns = models. Cell = forecast price (change vs no-change)."""
    models = order_models(latest["model"].unique())
    asof = latest["asof_date"].max()
    cur = latest[latest["asof_date"] == asof]
    p0 = cur["price_now"].iloc[0]
    rows = []
    for h, g in cur.groupby("horizon"):
        g = g.set_index("model")
        cells = [f"+{h}d", g["target_date_approx"].iloc[0].strftime("%Y-%m-%d")]
        for m in models:
            if m in g.index:
                r = g.loc[m]
                cells.append(f"{r['forecast_price']:.2f} <span style='color:#777'>({_pct(np.exp(r['forecast_return']) - 1)})</span>")
            else:
                cells.append("n/a")
        rows.append(cells)
    return _html_table(["Horizon", "Target date"] + [label(m) for m in models], rows), asof, p0


def last_days_table(log, price, n=7, horizon=1):
    """For the last n trading days: actual price and each model's forecast made `horizon` days earlier."""
    days = price.dropna().index
    pos = {d: i for i, d in enumerate(days)}
    models = order_models(log["model"].unique())
    sub = log[log["horizon"] == horizon]
    rows = []
    for d in days[-n:][::-1]:
        i = pos[d]
        cells = [d.strftime("%Y-%m-%d"), f"{price.loc[d]:.2f}"]
        origin = days[i - horizon] if i - horizon >= 0 else None
        for m in models:
            r = sub[(sub["asof_date"] == origin) & (sub["model"] == m)] if origin is not None else sub.iloc[0:0]
            if len(r):
                f = float(r["forecast_price"].iloc[0])
                err = price.loc[d] / f - 1
                cells.append(f"{f:.2f} <span style='color:#777'>({_pct(err)})</span>")
            else:
                cells.append("<span style='color:#aaa'>n/a</span>")
        rows.append(cells)
    return _html_table(["Date", "Actual"] + [f"{label(m)}" for m in models], rows)


def scorecard(log, logp):
    """Realised accuracy of logged forecasts: RMSE ratio vs no-change, hit-rate, by model and horizon."""
    days = logp.dropna().index
    pos = pd.Series(np.arange(len(days)), index=days)
    d = log.copy()
    i = d["asof_date"].map(pos)
    j = i + d["horizon"]
    ok = i.notna() & (j < len(days))
    d = d[ok].copy()
    if d.empty:
        return None
    d["actual"] = logp.to_numpy()[j[ok].astype(int)] - logp.to_numpy()[i[ok].astype(int)]
    d["err"] = d["actual"] - d["forecast_return"]
    rows = []
    for (h, m), g in d.groupby(["horizon", "model"]):
        rw = d[(d["horizon"] == h) & (d["model"] == "RW")]
        rmse = np.sqrt((g["err"] ** 2).mean())
        rmse_rw = np.sqrt((rw["actual"] ** 2).mean()) if len(rw) else np.nan
        rows.append({"horizon": h, "model": m, "n": len(g), "RMSE": rmse,
                     "ratio_vs_RW": rmse / rmse_rw if rmse_rw else np.nan,
                     "hit_rate": float(((g["forecast_return"] > 0) == (g["actual"] > 0)).mean())
                     if (g["forecast_return"] != 0).any() else np.nan})
    return pd.DataFrame(rows)


def scorecard_html(sc, min_n=10):
    if sc is None or sc.empty:
        return "<p style='font-size:12px;color:#777'>Scorecard: no forecast has a realised outcome yet.</p>"
    models = [m for m in order_models(sc["model"].unique()) if m != "RW"]
    rows = []
    for h in sorted(sc["horizon"].unique()):
        cells = [f"+{h}d"]
        n = int(sc[sc["horizon"] == h]["n"].max())
        for m in models:
            r = sc[(sc["horizon"] == h) & (sc["model"] == m)]
            cells.append("n/a" if r.empty or r["n"].iloc[0] < min_n else f"{r['ratio_vs_RW'].iloc[0]:.3f}")
        cells.append(str(n))
        rows.append(cells)
    return _html_table(["Horizon"] + [label(m) for m in models] + ["n"], rows)


def make_chart(price, latest, n_days=60):
    """Last n_days of actual prices, then forecasts out to the longest horizon, with a +-1 sd band."""
    asof = latest["asof_date"].max()
    cur = latest[latest["asof_date"] == asof]
    hist = price.dropna().loc[:asof].iloc[-n_days:]
    p0 = float(hist.iloc[-1])
    models = order_models(cur["model"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(hist.index, hist.values, color="black", lw=2, label="Actual")
    band = cur[cur["model"] == "RW"].sort_values("horizon") if "RW" in set(cur["model"]) else cur[cur["model"] == models[0]].sort_values("horizon")
    if band["sigma"].notna().any():
        xs = [asof] + list(band["target_date_approx"])
        lo = [p0] + list(p0 * np.exp(-band["sigma"].to_numpy()))
        hi = [p0] + list(p0 * np.exp(band["sigma"].to_numpy()))
        ax.fill_between(xs, lo, hi, color="grey", alpha=0.18, label="no-change +-1 sd (recent volatility)")
    ci = 0
    for m in models:
        g = cur[cur["model"] == m].sort_values("horizon")
        xs = [asof] + list(g["target_date_approx"])
        ys = [p0] + list(g["forecast_price"])
        if m == "RW":
            ax.plot(xs, ys, color="grey", ls="--", lw=1.2, label=label(m))
        else:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=cmap(ci % 10), label=label(m)); ci += 1
    ax.axvline(asof, color="k", lw=0.8, ls=":")
    ax.set_ylabel("Price (USD/bbl)")
    ax.set_title(f"Last {n_days} trading days and forecasts to +{int(cur['horizon'].max())} days (as of {asof.date()})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(fontsize=8, ncol=3, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    return buf.getvalue()
