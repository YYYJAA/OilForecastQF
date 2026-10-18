"""Outputs of the frozen-model evaluation: the project report and registration tables, plus the e-mail section.

PROJECT outputs:

  Outputs/frozen_eval/report.html                 standalone report: tables + charts (open it in a browser)
  Outputs/frozen_eval/registration_accuracy.csv   one row per model: accuracy across horizons (spreadsheet-ready)
  Outputs/frozen_eval/registration_robustness.csv one row per model and horizon: regimes, years, windows
  Outputs/frozen_eval/registration_summary.md     the same two tables as markdown, plus the test design

Written by scripts/frozen_eval.py after the metrics are computed.
"""
import base64
import json

import numpy as np
import pandas as pd

from . import reporting as rp
from .config import OUTPUT_DIR

FOLDER = OUTPUT_DIR / "frozen_eval"
HEADLINE = (1, 5, 20, 60)                       # horizons shown in the compact registration tables
BENCH = ("HistMean", "AR")


def _grey(t):
    return f"<span style='color:#999'>{t}</span>"


def _f(x, fmt="{:.3f}", na="n/a"):
    return na if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


# --------------------------------------------------------------------------- html tables for the report
def _matrix(m, models, col, fmt, horizons=None, star=False):
    rows = []
    hs = horizons or sorted(m["horizon"].unique())
    for h in hs:
        cells = [f"+{h}d"]
        for mod in models:
            r = m[(m["horizon"] == h) & (m["model"] == mod)]
            if r.empty or pd.isna(r[col].iloc[0]):
                cells.append(_grey("no call" if col in ("dir_acc", "f1_up", "f1_down") else "n/a")); continue
            v = r[col].iloc[0]
            txt = fmt.format(v)
            if star and pd.notna(r["dm_p_vs_no_change"].iloc[0]) and r["dm_p_vs_no_change"].iloc[0] < 0.10:
                txt += "*"
            cells.append(txt)
        rows.append(cells)
    return rp._html_table(["Horizon"] + [rp.label(x) for x in models], rows)


def _robust_html(rob, t, models, horizons=HEADLINE):
    out = ""
    for kind, title in (("regime", "By volatility regime at the forecast origin"), ("year", "By calendar year"),
                        ("window", "By consecutive window of the test period")):
        r = rob[(rob["target"] == t) & (rob["kind"] == kind)]
        if r.empty:
            continue
        slices = list(dict.fromkeys(r.sort_values("slice")["slice"]))
        out += f"<h4 style='margin:12px 0 2px'>{title} (RMSE / RMSE of no change)</h4>"
        for h in horizons:
            rows = []
            for mod in models:
                cells = [rp.label(mod)]
                for sl in slices:
                    x = r[(r["horizon"] == h) & (r["model"] == mod) & (r["slice"] == sl)]
                    cells.append(_grey("n/a") if x.empty or pd.isna(x["rmse_ratio"].iloc[0]) else f"{x['rmse_ratio'].iloc[0]:.3f}")
                rows.append(cells)
            out += f"<p style='font-size:12px;color:#555;margin:6px 0 2px'>Horizon +{h}d</p>" + \
                   rp._html_table(["Model"] + slices, rows)
    return out


def _trade_html(tr, H, models, cfg):
    rows = []
    for mod in models + ["Buy & hold"]:
        hh = 0 if mod == "Buy & hold" else H
        r = tr[(tr["model"] == mod) & (tr["horizon"] == hh)]
        if r.empty:
            continue
        r = r.iloc[0]
        flat = r["days_in_market"] == 0
        rows.append([rp.label(mod) if mod != "Buy & hold" else mod,
                     _grey("never trades") if flat else f"{100 * r['ann_return']:+.1f}%",
                     "" if flat else f"{100 * r['ann_vol']:.0f}%", "" if flat else f"{r['sharpe']:.2f}",
                     "" if flat else f"{100 * r['max_drawdown']:.0f}%", "" if flat else f"{r['turnover']:.1f}",
                     "" if flat else f"{100 * r['total_cost']:.2f}%"])
    return rp._html_table(["Strategy", "Annualised return", "Volatility", "Sharpe", "Max drawdown", "Turnover",
                           "Costs paid"], rows)


def _sharpe_html(tr, models, horizons):
    rows = []
    bh = tr[tr["model"] == "Buy & hold"]
    for mod in models:
        cells = [rp.label(mod)]
        for h in horizons:
            r = tr[(tr["model"] == mod) & (tr["horizon"] == h)]
            cells.append(_grey("flat") if r.empty or r["days_in_market"].iloc[0] == 0 else f"{r['sharpe'].iloc[0]:.2f}")
        rows.append(cells)
    if len(bh):
        rows.append(["Buy & hold"] + [f"{bh['sharpe'].iloc[0]:.2f}"] * len(horizons))
    return rp._html_table(["Sharpe ratio"] + [f"+{h}d" for h in horizons], rows)


def _img(path):
    return (f"<img style='max-width:100%;margin:6px 0' src='data:image/png;base64,"
            f"{base64.b64encode(path.read_bytes()).decode()}'>") if path.exists() else ""


# --------------------------------------------------------------------------- registration tables
def registration_tables(m, rob, summary):
    period = f"{summary['first_origin']} to {summary['last_origin']}"
    acc, rb = [], []
    for t in m["target"].unique():
        mt = m[m["target"] == t]
        order = [x for x in rp.order_models(mt["model"].unique())]
        rows = ["No change"] + [rp.label(x) for x in order if x != "RW"]
        for mod_label in rows:
            mod = "RW" if mod_label == "No change" else next(k for k in order if rp.label(k) == mod_label)
            row = {"Series": rp.target_label(t), "Model": mod_label,
                   "Type": "benchmark" if mod in ("RW",) + BENCH else "model", "Test period": period}
            for h in HEADLINE:
                if mod == "RW":
                    r = mt[mt["horizon"] == h].iloc[:1]
                    row.update({f"RMSE vs no change h={h}": 1.0, f"MAPE % h={h}": r["mape_no_change_pct"].iloc[0] if len(r) else np.nan,
                                f"Direction % h={h}": np.nan, f"DM p h={h}": np.nan, f"Band coverage % h={h}": np.nan,
                                f"F1 up h={h}": np.nan})
                    if h == HEADLINE[0]:
                        row["Forecast dates h=1"] = int(r["n"].iloc[0]) if len(r) else np.nan
                    continue
                r = mt[(mt["model"] == mod) & (mt["horizon"] == h)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if h == HEADLINE[0]:
                    row["Forecast dates h=1"] = int(r["n"])
                row.update({f"RMSE vs no change h={h}": r["rmse_ratio"], f"MAPE % h={h}": r["mape_pct"],
                            f"Direction % h={h}": 100 * r["dir_acc"] if pd.notna(r["dir_acc"]) else np.nan,
                            f"DM p h={h}": r["dm_p_vs_no_change"],
                            f"Band coverage % h={h}": 100 * r["coverage_1sd"] if pd.notna(r["coverage_1sd"]) else np.nan,
                            f"F1 up h={h}": r["f1_up"]})
            acc.append(row)
            for h in HEADLINE:
                if mod == "RW":
                    continue
                x = rob[(rob["target"] == t) & (rob["model"] == mod) & (rob["horizon"] == h)]
                if x.empty:
                    continue

                def pick(kind, key=None):
                    y = x[x["kind"] == kind]
                    if key is not None:
                        y = y[y["slice"] == key]
                    return y
                win = pick("window")
                rb.append({"Series": rp.target_label(t), "Model": mod_label, "Horizon (days)": h,
                           "High-volatility RMSE ratio": _first(pick("regime", "high volatility")),
                           "Low-volatility RMSE ratio": _first(pick("regime", "low volatility")),
                           **{f"{yr} RMSE ratio": _first(pick("year", yr)) for yr in sorted(x[x["kind"] == "year"]["slice"].unique())},
                           "Worst window RMSE ratio": win["rmse_ratio"].max() if len(win) else np.nan,
                           "Best window RMSE ratio": win["rmse_ratio"].min() if len(win) else np.nan,
                           "Windows better than no change": f"{int((win['rmse_ratio'] < 1).sum())} of {len(win)}"})
    return pd.DataFrame(acc), pd.DataFrame(rb)


def _first(df):
    return df["rmse_ratio"].iloc[0] if len(df) else np.nan


def _md(df, floatfmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (floatfmt.get(c.split(" h=")[0], "{:.3f}")).format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main entry
def write_outputs(cfg):
    s = json.loads((FOLDER / "summary.json").read_text())
    m = pd.read_csv(FOLDER / "metrics.csv")
    tr = pd.read_csv(FOLDER / "trading.csv")
    rob = pd.read_csv(FOLDER / "robustness.csv")
    acc, rb = registration_tables(m, rob, s)
    acc.to_csv(FOLDER / "registration_accuracy.csv", index=False)
    rb.to_csv(FOLDER / "registration_robustness.csv", index=False)

    fmt = {"RMSE vs no change": "{:.3f}", "MAPE %": "{:.2f}", "Direction %": "{:.0f}", "DM p": "{:.2f}",
           "Band coverage %": "{:.0f}", "F1 up": "{:.2f}", "Forecast dates": "{:.0f}"}
    md = [f"# Out-of-sample performance summary: oil price forecasting\n",
          f"**Test design.** Models were estimated only on data up to {s['cutoff']} and then applied day by day from "
          f"{s['first_origin']} to {s['last_origin']} ({s['n_origins']} forecast dates), with no refitting. Forecasts are "
          f"expected cumulative log returns, converted to prices. Benchmarks: no change (random walk), historical mean, "
          f"direct AR. RMSE vs no change below 1 means better than the no-change forecast; DM p is the Diebold-Mariano "
          f"p-value against no change (HAC, h-1 lags); band coverage is the share of outcomes inside the model's +-1 sd "
          f"band (68% if well calibrated). Window and weight choices come from the notebooks' own 2022-2023 validation; "
          f"some design decisions were made after seeing later data, so the test is clean for estimation but not pristine.\n",
          "## Accuracy by horizon\n", _md(acc, fmt), "\n\n## Robustness (RMSE / RMSE of no change; below 1 is better)\n",
          _md(rb, fmt), "\n"]
    (FOLDER / "registration_summary.md").write_text("\n".join(md))

    # ---- report.html
    H, CH = int(s.get("trade_horizon_chart", 3)), int(s.get("chart_horizon", 3))
    body = [f"<h1 style='margin-bottom:2px'>Oil price forecasting: out-of-sample evaluation</h1>",
            f"<p style='font-size:13px;color:#444;max-width:900px'>Models <b>estimated only on data up to {s['cutoff']}</b> (no "
            f"refitting since), then applied day by day from {s['first_origin']} to {s['last_origin']} "
            f"({s['n_origins']} forecast dates). Benchmarks: no change, historical mean, direct AR (also estimated "
            f"on data up to the cutoff). Window and weight choices come from the notebooks' own 2022-2023 validation; some "
            f"design decisions were made after seeing later data, so the test is clean for estimation but not pristine. "
            f"This is a deliberately stale test model (skill plus ageing). Generated {s['generated_at_utc'][:10]}.</p>"]
    for t in m["target"].unique():
        mt, tt = m[m["target"] == t], tr[tr["target"] == t]
        models = [x for x in rp.order_models(mt["model"].unique()) if x != "RW"]
        trade_models = [x for x in models if x in set(tt["model"]) and x not in BENCH]
        body.append(f"<h2 style='border-top:2px solid #333;padding-top:8px'>{rp.target_label(t)}</h2>")
        body.append(_img(FOLDER / f"rolling_{t}.png"))
        body.append("<h3>Accuracy: RMSE of the cumulative return / RMSE of no change</h3>"
                    "<p style='font-size:12px;color:#555'>Below 1 = better than no change; * = Diebold-Mariano p &lt; 0.10.</p>"
                    + _matrix(mt, models, "rmse_ratio", "{:.3f}", star=True))
        body.append("<h3>Accuracy in price terms: mean absolute percentage error (%)</h3>"
                    "<p style='font-size:12px;color:#555'>No-change MAPE for reference: "
                    + ", ".join(f"+{h}d {mt[mt['horizon'] == h]['mape_no_change_pct'].iloc[0]:.2f}%"
                                for h in sorted(mt["horizon"].unique())) + ".</p>"
                    + _matrix(mt, models, "mape_pct", "{:.2f}"))
        body.append("<h3>Mean absolute error / that of no change</h3>" + _matrix(mt, models, "mae_ratio", "{:.3f}"))
        body.append("<h3>Direction: share correct</h3><p style='font-size:12px;color:#555'>\"no call\" = the model "
                    "forecast exactly no change.</p>" + _matrix(mt, models, "dir_acc", "{:.0%}"))
        body.append("<h3>Up-move F1 score</h3>" + _matrix(mt, models, "f1_up", "{:.2f}")
                    + "<h3>Down-move F1 score</h3>" + _matrix(mt, models, "f1_down", "{:.2f}"))
        cov_models = [x for x in ["ARMA-GARCH", "MS-AR", "MS-ARX"] if x in set(mt["model"])] + (["RW"] if "RW" in set(mt["model"]) else [])
        if cov_models:
            body.append("<h3>Uncertainty bands: share of outcomes inside +-1 sd (68% if calibrated)</h3>"
                        + _matrix(mt, cov_models, "coverage_1sd", "{:.0%}"))
        body.append("<h3>Robustness</h3>" + _robust_html(rob, t, [x for x in models], HEADLINE))
        body.append(f"<h3>Economic value of a simple trading rule</h3><p style='font-size:12px;color:#555'>Long or short on the "
                    f"+{H}-day forecast (flat if within {s['trade_band']:g} sd of zero), overlapping tranches, "
                    f"{s['cost_bps']:g} bp per unit traded, same dates.</p>" + _trade_html(tt, H, trade_models, cfg))
        hz = [h for h in (1, 3, 5, 20, 60) if h in set(tt["horizon"])]
        body.append("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Sharpe ratio at other horizons:</p>"
                    + _sharpe_html(tt, trade_models, hz) + _img(FOLDER / f"equity_{t}.png"))
    html = ("<html><head><meta charset='utf-8'><title>Oil forecasting: out-of-sample evaluation</title>" + rp.CSS + "</head>"
            "<body style='font-family:Arial,sans-serif;max-width:980px;margin:20px auto;padding:0 12px'>"
            + "".join(body) + "</body></html>")
    (FOLDER / "report.html").write_text(html)
    return FOLDER / "report.html"


# --------------------------------------------------------------------------- e-mail section (as sent daily)
def _ratio_table(m, models):
    rows = []
    for h in sorted(m["horizon"].unique()):
        cells = [f"+{h}d"]
        for mod in models:
            r = m[(m["horizon"] == h) & (m["model"] == mod)]
            if r.empty or pd.isna(r["rmse_ratio"].iloc[0]):
                cells.append(_grey("n/a")); continue
            v, p = r["rmse_ratio"].iloc[0], r["dm_p_vs_no_change"].iloc[0]
            star = "*" if pd.notna(p) and p < 0.10 else ""
            cells.append(f"{v:.3f}{star}" if abs(v - 1) > 0.0005 else _grey(f"{v:.3f}"))
        cells.append(str(int(m[m["horizon"] == h]["n"].max())))
        rows.append(cells)
    return rp._html_table(["Horizon"] + [rp.label(x) for x in models] + ["n"], rows)


def _dir_table(m, models):
    rows = []
    for h in sorted(m["horizon"].unique()):
        cells = [f"+{h}d"]
        for mod in models:
            r = m[(m["horizon"] == h) & (m["model"] == mod)]
            if r.empty or pd.isna(r["dir_acc"].iloc[0]):
                cells.append(_grey("no call"))
            else:
                cells.append(f"{100 * r['dir_acc'].iloc[0]:.0f}%")
        rows.append(cells)
    return rp._html_table(["Horizon"] + [rp.label(x) for x in models], rows)


def _trade_table(tr, H, models):
    rows = []
    for mod in models + ["Buy & hold"]:
        hh = 0 if mod == "Buy & hold" else H
        r = tr[(tr["model"] == mod) & (tr["horizon"] == hh)]
        if r.empty:
            continue
        r = r.iloc[0]
        flat = r["days_in_market"] == 0
        rows.append([rp.label(mod) if mod != "Buy & hold" else mod,
                     _grey("never trades") if flat else f"{100 * r['ann_return']:+.1f}%",
                     "" if flat else f"{100 * r['ann_vol']:.0f}%",
                     "" if flat else f"{r['sharpe']:.2f}",
                     "" if flat else f"{100 * r['max_drawdown']:.0f}%",
                     "" if flat else f"{100 * r['days_in_market']:.0f}%",
                     "" if flat else f"{100 * r['total_cost']:.2f}%"])
    return rp._html_table(["Strategy", "Annualised return", "Volatility", "Sharpe", "Max drawdown",
                           "Days in market", "Costs paid"], rows)


def _coverage_table(m, models):
    rows = []
    for h in sorted(m["horizon"].unique()):
        cells = [f"+{h}d"]
        for mod in models:
            r = m[(m["horizon"] == h) & (m["model"] == mod)]
            v = r["coverage_1sd"].iloc[0] if len(r) else float("nan")
            cells.append(_grey("n/a") if pd.isna(v) else f"{100 * v:.0f}%")
        rows.append(cells)
    return rp._html_table(["Horizon"] + [("Recent volatility" if x == "RW" else rp.label(x)) for x in models], rows)


def _sharpe_matrix(tr, models, horizons):
    rows = []
    bh = tr[(tr["model"] == "Buy & hold")]
    for mod in models:
        cells = [rp.label(mod)]
        for h in horizons:
            r = tr[(tr["model"] == mod) & (tr["horizon"] == h)]
            cells.append(_grey("flat") if r.empty or r["days_in_market"].iloc[0] == 0 else f"{r['sharpe'].iloc[0]:.2f}")
        rows.append(cells)
    if len(bh):
        rows.append(["Buy & hold"] + [f"{bh['sharpe'].iloc[0]:.2f}"] * len(horizons))
    return rp._html_table(["Sharpe ratio"] + [f"+{h}d" for h in horizons], rows)


def _robust_compact(rob, t, models, horizons):
    r = rob[rob["target"] == t]
    out = ""
    for h in horizons:
        rows = []
        for mod in models:
            x = r[(r["model"] == mod) & (r["horizon"] == h)]
            if x.empty:
                continue

            def val(kind, key=None):
                y = x[x["kind"] == kind]
                if key is not None:
                    y = y[y["slice"] == key]
                return y["rmse_ratio"].iloc[0] if len(y) else np.nan
            win = x[x["kind"] == "window"]["rmse_ratio"]
            years = sorted(x[x["kind"] == "year"]["slice"].unique())
            cells = [rp.label(mod), _f(val("regime", "high volatility")), _f(val("regime", "low volatility"))]
            cells += [_f(val("year", yr)) for yr in years]
            cells += [_f(win.max()) if len(win) else "n/a", f"{int((win < 1).sum())} of {len(win)}"]
            rows.append(cells)
        years = sorted(r[r["kind"] == "year"]["slice"].unique())
        out += (f"<p style='font-size:12px;color:#555;margin:8px 0 2px'>Horizon +{h}d</p>"
                + rp._html_table(["Model", "High vol", "Low vol"] + years + ["Worst window", "Windows better"], rows))
    return out


FAMILIES = {"LASSO": ["LASSO_1se", "LASSO_RS"]}      # variants that are shown as one column when they behave the same


def _merge_variants(mt, models):
    """{kept model: [variants]} for variants whose results are indistinguishable from the kept model's
    (RMSE ratio within 0.002 and direction within 3 points at every horizon)."""
    merged = {}
    for keep, variants in FAMILIES.items():
        if keep not in models:
            continue
        a = mt[mt["model"] == keep].set_index("horizon")
        same = []
        for v in variants:
            if v not in models:
                continue
            b = mt[mt["model"] == v].set_index("horizon")
            hs = a.index.intersection(b.index)
            d_r = (a.loc[hs, "rmse_ratio"] - b.loc[hs, "rmse_ratio"]).abs().max()
            d_a = (a.loc[hs, "dir_acc"].fillna(0.5) - b.loc[hs, "dir_acc"].fillna(0.5)).abs().max()
            if d_r <= 0.002 and d_a <= 0.03:
                same.append(v)
        if same:
            merged[keep] = same
    return merged


def section(cfg):
    """E-mail section: returns (html, {cid: png_bytes}); empty when the evaluation has not been run."""
    sp = FOLDER / "summary.json"
    if not sp.exists():
        return "", {}
    s = json.loads(sp.read_text())
    m = pd.read_csv(FOLDER / "metrics.csv")
    tr = pd.read_csv(FOLDER / "trading.csv")
    rob = pd.read_csv(FOLDER / "robustness.csv") if (FOLDER / "robustness.csv").exists() else None
    H = int(s.get("trade_horizon_chart", cfg.frozen_trade_horizon))
    CH = int(s.get("chart_horizon", cfg.frozen_chart_horizon))
    images, parts = {}, []
    rp.OVERRIDES.clear()
    targets = [cfg.target] + [t for t in m["target"].unique() if t != cfg.target]
    for t in [x for x in targets if x in set(m["target"])]:
        mt, tt = m[m["target"] == t], tr[tr["target"] == t]
        models = [x for x in rp.order_models(mt["model"].unique()) if x != "RW"]
        merged = _merge_variants(mt, models)
        rp.OVERRIDES.clear()
        for keep, same in merged.items():                       # one column instead of near-identical variants
            rp.OVERRIDES[keep] = (rp.label(keep) + "<br><span style='font-weight:400;font-size:10.5px'>(+ "
                                  + ", ".join(rp.label(v).replace("LASSO ", "") for v in same) + ")</span>")
        models = [x for x in models if x not in {v for vs in merged.values() for v in vs}]
        trade_models = [x for x in models if x in set(tt["model"]) and x not in BENCH]
        cov_models = [x for x in ["RW", "ARMA-GARCH", "MS-AR", "MS-ARX"] if x in set(mt["model"])]
        block = f"<h3>{rp.target_label(t)}</h3>"
        f = FOLDER / f"rolling_{t}.png"
        if f.exists():
            images[f"frozen_rolling_{t}"] = f.read_bytes()
            block += (f"<p style='font-size:12px;color:#555;margin:2px 0'>Rolling {CH}-day-ahead forecast made every day by the "
                      f"frozen models, against what happened:</p><img src='cid:frozen_rolling_{t}' style='max-width:100%'><br>")
        block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Accuracy: RMSE of the cumulative return divided by "
                  "the RMSE of \"no change\" (below 1 = better; * = Diebold-Mariano p &lt; 0.10; grey = same as no change):</p>"
                  + _ratio_table(mt, models))
        block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Direction correct (\"no call\" = the model "
                  "forecast exactly no change):</p>" + _dir_table(mt, models))
        block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Price error: mean absolute percentage error (%); "
                  "no change for reference: " + ", ".join(f"+{h}d {mt[mt['horizon'] == h]['mape_no_change_pct'].iloc[0]:.2f}%"
                                                         for h in sorted(mt['horizon'].unique())) + ":</p>"
                  + _matrix(mt, models, "mape_pct", "{:.2f}"))
        block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Up-move F1 score:</p>"
                  + _matrix(mt, models, "f1_up", "{:.2f}")
                  + "<p style='font-size:12px;color:#555;margin:10px 0 2px'>Down-move F1 score:</p>"
                  + _matrix(mt, models, "f1_down", "{:.2f}"))
        if len(cov_models) > 1 or "ARMA-GARCH" in cov_models:
            block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Share of outcomes inside the +-1 sd band "
                      "(68% if well calibrated):</p>" + _coverage_table(mt, cov_models))
        block += (f"<p style='font-size:12px;color:#555;margin:10px 0 2px'>Economic value: long or short on the +{H}-day "
                  f"forecast (flat if it is within {cfg.trade_band:g} sd of zero), {cfg.cost_bps:g} bp per unit traded, "
                  f"same dates:</p>" + _trade_table(tt, H, trade_models))
        hz = [h for h in (1, 3, 5, 20, 60) if h in set(tt["horizon"])]
        block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Sharpe ratio of the same rule at other "
                  "horizons:</p>" + _sharpe_matrix(tt, trade_models, hz))
        if rob is not None:
            block += ("<p style='font-size:12px;color:#555;margin:12px 0 2px'><b>Robustness</b>: RMSE / RMSE of no change in "
                      "sub-periods of the test (below 1 = better than no change). Volatility regime is judged at the forecast "
                      "date; windows are four consecutive equal slices of the test period.</p>"
                      + _robust_compact(rob, t, models, (1, 5, 20)))
        f = FOLDER / f"equity_{t}.png"
        if f.exists() and t == cfg.target:
            images[f"frozen_equity_{t}"] = f.read_bytes()
            block += f"<img src='cid:frozen_equity_{t}' style='max-width:100%;margin-top:8px'><br>"
        parts.append(block)
    intro = (f"<h2 style='border-top:2px solid #333;padding-top:10px;margin-bottom:2px'>Frozen-model check</h2>"
             f"<p style='font-size:12.5px;color:#444;margin-top:0'>Models <b>estimated only on data up to {s['cutoff']}</b> "
             f"(no refitting since), then applied day by day from {s['first_origin']} to {s['last_origin']} "
             f"({s['n_origins']} forecast dates). Window and weight choices come from the notebooks' own 2022-2023 "
             f"validation; nothing after the cutoff was used to fit or choose. This is a deliberately stale test model, "
             f"not the production model above, so it measures skill plus ageing. Updated monthly "
             f"(last run {s['generated_at_utc'][:10]}).</p>")
    rp.OVERRIDES.clear()
    return intro + "".join(parts), images
