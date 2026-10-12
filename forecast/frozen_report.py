"""E-mail section for the frozen-model evaluation (reads Outputs/frozen_eval/*, written by scripts/frozen_eval.py)."""
import json

import pandas as pd

from . import reporting as rp
from .config import OUTPUT_DIR

FOLDER = OUTPUT_DIR / "frozen_eval"


def _grey(t):
    return f"<span style='color:#999'>{t}</span>"


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


def section(cfg):
    """Returns (html, {cid: png_bytes}); empty when the evaluation has not been run."""
    sp = FOLDER / "summary.json"
    if not sp.exists():
        return "", {}
    s = json.loads(sp.read_text())
    m = pd.read_csv(FOLDER / "metrics.csv")
    tr = pd.read_csv(FOLDER / "trading.csv")
    H = int(s.get("trade_horizon_chart", cfg.frozen_trade_horizon))
    CH = int(s.get("chart_horizon", cfg.frozen_chart_horizon))
    images, parts = {}, []
    targets = [cfg.target] + [t for t in m["target"].unique() if t != cfg.target]
    for t in [x for x in targets if x in set(m["target"])]:
        mt, tt = m[m["target"] == t], tr[tr["target"] == t]
        models = [x for x in rp.order_models(mt["model"].unique()) if x != "RW"]
        trade_models = [x for x in models if x in set(tt["model"])]
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
        if len(cov_models) > 1 or "ARMA-GARCH" in cov_models:
            block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Share of outcomes inside the +-1 sd band "
                      "(68% if well calibrated):</p>" + _coverage_table(mt, cov_models))
        block += (f"<p style='font-size:12px;color:#555;margin:10px 0 2px'>Economic value: long or short on the +{H}-day "
                  f"forecast (flat if it is within {cfg.trade_band:g} sd of zero), {cfg.cost_bps:g} bp per unit traded, "
                  f"same dates:</p>" + _trade_table(tt, H, trade_models))
        hz = [h for h in (1, 3, 5, 20, 60) if h in set(tt["horizon"])]
        block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Sharpe ratio of the same rule at other "
                  "horizons:</p>" + _sharpe_matrix(tt, trade_models, hz))
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
    return intro + "".join(parts), images
