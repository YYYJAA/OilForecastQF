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


def section(cfg):
    """Returns (html, {cid: png_bytes}); empty when the evaluation has not been run."""
    sp = FOLDER / "summary.json"
    if not sp.exists():
        return "", {}
    s = json.loads(sp.read_text())
    m = pd.read_csv(FOLDER / "metrics.csv")
    tr = pd.read_csv(FOLDER / "trading.csv")
    H = int(s.get("trade_horizon_chart", cfg.frozen_trade_horizon))
    images, parts = {}, []
    targets = [cfg.target] + [t for t in m["target"].unique() if t != cfg.target]
    for t in [x for x in targets if x in set(m["target"])]:
        mt, tt = m[m["target"] == t], tr[tr["target"] == t]
        models = [x for x in rp.order_models(mt["model"].unique()) if x != "RW"]
        trade_models = [x for x in models if x in set(tt["model"])]
        block = f"<h3>{rp.target_label(t)}</h3>"
        block += ("<p style='font-size:12px;color:#555;margin:2px 0'>RMSE of the cumulative return divided by the RMSE of "
                  "\"no change\" (below 1 = better; * = Diebold-Mariano p &lt; 0.10; grey = same as no change):</p>"
                  + _ratio_table(mt, models))
        block += ("<p style='font-size:12px;color:#555;margin:10px 0 2px'>Direction correct (\"no call\" = the model "
                  "forecast exactly no change):</p>" + _dir_table(mt, models))
        block += (f"<p style='font-size:12px;color:#555;margin:10px 0 2px'>Economic value: go long or short on the "
                  f"+{H}-day forecast (flat if it is within {cfg.trade_band:g} sd of zero), {cfg.cost_bps:g} bp per unit "
                  f"traded, evaluated on the same dates:</p>" + _trade_table(tt, H, trade_models))
        for kind, cap in (("fan", f"What the frozen models predicted on {s['fan_origin']} vs what happened"),
                          ("aligned", None), ("equity", None)):
            f = FOLDER / f"{kind}_{t}.png"
            if kind != "fan" and t != cfg.target:
                continue
            if f.exists():
                cid = f"frozen_{kind}_{t}"
                images[cid] = f.read_bytes()
                block += f"<img src='cid:{cid}' style='max-width:100%;margin-top:8px'><br>"
        parts.append(block)
    intro = (f"<h2 style='border-top:2px solid #333;padding-top:10px;margin-bottom:2px'>Frozen-model check</h2>"
             f"<p style='font-size:12.5px;color:#444;margin-top:0'>Models <b>estimated only on data up to {s['cutoff']}</b> "
             f"(no refitting since), then applied day by day from {s['first_origin']} to {s['last_origin']} "
             f"({s['n_origins']} forecast dates). Window and weight choices come from the notebooks' own 2022-2023 "
             f"validation; nothing after the cutoff was used to fit or choose. This is a deliberately stale test model, "
             f"not the production model above, so it measures skill plus ageing. Updated monthly "
             f"(last run {s['generated_at_utc'][:10]}).</p>")
    return intro + "".join(parts), images
