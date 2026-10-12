"""Frozen-model evaluation: how would the models have done if they had stopped learning on a fixed date?

1. Every plugin is trained ONLY on data up to cfg.frozen_cutoff (default 2024-12-31): all parameters, penalties and
   regime estimates come from that period. Window / specification choices come from the notebooks' own 2022-2023
   validation (see PIPELINE.md); nothing after the cutoff is used to estimate or choose anything.
2. From the day after the cutoff to today the frozen parameters are applied daily: each origin uses only data known on
   that day (features, regime filter), never refitting.
3. Forecasts are scored against what happened (RMSE vs no change, direction, band coverage, Diebold-Mariano) and turned
   into a simple trading rule (long / short / flat, overlapping tranches, transaction costs) for the economic value.

Writes Outputs/frozen_eval/{forecasts,metrics,trading,fan}.csv, summary.json and PNG charts.

Usage:  python scripts/frozen_eval.py [--cutoff 2024-12-31] [--fan-origin 2026-07-01]
This is NOT the production model: it is a stale model by design (it measures skill plus ageing).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forecast import evaluation as ev  # noqa: E402
from forecast import reporting as rp  # noqa: E402
from forecast.config import OUTPUT_DIR, Config  # noqa: E402
from forecast.data import load_inputs, truncate_inputs  # noqa: E402
from forecast.registry import REGISTRY  # noqa: E402
import forecast.plugins  # noqa: E402,F401

OUT = OUTPUT_DIR / "frozen_eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--fan-origin", default=None)
    args = ap.parse_args()
    cfg = Config()
    cutoff = pd.Timestamp(args.cutoff or cfg.frozen_cutoff)
    fan_origin = pd.Timestamp(args.fan_origin or cfg.frozen_fan_origin)
    OUT.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(cfg)
    train_inputs = truncate_inputs(inputs, cutoff)
    days = inputs.prices[cfg.target].dropna().index
    origins = [d for d in days if d > cutoff]
    if len(origins) < 60:
        raise SystemExit("fewer than 60 origins after the cutoff")
    parts, failed, trained = [], [], {}
    t0 = time.time()
    for name, plugin in REGISTRY.items():
        try:
            art = plugin.train(train_inputs, cfg)                    # parameters from data <= cutoff only
            assert pd.Timestamp(art["train_end"]) <= cutoff, "artifact trained past the cutoff"
            df = plugin.rolling_predict(art, inputs, cfg, origins)   # frozen parameters, daily data
            if "target" not in df:
                df["target"] = cfg.target
            if "sigma_kind" not in df:
                df["sigma_kind"] = "rv20"
            df["plugin"] = name
            parts.append(df)
            trained[name] = art["train_end"]
            print(f"[frozen] {name}: trained through {art['train_end']}, {df['asof_date'].nunique()} origins "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        except Exception as exc:                                     # one plugin failing must not stop the rest
            failed.append({"plugin": name, "error": str(exc)[:300]})
            print(f"[frozen] {name} FAILED: {exc}", file=sys.stderr)
    if not parts:
        return 1
    fc = pd.concat(parts, ignore_index=True)
    fc["asof_date"] = pd.to_datetime(fc["asof_date"])
    fc = fc.drop_duplicates(["asof_date", "target", "plugin", "model", "horizon"]).sort_values(
        ["target", "model", "horizon", "asof_date"])
    prices = {t: inputs.prices[t].dropna() for t in fc["target"].unique()}
    prices = {t: p[p > 0] for t, p in prices.items()}

    metrics = ev.forecast_metrics(fc, prices)
    trading, curves = ev.trading_metrics(fc, prices, cfg.cost_bps, cfg.trade_band)
    fc.to_csv(OUT / "forecasts.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    trading.to_csv(OUT / "trading.csv", index=False)

    # ---- one-origin fan: what the frozen models said on fan_origin vs what happened
    fan_rows = []
    for t, p in prices.items():
        idx = p.index
        o = idx[idx.searchsorted(fan_origin)] if idx.searchsorted(fan_origin) < len(idx) else None
        if o is None:
            continue
        sub = fc[(fc["target"] == t) & (fc["asof_date"] == o)]
        pos = idx.get_loc(o)
        for r in sub.itertuples():
            tgt = idx[pos + r.horizon] if pos + r.horizon < len(idx) else pd.NaT
            fan_rows.append({"target": t, "origin": o, "model": r.model, "horizon": r.horizon, "target_date": tgt,
                             "price_origin": float(p.loc[o]), "forecast_price": float(p.loc[o] * np.exp(r.forecast_return)),
                             "sigma": r.sigma, "actual_price": float(p.iloc[pos + r.horizon]) if pos + r.horizon < len(idx) else np.nan})
    fan = pd.DataFrame(fan_rows)
    fan.to_csv(OUT / "fan.csv", index=False)

    # ---- charts
    cmap = plt.get_cmap("tab10")
    models_all = rp.order_models(fc["model"].unique())
    colour = {m: cmap(k % 10) for k, m in enumerate(models_all)}
    for t, p in prices.items():
        sub = fan[fan["target"] == t]
        if len(sub):
            o = sub["origin"].iloc[0]
            pos = p.index.get_loc(o)
            end = min(pos + 60, len(p) - 1)
            fig, ax = plt.subplots(figsize=(10, 4.6))
            hist = p.iloc[max(0, pos - 40): end + 1]
            ax.plot(hist.index, hist.values, color="black", lw=2, label="Actual")
            ax.axvline(o, color="k", ls=":", lw=0.8)
            for m in models_all:
                g = sub[sub["model"] == m].sort_values("horizon")
                if g.empty:
                    continue
                xs = [o] + [d for d in g["target_date"]]
                ys = [g["price_origin"].iloc[0]] + list(g["forecast_price"])
                ok = [pd.notna(x) for x in xs]
                ax.plot([x for x, k in zip(xs, ok) if k], [y for y, k in zip(ys, ok) if k], marker="o", ms=4,
                        lw=1.6 if m != "RW" else 1.2, ls="--" if m == "RW" else "-", color=colour[m],
                        label=rp.label(m))
            ax.set_title(f"{rp.target_label(t)}: forecasts made on {o.date()} by models frozen at {cutoff.date()}, "
                         f"vs what happened", fontsize=10.5)
            ax.set_ylabel("USD/bbl"); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=3, loc="upper left")
            fig.tight_layout(); fig.savefig(OUT / f"fan_{t}.png", dpi=130); plt.close(fig)

        # forecast made `H` trading days earlier, drawn at the date it predicted
        H = cfg.frozen_trade_horizon
        fig, ax = plt.subplots(figsize=(10, 4.4))
        test = p.loc[p.index > cutoff]
        ax.plot(test.index, test.values, color="black", lw=2, label="Actual")
        pos_all = {d: i for i, d in enumerate(p.index)}
        for m in models_all:
            g = fc[(fc["target"] == t) & (fc["model"] == m) & (fc["horizon"] == H)].sort_values("asof_date")
            if g.empty:
                continue
            tgt = [p.index[pos_all[d] + H] if d in pos_all and pos_all[d] + H < len(p) else pd.NaT for d in g["asof_date"]]
            price_f = [p.loc[d] * np.exp(f) if d in pos_all else np.nan for d, f in zip(g["asof_date"], g["forecast_return"])]
            s = pd.Series(price_f, index=tgt).dropna()
            s = s[s.index.notna()]
            ax.plot(s.index, s.values, lw=1.0 if m != "RW" else 1.2, ls="--" if m == "RW" else "-", color=colour[m],
                    label=rp.label(m), alpha=0.9)
        ax.set_title(f"{rp.target_label(t)}: +{H}-day forecast made {H} trading days earlier, drawn at the date it "
                     f"predicts (models frozen at {cutoff.date()})", fontsize=10)
        ax.set_ylabel("USD/bbl"); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=4, loc="upper left")
        fig.tight_layout(); fig.savefig(OUT / f"aligned_{t}.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.2))
        for (tt, m, h), c in curves.items():
            if tt == t and h in (0, H) and m != "RW":
                ax.plot(c.index, c.values, lw=2.2 if m == "Buy & hold" else 1.2, color="black" if m == "Buy & hold" else colour.get(m, "grey"),
                        ls="--" if m == "Buy & hold" else "-", label=rp.label(m) if m != "Buy & hold" else m)
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_title(f"{rp.target_label(t)}: growth of 1 USD, long/short rule on the +{H}-day forecast, "
                     f"{cfg.cost_bps:.0f} bp costs", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=4, loc="upper left")
        fig.tight_layout(); fig.savefig(OUT / f"equity_{t}.png", dpi=130); plt.close(fig)

    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "cutoff": str(cutoff.date()), "first_origin": str(min(origins).date()), "last_origin": str(max(origins).date()),
               "n_origins": len(origins), "fan_origin": str(fan_origin.date()), "cost_bps": cfg.cost_bps,
               "trade_band": cfg.trade_band, "trade_horizon_chart": cfg.frozen_trade_horizon,
               "models": models_all, "horizons": list(cfg.horizons), "trained_through": trained, "failed": failed,
               "seconds": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[frozen] done in {time.time() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
