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
import dataclasses
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

from forecast import benchmarks as bm  # noqa: E402
from forecast import evaluation as ev  # noqa: E402
from forecast import frozen_report  # noqa: E402
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
    cfg = dataclasses.replace(cfg, horizons=list(cfg.frozen_horizons))    # evaluate short AND long horizons

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
    plugin_models = rp.order_models(fc["model"].unique())            # production models only (charts use these)
    origins_by_target = {t: sorted(fc[fc["target"] == t]["asof_date"].unique()) for t in prices}
    bench = bm.benchmark_forecasts(prices, cutoff, origins_by_target, cfg.frozen_horizons, cfg.series_start)
    fc = pd.concat([fc, bench], ignore_index=True).sort_values(["target", "model", "horizon", "asof_date"])

    metrics = ev.forecast_metrics(fc, prices)
    robust = ev.robustness(fc, prices, cutoff)
    robust.to_csv(OUT / "robustness.csv", index=False)
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
    models_all = plugin_models
    colour = {m: cmap(k % 10) for k, m in enumerate(models_all)}
    H, ZOOM = cfg.frozen_chart_horizon, cfg.frozen_zoom_days
    for t, p in prices.items():
        pos_all = {d: i for i, d in enumerate(p.index)}

        def target_date(d, h=H):
            i = pos_all.get(d)
            return p.index[i + h] if i is not None and i + h < len(p) else pd.NaT

        # ---------- rolling H-day-ahead chart (two panels)
        fig, ax = plt.subplots(2, 1, figsize=(11, 7.2), gridspec_kw={"height_ratios": [1, 1]})
        series = {}
        for m in models_all:
            g = fc[(fc["target"] == t) & (fc["model"] == m) & (fc["horizon"] == H)].sort_values("asof_date")
            if g.empty:
                continue
            tgt = [target_date(d) for d in g["asof_date"]]
            series[m] = pd.DataFrame({"ret": g["forecast_return"].to_numpy(), "sigma": g["sigma"].to_numpy(),
                                      "p0": [p.loc[d] if d in pos_all else np.nan for d in g["asof_date"]]},
                                     index=pd.DatetimeIndex(tgt)).dropna(subset=["ret"])
            series[m] = series[m][series[m].index.notna()]
        actual_h = np.log(p.shift(-H) / p)                              # indexed by origin
        act = pd.Series({target_date(d): v for d, v in actual_h.dropna().items() if d > cutoff}).sort_index()
        act = act[act.index.notna()]
        garch = series.get("ARMA-GARCH")
        flat = {m: s_ for m, s_ in series.items()
                if (s_["ret"].abs().max() < 0.0005)}                    # forecasts equal to "no change"
        # panel 1: price, last ZOOM target dates
        zoom_idx = act.index[-ZOOM:]
        lo = zoom_idx[0]
        px = p.loc[lo:]
        ax[0].plot(px.index, px.values, color="black", lw=2, label="Actual price")
        if garch is not None:
            gz = garch.loc[garch.index >= lo]
            ax[0].fill_between(gz.index, gz["p0"] * np.exp(-gz["sigma"]), gz["p0"] * np.exp(gz["sigma"]),
                               color="tab:blue", alpha=0.13, label=f"ARMA-GARCH +-1 sd of the {H}-day move")
        for m, s_ in series.items():
            sz = s_.loc[s_.index >= lo]
            if m in flat and m != "RW":
                continue
            ax[0].plot(sz.index, sz["p0"] * np.exp(sz["ret"]), lw=1.1 if m != "RW" else 1.2,
                       ls="--" if m == "RW" else "-", color=colour[m], label=rp.label(m))
        if flat:
            ax[0].plot([], [], color="grey", lw=0, label="Forecast = no change: " +
                       ", ".join(rp.label(m) for m in flat if m != "RW"))
        ax[0].set_title(f"{rp.target_label(t)}: {H}-day-ahead forecasts made {H} trading days earlier, drawn at the date "
                        f"they predict (last {ZOOM} days; models frozen at {cutoff.date()})", fontsize=10)
        ax[0].set_ylabel("USD/bbl"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7.5, ncol=3, loc="upper left")
        # panel 2: the H-day return itself, full test period
        ax[1].plot(act.index, act.values * 100, color="black", lw=0.8, alpha=0.75, label=f"Actual {H}-day return")
        if garch is not None:
            ax[1].fill_between(garch.index, -garch["sigma"] * 100, garch["sigma"] * 100, color="tab:blue", alpha=0.15,
                               label="ARMA-GARCH +-1 sd band")
        for m, s_ in series.items():
            if m in flat or m == "ARMA-GARCH":                       # ARMA-GARCH is its band, not a mean forecast
                continue
            ax[1].plot(s_.index, s_["ret"] * 100, lw=1.0, color=colour[m], label=rp.label(m))
        ax[1].axhline(0, color="k", lw=0.7, label="No change (0%)")
        cov = float((act.reindex(garch.index).abs() <= garch["sigma"]).mean()) if garch is not None else np.nan
        ax[1].set_title(f"The {H}-day return and the forecast bands over the whole test period"
                        + (f"; actual inside the GARCH band {100 * cov:.0f}% of days (68% if calibrated)" if cov == cov else ""),
                        fontsize=10)
        ax[1].set_ylabel("%"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7.5, ncol=4, loc="upper left")
        fig.tight_layout(); fig.savefig(OUT / f"rolling_{t}.png", dpi=130); plt.close(fig)

        # ---------- equity curves of the trading rule
        TH = cfg.frozen_trade_horizon
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for (tt, m, h), c in curves.items():
            if tt == t and h in (0, TH) and m != "RW" and m not in ("HistMean", "AR"):
                if m != "Buy & hold" and trading[(trading["target"] == t) & (trading["model"] == m)
                                                 & (trading["horizon"] == TH)]["days_in_market"].max() == 0:
                    continue                                          # never trades: a flat line adds nothing
                ax.plot(c.index, c.values, lw=2.2 if m == "Buy & hold" else 1.2,
                        color="black" if m == "Buy & hold" else colour.get(m, "grey"),
                        ls="--" if m == "Buy & hold" else "-", label=rp.label(m) if m != "Buy & hold" else m)
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_title(f"{rp.target_label(t)}: growth of 1 USD, long/short rule on the {TH}-day forecast, "
                     f"{cfg.cost_bps:.0f} bp costs (models that never trade are omitted)", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=4, loc="upper left")
        fig.tight_layout(); fig.savefig(OUT / f"equity_{t}.png", dpi=130); plt.close(fig)

    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "cutoff": str(cutoff.date()), "first_origin": str(min(origins).date()), "last_origin": str(max(origins).date()),
               "n_origins": len(origins), "fan_origin": str(fan_origin.date()), "cost_bps": cfg.cost_bps,
               "trade_band": cfg.trade_band, "trade_horizon_chart": cfg.frozen_trade_horizon,
               "chart_horizon": cfg.frozen_chart_horizon,
               "models": plugin_models, "benchmarks": ["HistMean", "AR"], "horizons": list(cfg.frozen_horizons), "trained_through": trained, "failed": failed,
               "seconds": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    frozen_report.write_outputs(cfg)                                 # report.html + registration tables
    print(f"[frozen] done in {time.time() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
