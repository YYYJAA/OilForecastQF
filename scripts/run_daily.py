"""Daily forecast: load the saved artifact of every plugin, forecast all horizons, update the log.

Writes  Outputs/latest_forecasts.csv   today's forecasts, all models and horizons
        Outputs/forecast_log.csv       every forecast ever produced (deduplicated)
        run_status.json                plugins that failed, stale models (read by the e-mail step)

Usage:  python scripts/run_daily.py
        python scripts/run_daily.py --backfill 30     seed the log with the last 30 trading days
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forecast.config import MODELS_DIR, OUTPUT_DIR, Config  # noqa: E402
from forecast.data import load_inputs  # noqa: E402
from forecast.registry import REGISTRY  # noqa: E402
import forecast.plugins  # noqa: E402,F401

STALE_DAYS = 100          # warn if a model has not been retrained for this long (quarterly = ~91)
KEY = ["asof_date", "plugin", "model", "horizon"]


def decorate(df, plugin, art, inputs, cfg, source):
    px = inputs.prices[cfg.target]
    out = df.copy()
    out["asof_date"] = pd.to_datetime(out["asof_date"])
    out["plugin"] = plugin
    out["source"] = source
    out["price_now"] = out["asof_date"].map(px)
    out["forecast_price"] = out["price_now"] * np.exp(out["forecast_return"])
    out["target_date_approx"] = [d + pd.offsets.BDay(int(h)) for d, h in zip(out["asof_date"], out["horizon"])]
    out["model_trained_at"] = art["trained_at"]
    out["model_train_end"] = art["train_end"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0, help="also compute forecasts for the last N origins")
    ap.add_argument("--refit-every", type=int, default=5)
    args = ap.parse_args()

    cfg = Config()
    inputs = load_inputs(cfg)
    OUTPUT_DIR.mkdir(exist_ok=True)
    status = {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "failed_plugins": [], "stale_plugins": [], "target": cfg.target,
              "data_last_date": str(inputs.prices[cfg.target].dropna().index.max().date())}
    live_parts, back_parts = [], []
    for name, plugin in REGISTRY.items():
        path = MODELS_DIR / f"{name}.joblib"
        try:
            if not path.exists():
                raise FileNotFoundError(f"{path.name} not found: run scripts/train_models.py")
            art = joblib.load(path)
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(art["trained_at"])).days
            if age > STALE_DAYS:
                status["stale_plugins"].append({"plugin": name, "age_days": age})
            if args.backfill:
                back_parts.append(decorate(plugin.backtest(inputs, cfg, args.backfill, args.refit_every),
                                           name, art, inputs, cfg, "backfill"))
            live_parts.append(decorate(plugin.predict(art, inputs, cfg), name, art, inputs, cfg, "live"))
            print(f"[daily] {name}: ok (trained through {art['train_end']})")
        except Exception as exc:
            print(f"[daily] {name} FAILED: {exc}", file=sys.stderr)
            status["failed_plugins"].append({"plugin": name, "error": str(exc)[:300]})

    if live_parts:
        live = pd.concat(live_parts, ignore_index=True)
        live.to_csv(OUTPUT_DIR / "latest_forecasts.csv", index=False)
        lp = OUTPUT_DIR / "forecast_log.csv"
        old = pd.read_csv(lp, parse_dates=["asof_date", "target_date_approx"]) if lp.exists() else pd.DataFrame()
        parts = [p for p in [pd.concat(back_parts, ignore_index=True) if back_parts else None, old, live] if p is not None and len(p)]
        log = pd.concat(parts, ignore_index=True)
        log = log.drop_duplicates(KEY, keep="last").sort_values(KEY)      # live rows win over backfill
        log.to_csv(lp, index=False)
        print(f"[daily] forecast log: {len(log):,} rows, {log['asof_date'].nunique()} origins")
    (ROOT / "run_status.json").write_text(json.dumps(status, indent=2))
    return 0 if live_parts else 1


if __name__ == "__main__":
    sys.exit(main())
