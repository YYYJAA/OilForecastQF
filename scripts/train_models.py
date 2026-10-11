"""Quarterly (re)training: fit every registered plugin on the latest data and save the artifacts.

Hyper-parameters are picked inside the fit with time-series cross-validation on the training window
only (LASSO penalty, number of PCA factors, regime HMM re-estimation), so every retrain re-selects them.

Writes  models/<plugin>.joblib        the fitted artifact (used by run_daily.py)
        models/<plugin>_meta.json     human-readable summary (train date, penalties, selected features)
        models/training_history.csv   one row per retrain

Usage:  python scripts/train_models.py [--plugins lasso_family]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forecast.config import MODELS_DIR, Config  # noqa: E402
from forecast.data import load_inputs  # noqa: E402
from forecast.registry import REGISTRY  # noqa: E402
import forecast.plugins  # noqa: E402,F401  (registers plugins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins", nargs="*", default=None, help="plugin names (default: all registered)")
    args = ap.parse_args()
    cfg = Config()
    inputs = load_inputs(cfg)
    MODELS_DIR.mkdir(exist_ok=True)
    names = args.plugins or list(REGISTRY)
    hist_rows, reports, failed = [], [], []
    for name in names:
        t0 = time.time()
        old_path = MODELS_DIR / f"{name}.joblib"
        old = None
        if old_path.exists():
            try:
                old = joblib.load(old_path)
            except Exception as exc:
                print(f"[train] could not load previous {name} artifact: {exc}", file=sys.stderr)
        try:
            art = REGISTRY[name].train(inputs, cfg)
        except Exception as exc:                                  # one failing plugin must not stop the others
            print(f"[train] {name} FAILED: {exc}", file=sys.stderr)
            failed.append(name)
            continue
        joblib.dump(art, MODELS_DIR / f"{name}.joblib")
        meta = {k: v for k, v in art.items() if k not in ("fits", "hmm", "cfg")}
        meta["hmm_fit_date"] = art.get("hmm", {}).get("fit_date")
        (MODELS_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        try:
            change = REGISTRY[name].describe_change(old, art, inputs, cfg)
        except Exception as exc:
            change = {"headline": [f"(change summary unavailable: {exc})"], "tables": []}
        reports.append({"plugin": name, "trained_at": art["trained_at"], "train_end": art["train_end"],
                        "previous_train_end": old["train_end"] if old else None,
                        "previous_trained_at": old["trained_at"] if old else None,
                        "seconds": round(time.time() - t0, 1), **change})
        hist_rows.append({"plugin": name, "trained_at": art["trained_at"], "train_end": art["train_end"],
                          "seconds": round(time.time() - t0, 1)})
        print(f"[train] {name}: trained through {art['train_end']} in {time.time() - t0:.0f}s")
    if reports:
        (MODELS_DIR / "retrain_report.json").write_text(json.dumps(
            {"generated_at_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "reports": reports,
             "failed": failed}, indent=2, default=str))
    if hist_rows:
        hp = MODELS_DIR / "training_history.csv"
        new = pd.DataFrame(hist_rows)
        pd.concat([pd.read_csv(hp), new]).to_csv(hp, index=False) if hp.exists() else new.to_csv(hp, index=False)
    return 1 if failed and not hist_rows else 0


if __name__ == "__main__":
    sys.exit(main())
