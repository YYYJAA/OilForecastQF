"""Regime-dependent LASSO (the team's lasson.ipynb), packaged for scripted training and daily prediction.

Model, per horizon h (one direct model per horizon):  y = 100 * (log P[t+h] - log P[t])
    y = beta'z + gamma'(z d) + delta d + e,   d = causal calm / turbulent label from a two-state Markov-switching model of daily
returns (each regime lasts at least 120 days in expectation), z = the ~20 standardised predictors, no intercept (with nothing
selected the forecast is "no change"). The penalty is chosen by time-ordered cross-validation inside the window, with an
h-day gap. The forecast is shrunk by half (equal-weight combination with no change).

Pipeline use: parameters are estimated at each quarterly retrain on the last 1,260 trading days. Between retrains they are
FIXED; every day the regime filter and label are continued with the new return and the forecast uses that day's features.
The interval is the standardised-error spread of the window times today's 20-day volatility (grows in turbulent markets).

Outputs two models per series: RW (no change, volatility band) and LASSO. Fitted for every series in cfg.targets; the Brent
settings are applied to WTI (the Brent-WTI spread predictor is sign-reversed).
"""
import inspect
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
import warnings

from ..data import target_inputs  # noqa: F401  (used through build_panel)
from ..registry import ForecastPlugin, register
from . import _tabular as T

MODEL, VERSION = "LASSO", "1"
ALPHA_ARG = "n_alphas" if "n_alphas" in inspect.signature(LassoCV).parameters else "alphas"
LABELS = {"brent_fut": "Brent", "wti_fut": "WTI"}


def _std_params(A, lab, train):
    muA, sdA = np.nanmean(A[train], 0), np.nanstd(A[train], 0)
    D = _raw_design(A, lab, muA, sdA)
    return muA, sdA, D[train].mean(0), D[train].std(0)


def _raw_design(A, lab, muA, sdA):
    Z = np.nan_to_num((A - muA) / np.where(sdA > 0, sdA, 1.0))
    return np.column_stack([Z, Z * lab[:, None], lab])


def _design(A, lab, std):
    muA, sdA, m, s = std
    D = _raw_design(A, lab, muA, sdA)
    return np.where(s > 0, (D - m) / np.where(s > 0, s, 1.0), 0.0)


def _lasso(D, y, rows, h, cfg):
    rows = rows[np.isfinite(y[rows])]
    m = LassoCV(cv=TimeSeriesSplit(n_splits=cfg.lr_cv_folds, gap=h), eps=cfg.lr_eps, fit_intercept=False,
                max_iter=5000, tol=1e-3, **{ALPHA_ARG: cfg.lr_n_alphas})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(D[rows], y[rows])
    return m, rows


def _fit(P, i0, cfg, warm=None):
    """Estimate everything with data up to position i0 (rows s with s + h <= i0 only)."""
    lo = max(i0 - cfg.window, P.first_valid())
    if i0 - lo < 500:
        raise ValueError(f"only {i0 - lo} usable rows before the training end (need at least 500)")
    rm = T.RegimeModel(cfg.lr_min_duration).fit(P.rr[lo + 1:i0 + 1], warm=warm)        # returns known at i0
    filt, _, _ = T._ms_filter(np.nan_to_num(P.rr[lo:i0 + 1]), rm.mu, rm.sig, rm.P, rm.stationary(rm.P))
    machine = T.RegimeLabel(cfg.lr_min_duration, cfg.lr_enter_p, cfg.lr_exit_p, cfg.lr_confirm)
    lab = np.array([machine.step(v) for v in filt[:, 1]], float)
    std = _std_params(P.A[lo:i0 + 1], lab, np.arange(0, i0 - lo))
    D = _design(P.A[lo:i0 + 1], lab, std)
    scale0 = 100 * P.rv[lo:i0 + 1]
    per_h = {}
    for h in cfg.horizons:
        y = P.Y[h][lo:i0 + 1]
        m, rows = _lasso(D, y, np.arange(0, i0 - lo - h + 1), h, cfg)
        z = (y[rows] - cfg.lr_shrink * m.predict(D[rows])) / (scale0[rows] * np.sqrt(h / 252))
        per_h[int(h)] = {"coef": m.coef_.copy(), "alpha": float(m.alpha_), "zsd": T.interval_sigma(z),
                         "n_terms": int((m.coef_ != 0).sum())}
    return {"i0": int(i0), "date": str(P.dates[i0].date()), "lo": int(lo), "window_start": str(P.dates[lo].date()),
            "n_obs": int(i0 - lo), "names": list(P.X.columns), "std": std, "per_h": per_h,
            "regime": rm.params(), "prior": (filt[-1] @ rm.P).tolist(), "p_last": float(filt[-1, 1]),
            "lab_last": float(lab[-1]), "label_state": machine.snapshot(),
            "_rm": rm}                                          # kept for the warm start of the next refit (dropped before saving)


def _apply(fit, P, positions, cfg):
    """Forecast rows at `positions` (>= the training end) with the fixed parameters in `fit`."""
    i0 = P.pos(fit["date"])
    if i0 is None:
        raise ValueError(f"training date {fit['date']} not in the data")
    pos = np.asarray(sorted(int(p) for p in positions if p >= i0))
    if len(pos) == 0:
        return pd.DataFrame()
    p_new, lab_new, _ = T.regime_path(P.rr[i0 + 1:pos.max() + 1], fit["regime"], fit["prior"], fit["label_state"], T.lr_params(cfg))
    p_all = np.r_[fit["p_last"], p_new]
    lab_all = np.r_[fit["lab_last"], lab_new]
    off = pos - i0
    D = _design(P.A[pos], lab_all[off], fit["std"])
    rows = []
    for h, par in fit["per_h"].items():
        raw = D @ par["coef"]
        f = cfg.lr_shrink * raw / 100
        sd = par["zsd"] * P.rv[pos] * np.sqrt(h / 252)
        for k, p in enumerate(pos):
            d = P.dates[p]
            rows.append({"asof_date": d, "model": MODEL, "horizon": int(h), "forecast_return": float(f[k]),
                         "sigma": float(sd[k]), "sigma_kind": "resid_rv", "raw_forecast_return": float(raw[k] / 100),
                         "regime_p": float(p_all[off[k]])})
            rows.append({"asof_date": d, "model": "RW", "horizon": int(h), "forecast_return": 0.0,
                         "sigma": float(P.rv[p] * np.sqrt(h / 252)), "sigma_kind": "rv20", "raw_forecast_return": np.nan,
                         "regime_p": float(p_all[off[k]])})
    return pd.DataFrame(rows)


def _light(fit):
    """Artifact copy without the helper objects."""
    return {k: v for k, v in fit.items() if k != "_rm"}


def _summary(fit):
    r = fit["regime"]
    return {"window_start": fit["window_start"], "n_obs": fit["n_obs"], "train_end": fit["date"],
            "vol_calm_ann_pct": round(100 * r["sig"][0] * np.sqrt(252) / 100, 1),
            "vol_turb_ann_pct": round(100 * r["sig"][1] * np.sqrt(252) / 100, 1),
            "regime_now": "turbulent" if fit["lab_last"] == 1 else "calm",
            "terms_kept": {h: p["n_terms"] for h, p in fit["per_h"].items()},
            "penalty": {h: round(p["alpha"], 5) for h, p in fit["per_h"].items()}}


def _series(art, cfg):
    return art["fits"]


class RegimeLasso(ForecastPlugin):
    name = "lasso_regime"

    def train(self, inputs, cfg):
        fits, summ, errs = {}, {}, {}
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            try:
                P = T.build_panel(inputs, cfg, t)
                fit = _fit(P, len(P.dates) - 1, cfg)
                fits[t], summ[t] = _light(fit), _summary(fit)
            except Exception as exc:                                # the other series is still usable
                errs[t] = str(exc)[:200]
        if not fits:
            raise RuntimeError(f"regime LASSO could not be fitted: {errs}")
        first = cfg.target if cfg.target in fits else next(iter(fits))
        return {"plugin": self.name, "version": VERSION, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_end": max(f["date"] for f in fits.values()), "target": first, "targets": list(fits),
                "by_target": summ, "errors": errs, "fits": fits}

    def predict(self, art, inputs, cfg):
        parts = []
        for t, fit in _series(art, cfg).items():
            if t not in inputs.prices.columns:
                continue
            P = T.build_panel(inputs, cfg, t)
            df = _apply(fit, P, [len(P.dates) - 1], cfg)
            df["target"] = t
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def rolling_predict(self, art, inputs, cfg, origins):
        parts = []
        for t, fit in _series(art, cfg).items():
            if t not in inputs.prices.columns:
                continue
            P = T.build_panel(inputs, cfg, t)
            pos = [P.pos(d) for d in origins]
            df = _apply(fit, P, [p for p in pos if p is not None], cfg)
            df["target"] = t
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def backtest(self, inputs, cfg, n_days, refit_every=5):
        step = max(int(refit_every), 1)
        parts = []
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            P = T.build_panel(inputs, cfg, t)
            last = len(P.dates) - 1
            starts = list(range(last - n_days + 1, last + 1, step))
            warm = None
            for k, i in enumerate(starts):
                fit = _fit(P, i, cfg, warm)
                warm = fit["_rm"]
                end = starts[k + 1] if k + 1 < len(starts) else last + 1
                df = _apply(fit, P, range(i, end), cfg)
                df["target"] = t
                parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def describe_change(self, old, new, inputs, cfg):
        head, tables = [], []
        olds = (old or {}).get("by_target", {})
        for t, s in new["by_target"].items():
            nm = LABELS.get(t, t)
            o = olds.get(t)
            head.append(f"[{nm}] New training data run through {s['train_end']}"
                        + (f" (previous model: {o['train_end']})." if o else " (first training)."))
            head.append(f"[{nm}] Estimation window: {s['window_start']} to {s['train_end']} ({s['n_obs']} trading days). Regimes: calm "
                        f"volatility {s['vol_calm_ann_pct']:.0f}% a year, turbulent {s['vol_turb_ann_pct']:.0f}%; the market is "
                        f"currently {s['regime_now']}.")
            kept = ", ".join(f"{h}d: {n}" for h, n in s["terms_kept"].items())
            head.append(f"[{nm}] Terms the LASSO keeps (of {2 * len(new['fits'][t]['names']) + 1}) by horizon: {kept}.")
            fit = new["fits"][t]
            h20 = 20 if 20 in fit["per_h"] else next(iter(fit["per_h"]))
            c, k = fit["per_h"][h20]["coef"], len(fit["names"])
            rows = [[fit["names"][j], f"{c[j]:+.3f}", f"{c[j] + c[k + j]:+.3f}"] for j in np.argsort(-np.abs(c[:k] + c[k:2 * k]))[:8]
                    if c[j] != 0 or c[k + j] != 0]
            if rows:
                tables.append({"title": f"[{nm}] Largest terms in the {h20}-day LASSO (standardised effect)",
                               "header": ["Predictor", "Calm market", "Turbulent market"], "rows": rows})
        return {"headline": head, "tables": tables}


register(RegimeLasso())
