"""XGBoost (the team's xgbn.ipynb), packaged for scripted training and daily prediction.

Per horizon h (one direct model per horizon):  y = 100 * (log P[t+h] - log P[t]) = F_h(x_t),  F_h = shallow gradient-boosted trees
(depth 3, learning rate 0.03, up to 500 trees stopped early on an inner validation block, strong L2 penalty, 70% row and column
subsampling, starting value 0 so that "no signal" means "no change"). The final forecast is shrunk by half.

Features: the same ~20 change / z-score predictors as the regime LASSO (on the days both Brent and WTI trade), reduced by PCA
variable selection (Jolliffe B4: one original variable per principal component with eigenvalue > 1; target never used), plus two
regime inputs: P(turbulent) from a two-state Markov-switching model of returns and a causal calm / turbulent label (regimes last
at least one quarter). Rows for horizon h: s + h <= origin.

Pipeline use: trained at each quarterly retrain on the last 1,260 trading days; between retrains the model is FIXED and runs on
each day's features. Interval: spread of the standardised in-window errors times today's 20-day volatility.
Fitted for every series in cfg.targets. Output model name: XGB.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from ..registry import ForecastPlugin, register
from . import _tabular as T

MODEL, VERSION = "XGB", "1"
LABELS = {"brent_fut": "Brent", "wti_fut": "WTI"}


def pca_select(A, names, min_eig=1.0):
    """Jolliffe B4: one original variable per retained principal component. Returns column indices into `names`."""
    D = pd.DataFrame(A, columns=names)
    D = D.loc[:, D.notna().mean() > 0.8]
    D = D.fillna(D.mean())
    Z = ((D - D.mean()) / D.std().replace(0, np.nan)).dropna(axis=1)
    eig, vec = np.linalg.eigh(np.corrcoef(Z.to_numpy().T))
    order = np.argsort(eig)[::-1]
    eig, vec = eig[order], vec[:, order]
    k = max(1, int((eig > min_eig).sum()))
    load = pd.DataFrame(vec[:, :k], index=Z.columns)
    chosen = []
    for j in range(k):
        for v in load.iloc[:, j].abs().sort_values(ascending=False).index:
            if v not in chosen:
                chosen.append(v)
                break
    return [list(names).index(v) for v in chosen]


def _make(cfg):
    return xgb.XGBRegressor(n_estimators=cfg.xgb_n_estimators, learning_rate=cfg.xgb_learning_rate, max_depth=cfg.xgb_max_depth,
                            min_child_weight=cfg.xgb_min_child_weight, subsample=cfg.xgb_subsample,
                            colsample_bytree=cfg.xgb_colsample, reg_lambda=cfg.xgb_reg_lambda, base_score=0.0,
                            tree_method="hist", objective="reg:squarederror", random_state=cfg.xgb_seed, n_jobs=4,
                            early_stopping_rounds=cfg.xgb_early_stopping)


def _fit(P, i0, cfg):
    lo = max(i0 - cfg.window, P.first_valid())
    if i0 - lo < 500:
        raise ValueError(f"only {i0 - lo} usable rows before the training end (need at least 500)")
    sel = pca_select(P.A[lo + 1:i0 + 1].astype(np.float32), P.X.columns, cfg.xgb_pca_min_eig)
    inputs = [P.X.columns[i] for i in sel]
    regime = None
    if cfg.xgb_use_regime:                                      # P(turbulent) and the label enter after the PCA selection
        rm = T.RegimeModel(cfg.xgb_min_duration).fit(P.rr[lo + 1:i0 + 1])
        filt, _, _ = T._ms_filter(np.nan_to_num(P.rr[lo:i0 + 1]), rm.mu, rm.sig, rm.P, rm.stationary(rm.P))
        machine = T.RegimeLabel(*T.xgb_params(cfg))
        lab = np.array([machine.step(v) for v in filt[:, 1]], float)
        R = np.full((len(P.dates), 2), np.nan)
        R[lo:i0 + 1, 0], R[lo:i0 + 1, 1] = filt[:, 1], lab
        As = np.column_stack([P.A[:, sel], R]).astype(np.float32)
        inputs += ["p_turbulent", "regime_label"]
        regime = {"params": rm.params(), "prior": (filt[-1] @ rm.P).tolist(), "p_last": float(filt[-1, 1]),
                  "lab_last": float(lab[-1]), "label_state": machine.snapshot()}
    else:
        As = P.A[:, sel].astype(np.float32)
    per_h = {}
    for h in cfg.horizons:
        y = P.Y[h]
        scale = 100 * P.rv * np.sqrt(h / 252)
        hi = i0 - h + 1                                        # rows lo..hi-1 (targets observed at the origin)
        v0 = hi - cfg.xgb_inner_val
        fit_rows, val_rows = np.arange(lo, v0 - h), np.arange(v0, hi)         # fit | gap of h days | validation
        fit_rows, val_rows = fit_rows[np.isfinite(y[fit_rows])], val_rows[np.isfinite(y[val_rows])]
        m = _make(cfg)
        m.fit(As[fit_rows], y[fit_rows], eval_set=[(As[val_rows], y[val_rows])], verbose=False)
        rows = np.r_[fit_rows, val_rows]
        z = (y[rows] - cfg.xgb_shrink * m.predict(As[rows])) / scale[rows]
        gain = m.get_booster().get_score(importance_type="total_gain")
        per_h[int(h)] = {"model": m, "zsd": T.interval_sigma(z), "trees": int(m.best_iteration) + 1,
                         "gain": {inputs[int(f[1:])]: float(g) for f, g in gain.items()}}
    return {"i0": int(i0), "date": str(P.dates[i0].date()), "lo": int(lo), "window_start": str(P.dates[lo].date()),
            "n_obs": int(i0 - lo), "names": list(P.X.columns), "sel": [int(s) for s in sel], "inputs": inputs,
            "regime": regime, "per_h": per_h}


def _apply(fit, P, positions, cfg):
    i0 = P.pos(fit["date"])
    if i0 is None:
        raise ValueError(f"training date {fit['date']} not in the {P.target} data")
    pos = np.asarray(sorted(int(p) for p in positions if p >= i0))
    if len(pos) == 0:
        return pd.DataFrame()
    x = P.A[pos][:, fit["sel"]]
    p_now = np.full(len(pos), np.nan)
    if fit["regime"] is not None:                              # continue the regime filter and label with fixed parameters
        rg = fit["regime"]
        p_new, lab_new, _ = T.regime_path(P.rr[i0 + 1:pos.max() + 1], rg["params"], rg["prior"], rg["label_state"], T.xgb_params(cfg))
        p_all, lab_all = np.r_[rg["p_last"], p_new], np.r_[rg["lab_last"], lab_new]
        p_now = p_all[pos - i0]
        x = np.column_stack([x, p_now, lab_all[pos - i0]])
    x = x.astype(np.float32)
    rows = []
    for h, par in fit["per_h"].items():
        raw = par["model"].predict(x)
        f = cfg.xgb_shrink * raw / 100
        sd = par["zsd"] * P.rv[pos] * np.sqrt(h / 252)
        for k, p in enumerate(pos):
            rows.append({"asof_date": P.dates[p], "model": MODEL, "horizon": int(h), "forecast_return": float(f[k]),
                         "sigma": float(sd[k]), "sigma_kind": "resid_rv", "raw_forecast_return": float(raw[k] / 100),
                         "regime_p": float(p_now[k])})
    return pd.DataFrame(rows)


def _summary(fit):
    gain = {}
    for p in fit["per_h"].values():
        for k, v in p["gain"].items():
            gain[k] = gain.get(k, 0.0) + v
    top = sorted(gain, key=gain.get, reverse=True)[:5]
    return {"window_start": fit["window_start"], "n_obs": fit["n_obs"], "train_end": fit["date"],
            "variables_kept": [fit["names"][i] for i in fit["sel"]], "regime_inputs": fit["regime"] is not None,
            "trees": {h: p["trees"] for h, p in fit["per_h"].items()}, "top_by_gain": top}


class XGBoostPlugin(ForecastPlugin):
    name = "xgboost"

    def train(self, inputs, cfg):
        fits, summ, errs = {}, {}, {}
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            try:
                P = T.build_panel(inputs, cfg, t, common=True)
                fits[t] = _fit(P, len(P.dates) - 1, cfg)
                summ[t] = _summary(fits[t])
            except Exception as exc:
                errs[t] = str(exc)[:200]
        if not fits:
            raise RuntimeError(f"XGBoost could not be fitted: {errs}")
        first = cfg.target if cfg.target in fits else next(iter(fits))
        return {"plugin": self.name, "version": VERSION, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_end": max(f["date"] for f in fits.values()), "target": first, "targets": list(fits),
                "by_target": summ, "errors": errs, "fits": fits}

    def _run(self, art, inputs, cfg, pick):
        parts = []
        for t, fit in art["fits"].items():
            if t not in inputs.prices.columns:
                continue
            P = T.build_panel(inputs, cfg, t, common=True)
            df = _apply(fit, P, pick(P), cfg)
            df["target"] = t
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def predict(self, art, inputs, cfg):
        return self._run(art, inputs, cfg, lambda P: [len(P.dates) - 1])

    def rolling_predict(self, art, inputs, cfg, origins):
        return self._run(art, inputs, cfg, lambda P: [p for p in (P.pos(d) for d in origins) if p is not None])

    def backtest(self, inputs, cfg, n_days, refit_every=5):
        step = max(int(refit_every), 1)
        parts = []
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            P = T.build_panel(inputs, cfg, t, common=True)
            last = len(P.dates) - 1
            starts = list(range(last - n_days + 1, last + 1, step))
            for k, i in enumerate(starts):
                fit = _fit(P, i, cfg)
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
            head.append(f"[{nm}] Estimation window: {s['window_start']} to {s['train_end']} ({s['n_obs']} trading days). PCA keeps "
                        f"{len(s['variables_kept'])} of {len(new['fits'][t]['names'])} variables"
                        + (", plus the regime probability and label." if s.get("regime_inputs") else "."))
            kept_old = set((o or {}).get("variables_kept", []))
            added = [v for v in s["variables_kept"] if v not in kept_old] if o else []
            dropped = [v for v in kept_old if v not in s["variables_kept"]]
            if o and (added or dropped):
                head.append(f"[{nm}] Variable set changed: added {added or 'none'}; dropped {dropped or 'none'}.")
            head.append(f"[{nm}] Trees kept by early stopping: " + ", ".join(f"{h}d: {n}" for h, n in s["trees"].items()) + ".")
            tables.append({"title": f"[{nm}] Variables kept by PCA selection and the most-used ones",
                           "header": ["Variable kept", "Among the top 5 by gain"],
                           "rows": [[v, "yes" if v in s["top_by_gain"] else ""] for v in s["variables_kept"]]})
        return {"headline": head, "tables": tables}


register(XGBoostPlugin())
