"""RW / LASSO / LASSO_1se / PCA / targeted PCA / regime-dependent LASSO / equal-weight combination.

This is the team's lasso.ipynb model set, packaged for scripted training and daily prediction.
Regime probabilities come from the real-time Gaussian HMM; between quarterly retrains the HMM
parameters are fixed and the filter is simply continued with each new day.

The set is fitted separately for every series in cfg.targets (Brent and WTI futures). The notebook's settings were chosen
on Brent; the same settings are applied to WTI. Artifact layout: art["by_target"][series] holds what used to be the
whole artifact (older single-series artifacts are still read as Brent-only).
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..data import target_inputs
from ..features import add_regime_features, build_features, train_rows
from ..models import FITTERS
from ..regime import continue_filter, hmm_inputs, run_regime_id
from ..registry import ForecastPlugin, register

MODELS = ["RW", "LASSO", "LASSO_1se", "PCA_reg", "TPCA", "LASSO_RS"]
COMBO = "Combo_EW"
VERSION = "2"


def _panel(inputs, cfg, target):
    X, Y, logp = build_features(inputs.prices, inputs.eia, inputs.weekly, target, cfg)
    P, hmm_log = run_regime_id(logp, cfg)
    return add_regime_features(X, P, cfg), Y, logp, P, hmm_log


def _series(art, cfg):
    """{series: single-series artifact}; an older artifact (one series, no by_target) is read as cfg.target."""
    return art["by_target"] if "by_target" in art else {art.get("target", cfg.target): art}


def _fit_at(X, Y, i, cfg):
    """Fit every model for every horizon with rows observable at origin i."""
    fits = {}
    for h in cfg.horizons:
        rows = train_rows(i, h, cfg.window)
        if len(rows) < 0.8 * cfg.window:
            continue
        ctx = {}
        Xtr, ytr = X.iloc[rows], Y[h].iloc[rows]
        fits[h] = {m: FITTERS[m](Xtr, ytr, h, cfg, ctx) for m in MODELS}
    return fits


def _predict_row(fits, x, cfg, asof):
    sigma_d = float(x.get("own_rv20", np.nan)) / np.sqrt(252)
    rows = []
    for h, fh in fits.items():
        preds = {m: fh[m].predict(x) for m in MODELS}
        preds[COMBO] = float(np.mean([preds[m] for m in cfg.combo_pool if m in preds]))
        for m, v in preds.items():
            rows.append({"asof_date": asof, "model": m, "horizon": int(h),
                         "forecast_return": float(v), "sigma": sigma_d * np.sqrt(h)})
    return rows


class LassoFamily(ForecastPlugin):
    name = "lasso_family"

    # ------------------------------------------------------------------ one series
    def _train_one(self, inputs, cfg, target):
        X, Y, logp, P, hmm_log = _panel(inputs, cfg, target)
        i = len(X) - 1
        fits = _fit_at(X, Y, i, cfg)
        last = hmm_log[-1]
        windows = {}
        for h in fits:
            rows = train_rows(i, h, cfg.window)
            windows[int(h)] = {"start": str(X.index[rows[0]].date()), "end": str(X.index[rows[-1]].date()),
                               "n_rows": int(len(rows))}
        meta = {str(h): {m: {"n_selected": len(f.selected), **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                                 for k, v in f.info.items()}}
                         for m, f in fh.items()} for h, fh in fits.items()}
        top = {str(h): {m: sorted(zip(f.cols, map(float, f.coef)), key=lambda t: -abs(t[1]))[:8]
                        for m, f in fh.items() if m.startswith(("LASSO", "PCA", "TPCA")) and len(f.cols)}
               for h, fh in fits.items()}
        return {
            "plugin": self.name, "version": VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_end": str(X.index[i].date()), "n_obs_window": cfg.window,
            "target": target, "cfg": cfg.to_dict(), "fits": fits, "windows": windows,
            "data_start": str(X.index[0].date()),
            "hmm": {"date": X.index[i], "alpha": P.iloc[i].to_numpy(dtype=float),
                    "A": last["A"], "mu": last["mu"], "var": last["var"], "fit_date": str(last["date"].date())},
            "meta": meta, "top_coefficients": top,
        }

    def _predict_one(self, art, inputs, cfg, target):
        X, Y, logp = build_features(inputs.prices, inputs.eia, inputs.weekly, target, cfg)
        hm = art["hmm"]
        H = hmm_inputs(logp, cfg.hmm_inputs)
        P = pd.DataFrame(np.nan, index=X.index, columns=[f"state{k}" for k in range(len(hm["alpha"]))])
        alpha = np.asarray(hm["alpha"], float)
        if hm["date"] in P.index:
            P.loc[hm["date"]] = alpha
        for d in X.index[X.index > hm["date"]]:        # continue the filter with fixed parameters
            row = H.loc[d].to_numpy(dtype=float)
            if np.isfinite(row).all():
                alpha = continue_filter(alpha, row, hm["A"], hm["mu"], hm["var"])
            P.loc[d] = alpha
        X = add_regime_features(X, P, cfg)
        asof = X.index[-1]
        df = pd.DataFrame(_predict_row(art["fits"], X.iloc[-1], cfg, asof))
        df["regime_p"] = float(X["regime_p"].iloc[-1])
        df["target"] = target
        return df

    def _rolling_one(self, art, inputs, cfg, target, origins):
        """Fixed parameters at every origin. Features are causal (checked by the look-ahead test), so they are built
        once; the regime filter is continued once through all days after the training end."""
        X, Y, logp = build_features(inputs.prices, inputs.eia, inputs.weekly, target, cfg)
        hm = art["hmm"]
        H = hmm_inputs(logp, cfg.hmm_inputs)
        P = pd.DataFrame(np.nan, index=X.index, columns=[f"state{k}" for k in range(len(hm["alpha"]))])
        alpha = np.asarray(hm["alpha"], float)
        if hm["date"] in P.index:
            P.loc[hm["date"]] = alpha
        for d in X.index[X.index > hm["date"]]:
            row = H.loc[d].to_numpy(dtype=float)
            if np.isfinite(row).all():
                alpha = continue_filter(alpha, row, hm["A"], hm["mu"], hm["var"])
            P.loc[d] = alpha
        X = add_regime_features(X, P, cfg)
        rows = []
        for d in origins:
            if d in X.index:
                r = _predict_row(art["fits"], X.loc[d], cfg, d)
                for x in r:
                    x["regime_p"] = float(X.loc[d, "regime_p"])
                    x["target"] = target
                rows += r
        return pd.DataFrame(rows)

    def _backtest_one(self, inputs, cfg, target, n_days, refit_every):
        X, Y, logp, P, hmm_log = _panel(inputs, cfg, target)
        last = len(X) - 1
        rows, fits = [], None
        for k, i in enumerate(range(last - n_days + 1, last + 1)):
            if fits is None or k % refit_every == 0:
                fits = _fit_at(X, Y, i, cfg)
            rows += _predict_row(fits, X.iloc[i], cfg, X.index[i])
        df = pd.DataFrame(rows)
        df["regime_p"] = df["asof_date"].map(X["regime_p"])
        df["target"] = target
        return df

    # ------------------------------------------------------------------ all series (public interface)
    def train(self, inputs, cfg):
        by = {t: self._train_one(target_inputs(inputs, cfg, t), cfg, t) for t in cfg.targets if t in inputs.prices.columns}
        if not by:
            raise RuntimeError("no target series available for the LASSO family")
        first = by[cfg.target] if cfg.target in by else next(iter(by.values()))
        return {"plugin": self.name, "version": VERSION, "trained_at": first["trained_at"],
                "train_end": max(a["train_end"] for a in by.values()), "target": cfg.target,
                "targets": list(by), "by_target": by}

    def predict(self, art, inputs, cfg):
        parts = [self._predict_one(a, target_inputs(inputs, cfg, t), cfg, t) for t, a in _series(art, cfg).items()]
        return pd.concat(parts, ignore_index=True)

    def rolling_predict(self, art, inputs, cfg, origins):
        parts = [self._rolling_one(a, target_inputs(inputs, cfg, t), cfg, t, origins) for t, a in _series(art, cfg).items()]
        return pd.concat(parts, ignore_index=True)

    def backtest(self, inputs, cfg, n_days, refit_every=5):
        parts = [self._backtest_one(target_inputs(inputs, cfg, t), cfg, t, n_days, refit_every)
                 for t in cfg.targets if t in inputs.prices.columns]
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------ retrain report
    def describe_change(self, old, new, inputs, cfg):
        head, tables = [], []
        olds = _series(old, cfg) if old is not None else {}
        for t, nt in _series(new, cfg).items():
            h_, t_ = self._describe_one(olds.get(t), nt, target_inputs(inputs, cfg, t), cfg, t)
            label = {"brent_fut": "Brent", "wti_fut": "WTI"}.get(t, t)
            head += [f"[{label}] {x}" for x in h_]
            tables += [{**tb, "title": f"[{label}] {tb['title']}"} for tb in t_]
        return {"headline": head, "tables": tables}

    def _describe_one(self, old, new, inputs, cfg, target):
        def fmt(x, d=4):
            return "n/a" if x is None else f"{x:.{d}g}"
        head, tables = [], []
        w = new.get("windows", {})
        ow = (old or {}).get("windows", {})
        head.append(f"New training data run through {new['train_end']}"
                    + (f" (previous model: {old['train_end']})." if old else " (first training)."))
        if w:
            h0 = min(w)
            head.append(f"Estimation window: {cfg.window} trading days (about {cfg.window / 252:.0f} years); "
                        f"for the 1-day horizon it covers {w[h0]['start']} to {w[h0]['end']}. "
                        f"The regime (HMM) model is estimated on all data since {new.get('data_start')} "
                        f"and was last re-estimated on {new['hmm']['fit_date']}.")
        tables.append({"title": "Training window by horizon",
                       "header": ["Horizon", "Window start", "Window end", "Rows", "Previous window end"],
                       "rows": [[f"+{h}d", w[h]["start"], w[h]["end"], w[h]["n_rows"],
                                 ow.get(h, {}).get("end", "-")] for h in sorted(w)]})

        rows, n_changed, n_same = [], 0, 0
        for h in sorted(new["fits"]):
            for m in MODELS:
                nf = new["fits"][h][m]
                of = (old or {}).get("fits", {}).get(h, {}).get(m)
                if not (len(nf.cols) or (of is not None and len(of.cols))):
                    continue
                ns, os_ = set(nf.selected), set(of.selected) if of is not None else set()
                added, dropped = sorted(ns - os_), sorted(os_ - ns)
                n_changed += bool(added or dropped)
                ni, oi = nf.info, (of.info if of is not None else {})
                if m.startswith("LASSO"):
                    par = f"penalty {fmt(oi.get('alpha'))} -> {fmt(ni.get('alpha'))}"
                else:
                    par = f"factors k {fmt(oi.get('k'), 3)} -> {fmt(ni.get('k'), 3)}"
                k_changed = (m in ("PCA_reg", "TPCA") and of is not None and ni.get("k") != oi.get("k"))
                if of is not None and not (added or dropped or k_changed):
                    n_same += 1
                    continue
                rows.append([f"+{h}d", m, f"{len(os_) if of is not None else '-'} -> {len(ns)}", par,
                             ", ".join(added[:5]) + (" ..." if len(added) > 5 else "") or "-",
                             ", ".join(dropped[:5]) + (" ..." if len(dropped) > 5 else "") or "-"])
        if rows:
            tables.append({"title": "Model fits whose selected variables or number of factors changed"
                                    f" ({n_same} other fits kept the same variables; their penalties were simply re-estimated)",
                           "header": ["Horizon", "Model", "Selected vars", "Hyper-parameter", "Added", "Dropped"],
                           "rows": rows})
        else:
            head.append(f"None of the {n_same} model fits changed its selected variables or factor count.")
        head.append(f"{n_changed} model fits changed their selected variables." if old else
                    "First training: no earlier model to compare with.")

        # forecasts: old model vs new model on the same (latest) data
        if old is not None:
            try:
                a, b = self._predict_one(old, inputs, cfg, target), self._predict_one(new, inputs, cfg, target)
                m = a.merge(b, on=["model", "horizon"], suffixes=("_old", "_new"))
                m = m[m["model"] != "RW"]            # the no-change forecast cannot change
                px = float(inputs.prices[target].dropna().iloc[-1])
                m["price_old"], m["price_new"] = px * np.exp(m["forecast_return_old"]), px * np.exp(m["forecast_return_new"])
                fr = [[f"+{int(r.horizon)}d", r.model, f"{r.price_old:.2f}", f"{r.price_new:.2f}",
                       f"{100 * (r.price_new / r.price_old - 1):+.2f}%"] for r in m.sort_values(["horizon", "model"]).itertuples()]
                tables.append({"title": f"Forecast impact: old vs new model, same data (current price {px:.2f})",
                               "header": ["Horizon", "Model", "Old model", "New model", "Change"], "rows": fr})
                head.append(f"Largest forecast change: {m.assign(d=(m.price_new / m.price_old - 1).abs()).d.max():.1%}.")
            except Exception as exc:                      # the report must never block a retrain
                head.append(f"(Could not compare forecasts: {exc})")

        # regime model
        def reg(a):
            h = a["hmm"]
            vol = np.sqrt(252 * np.asarray(h["var"])[:, 0])
            stay = np.diag(np.asarray(h["A"]))
            return vol, stay, float(np.asarray(h["alpha"])[-1])
        nv, ns_, np_ = reg(new)
        ov, os2, op = reg(old) if old else (None, None, None)
        k = len(nv)
        tables.append({"title": "Regime model (2-state HMM on daily returns)",
                       "header": ["State", "Annual volatility (old -> new)", "Stay probability (old -> new)"],
                       "rows": [[("calm" if j == 0 else "turbulent") if k == 2 else f"state {j}",
                                 f"{(f'{ov[j]:.0f}% -> ' if ov is not None else '')}{nv[j]:.0f}%",
                                 f"{(f'{os2[j]:.3f} -> ' if os2 is not None else '')}{ns_[j]:.3f}"] for j in range(k)]
                                + [["P(turbulent) today", "", f"{(f'{op:.2f} -> ' if op is not None else '')}{np_:.2f}"]]})
        return head, tables


register(LassoFamily())
