"""MS-AR / MS-ARX: two-regime Markov-switching autoregressions on Brent daily returns.

Ported from the team's msar.ipynb (same estimation, admissibility checks, exact conditional-mean recursion and
simulation), packaged as a plugin:  fit at each retrain, then between retrains only *filter* the new data through the
fixed parameters (the notebook's rule: refit every 63 trading days, filter daily).

    MS-AR   AR(1), switching mean and variance, shared AR coefficient; window 1260 returns; weight 1.0
    MS-ARX  AR(1) + lagged WTI dollar change, DXY log return and Brent-WTI spread (shared coefficients, switching
            variance); window 756 returns; the expected cumulative return is multiplied by the weight 0.25

forecast_return = weight * exact expected cumulative log return;  sigma = standard deviation of simulated cumulative
log returns (regime mixture; parameter and predictor-path uncertainty excluded, as in the notebook).

Fitted separately for every series in cfg.targets (Brent and WTI futures). For MS-ARX the "other oil benchmark" is WTI when
forecasting Brent and Brent when forecasting WTI; the notebook's window and mean weight were chosen on Brent and are reused for
WTI. Artifact layout: art["series"][series]["models"]; older single-series artifacts are read as Brent only.

Requires statsmodels >= 0.15 (parameter layout and `rng` argument); the artifact stores the version it was fitted with.
"""
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from ..data import target_inputs
from ..registry import ForecastPlugin, register

VERSION = "1"
FEATURES = ("other_change_usd", "dxy_log_return_pct", "spread_usd")      # other = WTI for Brent, Brent for WTI
OTHER = {"brent_fut": "wti_fut", "wti_fut": "brent_fut"}


def _need_statsmodels():
    major_minor = tuple(int(x) for x in statsmodels.__version__.split(".")[:2])
    if major_minor < (0, 15):
        raise RuntimeError(f"MS-AR needs statsmodels >= 0.15 (found {statsmodels.__version__}); "
                           "see requirements.txt")


# --------------------------------------------------------------------------- data
def _panel(inputs, cfg, target):
    """Prices, % log returns, lagged inputs X_t = f(t-1) and the 'current' features, on the target's own calendar."""
    px = inputs.prices[target].dropna()
    px = px[px > 0].astype(float)
    other = OTHER[target]
    quoted = inputs.prices[[other, "dxy"]].reindex(px.index)
    filled = quoted.ffill(limit=cfg.msar_fill_limit)            # past quotes only, never back-filled
    current = pd.DataFrame({
        "other_change_usd": filled[other].diff(),
        "dxy_log_return_pct": np.log(filled["dxy"]).diff() * 100,
        "spread_usd": px - filled[other],
    }, index=px.index)
    lagged = current.shift(1)                                    # the return on day t is explained by day t-1 inputs
    returns = np.log(px).diff().to_numpy() * 100
    return px, returns, lagged, current


def _scaler(raw_x):
    means, scales = raw_x.mean(axis=0), raw_x.std(axis=0, ddof=0)
    if not np.isfinite(raw_x).all() or (scales <= 1e-8).any():
        raise ValueError("non-finite or constant predictor in the estimation window")
    return {"means": means, "scales": scales}


def _standardize(raw, sc):
    return (np.asarray(raw) - sc["means"]) / sc["scales"]


def _future_x(current_row, horizon, sc):
    """Conditional frozen-input path: first step uses origin-known values, later WTI/DXY changes are 0 and the
    spread stays at its origin value (a scenario, not a joint price forecast)."""
    raw = np.tile(np.asarray(current_row, float), (horizon, 1))
    raw[1:, :2] = 0.0
    return _standardize(raw, sc)


# --------------------------------------------------------------------------- baseline MS-AR(1)
def _make_baseline(history):
    return MarkovAutoregression(np.asarray(history, float), k_regimes=2, order=1, trend="c", switching_trend=True,
                                switching_ar=False, switching_variance=True)


def _unpack_baseline(model, params):
    mu = np.array([params[model.parameters[s, "exog"]][0] for s in range(2)])
    phi = np.array([params[model.parameters[s, "autoregressive"]] for s in range(2)])
    var = np.array([params[model.parameters[s, "variance"]][0] for s in range(2)])
    return mu, phi, var, model.regime_transition_matrix(params)[:, :, 0]


def _mean_square_radius(phi, transition):
    p = phi.shape[1]
    comps = []
    for row in phi:
        a = np.zeros((p, p)); a[0] = row
        if p > 1:
            a[1:, :-1] = np.eye(p - 1)
        comps.append(a)
    op = np.block([[transition[j, i] * np.kron(comps[j], comps[j]) for i in range(2)] for j in range(2)])
    return float(np.max(np.abs(np.linalg.eigvals(op))))


def _fit_baseline(history, seed, warm=None):
    model = _make_baseline(history)
    accepted, attempts = [], []
    for n, start in enumerate([None, warm] if warm is not None else [None, None]):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(start_params=start, search_reps=5 if start is None else 0, maxiter=300, disp=False,
                                cov_type="none", rng=np.random.default_rng(seed + n))
            mu, phi, var, trans = _unpack_baseline(model, fit.params)
            occ = np.asarray(fit.filtered_marginal_probabilities).mean(axis=0)
            radius = _mean_square_radius(phi, trans)
            good = (bool(fit.mle_retvals.get("converged", False)) and np.isfinite(fit.llf) and np.isfinite(fit.params).all()
                    and (var > 1e-6).all() and occ.min() >= 0.01 and np.diag(trans).min() > 0.001
                    and np.diag(trans).max() < 0.99999 and radius < 0.999)
            attempts.append({"start": "cold" if start is None else "warm", "accepted": bool(good), "llf": float(fit.llf),
                             "radius": radius})
            if good:
                accepted.append(fit)
        except Exception as exc:                                  # a failed attempt is recorded, not fatal
            attempts.append({"accepted": False, "error": str(exc)[:200]})
    if not accepted:
        raise RuntimeError(f"no admissible MS-AR fit: {attempts}")
    best = max(accepted, key=lambda f: f.llf)
    return np.asarray(best.params), attempts


def _exact_baseline(model, params, joint, history, horizon):
    mu, phi, var, trans = _unpack_baseline(model, params)
    p = model.order
    probs = joint.sum(axis=tuple(range(1, joint.ndim)))
    idx = np.indices(joint.shape)
    last_y = np.asarray(history[-p:])[::-1]
    moments = np.zeros((2, p))
    for s in range(2):
        for lag in range(p):
            moments[s, lag] = np.sum(joint * (idx[0] == s) * (last_y[lag] - mu[idx[lag]]))
    daily = np.zeros(horizon)
    for step in range(horizon):
        nxt = trans @ probs
        mixed = trans @ moments
        newest = np.sum(phi * mixed, axis=1)
        daily[step] = np.dot(nxt, mu) + newest.sum()
        moments = np.column_stack([newest, mixed[:, :-1]])
        probs = nxt
    return np.cumsum(daily) / 100.0


def _simulate_baseline(model, params, joint, history, horizon, n_sim, seed):
    mu, phi, var, trans = _unpack_baseline(model, params)
    rng = np.random.default_rng(seed)
    flat = np.maximum(joint.ravel(), 0); flat = flat / flat.sum()
    draw = rng.choice(flat.size, n_sim, p=flat)
    states = np.column_stack(np.unravel_index(draw, joint.shape))
    p = model.order
    past_states = states[:, :p].copy()
    past_ret = np.tile(np.asarray(history[-p:])[::-1], (n_sim, 1))
    out, total = np.zeros((n_sim, horizon)), np.zeros(n_sim)
    for step in range(horizon):
        new_state = (rng.random(n_sim) >= trans[0, past_states[:, 0]]).astype(int)
        mean = mu[new_state] + np.sum(phi[new_state] * (past_ret - mu[past_states]), axis=1)
        y = mean + np.sqrt(var[new_state]) * rng.standard_normal(n_sim)
        total += y / 100.0
        out[:, step] = total
        past_ret = np.column_stack([y, past_ret[:, :-1]])
        past_states = np.column_stack([new_state, past_states[:, :-1]])
    if not np.isfinite(out).all() or np.max(np.abs(out)) > 20:
        raise RuntimeError("explosive or non-finite simulation")
    return out


# --------------------------------------------------------------------------- MS-ARX (AR(1) + lagged inputs)
def _make_arx(history, x):
    return MarkovAutoregression(np.asarray(history, float), k_regimes=2, order=1, trend="c", switching_trend=False,
                                switching_ar=False, switching_variance=True, exog=x, switching_exog=False)


def _unpack_arx(model, params):
    intercept, beta = np.zeros(2), np.zeros((2, model.k_exog))
    for s in range(2):
        c = np.asarray(params[model.parameters[s, "exog"]])
        if model.k_trend:
            intercept[s] = c[0]
        beta[s] = c[model.k_trend:]
    phi = np.array([params[model.parameters[s, "autoregressive"]][0] for s in range(2)])
    var = np.array([params[model.parameters[s, "variance"]][0] for s in range(2)])
    return intercept, beta, phi, var, model.regime_transition_matrix(params)[:, :, 0]


def _fit_arx(history, x, seed, warm=None):
    model = _make_arx(history, x)
    accepted, attempts = [], []
    for n, start in enumerate([None, warm] if warm is not None else [None, None]):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(start_params=start, search_reps=4 if start is None else 0, maxiter=350, disp=False,
                                cov_type="none", rng=np.random.default_rng(seed + n))
            _, _, phi, var, trans = _unpack_arx(model, fit.params)
            occ = np.asarray(fit.filtered_marginal_probabilities).mean(axis=0)
            good = (bool(fit.mle_retvals.get("converged", False)) and np.isfinite(fit.params).all() and np.isfinite(fit.llf)
                    and (var > 1e-6).all() and occ.min() >= 0.01 and np.max(np.abs(phi)) < 0.999
                    and np.diag(trans).min() > 0.001 and np.diag(trans).max() < 0.99999)
            attempts.append({"start": "cold" if start is None else "warm", "accepted": bool(good), "llf": float(fit.llf)})
            if good:
                accepted.append(fit)
        except Exception as exc:
            attempts.append({"accepted": False, "error": str(exc)[:200]})
    if not accepted:
        raise RuntimeError(f"no admissible MS-ARX fit: {attempts}")
    best = max(accepted, key=lambda f: f.llf)
    return np.asarray(best.params), attempts


def _exact_arx(model, params, probs, y_last, x_last, future_x, horizon):
    intercept, beta, phi, var, trans = _unpack_arx(model, params)
    probs = np.asarray(probs, float).copy()
    moments = probs * (y_last - (intercept + beta @ x_last))
    daily = np.zeros(horizon)
    for step in range(horizon):
        moments = phi * (trans @ moments)
        probs = trans @ probs
        daily[step] = probs @ (intercept + beta @ future_x[step]) + moments.sum()
    return np.cumsum(daily) / 100.0


def _simulate_arx(model, params, probs, y_last, x_last, future_x, horizon, n_sim, seed):
    intercept, beta, phi, var, trans = _unpack_arx(model, params)
    rng = np.random.default_rng(seed)
    probs = np.maximum(np.asarray(probs, float), 0); probs = probs / probs.sum()
    state = rng.choice(2, n_sim, p=probs)
    centred = y_last - (intercept + beta @ x_last)[state]
    out, total = np.zeros((n_sim, horizon)), np.zeros(n_sim)
    for step in range(horizon):
        state = (rng.random(n_sim) >= trans[0, state]).astype(int)
        centred = phi[state] * centred + np.sqrt(var[state]) * rng.standard_normal(n_sim)
        total += ((intercept + beta @ future_x[step])[state] + centred) / 100.0
        out[:, step] = total
    if not np.isfinite(out).all() or np.max(np.abs(out)) > 20:
        raise RuntimeError("explosive or non-finite simulation")
    return out


# --------------------------------------------------------------------------- plugin
def _window(returns, lagged, i, window):
    """The last `window` returns ending at i. A shorter history is accepted down to 80% of the window (WTI starts in
    2020-06), never fewer."""
    lo = max(2, i - window + 1)
    if i - lo + 1 < 0.8 * window:
        raise ValueError("not enough history for the estimation window")
    history = returns[lo:i + 1]
    raw_x = lagged.iloc[lo:i + 1].to_numpy()
    if not np.isfinite(history).all() or not np.isfinite(raw_x).all():
        raise ValueError("missing returns or predictors inside the estimation window")
    return history, raw_x, lo


class MSAR(ForecastPlugin):
    name = "msar"

    # ---- fitting
    def _fit_one(self, spec, px, returns, lagged, i, seed):
        history, raw_x, lo = _window(returns, lagged, i, spec["window"])
        if spec["family"] == "msarx":
            sc = _scaler(raw_x)
            params, attempts = _fit_arx(history, _standardize(raw_x, sc), seed)
        else:
            sc = None
            params, attempts = _fit_baseline(history, seed)
        return {"family": spec["family"], "window": int(spec["window"]), "weight": float(spec["weight"]),
                "params": params, "scaler": sc, "fit_date": str(px.index[i].date()),
                "window_start": str(px.index[lo].date()), "n_obs": int(len(history)), "attempts": attempts}

    def _train_one(self, inputs, cfg, target):
        px, returns, lagged, current = _panel(inputs, cfg, target)
        i = len(px) - 1
        return {"models": {name: self._fit_one(spec, px, returns, lagged, i, cfg.msar_seed + 99999)
                           for name, spec in cfg.msar_models.items()}, "train_end": str(px.index[i].date())}

    def train(self, inputs, cfg):
        _need_statsmodels()
        series = {}
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            try:
                series[t] = self._train_one(target_inputs(inputs, cfg, t), cfg, t)
            except Exception as exc:                              # one series failing must not lose the other
                print(f"[msar] {t} skipped: {exc}")
        if not series:
            raise RuntimeError("MS-AR could not be fitted for any series")
        return {"plugin": self.name, "version": VERSION, "statsmodels": statsmodels.__version__,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_end": max(v["train_end"] for v in series.values()), "target": cfg.target,
                "series": series, "refit_every": cfg.msar_refit_every,
                "note": "specification (window, mean weight) selected by the notebook on 2022-2023 validation (Brent); "
                        "parameters are re-estimated at each retrain"}

    # ---- forecasting from fixed parameters
    def _forecast(self, name, par, px, returns, lagged, current, i, cfg, target):
        horizons = list(cfg.horizons)
        H = max(horizons)
        history, raw_x, _ = _window(returns, lagged, i, par["window"])
        seed = cfg.msar_seed + int(i)
        if par["family"] == "msarx":
            sc = par["scaler"]
            x = _standardize(raw_x, sc)
            fut = _future_x(current.iloc[i].to_numpy(), H, sc)
            model = _make_arx(history, x)
            filt = model.filter(par["params"], return_raw=True)
            probs = np.asarray(filt.filtered_marginal_probabilities[:, -1])
            raw = _exact_arx(model, par["params"], probs, history[-1], x[-1], fut, H)
            paths = _simulate_arx(model, par["params"], probs, history[-1], x[-1], fut, H, cfg.msar_n_sim, seed)
            high = int(np.argmax(_unpack_arx(model, par["params"])[3]))
            hp = float(probs[high])
        else:
            model = _make_baseline(history)
            filt = model.filter(par["params"], return_raw=True)
            joint = np.asarray(filt.filtered_joint_probabilities[..., -1])
            raw = _exact_baseline(model, par["params"], joint, history, H)
            paths = _simulate_baseline(model, par["params"], joint, history, H, cfg.msar_n_sim, seed)
            marg = joint.sum(axis=tuple(range(1, joint.ndim)))
            hp = float(marg[int(np.argmax(_unpack_baseline(model, par["params"])[2]))])
        w = par["weight"]
        paths = paths + (w - 1.0) * raw[None, :]                  # mean adjustment, noise not shrunk
        return [{"asof_date": px.index[i], "target": target, "model": name, "horizon": int(h),
                 "forecast_return": float(w * raw[h - 1]), "sigma": float(paths[:, h - 1].std(ddof=1)),
                 "sigma_kind": "msar", "raw_forecast_return": float(raw[h - 1]), "regime_p": hp} for h in horizons]

    @staticmethod
    def _series_of(art, cfg):
        return art["series"] if "series" in art else {art.get("target", cfg.target): {"models": art["models"]}}

    def _predict_one(self, models, inputs, cfg, target):
        px, returns, lagged, current = _panel(inputs, cfg, target)
        i = len(px) - 1
        rows = []
        for name, par in models.items():
            rows += self._forecast(name, par, px, returns, lagged, current, i, cfg, target)
        return pd.DataFrame(rows)

    def _check_version(self, art):
        _need_statsmodels()
        if art.get("statsmodels", "0.15").split(".")[:2] != statsmodels.__version__.split(".")[:2]:
            raise RuntimeError(f"MS-AR artifact was fitted with statsmodels {art.get('statsmodels')}, "
                               f"found {statsmodels.__version__}: retrain (parameter layout may differ)")

    def predict(self, art, inputs, cfg):
        self._check_version(art)
        parts = []
        for t, v in self._series_of(art, cfg).items():
            if t in inputs.prices.columns:
                parts.append(self._predict_one(v["models"], target_inputs(inputs, cfg, t), cfg, t))
        return pd.concat(parts, ignore_index=True)

    def backtest(self, inputs, cfg, n_days, refit_every=5):
        """Past origins: parameters fitted at the first origin (data up to then), filtered forward to each origin.
        The notebook's own 63-day refit is used instead of the generic refit_every (fits are slow)."""
        _need_statsmodels()
        rows = []
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            ti = target_inputs(inputs, cfg, t)
            try:
                px, returns, lagged, current = _panel(ti, cfg, t)
                last = len(px) - 1
                fits = None
                for k, i in enumerate(range(last - n_days + 1, last + 1)):
                    if fits is None or k % cfg.msar_refit_every == 0:
                        fits = {name: self._fit_one(spec, px, returns, lagged, i, cfg.msar_seed + int(i))
                                for name, spec in cfg.msar_models.items()}
                    for name, par in fits.items():
                        rows += self._forecast(name, par, px, returns, lagged, current, i, cfg, t)
            except Exception as exc:
                print(f"[msar] backtest for {t} skipped: {exc}")
        return pd.DataFrame(rows)

    # ---- retrain report
    def describe_change(self, old, new, inputs, cfg):
        head = [f"Estimated on data through {new['train_end']}"
                + (f" (previous: {old['train_end']})." if old else " (first training).")]
        head.append("Window and mean weight are the notebook's validation choices (made on Brent) and are not re-selected "
                    "here; only the regime and AR parameters are re-estimated.")
        tables = []
        olds = self._series_of(old, cfg) if old is not None else {}
        for t, v in self._series_of(new, cfg).items():
            label = {"brent_fut": "Brent", "wti_fut": "WTI"}.get(t, t)
            ti = target_inputs(inputs, cfg, t)
            px, returns, lagged, current = _panel(ti, cfg, t)
            rows = []
            for name, par in v["models"].items():
                history, raw_x, _ = _window(returns, lagged, len(px) - 1, par["window"])
                if par["family"] == "msarx":
                    m = _make_arx(history, _standardize(raw_x, par["scaler"]))
                    intercept, beta, phi, var, trans = _unpack_arx(m, par["params"])
                    extra = ", ".join(f"{f.replace('_usd', '').replace('_log_return_pct', '')} {b:+.3f}"
                                      for f, b in zip(FEATURES, beta[0]))
                else:
                    m = _make_baseline(history)
                    intercept, phi_, var, trans = _unpack_baseline(m, par["params"])
                    phi, extra = phi_[:, 0], "-"
                order = np.argsort(var)
                rows.append([name, f"{par['window_start']} -> {par['fit_date']} ({par['n_obs']})",
                             " / ".join(f"{np.sqrt(var[s_]):.2f}%" for s_ in order),
                             " / ".join(f"{1 / (1 - trans[s_, s_]):.0f}d" for s_ in order),
                             f"{float(np.mean(phi)):.3f}", extra, f"{par['weight']:.2f}"])
            tables.append({"title": f"[{label}] MS-AR / MS-ARX parameters (regimes listed low vol / high vol)",
                           "header": ["Model", "Estimation window (obs)", "Shock sd (daily)", "Expected duration",
                                      "AR(1)", "Predictor coefficients (standardised)", "Mean weight"], "rows": rows})
            try:
                if t in olds:
                    a_ = self._predict_one(olds[t]["models"], ti, cfg, t)
                    b_ = self._predict_one(v["models"], ti, cfg, t)
                    mm = a_.merge(b_, on=["model", "horizon"], suffixes=("_old", "_new"))
                    px0 = float(ti.prices[t].dropna().iloc[-1])
                    mm = mm[mm["horizon"].isin([1, 20, 60])]
                    tables.append({"title": f"[{label}] Forecast impact: old vs new parameters, same data (price {px0:.2f})",
                                   "header": ["Model", "Horizon", "Old", "New", "Change"],
                                   "rows": [[r.model, f"+{int(r.horizon)}d", f"{px0 * np.exp(r.forecast_return_old):.2f}",
                                             f"{px0 * np.exp(r.forecast_return_new):.2f}",
                                             f"{100 * (np.exp(r.forecast_return_new - r.forecast_return_old) - 1):+.2f}%"]
                                            for r in mm.itertuples()]})
            except Exception as exc:
                head.append(f"[{label}] (Could not compare forecasts: {exc})")
        return {"headline": head, "tables": tables}


register(MSAR())
