"""MS-AR / MS-ARX: two-regime Markov-switching autoregressions on daily log returns (the team's msar_common.ipynb).

    MS-AR   AR(1) with a switching mean and variance (regimes ordered calm / turbulent), shared AR coefficient
    MS-ARX  the same plus three lagged inputs with shared coefficients: the other oil benchmark's % return, the dollar
            index % return and the Brent-WTI spread in USD (each known at t-1, standardised on the window, clipped at +-10)

Both use a 1,260-day estimation window and a COMMON calendar (days on which Brent and WTI futures both trade, so the two
series use the same dates). The point forecast is the exact regime-mixture mean of the cumulative return, shrunk by half
(an equal-weight combination with no change); the band is the spread of simulated cumulative returns (regime mixture; the
parameter uncertainty is not included, as in the notebook).

Pipeline use: parameters are estimated at each quarterly retrain. Between retrains they are FIXED: every day the regime filter is
run through the new data, and the forecast starts from today's filtered regime probabilities and inputs.
Fitted for every series in cfg.targets; for MS-ARX the "other benchmark" is WTI for Brent and Brent for WTI. The settings were
chosen by the team on Brent and are reused for WTI. Requires statsmodels >= 0.15.
"""
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from ..registry import ForecastPlugin, register

VERSION = "2"
MODELS = {"MS-AR": False, "MS-ARX": True}               # name -> uses the lagged inputs
OTHER = {"brent_fut": "wti_fut", "wti_fut": "brent_fut"}
LABELS = {"brent_fut": "Brent", "wti_fut": "WTI"}
FEATURES = ("other_oil_return_pct", "dxy_return_pct", "spread_usd")


def _need_statsmodels():
    if tuple(int(x) for x in statsmodels.__version__.split(".")[:2]) < (0, 15):
        raise RuntimeError(f"MS-AR needs statsmodels >= 0.15 (found {statsmodels.__version__}); see requirements.txt")


# --------------------------------------------------------------------------- data
class Panel:
    def __init__(self, dates, price, r, x_lag, x_now, target):
        self.dates, self.price, self.r, self.x_lag, self.x_now, self.target = dates, price, r, x_lag, x_now, target

    def pos(self, date):
        i = self.dates.searchsorted(pd.Timestamp(date))
        return int(i) if i < len(self.dates) and self.dates[i] == pd.Timestamp(date) else None


def _panel(inputs, cfg, target):
    """Common calendar (both futures quoted and positive); returns in %, inputs known at t-1 (x_lag) and at t (x_now).
    The target's own history is cut at cfg.series_start (WTI settled at -37.63 on 2020-04-20)."""
    px = inputs.prices
    both = px[["brent_fut", "wti_fut"]].where(px[["brent_fut", "wti_fut"]] > 0).dropna()
    dxy = px["dxy"].where(px["dxy"] > 0)
    dxy = dxy.reindex(dxy.index.union(both.index)).ffill(limit=cfg.msar_fill_limit).reindex(both.index)
    other = OTHER[target]
    ret = 100 * np.log(both).diff()
    now = pd.DataFrame({FEATURES[0]: ret[other], FEATURES[1]: 100 * np.log(dxy).diff(),
                        FEATURES[2]: both["brent_fut"] - both["wti_fut"]}, index=both.index)
    start = cfg.series_start.get(target)
    keep = both.index >= pd.Timestamp(start) if start else np.ones(len(both), bool)
    d = both.index[keep]
    return Panel(d, both[target][keep], ret[target][keep].to_numpy(), now.shift(1)[keep].to_numpy(), now[keep].to_numpy(), target)


# --------------------------------------------------------------------------- estimation
def _stdz(a, m, s):
    return np.clip(np.nan_to_num((a - m) / s, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)


def _model(y, X):
    return MarkovAutoregression(y, k_regimes=2, order=1, exog=X, switching_ar=False, switching_variance=True,
                                switching_exog=False)


def _fit_ms(y, X, prev):
    mod = _model(y, X)
    attempts = ([dict(start_params=prev)] if prev is not None else []) + [dict(), dict(search_reps=5),
                                                                        dict(method="nm", maxiter=2000)]
    for kw in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = mod.fit(disp=False, **kw)
            if np.isfinite(res.llf) and np.all(np.isfinite(res.params)):
                return mod, np.asarray(res.params), float(res.llf)
        except Exception:
            pass
    if prev is None:
        raise RuntimeError("Markov-switching fit failed")
    return mod, np.asarray(prev), float("nan")                 # keep the previous parameters


def _unpack(names, p):
    d = dict(zip(names, p))
    mu = np.array([d["const[0]"], d["const[1]"]])
    sig = np.sqrt([d["sigma2[0]"], d["sigma2[1]"]])
    P = np.array([[d["p[0->0]"], 1 - d["p[0->0]"]], [d["p[1->0]"], 1 - d["p[1->0]"]]])
    beta = np.array([d[k] for k in names if k.startswith("x")])
    order = np.argsort(sig)                                      # 0 = calm, 1 = turbulent
    return {"mu": mu[order], "sig": sig[order], "P": P[order][:, order], "beta": beta, "phi": float(d["ar.L1"]),
            "order": order}


def _fit(P, i0, use_x, cfg, prev=None):
    lo = max(i0 - cfg.msar_window + 1, 1)
    if i0 - lo < 500:
        raise ValueError(f"only {i0 - lo} returns before the training end (need at least 500)")
    y = np.nan_to_num(P.r[lo:i0 + 1])
    if use_x:
        m_, s_ = np.nanmean(P.x_lag[lo:i0 + 1], 0), np.nanstd(P.x_lag[lo:i0 + 1], 0)
        s_ = np.where(s_ > 1e-12, s_, 1.0)
        X = _stdz(P.x_lag[lo:i0 + 1], m_, s_)
    else:
        m_ = s_ = X = None
    mod, params, llf = _fit_ms(y, X, prev)
    par = _unpack(mod.param_names, params)
    return {"i0": int(i0), "date": str(P.dates[i0].date()), "lo": int(lo), "window_start": str(P.dates[lo].date()),
            "n_obs": int(i0 - lo + 1), "use_x": bool(use_x), "params": params, "par": par,
            "scaler": None if m_ is None else (m_, s_), "loglik": llf}


# --------------------------------------------------------------------------- forecasting
def _forecast_one(y, xb, filt, par, H, n_sim, rng, xnext):
    """Exact cumulative conditional mean and simulated cumulative returns (percent) for ONE origin."""
    mu, sig, P, phi = par["mu"], par["sig"], par["P"], par["phi"]
    Ez = y - filt @ mu - xb
    pk, mean = filt.copy(), np.zeros(H)
    for k in range(1, H + 1):
        pk = pk @ P
        mean[k - 1] = pk @ mu + phi ** k * Ez + (xnext if k == 1 else 0.0)
    s = (rng.random(n_sim) < filt[1]).astype(int)
    z = np.full(n_sim, y - xb) - mu[s]
    cum, out = np.zeros(n_sim), np.zeros((n_sim, H))
    for k in range(1, H + 1):
        stay = rng.random(n_sim) < P[s, s]
        s = np.where(stay, s, 1 - s)
        z = phi * z + sig[s] * rng.standard_normal(n_sim)
        cum = cum + mu[s] + z + (xnext if k == 1 else 0.0)
        out[:, k - 1] = cum
    return np.cumsum(mean), out


def _apply(fit, name, P, positions, cfg):
    i0 = P.pos(fit["date"])
    if i0 is None:
        raise ValueError(f"training date {fit['date']} not in the {P.target} data")
    pos = np.asarray(sorted(int(p) for p in positions if p >= i0))
    if len(pos) == 0:
        return pd.DataFrame()
    lo, par, params = fit["lo"], fit["par"], fit["params"]
    end = pos.max() + 1
    if fit["use_x"]:
        m_, s_ = fit["scaler"]
        Xs = _stdz(P.x_lag[lo:end], m_, s_)
        xn = _stdz(P.x_now, m_, s_)
    else:
        Xs = xn = None
    fmod = _model(np.nan_to_num(P.r[lo:end]), Xs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filt_all = np.asarray(fmod.filter(params).filtered_marginal_probabilities)[:, par["order"]]
    H, rows = max(cfg.horizons), []
    for p in pos:
        filt = filt_all[p - lo - 1]                              # row j <-> the return at position lo + 1 + j
        xb = float(Xs[p - lo] @ par["beta"]) if fit["use_x"] else 0.0
        xnext = float(xn[p] @ par["beta"]) if fit["use_x"] else 0.0
        rng = np.random.default_rng([cfg.msar_seed, int(p)])     # same draws whether run daily or in a batch
        cum_mean, sims = _forecast_one(float(np.nan_to_num(P.r[p])), xb, filt, par, H, cfg.msar_n_sim, rng, xnext)
        for h in cfg.horizons:
            rows.append({"asof_date": P.dates[p], "model": name, "horizon": int(h),
                         "forecast_return": float(cfg.msar_shrink * cum_mean[h - 1] / 100),
                         "sigma": float(sims[:, h - 1].std(ddof=1) / 100), "sigma_kind": "msar",
                         "raw_forecast_return": float(cum_mean[h - 1] / 100), "regime_p": float(filt[1])})
    return pd.DataFrame(rows)


def _summary(fits):
    out = {}
    for name, f in fits.items():
        par = f["par"]
        out[name] = {"window_start": f["window_start"], "n_obs": f["n_obs"], "train_end": f["date"],
                     "vol_calm_ann_pct": round(float(par["sig"][0] * np.sqrt(252)), 1),
                     "vol_turb_ann_pct": round(float(par["sig"][1] * np.sqrt(252)), 1),
                     "p_stay_calm": round(float(par["P"][0, 0]), 4), "p_stay_turb": round(float(par["P"][1, 1]), 4),
                     "ar_coef": round(par["phi"], 4), "loglik": round(f["loglik"], 1)}
        if f["use_x"]:
            out[name]["input_effects"] = {k: round(float(b), 4) for k, b in zip(FEATURES, par["beta"])}
    return out


class MSARPlugin(ForecastPlugin):
    name = "msar"

    def train(self, inputs, cfg):
        _need_statsmodels()
        fits, summ, errs = {}, {}, {}
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            try:
                P = _panel(inputs, cfg, t)
                i0 = len(P.dates) - 1
                fits[t] = {name: _fit(P, i0, ux, cfg) for name, ux in MODELS.items()}
                summ[t] = _summary(fits[t])
            except Exception as exc:
                errs[t] = str(exc)[:200]
        if not fits:
            raise RuntimeError(f"MS-AR could not be fitted: {errs}")
        first = cfg.target if cfg.target in fits else next(iter(fits))
        return {"plugin": self.name, "version": VERSION, "statsmodels": statsmodels.__version__,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_end": max(f["MS-AR"]["date"] for f in fits.values()), "target": first, "targets": list(fits),
                "series": summ, "errors": errs, "fits": fits,
                "note": "common Brent/WTI calendar, 1260-day window, forecast x 0.5"}

    def _run(self, art, inputs, cfg, pick):
        _need_statsmodels()
        parts = []
        for t, fits in art["fits"].items():
            if t not in inputs.prices.columns:
                continue
            P = _panel(inputs, cfg, t)
            for name, fit in fits.items():
                df = _apply(fit, name, P, pick(P), cfg)
                if len(df):
                    df["target"] = t
                    parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def predict(self, art, inputs, cfg):
        return self._run(art, inputs, cfg, lambda P: [len(P.dates) - 1])

    def rolling_predict(self, art, inputs, cfg, origins):
        return self._run(art, inputs, cfg, lambda P: [p for p in (P.pos(d) for d in origins) if p is not None])

    def backtest(self, inputs, cfg, n_days, refit_every=5):
        _need_statsmodels()
        step = max(int(refit_every), 1)
        parts = []
        for t in cfg.targets:
            if t not in inputs.prices.columns:
                continue
            P = _panel(inputs, cfg, t)
            last = len(P.dates) - 1
            starts = list(range(last - n_days + 1, last + 1, step))
            prev = {n: None for n in MODELS}
            for k, i in enumerate(starts):
                end = starts[k + 1] if k + 1 < len(starts) else last + 1
                for name, ux in MODELS.items():
                    fit = _fit(P, i, ux, cfg, prev[name])
                    prev[name] = fit["params"]
                    df = _apply(fit, name, P, range(i, end), cfg)
                    df["target"] = t
                    parts.append(df)
        return pd.concat(parts, ignore_index=True)

    def describe_change(self, old, new, inputs, cfg):
        head, tables = [], []
        olds = (old or {}).get("series", {}) if (old or {}).get("version") == VERSION else {}
        for t, models in new["series"].items():
            nm = LABELS.get(t, t)
            for name, s in models.items():
                o = olds.get(t, {}).get(name)
                head.append(f"[{nm}] {name}: trained through {s['train_end']}"
                            + (f" (previous: {o['train_end']})" if o else " (first training with this specification)")
                            + f"; window {s['window_start']} to {s['train_end']} ({s['n_obs']} returns). Volatility: calm "
                              f"{s['vol_calm_ann_pct']:.0f}% a year, turbulent {s['vol_turb_ann_pct']:.0f}%; the chance of staying "
                              f"in each regime a day is {100 * s['p_stay_calm']:.1f}% / {100 * s['p_stay_turb']:.1f}%.")
            rows = []
            for name, s in models.items():
                o = olds.get(t, {}).get(name, {})
                for key, lab in (("vol_calm_ann_pct", "Calm volatility (% a year)"), ("vol_turb_ann_pct", "Turbulent volatility (% a year)"),
                                 ("ar_coef", "AR coefficient")):
                    rows.append([name, lab, "n/a" if key not in o else f"{o[key]:g}", f"{s[key]:g}"])
                for k, v in s.get("input_effects", {}).items():
                    ov = o.get("input_effects", {}).get(k)
                    rows.append([name, f"Input effect: {k}", "n/a" if ov is None else f"{ov:g}", f"{v:g}"])
            tables.append({"title": f"[{nm}] MS-AR parameters", "header": ["Model", "Parameter", "Previous", "New"], "rows": rows})
        return {"headline": head, "tables": tables}


register(MSARPlugin())
