"""ARMA-GARCH: zero-mean ARMA on daily log returns, GARCH(1,1) with Student-t errors on its residuals.

Specification from the team's gp_arma_garch.ipynb (MA(2), GARCH(1,1)-t). The notebook refits both models
every day for a 1-day forecast. Here the parameters are estimated at each (quarterly) retrain and, between
retrains, simply *filtered* through the new data, which gives the same one-step quantities with fixed
parameters. Forecasts are produced for every horizon:

    mean_h   = sum of the ARMA step forecasts 1..h
    sigma_h  = sqrt( sum_k c_k^2 * v_k ),  v_k = GARCH conditional variance of the shock k steps ahead,
               c_k = sum of the ARMA impulse responses up to lag h-k  (exact variance of the h-day return)

The model runs on every series in cfg.arma_garch_targets (each series is its own panel in the e-mail).
Returns are handled in percent internally and converted back to decimal log returns.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma2ma

from ..registry import ForecastPlugin, register

MODEL = "ARMA-GARCH"
VERSION = "1"


def _returns_pct(price: pd.Series) -> pd.Series:
    """Daily log returns in %, non-positive prices removed (same treatment as the notebook)."""
    p = price.dropna()                      # trading days of this series only (the price table is an outer join)
    p = p.where(p > 0)                      # negative / zero prices (WTI, 2020-04-20) -> missing
    return (np.log(p).diff().dropna() * 100.0)


def _arma(ret: pd.Series, cfg):
    p, q = cfg.arma_order
    return ARIMA(ret.reset_index(drop=True), order=(p, 0, q), trend="n")


def _garch(resid, cfg):
    gp, gq = cfg.garch_pq
    return arch_model(np.asarray(resid), mean="Zero", vol="GARCH", p=gp, q=gq, dist=cfg.garch_dist)


def _fit_series(ret: pd.Series, cfg) -> dict:
    arma_res = _arma(ret, cfg).fit()
    g_res = _garch(arma_res.resid, cfg).fit(disp="off", show_warning=False)
    p, q = cfg.arma_order
    names = list(g_res.params.index)
    a, b = float(g_res.params.get("alpha[1]", np.nan)), float(g_res.params.get("beta[1]", np.nan))
    omega = float(g_res.params.get("omega", np.nan))
    return {
        "arma_params": np.asarray(arma_res.params, float), "garch_params": np.asarray(g_res.params, float),
        "garch_param_names": names, "arma_param_names": list(arma_res.params.index),
        "converged_arma": bool(arma_res.mle_retvals.get("converged", True)),
        "converged_garch": bool(getattr(g_res, "convergence_flag", 0) == 0),
        "n_obs": int(len(ret)), "first_date": str(ret.index[0].date()), "last_date": str(ret.index[-1].date()),
        "aic_arma": float(arma_res.aic), "aic_garch": float(g_res.aic),
        "persistence": a + b, "uncond_vol_daily_pct": float(np.sqrt(omega / (1 - a - b))) if a + b < 1 else float("nan"),
        "nu": float(g_res.params.get("nu", np.nan)),
    }


def _forecast_series(par: dict, ret: pd.Series, cfg, horizons):
    """{h: (mean log return, sd of the h-day log return)} using fixed parameters and data up to ret.index[-1]."""
    p, q = cfg.arma_order
    H = max(horizons)
    am = _arma(ret, cfg).smooth(par["arma_params"])
    m = np.asarray(am.forecast(steps=H), float)                      # % per step
    gm = _garch(am.resid, cfg).fix(par["garch_params"])
    v = gm.forecast(horizon=H, reindex=False).variance.iloc[-1].to_numpy(dtype=float)   # %^2 per step
    ar = np.r_[1.0, -np.asarray(par["arma_params"][:p])] if p else np.array([1.0])
    ma = np.r_[1.0, np.asarray(par["arma_params"][p:p + q])] if q else np.array([1.0])
    psi_cum = np.cumsum(arma2ma(ar, ma, lags=H))                      # psi_cum[j] = psi_0 + ... + psi_j
    out = {}
    for h in horizons:
        c = psi_cum[h - np.arange(1, h + 1)]                          # weight of the shock k steps ahead
        out[int(h)] = (float(m[:h].sum()) / 100.0, float(np.sqrt((c ** 2 * v[:h]).sum())) / 100.0)
    return out


def _rows(target, asof, fc):
    return [{"asof_date": asof, "target": target, "model": MODEL, "horizon": h, "forecast_return": r,
             "sigma": s, "sigma_kind": "garch"} for h, (r, s) in fc.items()]


class ArmaGarch(ForecastPlugin):
    name = "arma_garch"

    def _targets(self, inputs, cfg):
        return [t for t in cfg.arma_garch_targets if t in inputs.prices.columns]

    def train(self, inputs, cfg):
        series = {}
        for t in self._targets(inputs, cfg):
            series[t] = _fit_series(_returns_pct(inputs.prices[t]), cfg)
        if not series:
            raise RuntimeError("no target series available for ARMA-GARCH")
        last = max(s["last_date"] for s in series.values())
        warn = [f"{t}: {k} did not converge" for t, s in series.items()
                for k in ("arma", "garch") if not s[f"converged_{k}"]]
        return {"plugin": self.name, "version": VERSION, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_end": last, "arma_order": list(cfg.arma_order), "garch_pq": list(cfg.garch_pq),
                "garch_dist": cfg.garch_dist, "series": series, "warnings": warn}

    def predict(self, art, inputs, cfg):
        rows = []
        for t, par in art["series"].items():
            if t not in inputs.prices.columns:
                continue
            ret = _returns_pct(inputs.prices[t])
            rows += _rows(t, ret.index[-1], _forecast_series(par, ret, cfg, cfg.horizons))
        return pd.DataFrame(rows)

    def backtest(self, inputs, cfg, n_days, refit_every=5):
        rows = []
        for t in art_targets(inputs, cfg):
            ret = _returns_pct(inputs.prices[t])
            par = None
            for k, i in enumerate(range(len(ret) - n_days, len(ret))):
                hist = ret.iloc[: i + 1]
                if par is None or k % refit_every == 0:
                    par = _fit_series(hist, cfg)
                rows += _rows(t, hist.index[-1], _forecast_series(par, hist, cfg, cfg.horizons))
        return pd.DataFrame(rows)

    def describe_change(self, old, new, inputs, cfg):
        head, tables = [], []
        head.append(f"Estimated on all data through {new['train_end']}"
                    + (f" (previous: {old['train_end']})." if old else " (first training)."))
        rows = []
        for t, s in new["series"].items():
            o = (old or {}).get("series", {}).get(t)
            def pair(key, d=3):
                nv = s.get(key)
                if o is None or o.get(key) is None:
                    return f"{nv:.{d}f}"
                return f"{o[key]:.{d}f} -> {nv:.{d}f}"
            ma = " / ".join(f"{(f'{x:.3f} -> ' if o is not None else '')}{y:.3f}"
                            for x, y in zip((o["arma_params"] if o is not None else s["arma_params"]), s["arma_params"])
                            ) if s["arma_params"].size else "-"
            names = s["garch_param_names"]
            gp = dict(zip(names, s["garch_params"]))
            rows.append([t, f"{s['n_obs']:,}", f"{s['first_date']} -> {s['last_date']}",
                         " | ".join(f"{n} {v:.3f}" for n, v in zip(s["arma_param_names"][:-1], s["arma_params"][:-1])) or "-",
                         f"alpha {gp.get('alpha[1]', float('nan')):.3f}, beta {gp.get('beta[1]', float('nan')):.3f}, nu {gp.get('nu', float('nan')):.1f}",
                         pair("persistence"), f"{s['uncond_vol_daily_pct'] * np.sqrt(252):.0f}%",
                         "yes" if s["converged_arma"] and s["converged_garch"] else "NO"])
        tables.append({"title": "ARMA-GARCH parameters",
                       "header": ["Series", "Observations", "Sample", "ARMA coefficients", "GARCH", "alpha+beta (old -> new)",
                                  "Long-run vol (ann.)", "Converged"], "rows": rows})
        if new.get("warnings"):
            head.append("WARNING: " + "; ".join(new["warnings"]) + " (parameters kept, check them).")
        try:
            if old is not None:
                a = self.predict(old, inputs, cfg)
                b = self.predict(new, inputs, cfg)
                m = a.merge(b, on=["target", "model", "horizon"], suffixes=("_old", "_new"))
                m = m[m["horizon"].isin([1, 20, 60])]
                tables.append({"title": "Volatility forecast impact (1-sd of the h-day return, same data)",
                               "header": ["Series", "Horizon", "Old model", "New model", "Change"],
                               "rows": [[r.target, f"+{int(r.horizon)}d", f"{100 * r.sigma_old:.2f}%", f"{100 * r.sigma_new:.2f}%",
                                         f"{100 * (r.sigma_new / r.sigma_old - 1):+.1f}%"] for r in m.itertuples()]})
        except Exception as exc:
            head.append(f"(Could not compare forecasts: {exc})")
        return {"headline": head, "tables": tables}


def art_targets(inputs, cfg):
    return [t for t in cfg.arma_garch_targets if t in inputs.prices.columns]


register(ArmaGarch())
