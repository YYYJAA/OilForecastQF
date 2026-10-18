"""Benchmark forecasts for the frozen evaluation (not production models, never in the daily e-mail).

HistMean  the average h-day log return observed up to the cutoff, held fixed.
AR        direct AR(p) regression of the h-day return on the last p daily returns; p chosen by BIC and coefficients
          estimated on data up to the cutoff only, then held fixed.
Both use only training rows whose h-day target was already realised at the cutoff (the purge rule).
"""
import numpy as np
import pandas as pd

PMAX = 10


def benchmark_forecasts(prices: dict, cutoff, origins_by_target: dict, horizons, series_start=None) -> pd.DataFrame:
    rows = []
    cutoff = pd.Timestamp(cutoff)
    for t, p in prices.items():
        p = p.dropna()
        p = p[p > 0]
        if series_start and series_start.get(t):
            p = p[p.index >= pd.Timestamp(series_start[t])]      # same data start as the models (WTI: after April 2020)
        lp = np.log(p)
        r = lp.diff().fillna(0.0)
        lags = pd.DataFrame({j: r.shift(j) for j in range(PMAX)}).fillna(0.0)          # r_t, r_{t-1}, ...
        end = p.index.searchsorted(cutoff, side="right") - 1                         # last position <= cutoff
        origins = [d for d in origins_by_target[t] if d in lags.index]
        for h in horizons:
            y = (lp.shift(-h) - lp)
            tr = np.arange(PMAX, end - h + 1)                                         # s + h <= cutoff
            ytr = y.iloc[tr].to_numpy()
            ok = np.isfinite(ytr)
            mean_h = float(ytr[ok].mean())
            best = (np.inf, 1, None)
            for k in range(1, PMAX + 1):
                Z = np.column_stack([np.ones(ok.sum()), lags.iloc[tr][list(range(k))].to_numpy()[ok]])
                beta, *_ = np.linalg.lstsq(Z, ytr[ok], rcond=None)
                rss = float(np.sum((ytr[ok] - Z @ beta) ** 2))
                n = ok.sum()
                bic = n * np.log(rss / n) + (k + 1) * np.log(n)
                if bic < best[0]:
                    best = (bic, k, beta)
            _, k, beta = best
            for d in origins:
                x = lags.loc[d, list(range(k))].to_numpy()
                for name, f in (("HistMean", mean_h), ("AR", float(beta[0] + x @ beta[1:]))):
                    rows.append({"asof_date": d, "target": t, "model": name, "horizon": int(h),
                                 "forecast_return": f, "sigma": np.nan, "sigma_kind": "none", "plugin": "benchmark"})
    return pd.DataFrame(rows)
