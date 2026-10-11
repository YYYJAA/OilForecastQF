"""Feature engineering (logic from the team's lasso.ipynb, unchanged).

X.loc[t] uses information available at the close of t.  Y[h].loc[t] = log P(t+h) - log P(t).
"""
import numpy as np
import pandas as pd

from .config import Config


def _safe_log(s):
    return np.log(s.where(s > 0))


def _iso(index):
    iso = index.isocalendar()
    return iso["year"].astype(int).to_numpy(), np.minimum(iso["week"].astype(int).to_numpy(), 52)


def seasonal_anomaly(s, years, min_years=3):
    """Value minus the mean of the same ISO week in the previous `years` years (prior years only)."""
    s = s.dropna()
    if s.empty:
        return s
    yy, ww = _iso(s.index)
    tab = pd.DataFrame({"y": yy, "w": ww, "v": s.to_numpy()}).groupby(["y", "w"])["v"].mean().unstack("w")
    tab = tab.reindex(range(int(tab.index.min()), int(tab.index.max()) + 1))
    base = tab.shift(1).rolling(years, min_periods=min_years).mean()
    long = base.rename_axis(index="y", columns="w").reset_index().melt(
        id_vars="y", var_name="w", value_name="b").set_index(["y", "w"])["b"]
    b = long.reindex(pd.MultiIndex.from_arrays([yy, ww])).to_numpy()
    return s - b


def asof_daily(rel, days, stale_days):
    """Latest released value on or before each day; missing if older than stale_days."""
    rel = rel.dropna().sort_index()
    out = np.full(len(days), np.nan)
    if rel.empty:
        return pd.Series(out, index=days)
    pos = rel.index.searchsorted(days, side="right") - 1
    ok = pos >= 0
    out[ok] = rel.to_numpy()[pos[ok]]
    age = np.full(len(days), np.inf)
    age[ok] = (days[ok] - rel.index[pos[ok]]).days
    out[age > stale_days] = np.nan
    return pd.Series(out, index=days)


def build_features(prices, eia, weekly, target, cfg: Config):
    days = prices[target].dropna().index
    full = prices.reindex(prices.index.union(days)).ffill(limit=cfg.ffill_limit)
    P = full.reindex(days)
    logp = _safe_log(P[target])
    r = logp.diff()
    feats, lag = {}, cfg.feature_lag

    # own price
    for k in (1, 5, 20, 60):
        feats[f"own_ret{k}"] = logp - logp.shift(k)
    feats["own_rv20"] = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    feats["own_rv60"] = r.rolling(60, min_periods=45).std() * np.sqrt(252)
    for m in (50, 200):
        ma = P[target].rolling(m, min_periods=int(0.8 * m)).mean()
        feats[f"own_gap_ma{m}"] = logp - _safe_log(ma)

    # other daily series
    for c in [c for c in P.columns if c != target and c not in cfg.exclude]:
        is_level = c in cfg.level_series
        x = P[c] if is_level else _safe_log(P[c])
        name = "d" if is_level else "ret"
        for k in (1, 5, 20):
            feats[f"{c}__{name}{k}"] = (x - x.shift(k)).shift(lag)
        if is_level:
            feats[f"{c}__lvl"] = x.shift(lag)

    # Brent-WTI spread
    if {"brent_fut", "wti_fut"} <= set(P.columns):
        spr = _safe_log(P["brent_fut"]) - _safe_log(P["wti_fut"])
        sl = 0 if target in ("brent_fut", "wti_fut") else lag
        feats["brent_wti__lvl"] = spr.shift(sl)
        feats["brent_wti__d5"] = (spr - spr.shift(5)).shift(sl)

    # 3-2-1 crack spread ($/bbl) scaled by WTI; gasoline and heating oil are in $/gal
    if {"gasoline", "heating_oil", "wti_fut"} <= set(P.columns):
        cl = P["wti_fut"].where(P["wti_fut"] > 0)
        crack = (2 * 42 * P["gasoline"] + 42 * P["heating_oil"] - 3 * cl) / 3 / cl
        crack = crack.where(crack.abs() < 2)
        feats["crack321__lvl"] = crack.shift(lag)
        feats["crack321__anom"] = seasonal_anomaly(crack, cfg.season_years).reindex(days).shift(lag)
        feats["crack321__d20"] = (crack - crack.shift(20)).shift(lag)

    # futures curve slope (EIA WTI contracts 1 and 4); EIA's files stop in 2024, so this goes missing
    if eia is not None and len(eia) and {"wti_c1", "wti_c4"} <= set(eia.columns):
        e = eia.reindex(eia.index.union(days)).ffill(limit=cfg.slow_ffill_limit).reindex(days)
        slope = np.log(e["wti_c1"].where(e["wti_c1"] > 0) / e["wti_c4"].where(e["wti_c4"] > 0))
        feats["wti_slope__lvl"] = slope
        feats["wti_slope__d5"] = slope - slope.shift(5)
        feats["wti_slope__d20"] = slope - slope.shift(20)

    # weekly series, entered from release dates
    def released(s, lagd):
        return s.set_axis(s.index + pd.Timedelta(days=lagd))
    if "crude_stocks" in weekly:
        s, lagd = weekly["crude_stocks"]
        ls = _safe_log(s)
        anom = seasonal_anomaly(ls, cfg.season_years)
        chg_anom = seasonal_anomaly(ls.diff(), cfg.season_years)
        feats["inv__anom"] = asof_daily(released(anom, lagd), days, cfg.weekly_stale_days)
        feats["inv__anom_d4"] = asof_daily(released(anom - anom.shift(4), lagd), days, cfg.weekly_stale_days)
        feats["inv__chg_anom"] = asof_daily(released(chg_anom, lagd), days, cfg.weekly_stale_days)
    if "wei" in weekly:
        s, lagd = weekly["wei"]
        feats["wei__lvl"] = asof_daily(released(s, lagd), days, cfg.weekly_stale_days)
        feats["wei__d4"] = asof_daily(released(s - s.shift(4), lagd), days, cfg.weekly_stale_days)

    # calendar seasonality
    doy = days.dayofyear.to_numpy()
    for k in (1, 2):
        feats[f"season_sin{k}"] = pd.Series(np.sin(2 * np.pi * k * doy / 365.25), index=days)
        feats[f"season_cos{k}"] = pd.Series(np.cos(2 * np.pi * k * doy / 365.25), index=days)

    X = pd.DataFrame(feats, index=days).replace([np.inf, -np.inf], np.nan)
    Y = {h: (logp.shift(-h) - logp).rename(f"y_h{h}") for h in cfg.horizons}
    return X, Y, logp


def train_rows(i, h, window):
    """Row j has y = logP(j+h) - logP(j), observed only at j+h, so j <= i-h (the purge rule)."""
    hi = i - h
    lo = max(0, hi - window + 1)
    return np.arange(lo, hi + 1) if hi >= 0 else np.array([], dtype=int)


def base_cols(X):
    """Everything except the regime probability and its interactions."""
    return [c for c in X.columns if c != "regime_p" and not c.startswith("rs__")]


def add_regime_features(X, P, cfg: Config):
    X = X.copy()
    X["regime_p"] = P.iloc[:, -1].reindex(X.index)
    for c in base_cols(X):
        if not c.startswith("season_") and any(s in c for s in cfg.rs_patterns):
            X[f"rs__{c}"] = X[c] * X["regime_p"]
    return X
