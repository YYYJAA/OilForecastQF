"""Feature panel and regime model shared by the regime-LASSO and XGBoost plugins.

Both come from the team's notebooks (lasson.ipynb, xgbn.ipynb), which use the same ~20 predictors, all known at the close of
day t and all changes or z-scores against the series' own past year (never raw levels). Here the panel is built for any target
series: for Brent the Brent-WTI spread is a predictor, for WTI the same spread with the sign reversed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import target_inputs

try:                                            # numba speeds up the regime filter; plain Python gives the same numbers
    from numba import njit
except Exception:                               # pragma: no cover - numba missing or broken
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

FRED_COLS = {"ust10y", "curve_10y2y", "t10y3m", "breakeven10y", "ovx", "epu"}     # may legitimately be <= 0
SHORT = {"brent_fut": "brent", "wti_fut": "wti"}
COUNTER = {"brent_fut": "wti_fut", "wti_fut": "brent_fut"}


def zscore(s, n=252):
    roll = s.rolling(n, min_periods=int(0.8 * n))
    return (s - roll.mean()) / roll.std()


@dataclass
class Panel:
    """Everything the models need, as arrays aligned on the target's own trading days."""
    target: str
    X: pd.DataFrame
    Y: dict                    # horizon -> 100 * (log P[t+h] - log P[t]) as a numpy array (percent)
    lp: pd.Series
    rv: np.ndarray             # 20-day realised volatility, annualised, decimal
    rr: np.ndarray             # daily log return in percent (first element NaN)
    A: np.ndarray              # X as float64
    dates: pd.DatetimeIndex
    price: pd.Series

    def pos(self, date):
        """Position of an exact date, or None."""
        i = self.dates.searchsorted(pd.Timestamp(date))
        return int(i) if i < len(self.dates) and self.dates[i] == pd.Timestamp(date) else None

    def first_valid(self):
        """First row where most features exist (the z-scores need a year of history)."""
        ok = np.flatnonzero((self.X.notna().mean(axis=1) > 0.7).to_numpy())
        return int(ok[0]) if len(ok) else len(self.X)


def build_panel(inputs, cfg, target) -> Panel:
    px = target_inputs(inputs, cfg, target).prices
    price = px[target].dropna()
    price = price[price > 0]
    other = px.drop(columns=[target]).copy()
    for c in other.columns:
        if c not in FRED_COLS:
            other[c] = other[c].where(other[c] > 0)
    other = other.reindex(other.index.union(price.index)).ffill(limit=cfg.ffill_limit).reindex(price.index)
    lp = np.log(price)
    r = lp.diff()
    vol = pd.DataFrame({"rv20": r.rolling(20, min_periods=15).std() * np.sqrt(252),
                        "rv60": r.rolling(60, min_periods=45).std() * np.sqrt(252)})
    nm = SHORT.get(target, target)
    f = {f"{nm}_ret{k}": lp.diff(k) for k in (1, 5, 20, 60)}
    f[f"{nm}_gap_ma50"] = lp - np.log(price.rolling(50, min_periods=40).mean())
    f[f"{nm}_gap_ma200"] = lp - np.log(price.rolling(200, min_periods=160).mean())
    f["rv_ratio"] = vol["rv20"] / vol["rv60"]
    f["rv60_z"] = zscore(vol["rv60"])
    if "vix" in other:
        f["vix_z"] = zscore(other["vix"])
    cp = COUNTER.get(target)
    if cp in other:
        f[f"{nm}_{SHORT[cp]}_z"] = zscore(lp - np.log(other[cp]))
    wti = price if target == "wti_fut" else other["wti_fut"] if "wti_fut" in other else None
    if wti is not None and {"gasoline", "heating_oil"} <= set(other.columns):
        crack = ((2 * 42 * other["gasoline"] + 42 * other["heating_oil"] - 3 * wti) / 3 / wti).where(lambda s: s.abs() < 2)
        f["crack_z"], f["crack_d20"] = zscore(crack), crack.diff(20)
    for c in ("natgas", "xle", "sp500", "dxy", "gold", "copper", "usdcad", "usdnok", "eem"):
        if c in other and other[c].notna().sum() > 500:
            f[f"{c}_ret20"] = np.log(other[c]).diff(20)
    if "t10y3m" in other:
        f["t10y3m_d60"] = other["t10y3m"].diff(60)
    if "ovx" in other:
        f["ovx_z"] = zscore(other["ovx"])
    if "breakeven10y" in other:
        f["breakeven_d20"] = other["breakeven10y"].diff(20)
    X = pd.DataFrame(f, index=price.index).replace([np.inf, -np.inf], np.nan)
    Y = {h: (100 * (lp.shift(-h) - lp)).to_numpy() for h in cfg.horizons}
    return Panel(target, X, Y, lp, vol["rv20"].to_numpy(), 100 * lp.diff().to_numpy(), X.to_numpy(float), price.index, price)


def rows_before(i0, h, lo):
    """Training rows s with s + h <= i0 (the target is already observed at the origin): the purge rule."""
    return np.arange(lo, i0 - h + 1)


def interval_sigma(z):
    """1-sd of the standardised errors (error / (20-day vol scaled to h days))."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    return float(z.std(ddof=1)) if len(z) > 5 else np.nan


# ---------------------------------------------------------------------------------------------------- regimes
@njit(cache=False)
def _ms_filter(y, mu, sig, P, xi0):
    # pred[t] = P(s_t | data before t), filt[t] = P(s_t | data up to t)
    T = y.shape[0]
    filt = np.empty((T, 2))
    pred = np.empty((T, 2))
    p0, p1, ll = xi0[0], xi0[1], 0.0
    c = 1.0 / np.sqrt(2.0 * np.pi)
    for t in range(T):
        pred[t, 0] = p0
        pred[t, 1] = p1
        z0 = (y[t] - mu[0]) / sig[0]
        z1 = (y[t] - mu[1]) / sig[1]
        a0 = p0 * c / sig[0] * np.exp(-0.5 * z0 * z0)
        a1 = p1 * c / sig[1] * np.exp(-0.5 * z1 * z1)
        s = max(a0 + a1, 1e-300)
        ll += np.log(s)
        g0 = a0 / s
        g1 = a1 / s
        filt[t, 0] = g0
        filt[t, 1] = g1
        p0 = g0 * P[0, 0] + g1 * P[1, 0]
        p1 = 1.0 - p0
    return filt, pred, ll


@njit(cache=False)
def _ms_smooth(filt, pred, P):
    T = filt.shape[0]
    sm = np.empty((T, 2))
    xs = np.zeros((2, 2))
    sm[T - 1, 0] = filt[T - 1, 0]
    sm[T - 1, 1] = filt[T - 1, 1]
    for t in range(T - 2, -1, -1):
        r0 = sm[t + 1, 0] / max(pred[t + 1, 0], 1e-300)
        r1 = sm[t + 1, 1] / max(pred[t + 1, 1], 1e-300)
        for i in range(2):
            a = filt[t, i] * P[i, 0] * r0
            b = filt[t, i] * P[i, 1] * r1
            sm[t, i] = a + b
            xs[i, 0] += a
            xs[i, 1] += b
    return sm, xs


class RegimeModel:
    """Two-state Markov-switching mean/variance model with a minimum expected regime length (EM, Hamilton filter, Kim smoother)."""

    def __init__(self, min_duration=120, max_iter=200, tol=1e-7):
        self.p_min, self.max_iter, self.tol = 1 - 1 / min_duration, max_iter, tol

    @staticmethod
    def stationary(P):
        den = 2 - P[0, 0] - P[1, 1]
        x0 = (1 - P[1, 1]) / den if den > 1e-12 else 0.5
        return np.array([x0, 1 - x0])

    def _em(self, y, mu, sig, P):
        floor, prev = 0.05 * y.std(), -np.inf
        for _ in range(self.max_iter):
            filt, pred, ll = _ms_filter(y, mu, sig, P, self.stationary(P))
            sm, xs = _ms_smooth(filt, pred, P)
            for j in range(2):
                w = sm[:, j] + 1e-10
                mu[j] = np.sum(w * y) / w.sum()
                sig[j] = max(np.sqrt(np.sum(w * (y - mu[j]) ** 2) / w.sum()), floor)
            P = xs / xs.sum(1, keepdims=True)
            for j in range(2):
                if P[j, j] < self.p_min:
                    P[j, j], P[j, 1 - j] = self.p_min, 1 - self.p_min
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        return mu, sig, P, _ms_filter(y, mu, sig, P, self.stationary(P))[2]

    def fit(self, y, warm=None):
        vol = pd.Series(y).rolling(20, min_periods=5).std().bfill().to_numpy()
        g = vol > np.median(vol)
        d = max(0.995, self.p_min)
        starts = [(np.array([y[~g].mean(), y[g].mean()]), np.array([y[~g].std(), y[g].std()]),
                   np.array([[d, 1 - d], [1 - d, d]]))]
        if warm is not None:
            starts.append((warm.mu.copy(), warm.sig.copy(), warm.P.copy()))
        mu, sig, P, ll = max((self._em(y, *[a.copy() for a in s]) for s in starts), key=lambda res: res[3])
        o = np.argsort(sig)                          # 0 = calm, 1 = turbulent
        self.mu, self.sig, self.P, self.loglik = mu[o], sig[o], P[o][:, o], ll
        return self

    def params(self):
        return {"mu": self.mu.tolist(), "sig": self.sig.tolist(), "P": self.P.tolist()}


class RegimeLabel:
    """Causal label, run day by day: switch only after `confirm` days beyond the threshold, and never before the current
    regime has lasted `min_duration` days."""

    def __init__(self, min_duration, enter_p, exit_p, confirm, state=None, run=0, since=None):
        self.min_duration, self.enter_p, self.exit_p, self.confirm = min_duration, enter_p, exit_p, confirm
        self.state, self.run = state, run
        self.since = min_duration if since is None else since

    def step(self, v):
        if self.state is None:
            self.state = int(v > 0.5)
        self.since += 1
        cross = v > self.enter_p if self.state == 0 else v < self.exit_p
        self.run = self.run + 1 if cross else 0
        if self.run >= self.confirm and self.since >= self.min_duration:
            self.state, self.run, self.since = 1 - self.state, 0, 0
        return self.state

    def snapshot(self):
        return {"state": self.state, "run": self.run, "since": self.since}


def regime_path(y_new, regime, prior, label_state, cfg):
    """Continue the regime filter and label with FIXED parameters through the returns y_new (days after the training end).
    Returns (P(turbulent) per day, label per day, state after the last day)."""
    if len(y_new) == 0:
        return np.array([]), np.array([]), None
    mu, sig, P = (np.asarray(regime[k], float) for k in ("mu", "sig", "P"))
    filt, _, _ = _ms_filter(np.nan_to_num(np.asarray(y_new, float)), mu, sig, P, np.asarray(prior, float))
    p = filt[:, 1]
    machine = RegimeLabel(cfg.lr_min_duration, cfg.lr_enter_p, cfg.lr_exit_p, cfg.lr_confirm, **label_state)
    lab = np.array([machine.step(v) for v in p], float)
    return p, lab, machine.snapshot()
