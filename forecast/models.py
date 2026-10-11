"""Linear forecasting models (logic from the team's lasso.ipynb, unchanged).

Every fitter returns a `Fit`: a standardised linear rule  y_hat = intercept + ((x - mu) / sd) @ coef.
Fits are small, picklable and can be applied to a new day's features without refitting.
"""
import inspect
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit

from .config import Config
from .features import base_cols

AR_COLS = ["own_ret1", "own_ret5", "own_ret20"]
_E = np.array([])


@dataclass
class Fit:
    cols: list
    mu: np.ndarray
    sd: np.ndarray
    coef: np.ndarray
    intercept: float
    selected: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def predict(self, x):
        """x: pandas Series of one day's features (missing columns / NaN -> window mean)."""
        if not self.cols:
            return self.intercept
        v = x.reindex(self.cols).to_numpy(dtype=float)
        z = (v - self.mu) / self.sd
        z = np.where(np.isfinite(z), z, 0.0)
        return float(self.intercept + z @ self.coef)


def bai_ng_k(S, N, T, kmax):
    ev = S ** 2
    tot = ev.sum()
    ic = [np.log(max(tot - ev[:k].sum(), 1e-12) / (N * T)) + k * (N + T) / (N * T) * np.log(min(N, T))
          for k in range(1, kmax + 1)]
    return int(np.argmin(ic)) + 1


def _prep(X, y, cols=None, max_nan=0.05):
    """Window data -> standardised matrix; columns screened on window data only."""
    df = X if cols is None else X[cols]
    A, yv, names = df.to_numpy(dtype=float), y.to_numpy(dtype=float), np.array(df.columns)
    ok = np.isfinite(yv)
    A, yv = A[ok], yv[ok]
    if cols is None:
        with np.errstate(invalid="ignore"):
            keep = (np.isnan(A).mean(0) <= max_nan) & (np.nan_to_num(np.nanstd(A, 0)) > 0)
        A, names = A[:, keep], names[keep]
    rows = np.isfinite(A).all(axis=1)
    A, yv = A[rows], yv[rows]
    mu, sd = A.mean(0), A.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return list(names), (A - mu) / sd, yv, mu, sd


def _ols(cols, Z, y, mu, sd, intercept=True):
    if Z.shape[1] == 0:
        return Fit([], mu, sd, _E, float(np.mean(y)) if intercept else 0.0)
    if not intercept:
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return Fit(cols, mu, sd, beta, 0.0, selected=list(cols))
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Z]), y, rcond=None)
    return Fit(cols, mu, sd, beta[1:], float(beta[0]), selected=list(cols))


def fit_rw(X, y, h, cfg, ctx):
    return Fit([], _E, _E, _E, 0.0)


def fit_histmean(X, y, h, cfg, ctx):
    return Fit([], _E, _E, _E, float(y.dropna().mean()))


def _make_lassocv(cfg, cv):
    kw = dict(cv=cv, eps=cfg.lasso_eps, max_iter=5000, tol=1e-3, fit_intercept=cfg.fit_intercept,
              random_state=cfg.seed)
    if "n_alphas" in inspect.signature(LassoCV).parameters:      # older sklearn
        return LassoCV(n_alphas=cfg.n_alphas, **kw)
    return LassoCV(alphas=cfg.n_alphas, **kw)


def _lasso(Xw, y, h, cfg, rule="min", ctx=None):
    # the CV path is shared by LASSO and LASSO_1se on the same window (cached in ctx)
    key = (Xw.index[0], Xw.index[-1], Xw.shape[1])
    cache = ctx.get("lasso_cv") if ctx is not None else None
    if cache is not None and cache[0] == key:
        _, cols, Z, yv, mu, sd, m = cache
    else:
        cols, Z, yv, mu, sd = _prep(Xw, y)
        m = _make_lassocv(cfg, TimeSeriesSplit(n_splits=cfg.cv_folds, gap=h)).fit(Z, yv)
        if ctx is not None:
            ctx["lasso_cv"] = (key, cols, Z, yv, mu, sd, m)
    coef, icpt, alpha = m.coef_, float(m.intercept_), float(m.alpha_)
    if rule == "1se":
        mse = m.mse_path_.mean(axis=1)
        se = m.mse_path_.std(axis=1) / np.sqrt(m.mse_path_.shape[1])
        j = int(np.argmin(mse))
        a1 = float(m.alphas_[mse <= mse[j] + se[j]].max())      # largest alpha within 1 SE
        if a1 > alpha:
            lm = Lasso(alpha=a1, max_iter=5000, tol=1e-3, fit_intercept=cfg.fit_intercept).fit(Z, yv)
            coef, icpt, alpha = lm.coef_, float(lm.intercept_), a1
    sel = [c for c, b in zip(cols, coef) if abs(b) > 1e-12]
    return Fit(cols, mu, sd, coef, icpt, selected=sel,
               info={"alpha": alpha, "n_selected": len(sel), "n_candidates": len(cols)})


def fit_lasso(X, y, h, cfg, ctx):
    return _lasso(X[base_cols(X)], y, h, cfg, "min", ctx)


def fit_lasso_1se(X, y, h, cfg, ctx):
    return _lasso(X[base_cols(X)], y, h, cfg, "1se", ctx)


def fit_lasso_rs(X, y, h, cfg, ctx):
    if "regime_p" not in X.columns:
        return fit_lasso(X, y, h, cfg, ctx)
    return _lasso(X, y, h, cfg)


def _factor_fit(cols, Z, yv, mu, sd, fac_idx, own_idx, cfg, extra_info=None):
    """Regress y on PCA factors of Z[:, fac_idx] plus Z[:, own_idx]; map back to a linear rule in Z."""
    Zf = Z[:, fac_idx]
    _, S, Vt = np.linalg.svd(Zf, full_matrices=False)
    kmax = max(1, min(cfg.pca_kmax, Zf.shape[1] - 1))
    k = cfg.pca_k or bai_ng_k(S, Zf.shape[1], Zf.shape[0], kmax)
    V = Vt[:k].T
    D = np.column_stack([Zf @ V, Z[:, own_idx]])
    if cfg.fit_intercept:
        D = np.column_stack([np.ones(len(yv)), D])
    beta, *_ = np.linalg.lstsq(D, yv, rcond=None)
    icpt, beta = (float(beta[0]), beta[1:]) if cfg.fit_intercept else (0.0, beta)
    coef = np.zeros(len(cols))
    coef[fac_idx] += V @ beta[:k]
    coef[own_idx] += beta[k:]
    return Fit(cols, mu, sd, coef, icpt, info={"k": k, **(extra_info or {})})


def fit_pca(X, y, h, cfg, ctx):
    cols, Z, yv, mu, sd = _prep(X[base_cols(X)], y)
    fac = [i for i, c in enumerate(cols) if not c.startswith(("own_", "season_"))]
    own = [i for i, c in enumerate(cols) if c in AR_COLS]
    if len(fac) < 3:
        return fit_histmean(X, y, h, cfg, ctx)
    return _factor_fit(cols, Z, yv, mu, sd, fac, own, cfg)


def _hac_t(Zf, yv, h):
    """t-statistics of univariate slopes with Newey-West variance (lag h-1)."""
    yc = yv - yv.mean()
    sxx = (Zf ** 2).sum(0)
    b = Zf.T @ yc / sxx
    g = Zf * (yc[:, None] - Zf * b)
    T, L = len(yc), max(h - 1, 0)
    S = (g * g).sum(0) / T
    for l in range(1, L + 1):
        S = S + 2 * (1 - l / (L + 1)) * (g[l:] * g[:-l]).sum(0) / T
    return b / np.sqrt(np.maximum(T * S, 1e-18) / sxx ** 2)


def fit_tpca(X, y, h, cfg, ctx):
    cols, Z, yv, mu, sd = _prep(X[base_cols(X)], y)
    fac = np.array([i for i, c in enumerate(cols) if not c.startswith(("own_", "season_"))])
    own = [i for i, c in enumerate(cols) if c in AR_COLS]
    if len(fac) == 0:
        return fit_histmean(X, y, h, cfg, ctx)
    t = _hac_t(Z[:, fac], yv, h)
    keep = fac[np.abs(t) > cfg.tpca_t]
    if len(keep) == 0:
        f = fit_histmean(X, y, h, cfg, ctx) if cfg.fit_intercept else fit_rw(X, y, h, cfg, ctx)
        f.info = {"k": 0, "n_targeted": 0}
        return f
    if len(keep) <= 2:                       # too few for PCA: plain OLS on them
        idx = list(keep) + own
        f = _ols([cols[i] for i in idx], Z[:, idx], yv, mu[idx], sd[idx], intercept=cfg.fit_intercept)
        f.info = {"k": len(keep), "n_targeted": len(keep)}
        return f
    return _factor_fit(cols, Z, yv, mu, sd, list(keep), own, cfg, {"n_targeted": len(keep)})


FITTERS = {"RW": fit_rw, "LASSO": fit_lasso, "LASSO_1se": fit_lasso_1se,
           "PCA_reg": fit_pca, "TPCA": fit_tpca, "LASSO_RS": fit_lasso_rs}
