"""Real-time 2-state Gaussian HMM (code from the team's lasso.ipynb).

Probabilities are *filtered* (use data up to t only). The state at the end of training is stored so the
daily run can continue filtering new days with fixed parameters, without re-estimating.
"""
import time

import numpy as np
import pandas as pd

from .config import Config


class GaussHMM:
    """Gaussian HMM with diagonal covariances fitted by EM (Baum-Welch) with scaling.
    tied_mean=True: states share one mean and differ only in variance (volatility regimes)."""

    def __init__(self, k=2, tied_mean=True, max_iter=200, tol=1e-7, var_floor=0.02):
        self.k, self.tied, self.max_iter, self.tol, self.var_floor = k, tied_mean, max_iter, tol, var_floor

    def _init(self, X):
        k = self.k
        s = pd.Series(X[:, 0]).rolling(20, min_periods=5).std().bfill().to_numpy()
        lab = np.digitize(s, np.quantile(s, np.linspace(0, 1, k + 1)[1:-1]))
        self.mu = np.vstack([X[lab == j].mean(0) for j in range(k)])
        if self.tied:
            self.mu[:] = X.mean(0)
        self.var = np.vstack([X[lab == j].var(0) for j in range(k)])
        cnt = np.ones((k, k)) + np.histogram2d(lab[:-1], lab[1:], bins=[np.arange(k + 1)] * 2)[0]
        self.A = cnt / cnt.sum(1, keepdims=True)
        self.pi = np.bincount(lab, minlength=k) / len(lab)

    def _B(self, X):
        logB = -0.5 * (((X[:, None, :] - self.mu[None]) ** 2) / self.var[None]
                       + np.log(2 * np.pi * self.var[None])).sum(2)
        c = logB.max(1, keepdims=True)
        return np.exp(logB - c), c.sum()

    def _forward(self, B, pi):
        T, A = len(B), self.A
        alpha, scale = np.empty_like(B), np.empty(T)
        a = pi * B[0]; scale[0] = a.sum(); alpha[0] = a / scale[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            s = a.sum(); scale[t] = s; alpha[t] = a / s
        return alpha, scale

    def _backward(self, B, scale):
        T, A = len(B), self.A
        beta = np.ones_like(B)
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / scale[t + 1]
        return beta

    def _em(self, X):
        floor = self.var_floor * X.var(0)
        prev, ll = -np.inf, -np.inf
        for _ in range(self.max_iter):
            B, c = self._B(X)
            alpha, scale = self._forward(B, self.pi)
            ll = np.log(scale).sum() + c
            beta = self._backward(B, scale)
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            xi = (alpha[:-1].T @ (B[1:] * beta[1:] / scale[1:, None])) * self.A
            self.A = xi / xi.sum(1, keepdims=True)
            self.pi = gamma[0]
            w = gamma.sum(0)
            self.mu = (gamma.T @ X) / w[:, None]
            if self.tied:
                self.mu[:] = X.mean(0)
            self.var = np.maximum((gamma.T @ X ** 2) / w[:, None] - self.mu ** 2, floor)
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        self.loglik, self.occupancy = ll, gamma.mean(0)
        return self

    def _copy(self):
        return {k: getattr(self, k).copy() for k in ("mu", "var", "A", "pi")}

    def fit(self, X, warm=None):
        cands = []
        if warm is not None:                          # warm start keeps state meanings stable
            for k_ in ("mu", "var", "A", "pi"):
                setattr(self, k_, getattr(warm, k_).copy())
            self._em(X); cands.append((self.loglik, self.occupancy.min(), self._copy()))
        if warm is None or cands[0][1] < 0.03:        # cold start if no warm start or degenerate
            self._init(X); self._em(X); cands.append((self.loglik, self.occupancy.min(), self._copy()))
        ok = [c for c in cands if c[1] >= 0.03] or cands
        best = max(ok, key=lambda c: c[0])
        for k_, v in best[2].items():
            setattr(self, k_, v)
        self.loglik = best[0]
        order = np.argsort(self.var[:, 0])            # last state = highest variance
        self.mu, self.var, self.pi = self.mu[order], self.var[order], self.pi[order]
        self.A = self.A[order][:, order]
        return self

    def filter(self, X):
        B, _ = self._B(X)
        return self._forward(B, self.pi)[0]


def hmm_inputs(logp, names):
    r = logp.diff() * 100
    cols = {"ret": r, "absret5": r.abs().rolling(5).mean(), "logrv20": np.log(r.rolling(20).std())}
    return pd.DataFrame({n: cols[n] for n in names})


def run_regime_id(logp, cfg: Config, verbose=True):
    """Expanding-window HMM re-estimated every cfg.hmm_refit days, filtered daily. Returns (P, log)."""
    H = hmm_inputs(logp, cfg.hmm_inputs).dropna()
    vals, idx = H.to_numpy(), H.index
    P = pd.DataFrame(np.nan, index=logp.index, columns=[f"state{k}" for k in range(cfg.hmm_k)])
    log, model, t0 = [], None, time.time()
    i = cfg.hmm_min_obs - 1
    while i < len(vals):
        end = min(i + cfg.hmm_refit, len(vals))
        lo = 0 if cfg.hmm_window is None else max(0, i + 1 - cfg.hmm_window)
        model = GaussHMM(cfg.hmm_k).fit(vals[lo:i + 1], warm=model)     # data up to i only
        alpha = model.filter(vals[lo:end])                              # row t uses data <= t
        P.loc[idx[i:end]] = alpha[i - lo:]
        log.append({"date": idx[i], "pos": logp.index.get_loc(idx[i]), "A": model.A.copy(),
                    "mu": model.mu.copy(), "var": model.var.copy()})
        i = end
    if verbose:
        print(f"[hmm] {len(log)} re-estimations ({time.time() - t0:.0f}s)")
    return P, log


def continue_filter(alpha_prev, x_new, A, mu, var):
    """One filtering step with fixed parameters: alpha_t = normalise((alpha_{t-1} @ A) * B(x_t))."""
    logB = -0.5 * (((x_new[None, :] - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(1)
    b = np.exp(logB - logB.max())
    a = (alpha_prev @ A) * b
    s = a.sum()
    return a / s if s > 0 else alpha_prev
