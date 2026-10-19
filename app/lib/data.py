"""Cached loaders and small calculations for the web app. Read-only: the app never downloads data or trains models;
everything comes from files the daily bot commits to the repo (Data/raw, Outputs, models)."""
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUT, FROZEN, RAW, MODELS = ROOT / "Outputs", ROOT / "Outputs" / "frozen_eval", ROOT / "Data" / "raw", ROOT / "models"

TARGETS = {"brent_fut": "Brent", "wti_fut": "WTI"}
PRICE_COL = {"brent_fut": "brent_fut", "wti_fut": "wti_fut"}
SPOT_COL = {"brent_fut": "brent_spot", "wti_fut": "wti_spot"}
SERIES_START = {"wti_fut": "2020-06-01"}                 # WTI futures went negative on 2020-04-20
ORDER = ["RW", "LASSO", "LASSO_1se", "PCA_reg", "TPCA", "LASSO_RS", "Combo_EW", "ARMA-GARCH", "MS-AR", "MS-ARX", "HistMean", "AR"]
LABELS = {"RW": "No change", "LASSO": "LASSO", "LASSO_1se": "LASSO 1-SE", "PCA_reg": "PCA reg.", "TPCA": "Targeted PCA",
          "LASSO_RS": "LASSO regime", "Combo_EW": "Combo (equal wt)", "ARMA-GARCH": "ARMA-GARCH", "MS-AR": "MS-AR",
          "MS-ARX": "MS-ARX", "HistMean": "Historical mean", "AR": "AR (direct)", "Buy & hold": "Buy & hold"}
BENCH = ("HistMean", "AR")
SHOW = {"HIGH": "Significant", "MEDIUM": "Suggestive", "LOW": "None yet", "n/a": "No call"}
RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
BAND_BY = {"Recent volatility": "RW", "ARMA-GARCH": "ARMA-GARCH", "MS-AR": "MS-AR", "MS-ARX": "MS-ARX"}
SAVED_H = [1, 5, 10, 20, 40, 60]


def label(m):
    return LABELS.get(m, m)


def order(models):
    return [m for m in ORDER if m in models] + sorted(m for m in models if m not in ORDER)


def _stamp(*paths):
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


def _csv(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _latest(_s):
    d = _csv(OUT / "latest_forecasts.csv", parse_dates=["asof_date"])
    return d


@st.cache_data(show_spinner=False)
def _log(_s):
    return _csv(OUT / "forecast_log.csv", parse_dates=["asof_date", "target_date_approx"])


@st.cache_data(show_spinner=False)
def _prices(_s):
    y = _csv(RAW / "yahoo_daily.csv", parse_dates=["date"]).set_index("date")
    f = _csv(RAW / "fred_daily.csv", parse_dates=["date"]).set_index("date")
    return y, f


@st.cache_data(show_spinner=False)
def _frozen(name, _s):
    f = FROZEN / name
    if name.endswith(".json"):
        return json.loads(f.read_text()) if f.exists() else {}
    dates = {"equity_curves.csv": ["date"], "rolling_forecasts.csv": ["asof_date"], "fan.csv": ["origin", "target_date"]}
    return _csv(f, parse_dates=dates.get(name, False))


@st.cache_data(show_spinner=False)
def _meta(_s):
    out = {}
    for p in MODELS.glob("*_meta.json"):
        try:
            out[p.stem.replace("_meta", "")] = json.loads(p.read_text())
        except Exception:
            pass
    return out


def latest():
    return _latest(_stamp(OUT / "latest_forecasts.csv"))


def log():
    return _log(_stamp(OUT / "forecast_log.csv"))


def prices():
    return _prices(_stamp(RAW / "yahoo_daily.csv", RAW / "fred_daily.csv"))


def frozen(name):
    return _frozen(name, _stamp(FROZEN / name))


def meta():
    return _meta(_stamp(*MODELS.glob("*_meta.json")))


def price_series(target, kind="futures"):
    y, f = prices()
    s = (y[PRICE_COL[target]] if kind == "futures" else f[SPOT_COL[target]]).dropna()
    s = s[s > 0]
    return s[s.index >= SERIES_START[target]] if target in SERIES_START else s


def freshness():
    """(last price date, forecasts generated at (UTC string or ''), forecast as-of date)."""
    lt = latest()
    if lt.empty:
        return None, "", None
    gen = str(lt["generated_at_utc"].iloc[0]) if "generated_at_utc" in lt else ""
    p = price_series("brent_fut")
    return (p.index.max() if len(p) else None), gen, lt["asof_date"].max()


# ------------------------------------------------------------------ forecasts at any horizon
def _interp(hs, vals, h, var=False):
    """Linear in horizon for the log return; the variance is linear in horizon for sigma (random-walk scaling)."""
    x = np.array([0.0] + list(hs))
    v = np.array([0.0] + list(vals), dtype=float)
    if var:
        return float(np.sqrt(max(np.interp(h, x, v ** 2), 0)))
    return float(np.interp(h, x, v))


def current(target):
    lt = latest()
    lt = lt[lt["target"] == target]
    return lt[lt["asof_date"] == lt["asof_date"].max()] if len(lt) else lt


def at_horizon(cur, h):
    """Rows model, ret (log), sigma at horizon h (1..60), interpolating between the saved horizons."""
    rows = []
    for m, g in cur.groupby("model"):
        g = g.sort_values("horizon")
        hs = g["horizon"].tolist()
        sg = g["sigma"].astype(float)
        rows.append({"model": m, "ret": _interp(hs, g["forecast_return"], h),
                     "sigma": _interp(hs, sg, h, var=True) if sg.notna().all() else np.nan,
                     "regime_p": g["regime_p"].iloc[0] if "regime_p" in g else np.nan})
    return pd.DataFrame(rows)


def no_change_models(cur, tol=1e-6):
    """Models whose forecast is (numerically) zero at every horizon: they forecast 'no change'."""
    out = []
    for m, g in cur.groupby("model"):
        if m != "RW" and (g["forecast_return"].abs() < tol).all():
            out.append(m)
    return out


def _phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def track_record(hit, n, h, ratio):
    """Same rule as forecast/signals.py: direction vs a coin flip with n/h effective observations, and error vs no change."""
    if hit is None or n is None or not np.isfinite(hit) or n <= 0:
        return "n/a", np.nan
    n_eff = max(n / max(h, 1), 1.0)
    p = 1 - _phi((hit - 0.5) / sqrt(0.25 / n_eff))
    if p < 0.05 and ratio < 1:
        return "HIGH", p
    if p < 0.10 or (hit > 0.55 and ratio < 1):
        return "MEDIUM", p
    return "LOW", p


def nearest(hs, h):
    return min(hs, key=lambda x: abs(x - h))


def signal_table(target, rows, h, band, models=None):
    """One row per model at horizon h with BUY/SELL/HOLD, noise ratio, P(up) and the frozen-test track record."""
    ev = frozen("metrics.csv")
    ev = ev[ev["target"] == target] if len(ev) else ev
    eh = sorted(ev["horizon"].unique()) if len(ev) else []
    out = []
    for r in rows.itertuples():
        if r.model == "RW" or (models is not None and r.model not in models):
            continue
        s = r.sigma
        z = r.ret / s if s and np.isfinite(s) else np.nan
        sig = "HOLD" if (not np.isfinite(z) or abs(r.ret) <= band * s) else ("BUY" if r.ret > 0 else "SELL")
        hit = n = ratio = np.nan
        if eh:
            m = ev[(ev["model"] == r.model) & (ev["horizon"] == nearest(eh, h))]
            if len(m):
                hit, n, ratio = m["dir_acc"].iloc[0], float(m["n"].iloc[0]), m["rmse_ratio"].iloc[0]
        conf, p = track_record(hit, n, nearest(eh, h) if eh else h, ratio)
        out.append({"model": r.model, "label": label(r.model), "signal": sig, "move": np.exp(r.ret) - 1, "ret": r.ret,
                    "sigma": s, "z": z, "p_up": _phi(z) if np.isfinite(z) else np.nan, "hit": hit, "ratio": ratio,
                    "conf": conf, "p": p})
    df = pd.DataFrame(out)
    if len(df):
        df["_r"] = df["conf"].map(RANK).fillna(0)
        df = df.sort_values(["_r", "p", "z"], ascending=[False, True, False], key=lambda c: c.abs() if c.name == "z" else c)
        df = df.drop(columns="_r").reset_index(drop=True)
    return df


def consensus(df):
    nb, ns, nh = (df["signal"] == "BUY").sum(), (df["signal"] == "SELL").sum(), (df["signal"] == "HOLD").sum()
    label_ = "LEAN BUY" if nb >= 2 and nb > ns else "LEAN SELL" if ns >= 2 and ns > nb else "NEUTRAL"
    return label_, int(nb), int(ns), int(nh)


def recent_vol_sigma(target, h):
    """1-sd move over h days from the last 20 daily returns (what the 'no change' band uses)."""
    p = price_series(target)
    r = np.log(p).diff().dropna().tail(20)
    return float(r.std() * np.sqrt(h)) if len(r) > 2 else np.nan
