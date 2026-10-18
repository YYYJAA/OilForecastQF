"""Out-of-sample accuracy and simple trading-rule value of logged forecasts.

forecasts: DataFrame with asof_date, target, model, horizon, forecast_return, sigma (one row per origin/model/horizon).
prices:    {target: price Series}. Realised h-day log return = log P[t+h] / log P[t] on the series' own trading days.
"""
import numpy as np
import pandas as pd
from scipy import stats


def realised(price: pd.Series, h: int) -> pd.Series:
    p = price.dropna()
    p = p[p > 0]
    return np.log(p.shift(-h) / p)


def dm_pvalue(err_bench, err_model, h):
    """Diebold-Mariano (squared error), Newey-West variance with h-1 lags, HLN correction. Two-sided p-value."""
    d = np.asarray(err_bench, float) ** 2 - np.asarray(err_model, float) ** 2
    n = len(d)
    if n < 30:
        return np.nan
    m = d.mean()
    v = np.mean((d - m) ** 2)
    for k in range(1, max(h, 1)):
        v += 2 * (1 - k / h) * np.mean((d[k:] - m) * (d[:-k] - m))
    s = m / np.sqrt(max(v, 1e-18) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(2 * stats.t.sf(abs(s), n - 1))


def forecast_metrics(fc: pd.DataFrame, prices: dict) -> pd.DataFrame:
    rows = []
    for (t, h), g in fc.groupby(["target", "horizon"]):
        p = prices[t].dropna()
        p = p[p > 0]
        y = realised(p, int(h))
        p_future = p.shift(-int(h))
        g = g.assign(actual=g["asof_date"].map(y), p0=g["asof_date"].map(p), p_h=g["asof_date"].map(p_future))
        g = g.dropna(subset=["actual"])
        for m, gm in g.groupby("model"):
            err = gm["actual"] - gm["forecast_return"]
            rmse, mae = float(np.sqrt((err ** 2).mean())), float(err.abs().mean())
            call = gm["forecast_return"] != 0
            up_a, up_f = gm["actual"] > 0, gm["forecast_return"] > 0
            rw_rmse = float(np.sqrt((gm["actual"] ** 2).mean()))
            mape = float((np.abs(gm["p0"] * np.exp(gm["forecast_return"]) - gm["p_h"]) / gm["p_h"]).mean() * 100)
            mape_nc = float((np.abs(gm["p0"] - gm["p_h"]) / gm["p_h"]).mean() * 100)
            if call.any():
                a_, f_ = up_a[call], up_f[call]
                tp, fp, fn = int((a_ & f_).sum()), int((~a_ & f_).sum()), int((a_ & ~f_).sum())
                tn = int((~a_ & ~f_).sum())
                prec = tp / (tp + fp) if tp + fp else np.nan
                rec = tp / (tp + fn) if tp + fn else np.nan
                f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else np.nan
                prec_d = tn / (tn + fn) if tn + fn else np.nan
                rec_d = tn / (tn + fp) if tn + fp else np.nan
                f1_d = 2 * prec_d * rec_d / (prec_d + rec_d) if prec_d == prec_d and rec_d == rec_d and prec_d + rec_d > 0 else np.nan
            else:
                prec = rec = f1 = f1_d = np.nan
            rows.append({
                "target": t, "model": m, "horizon": int(h), "n": len(gm), "rmse": rmse, "rmse_no_change": rw_rmse,
                "rmse_ratio": rmse / rw_rmse if rw_rmse else np.nan, "mae": mae,
                "mae_ratio": mae / float(gm["actual"].abs().mean()) if gm["actual"].abs().mean() else np.nan,
                "mape_pct": mape, "mape_no_change_pct": mape_nc,
                "dir_acc": float((np.sign(gm["forecast_return"][call]) == np.sign(gm["actual"][call])).mean()) if call.any() else np.nan,
                "share_directional": float(call.mean()),
                "precision_up": prec, "recall_up": rec, "f1_up": f1, "f1_down": f1_d,
                "coverage_1sd": float((err.abs() <= gm["sigma"]).mean()) if gm["sigma"].notna().any() else np.nan,
                "dm_p_vs_no_change": dm_pvalue(gm["actual"], err, int(h)) if m != "RW" else np.nan,
            })
    return pd.DataFrame(rows)


def robustness(fc: pd.DataFrame, prices: dict, cutoff, vol_window: int = 20, n_windows: int = 4) -> pd.DataFrame:
    """RMSE vs no change and direction accuracy in sub-samples of the test period:
    high / low volatility at the forecast origin (threshold = median trailing volatility up to the cutoff, so it
    never uses the test period), calendar years, and `n_windows` consecutive equal-size windows."""
    cutoff = pd.Timestamp(cutoff)
    out = []
    for (t, h), g in fc.groupby(["target", "horizon"]):
        p = prices[t].dropna()
        p = p[p > 0]
        ret = np.log(p).diff()
        vol = ret.rolling(vol_window).std()
        thr = vol.loc[:cutoff].median()
        y = realised(p, int(h))
        g = g.assign(actual=g["asof_date"].map(y), vol=g["asof_date"].map(vol)).dropna(subset=["actual"])
        dates = np.sort(g["asof_date"].unique())
        edges = np.array_split(dates, n_windows)
        label = {}
        for k, chunk in enumerate(edges, start=1):
            for d in chunk:
                label[d] = f"window {k} ({pd.Timestamp(chunk[0]).strftime('%Y-%m')} to {pd.Timestamp(chunk[-1]).strftime('%Y-%m')})"
        g["window"] = g["asof_date"].map(label)
        g["year"] = g["asof_date"].dt.year.astype(str)
        g["regime"] = np.where(g["vol"] > thr, "high volatility", "low volatility")
        for kind, col in (("regime", "regime"), ("year", "year"), ("window", "window")):
            for (m, lab), gm in g.groupby(["model", col]):
                err = gm["actual"] - gm["forecast_return"]
                nc = float(np.sqrt((gm["actual"] ** 2).mean()))
                call = gm["forecast_return"] != 0
                out.append({"target": t, "model": m, "horizon": int(h), "kind": kind, "slice": lab, "n": len(gm),
                            "rmse_ratio": float(np.sqrt((err ** 2).mean()) / nc) if nc else np.nan,
                            "dir_acc": float((np.sign(gm["forecast_return"][call]) == np.sign(gm["actual"][call])).mean())
                            if call.any() else np.nan})
    return pd.DataFrame(out)


def perf(ret: pd.Series, turnover: pd.Series, position: pd.Series, cost: float) -> dict:
    eq = (1 + ret).cumprod()
    n = len(ret)
    sd = ret.std()
    return {"days": n, "total_return": float(eq.iloc[-1] - 1),
            "ann_return": float(eq.iloc[-1] ** (252 / n) - 1) if n else np.nan,
            "ann_vol": float(sd * np.sqrt(252)),
            "sharpe": float(ret.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "turnover": float(turnover.sum()), "total_cost": float(turnover.sum() * cost),
            "days_in_market": float((position.abs() > 1e-12).mean())}


def trading_rule(fc_series: pd.DataFrame, price: pd.Series, h: int, cost: float, band: float):
    """fc_series: index asof_date with forecast_return, sigma. Long / short / flat on the sign of the forecast
    (flat if |forecast| <= band * sigma). Overlapping tranches: each day opens 1/h of capital in that day's direction
    and holds it h days, so the position is the mean of the last h signals. Return earned from t to t+1."""
    p = price.dropna()
    p = p[p > 0]
    nxt = (p.shift(-1) / p - 1).reindex(fc_series.index)
    f, s = fc_series["forecast_return"], fc_series["sigma"].fillna(0.0)
    sig = pd.Series(np.where(f.abs() > band * s, np.sign(f), 0.0), index=fc_series.index)
    pos = sig.rolling(h, min_periods=1).sum() / h
    turn = pos.diff().abs().fillna(pos.abs())
    ret = (pos * nxt - cost * turn)[nxt.notna()]
    return ret, turn[ret.index], pos[ret.index]


def trading_metrics(fc: pd.DataFrame, prices: dict, cost_bps=5.0, band=0.1):
    cost = cost_bps / 1e4
    rows, curves = [], {}
    for (t, m, h), g in fc.groupby(["target", "model", "horizon"]):
        g = g.sort_values("asof_date").set_index("asof_date")
        ret, turn, pos = trading_rule(g, prices[t], int(h), cost, band)
        rows.append({"target": t, "model": m, "horizon": int(h), **perf(ret, turn, pos, cost)})
        curves[(t, m, int(h))] = (1 + ret).cumprod()
    # buy and hold benchmark on the same dates
    for t in fc["target"].unique():
        idx = pd.DatetimeIndex(sorted(fc[fc["target"] == t]["asof_date"].unique()))
        p = prices[t].dropna()
        p = p[p > 0]
        nxt = (p.shift(-1) / p - 1).reindex(idx).dropna()
        turn = pd.Series(0.0, index=nxt.index); turn.iloc[0] = 1.0
        ret = nxt - cost * turn
        rows.append({"target": t, "model": "Buy & hold", "horizon": 0,
                     **perf(ret, turn, pd.Series(1.0, index=ret.index), cost)})
        curves[(t, "Buy & hold", 0)] = (1 + ret).cumprod()
    return pd.DataFrame(rows), curves
