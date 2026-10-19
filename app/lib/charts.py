"""Plotly figures. Every function takes already-loaded data and only draws."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import data as D
from .ui import BG, CARD, COLORS, GRID, SIG

SAME = 0.0025            # forecasts this close to 'no change' are drawn as one line


def _layout(fig, title=None, height=460, legend_below=True, **kw):
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", y=-0.18 if legend_below else 1.02, x=0, font=dict(size=11)), **kw)
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _path(cur, model, H, field="forecast_return", var=False):
    g = cur[cur["model"] == model].sort_values("horizon")
    if g.empty or (var and g["sigma"].isna().any()):
        return None
    col = "sigma" if var else field
    return np.array([D._interp(g["horizon"].tolist(), g[col], h, var=var) for h in range(0, H + 1)])


# ------------------------------------------------------------------ Today
def fan_chart(target, cur, H, models, band, k, window, show_band=True, show_rw=True, log_y=False, pct=False):
    hist = D.price_series(target).tail(window)
    p0 = float(cur["price_now"].iloc[0])
    d0 = cur["asof_date"].iloc[0]
    days = pd.bdate_range(d0, periods=H + 1)
    base = p0 if pct else 1.0
    f = (lambda x: 100 * (x / p0 - 1)) if pct else (lambda x: x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=f(hist.values), name="Actual", line=dict(color="#ffffff", width=2.4),
                             hovertemplate="%{y:.2f}"))
    bm = D.BAND_BY.get(band, "RW")
    if show_band:
        sg = _path(cur, bm, H, var=True)
        if sg is not None:
            up, lo = p0 * np.exp(k * sg), p0 * np.exp(-k * sg)
            fig.add_trace(go.Scatter(x=days, y=f(up), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=days, y=f(lo), fill="tonexty", fillcolor="rgba(86,180,233,0.17)", mode="lines", line=dict(width=0),
                                     name=f"{band} band (±{k:g} sd)", hovertemplate="%{y:.2f}"))
    flat, moving = [], []
    for m in models:
        r = _path(cur, m, H)
        if r is None:
            continue
        (flat if abs(r[-1]) < SAME else moving).append((m, r))
    if show_rw or flat:
        names = ["No change"] + [D.label(m) for m, _ in flat]
        fig.add_trace(go.Scatter(x=days, y=f(np.full(H + 1, p0)), name=" = ".join(names) if len(names) > 1 else "No change", mode="lines",
                                 line=dict(color=COLORS["RW"], dash="dash", width=2), hovertemplate="%{y:.2f}"))
    for m, r in moving:
        fig.add_trace(go.Scatter(x=days, y=f(p0 * np.exp(r)), name=D.label(m), line=dict(color=COLORS.get(m), width=2.2), mode="lines",
                                 customdata=100 * (np.exp(r) - 1), hovertemplate="%{y:.2f} (%{customdata:+.2f}% vs no change)"))
    fig.add_vline(x=d0, line_color=GRID, line_dash="dot")
    fig.update_yaxes(title="% vs today" if pct else "USD / bbl", type="log" if (log_y and not pct) else "linear")
    return _layout(fig, f"{D.TARGETS[target]} futures: recent prices and forecasts to {H} trading day{'s' if H > 1 else ''}", height=500)


def horizon_heatmap(cur, models):
    ms = [m for m in D.order(cur["model"].unique()) if m in models]
    hs = sorted(cur["horizon"].unique())
    z = [[100 * (np.exp(float(cur[(cur["model"] == m) & (cur["horizon"] == h)]["forecast_return"].iloc[0])) - 1)
          if len(cur[(cur["model"] == m) & (cur["horizon"] == h)]) else np.nan for h in hs] for m in ms]
    zz = np.array(z, dtype=float)
    lim = max(0.5, float(np.nanmax(np.abs(zz)))) if zz.size else 1
    fig = go.Figure(go.Heatmap(z=zz, x=[f"{h}d" for h in hs], y=[D.label(m) for m in ms], zmid=0, zmin=-lim, zmax=lim,
                               colorscale=[[0, SIG["SELL"]], [0.5, CARD], [1, SIG["BUY"]]], text=np.round(zz, 2),
                               texttemplate="%{text:+.2f}%", hovertemplate="%{y}, %{x}: %{z:+.2f}%<extra></extra>",
                               colorbar=dict(title="% vs now")))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Expected move by model and horizon (%, green = up, red = down)", height=60 + 42 * max(len(ms), 3),
                   hovermode="closest")


def uncertainty_chart(cur, H, k, models):
    p0 = float(cur["price_now"].iloc[0])
    rows = D.at_horizon(cur, H)
    rows = rows[rows["sigma"].notna() & rows["model"].isin(set(models) | {"RW"})]
    rows = rows.assign(o=rows["model"].map({m: i for i, m in enumerate(D.ORDER)})).sort_values("o")
    fig = go.Figure()
    for r in rows.itertuples():
        lo, hi, mid = p0 * np.exp(-k * r.sigma), p0 * np.exp(k * r.sigma), p0 * np.exp(r.ret)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[D.label(r.model)] * 2, mode="lines", line=dict(color=COLORS.get(r.model), width=10),
                                 showlegend=False, hovertemplate=f"{D.label(r.model)}: {lo:.2f} to {hi:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[mid], y=[D.label(r.model)], mode="markers", marker=dict(color="white", size=9),
                                 showlegend=False, hovertemplate=f"forecast {mid:.2f}<extra></extra>"))
    fig.add_vline(x=p0, line_dash="dot", line_color="#8fa3bb", annotation_text="today", annotation_position="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="USD / bbl")
    return _layout(fig, f"Price range after {H} trading day{'s' if H > 1 else ''} (±{k:g} sd; white dot = forecast)",
                   height=90 + 46 * max(len(rows), 3), hovermode="closest")


# ------------------------------------------------------------------ Track record
METRICS = {"RMSE ratio (vs no change)": ("rmse_ratio", 1.0, True), "MAE ratio (vs no change)": ("mae_ratio", 1.0, True),
           "Direction accuracy": ("dir_acc", 0.5, False), "F1 score (up)": ("f1_up", 0.5, False), "MAPE %": ("mape_pct", None, True)}


def accuracy_heatmap(met, target, models, metric):
    col, mid, lower_good = METRICS[metric]
    m = met[(met["target"] == target) & (met["model"].isin(models))]
    ms = [x for x in D.order(m["model"].unique())]
    hs = sorted(m["horizon"].unique())
    piv = m.pivot(index="model", columns="horizon", values=col).reindex(ms)[hs]
    dm = m.pivot(index="model", columns="horizon", values="dm_p_vs_no_change").reindex(ms)[hs]
    def star(p):
        return "" if pd.isna(p) or col not in ("rmse_ratio", "mae_ratio") else "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
    stars = dm.apply(lambda c: c.map(star))
    txt = piv.round(3).astype(str) + stars
    z = piv.values.astype(float)
    spread = max(0.02, float(np.nanmax(np.abs(z - mid)))) if mid is not None and z.size else None
    good, bad = SIG["BUY"], SIG["SELL"]
    scale = [[0, good], [0.5, CARD], [1, bad]] if lower_good else [[0, bad], [0.5, CARD], [1, good]]
    kw = dict(zmid=mid, zmin=mid - spread, zmax=mid + spread) if mid is not None else {}
    fig = go.Figure(go.Heatmap(z=z, x=[f"{h}d" for h in hs], y=[D.label(x) for x in ms], colorscale=scale,
                               text=txt.values, texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>",
                               **kw, colorbar=dict(title=col)))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, f"{D.TARGETS[target]}: {metric} (green = better; * p<0.10, ** p<0.05 vs no change)",
                   height=90 + 38 * max(len(ms), 3), hovermode="closest")


def coverage_chart(met, target):
    m = met[(met["target"] == target) & met["coverage_1sd"].notna()]
    fig = go.Figure()
    for mod, g in m.groupby("model"):
        g = g.sort_values("horizon")
        fig.add_trace(go.Scatter(x=g["horizon"], y=100 * g["coverage_1sd"], mode="lines+markers", name=D.label(mod),
                                 line=dict(color=COLORS.get(mod)), hovertemplate="%{y:.0f}%"))
    fig.add_hline(y=68.3, line_dash="dash", line_color="#ffffff", annotation_text="68% = perfectly calibrated")
    fig.update_yaxes(title="% of outcomes inside ±1 sd", range=[40, 95])
    fig.update_xaxes(title="Horizon (trading days)", type="log", tickvals=sorted(m["horizon"].unique()))
    return _layout(fig, f"{D.TARGETS[target]}: how often the actual move fell inside the model's ±1 sd band", height=380)


def rolling_chart(target, model, h, band_model, days=180):
    rf = D.frozen("rolling_forecasts.csv")
    rf = rf[(rf["target"] == target) & (rf["horizon"] == h)]
    p = D.price_series(target)
    pos = pd.Series(np.arange(len(p)), index=p.index)
    f = rf[rf["model"] == model].set_index("asof_date").sort_index()
    b = rf[rf["model"] == band_model].set_index("asof_date").sort_index()
    f = f[f.index.isin(p.index)]
    tgt = pd.Series([p.index[pos[d] + h] if pos[d] + h < len(p) else pd.NaT for d in f.index], index=f.index)
    act = pd.Series(np.log(p.shift(-h) / p).reindex(f.index))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.55, 0.45], vertical_spacing=0.12,
                        subplot_titles=(f"Price: forecast made {h} trading day{'s' if h > 1 else ''} earlier, drawn at the date it predicts",
                                        f"The {h}-day return itself (%), with the ±1 sd band"))
    ok = tgt.notna()
    zoom = p.index[-days:]
    fp = p.reindex(f.index) * np.exp(f["forecast_return"])
    bs = b["sigma"].reindex(f.index)
    z = ok & (tgt >= zoom[0])
    fig.add_trace(go.Scatter(x=zoom, y=p.loc[zoom], name="Actual", line=dict(color="#fff", width=2.4)), 1, 1)
    if bs.notna().any():
        up = (p.reindex(f.index) * np.exp(bs))[z]
        lo = (p.reindex(f.index) * np.exp(-bs))[z]
        fig.add_trace(go.Scatter(x=tgt[z], y=up, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=tgt[z], y=lo, fill="tonexty", fillcolor="rgba(86,180,233,0.17)", mode="lines", line=dict(width=0),
                                 name=f"{D.label(band_model)} band"), 1, 1)
    fig.add_trace(go.Scatter(x=tgt[z], y=fp[z], name=D.label(model), line=dict(color=COLORS.get(model), width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=act.index, y=100 * act, name=f"Actual {h}-day return", line=dict(color="#fff", width=1), opacity=0.8), 2, 1)
    if bs.notna().any():
        fig.add_trace(go.Scatter(x=bs.index, y=100 * bs, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), 2, 1)
        fig.add_trace(go.Scatter(x=bs.index, y=-100 * bs, fill="tonexty", fillcolor="rgba(86,180,233,0.17)", mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), 2, 1)
    if abs(f["forecast_return"]).max() > 1e-4:
        fig.add_trace(go.Scatter(x=f.index, y=100 * f["forecast_return"], name=f"{D.label(model)} forecast",
                                 line=dict(color=COLORS.get(model), width=1.6)), 2, 1)
    else:
        fig.add_hline(y=0, line_color=COLORS["RW"], row=2, col=1, annotation_text=f"{D.label(model)} = no change (0%)")
    fig.update_yaxes(title="USD / bbl", row=1, col=1)
    fig.update_yaxes(title="%", row=2, col=1)
    return _layout(fig, None, height=640, hovermode="x unified")


def equity(target, h, models, cost_bps, amount):
    eq = D.frozen("equity_curves.csv")
    eq = eq[(eq["target"] == target) & (eq["horizon"].isin([h, 0])) & (eq["model"].isin(set(models) | {"Buy & hold"}))]
    out = {}
    for m, g in eq.groupby("model"):
        g = g.sort_values("date")
        net = g["gross"].values - cost_bps / 1e4 * g["turn"].values
        out[m] = pd.Series(amount * np.cumprod(1 + net), index=g["date"].values), pd.Series(net, index=g["date"].values), float(g["turn"].sum())
    return out


def equity_stats(curves, amount, cost_bps):
    rows = []
    for m, (c, net, turn) in curves.items():
        n = len(net)
        tot = c.iloc[-1] / amount - 1
        ann = (c.iloc[-1] / amount) ** (252 / n) - 1 if n else np.nan
        vol = net.std() * np.sqrt(252)
        dd = (c / c.cummax() - 1).min()
        rows.append({"Model": D.label(m), "End value (USD)": c.iloc[-1], "Total return": tot, "Annualised": ann,
                     "Sharpe": (net.mean() * 252) / vol if vol > 0 else np.nan, "Max drawdown": dd,
                     "Turnover": turn, "Costs paid (USD)": turn * cost_bps / 1e4 * amount})
    return pd.DataFrame(rows)


def equity_chart(curves, amount):
    fig = go.Figure()
    for m, (c, _, turn) in curves.items():
        if m != "Buy & hold" and turn == 0:
            continue
        fig.add_trace(go.Scatter(x=c.index, y=c.values, name=D.label(m), line=dict(color=COLORS.get(m), width=3 if m == "Buy & hold" else 1.8,
                                                                                dash="dash" if m == "Buy & hold" else "solid")))
    fig.add_hline(y=amount, line_color=GRID)
    fig.update_yaxes(title="Portfolio value (USD)")
    return _layout(fig, None, height=420)


def robust_chart(rob, target, h, models, kind):
    r = rob[(rob["target"] == target) & (rob["horizon"] == h) & (rob["kind"] == kind) & (rob["model"].isin(models))]
    fig = go.Figure()
    for m in D.order(r["model"].unique()):
        g = r[r["model"] == m]
        fig.add_trace(go.Bar(x=g["slice"], y=g["rmse_ratio"], name=D.label(m), marker_color=COLORS.get(m),
                             hovertemplate="%{y:.3f}"))
    fig.add_hline(y=1, line_color="#fff", line_dash="dash", annotation_text="1.0 = no change")
    lo = float(r["rmse_ratio"].min()) if len(r) else 0.95
    hi = float(r["rmse_ratio"].max()) if len(r) else 1.05
    fig.update_yaxes(title="RMSE ratio (below 1 beats no change)", range=[min(0.95, lo - 0.02), max(1.05, hi + 0.02)])
    return _layout(fig, None, height=360, barmode="group")


# ------------------------------------------------------------------ Explore
def history_chart(target, kind, days):
    p = D.price_series(target, "futures" if kind != "Spot" else "spot")
    y, f = D.prices()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    sers = {"Futures": [("futures", "#f5a623")], "Spot": [("spot", "#56b4e9")], "Both": [("futures", "#f5a623"), ("spot", "#56b4e9")]}[kind]
    for k_, c in sers:
        s = D.price_series(target, k_)
        s = s.tail(days) if days else s
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{D.TARGETS[target]} {k_}", line=dict(color=c, width=1.8)), 1, 1)
    base = D.price_series(target)
    lr = np.log(base).diff()
    vol = (lr.rolling(20).std() * np.sqrt(252) * 100).dropna()
    vol = vol.tail(days) if days else vol
    fig.add_trace(go.Scatter(x=vol.index, y=vol.values, name="20-day realised vol (%)", line=dict(color="#a78bfa", width=1.4)), 2, 1)
    if "ovx" in f and f["ovx"].notna().any():
        o = f["ovx"].dropna()
        o = o[o.index >= vol.index.min()]
        fig.add_trace(go.Scatter(x=o.index, y=o.values, name="OVX (implied vol)", line=dict(color="#2fc79a", width=1.2)), 2, 1)
    fig.update_yaxes(title="USD / bbl", row=1, col=1)
    fig.update_yaxes(title="vol (%)", row=2, col=1)
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=1, label="1m", step="month", stepmode="backward"),
                                                 dict(count=6, label="6m", step="month", stepmode="backward"),
                                                 dict(count=1, label="1y", step="year", stepmode="backward"),
                                                 dict(step="all", label="all")], bgcolor=CARD), row=2, col=1)
    return _layout(fig, None, height=560)


def log_pairs(target, h, models):
    """Forecast-log rows with a known outcome: forecast price vs actual price at the h-th trading day after the origin."""
    lg = D.log()
    lg = lg[(lg["target"] == target) & (lg["horizon"] == h) & (lg["model"].isin(models))].drop_duplicates(["asof_date", "model"], keep="last")
    p = D.price_series(target)
    pos = pd.Series(np.arange(len(p)), index=p.index)
    rows = []
    for r in lg.itertuples():
        if r.asof_date not in pos.index or pos[r.asof_date] + h >= len(p):
            continue
        t = p.index[pos[r.asof_date] + h]
        act, p0 = float(p.iloc[pos[r.asof_date] + h]), float(p.loc[r.asof_date])
        fp = p0 * np.exp(r.forecast_return)
        rows.append({"model": r.model, "origin": r.asof_date, "target_date": t, "forecast": fp, "actual": act,
                     "err_pct": 100 * (fp / act - 1), "source": r.source, "p0": p0})
    return pd.DataFrame(rows)


def explorer_chart(pairs, models):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=("Forecast vs actual (at the target date)",
                                                                                      "Forecast error (%)"))
    if pairs.empty:
        return _layout(fig, None, height=380)
    a = pairs.drop_duplicates("target_date").sort_values("target_date")
    fig.add_trace(go.Scatter(x=a["target_date"], y=a["actual"], name="Actual", line=dict(color="#fff", width=2.4)), 1, 1)
    for m in models:
        g = pairs[pairs["model"] == m].sort_values("target_date")
        if len(g):
            fig.add_trace(go.Scatter(x=g["target_date"], y=g["forecast"], name=D.label(m), line=dict(color=COLORS.get(m), width=1.8)), 1, 1)
            fig.add_trace(go.Histogram(x=g["err_pct"], name=D.label(m), marker_color=COLORS.get(m), opacity=0.6, showlegend=False), 1, 2)
    fig.update_layout(barmode="overlay")
    return _layout(fig, None, height=400, hovermode="closest")
