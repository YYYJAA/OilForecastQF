"""Today: current forecasts, the consensus signal and an interactive fan chart."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from lib import charts as C
from lib import data as D
from lib import ui

ui.setup()
targets, models = ui.sidebar()

st.title("Oil Forecast Monitor")

# ------------------------------------------------------------------ controls
st.session_state.setdefault("w_h", 5)
H = int(st.session_state["w_h"])          # the slider sits under each chart; it writes w_h through _sync_h


def _sync_h(target):
    st.session_state["w_h"] = st.session_state[f"h_{target}"]


with st.expander("Chart options"):
    c1, c2, c3 = st.columns(3)
    band = c1.selectbox("Uncertainty band", list(D.BAND_BY), help="Whose noise estimate draws the shaded band and the 'noise' in the table.")
    k = {"±1 sd (68%)": 1.0, "±2 sd (95%)": 2.0}[c2.radio("Band width", ["±1 sd (68%)", "±2 sd (95%)"], horizontal=True)]
    win = c3.selectbox("History shown", [0, 30, 90, 180, 365], index=0, key="w_window",
                       format_func=lambda d: "Auto" if d == 0 else f"last {d} days")
    t1, t2, t3, t4 = st.columns(4)
    show_band = t1.toggle("Show band", True)
    show_rw = t2.toggle("Show 'no change' line", True)
    as_pct = t3.toggle("Show % change", False)
    log_y = t4.toggle("Log scale", False, disabled=as_pct)
if win == 0:
    win = int(min(365, max(30, 4 * H)))


def render(target):
    name = D.TARGETS[target]
    cur = D.current(target)
    if cur.empty:
        st.info(f"No forecasts for {name} yet.")
        return
    p0 = float(cur["price_now"].iloc[0])
    asof = cur["asof_date"].iloc[0]
    allm = set(D.latest()["model"])
    missing = [D.label(m) for m in D.order(allm - set(cur["model"])) if m != "RW"]
    if missing:
        st.info(f"Live {name} forecasts so far come from {', '.join(D.label(m) for m in D.order(set(cur['model'])) if m != 'RW')} only; "
                f"not yet available for {name}: {', '.join(missing)}.")
    rows = D.at_horizon(cur, H)
    flat = [m for m in D.no_change_models(cur) if m in models]
    tbl = D.signal_table(target, rows, H, 0.1, [m for m in models if m not in flat])
    nb = ns = nh = 0
    label = "NEUTRAL"
    if len(tbl):
        label, nb, ns, nh = D.consensus(tbl)
    nh += 1 if flat else 0
    bm = D.BAND_BY[band]
    brow = rows[rows["model"] == bm]
    sg = float(brow["sigma"].iloc[0]) if len(brow) and np.isfinite(brow["sigma"].iloc[0]) else D.recent_vol_sigma(target, H)
    lo, hi = p0 * np.exp(-k * sg), p0 * np.exp(k * sg)
    prev = D.price_series(target)
    prev = prev[prev.index < asof]
    chg = p0 / float(prev.iloc[-1]) - 1 if len(prev) else np.nan
    rp = cur["regime_p"].dropna()
    reg_p = float(rp.mean()) if len(rp) else np.nan
    n_sel = len(tbl) + (1 if flat else 0)
    voters = tbl[tbl["signal"] == label.split()[-1]]["label"].tolist() if label != "NEUTRAL" else []

    # one-sentence summary
    if not n_sel:
        story = "No model is selected: tick at least one in the sidebar."
    elif label == "NEUTRAL" and nb == 0 and ns == 0:
        story = f"No actionable signal: every selected model's expected move is inside its noise."
    elif label == "NEUTRAL":
        story = f"Mixed picture: {nb} model(s) lean BUY and {ns} lean SELL, so there is no consensus."
    else:
        story = f"{nb if 'BUY' in label else ns} of {n_sel} models {label.lower()}: {', '.join(voters)}."
    story += (f" {name} is likely to stay between <b>{lo:.1f}</b> and <b>{hi:.1f}</b> over the next {H} trading day{'s' if H > 1 else ''} "
              f"(±{k:g} sd, {band.lower()}).")
    if np.isfinite(reg_p):
        story += " Volatility regime: " + ("<b>turbulent</b>." if reg_p > 0.5 else "calm.")
    ui.hero(f"{name} futures · as of {asof:%A %d %B %Y}", story)

    sigcol = ui.SIG[label]
    ui.kpis([
        ui.kpi("Last price", f"${p0:,.2f}", f"{100 * chg:+.2f}% on the day" if np.isfinite(chg) else ""),
        ui.kpi(f"Consensus ({H}d)", label, f"{nb} buy · {ns} sell · {nh} hold", sigcol),
        ui.kpi(f"Likely range ({H}d)", f"{lo:,.1f} – {hi:,.1f}", f"±{k:g} sd = {100 * (np.exp(k * sg) - 1):.1f}% either way"),
        ui.kpi("High-volatility regime", f"{100 * reg_p:.0f}%" if np.isfinite(reg_p) else "n/a",
               "chance we are in the turbulent state" if np.isfinite(reg_p) else "", "#f5a623" if reg_p > 0.5 else None),
    ])
    st.plotly_chart(C.fan_chart(target, cur, H, models, band, k, win, show_band, show_rw, log_y, as_pct),
                    use_container_width=True, config=ui.plotly_cfg(), key=f"fan_{target}")
    st.session_state[f"h_{target}"] = H
    sl, _ = st.columns([2, 3])
    sl.slider("How far ahead (trading days)", 1, 60, key=f"h_{target}", on_change=_sync_h, args=(target,),
              help="Drag to move the forecast horizon; between the saved horizons (1, 5, 10, 20, 40, 60) the forecast is interpolated.")

    a, b, c, d = st.tabs(["Models today", "By horizon", "Price ranges", "Last 7 days"])
    with a:
        if len(tbl) or flat:
            st.markdown(ui.signal_html(tbl, flat), unsafe_allow_html=True)
            st.markdown(f"<p class='note'>HOLD = the expected move is under 0.1 × its noise. <b>Track record</b> comes from the frozen test "
                        f"(models trained to 31 Dec 2024, applied since 2 Jan 2025), nearest evaluated horizon. "
                        f"<b>Chance up</b> is the model's own view.</p>", unsafe_allow_html=True)
        else:
            st.info("Select models in the sidebar.")
    with b:
        st.plotly_chart(C.horizon_heatmap(cur, models), use_container_width=True, config=ui.plotly_cfg(), key=f"hm_{target}")
    with c:
        st.plotly_chart(C.uncertainty_chart(cur, H, k, models), use_container_width=True, config=ui.plotly_cfg(), key=f"un_{target}")
    with d:
        pr = C.log_pairs(target, 1, [m for m in models if m not in flat])
        if pr.empty:
            st.info("No completed 1-day forecasts yet.")
        else:
            wide = pr.pivot_table(index=["origin", "target_date"], columns="model", values="forecast").reset_index()
            act = pr.drop_duplicates("origin").set_index("origin")[["actual", "p0"]]
            wide = wide.merge(act, left_on="origin", right_index=True).sort_values("origin").tail(7)
            wide = wide.rename(columns={m: D.label(m) for m in models}).rename(
                columns={"origin": "Forecast made", "target_date": "For day", "actual": "Actual", "p0": "No change"})
            for col in wide.columns[2:]:
                wide[col] = wide[col].round(2)
            wide["Forecast made"] = wide["Forecast made"].dt.strftime("%a %d %b")
            wide["For day"] = wide["For day"].dt.strftime("%a %d %b")
            st.dataframe(wide.iloc[::-1], hide_index=True, use_container_width=True)
            st.caption("Prices the models predicted one day ahead, next to what happened.")


if len(targets) == 1:
    render(targets[0])
else:
    tabs = st.tabs([D.TARGETS[t] for t in targets])
    for tb, t in zip(tabs, targets):
        with tb:
            render(t)
