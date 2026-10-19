"""Explore: price history, forecast-vs-actual explorer and downloads."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from lib import charts as C
from lib import data as D
from lib import ui

ui.setup("Explore · Oil Forecast Monitor", "🔎")
targets, models = ui.sidebar()
st.title("Explore")


def render(target):
    st.subheader("Price history")
    a, b = st.columns([1, 2])
    kind = a.radio("Series", ["Futures", "Spot", "Both"], horizontal=True, key=f"hk_{target}",
                   help="Futures are the continuous front-month contract (they jump when the contract rolls); spot comes from FRED "
                        "with about a week's delay.")
    days = b.select_slider("Window", [90, 180, 365, 730, 1260, 2520, 0], value=730, key=f"hw_{target}",
                           format_func=lambda d: "all" if d == 0 else f"{d} days")
    st.plotly_chart(C.history_chart(target, kind, days), use_container_width=True, config=ui.plotly_cfg(), key=f"hist_{target}")
    if target == "wti_fut":
        st.caption("WTI futures start on 1 June 2020: the series went negative on 20 April 2020, which breaks log returns.")

    st.subheader("Forecast vs actual explorer")
    a, b, c = st.columns(3)
    h = a.selectbox("Horizon (days)", [1, 5, 10, 20], key=f"eh_{target}")
    cands = [m for m in models if m != "RW"]
    m1 = b.selectbox("Model", cands, format_func=D.label, key=f"e1_{target}") if cands else None
    m2 = c.selectbox("Compare with", ["(none)"] + cands, format_func=lambda m: m if m == "(none)" else D.label(m), key=f"e2_{target}")
    if m1 is None:
        st.info("Select models in the sidebar.")
        return
    chosen = [m1] + ([m2] if m2 != "(none)" and m2 != m1 else [])
    pairs = C.log_pairs(target, h, chosen)
    if pairs.empty:
        st.info("No completed forecasts for this horizon yet.")
    else:
        st.plotly_chart(C.explorer_chart(pairs, chosen), use_container_width=True, config=ui.plotly_cfg(), key=f"ex_{target}")
        s = pairs.groupby("model")["err_pct"].agg(n="size", bias="mean", mae=lambda x: x.abs().mean()).reset_index()
        s["Model"] = s["model"].map(D.label)
        st.dataframe(s[["Model", "n", "bias", "mae"]].rename(columns={"n": "Forecasts", "bias": "Average error (%)", "mae": "Average |error| (%)"}),
                     hide_index=True, use_container_width=True,
                     column_config={"Average error (%)": st.column_config.NumberColumn(format="%.2f"),
                                    "Average |error| (%)": st.column_config.NumberColumn(format="%.2f")})
        if len(chosen) == 2:
            w = pairs.pivot_table(index="target_date", columns="model", values="err_pct").dropna().sort_index()
            if len(w):
                diff = (w[chosen[1]].abs() - w[chosen[0]].abs()).cumsum()
                st.line_chart(diff.rename("Cumulative |error| difference (pct points)"), height=200)
                st.caption(f"Up = {D.label(chosen[0])} has had the smaller error; down = {D.label(chosen[1])} has.")


if len(targets) == 1:
    render(targets[0])
else:
    for tb, t in zip(st.tabs([D.TARGETS[t] for t in targets]), targets):
        with tb:
            render(t)

st.subheader("Downloads")
files = [("Today's forecasts", D.OUT / "latest_forecasts.csv"), ("Forecast log", D.OUT / "forecast_log.csv"),
         ("Frozen test: accuracy by horizon", D.FROZEN / "registration_accuracy.csv"),
         ("Frozen test: robustness", D.FROZEN / "registration_robustness.csv"),
         ("Frozen test: metrics (all)", D.FROZEN / "metrics.csv"), ("Frozen test: trading", D.FROZEN / "trading.csv")]
cols = st.columns(3)
for i, (lab, p) in enumerate(files):
    if p.exists():
        cols[i % 3].download_button(lab, p.read_bytes(), file_name=p.name, mime="text/csv", use_container_width=True)
