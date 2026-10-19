"""Track record: the frozen out-of-sample test, calibration, robustness and economic value."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from lib import charts as C
from lib import data as D
from lib import ui

ui.setup("Track record · Oil Forecast Monitor", "📈")
targets, models = ui.sidebar()
summ, met, rob = D.frozen("summary.json"), D.frozen("metrics.csv"), D.frozen("robustness.csv")

st.title("Track record")
if met.empty:
    st.info("The frozen-model evaluation has not been run yet.")
    st.stop()
st.markdown(
    f"<div class='hero'><div class='t'>How we test</div><div class='s' style='font-size:17px'>Models were estimated only on data up to "
    f"<b>{summ.get('cutoff')}</b>, then applied day by day from <b>{summ.get('first_origin')}</b> to <b>{summ.get('last_origin')}</b> "
    f"({summ.get('n_origins')} forecast dates) with their parameters frozen. Nothing in this period was used to fit a model, "
    f"so the results below are out-of-sample. Benchmarks: no change, historical mean, direct AR.</div></div>", unsafe_allow_html=True)
shown = list(models) + [m for m in D.BENCH]


def render(target):
    name = D.TARGETS[target]
    st.subheader("1 · Accuracy by model and horizon", help=ui.HELP["ratio"])
    metric = st.radio("Metric", list(C.METRICS), horizontal=True, key=f"met_{target}")
    st.plotly_chart(C.accuracy_heatmap(met, target, shown, metric), use_container_width=True, config=ui.plotly_cfg(), key=f"ah_{target}")
    st.caption("Ratios below 1 mean the model's error was smaller than 'no change'. Beating a random walk in oil is hard; "
               "the stars mark the differences that are statistically distinguishable from no change.")

    st.subheader("2 · Are the uncertainty bands honest?", help=ui.HELP["coverage"])
    cm = met[met["model"].isin(models)]
    st.plotly_chart(C.coverage_chart(cm, target), use_container_width=True, config=ui.plotly_cfg(), key=f"cv_{target}")

    st.subheader("3 · Rolling forecast against what happened")
    hs = sorted(D.frozen("rolling_forecasts.csv")["horizon"].unique()) if len(D.frozen("rolling_forecasts.csv")) else [1]
    cands = [m for m in models if m != "RW"] or ["RW"]
    a, b, c, d = st.columns(4)
    mod = a.selectbox("Model", cands, format_func=D.label, key=f"rm_{target}")
    hh = b.selectbox("Horizon (days)", hs, index=min(1, len(hs) - 1), key=f"rh_{target}")
    bnd = c.selectbox("Band from", ["RW", "ARMA-GARCH", "MS-AR", "MS-ARX"], format_func=lambda m: "Recent volatility" if m == "RW" else m,
                      key=f"rb_{target}")
    days = d.selectbox("Zoom", [90, 180, 365, 450], index=1, format_func=lambda x: f"last {x} days", key=f"rz_{target}")
    st.plotly_chart(C.rolling_chart(target, mod, hh, bnd, days), use_container_width=True, config=ui.plotly_cfg(), key=f"ro_{target}")

    st.subheader("4 · Does it hold up across market conditions?")
    rh = st.select_slider("Horizon (days)", sorted(rob["horizon"].unique()), value=5, key=f"rbh_{target}")
    tabs = st.tabs(["Volatility regime", "Calendar year", "Time windows"])
    for tb, kind in zip(tabs, ["regime", "year", "window"]):
        with tb:
            st.plotly_chart(C.robust_chart(rob, target, rh, shown, kind), use_container_width=True, config=ui.plotly_cfg(),
                            key=f"rb_{kind}_{target}")

    st.subheader("5 · What would a trader have made?", help=ui.HELP["sharpe"])
    eq = D.frozen("equity_curves.csv")
    if eq.empty:
        st.info("Equity curves are produced by the next frozen evaluation run.")
    else:
        trade_models = [m for m in models if m != "RW"]
        a, b, c, d = st.columns([2, 1, 1, 1])
        pick = a.multiselect("Models", trade_models, default=[m for m in trade_models if m in ("LASSO", "XGB", "MS-AR", "MS-ARX")],
                             format_func=D.label, key=f"eqm_{target}")
        eh = b.selectbox("Horizon (days)", sorted(h for h in eq["horizon"].unique() if h > 0), index=2, key=f"eqh_{target}")
        cost = c.slider("Cost (bp per trade)", 0.0, 25.0, 5.0, 0.5, key=f"eqc_{target}")
        amt = d.number_input("Start (USD)", 1000, 1_000_000, 10_000, 1000, key=f"eqa_{target}")
        cur = C.equity(target, eh, pick, cost, amt)
        if cur:
            st.plotly_chart(C.equity_chart(cur, amt), use_container_width=True, config=ui.plotly_cfg(), key=f"eq_{target}")
            tbl = C.equity_stats(cur, amt, cost)
            st.dataframe(tbl.style.format({"End value (USD)": "{:,.0f}", "Total return": "{:+.1%}", "Annualised": "{:+.1%}",
                                           "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}", "Turnover": "{:.0f}",
                                           "Costs paid (USD)": "{:,.0f}"}), hide_index=True, use_container_width=True)
            st.caption("Rule: long / short when the forecast is bigger than 0.1 × its noise, otherwise flat; each day opens 1/h of capital "
                       "held for h days. Models that never trade are omitted. Replayed from the saved daily returns, so changing the cost "
                       "re-prices the same trades.")
        else:
            st.info("Select at least one model.")

    st.subheader("6 · Live scorecard (forecasts logged since the pipeline went live)")
    lh = st.selectbox("Horizon (days)", [1, 5, 10, 20], key=f"lh_{target}")
    pr = C.log_pairs(target, lh, [m for m in models])
    if pr.empty:
        st.info("Not enough completed forecasts yet.")
    else:
        pr["e_model"] = np.log(pr["forecast"] / pr["actual"]) ** 2
        pr["e_nc"] = np.log(pr["p0"] / pr["actual"]) ** 2
        t = pr.groupby("model").agg(n=("e_model", "size"), m=("e_model", "mean"), r=("e_nc", "mean")).reset_index()
        t["RMSE ratio"] = np.sqrt(t["m"] / t["r"])
        t["Model"] = t["model"].map(D.label)
        st.dataframe(t[["Model", "n", "RMSE ratio"]].rename(columns={"n": "Forecasts scored"}), hide_index=True,
                     use_container_width=True, column_config={"RMSE ratio": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Includes the back-filled forecasts the pipeline reconstructed before launch (made with then-available data).")


if len(targets) == 1:
    render(targets[0])
else:
    for tb, t in zip(st.tabs([D.TARGETS[t] for t in targets]), targets):
        with tb:
            render(t)
