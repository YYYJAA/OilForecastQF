"""Palette, CSS, sidebar controls (shared by every page) and small HTML components."""
import html
from datetime import timezone

import pandas as pd
import streamlit as st

from . import data as D

COLORS = {"RW": "#9aa5b1", "LASSO": "#56b4e9", "LASSO_1se": "#2a7fc1", "PCA_reg": "#e69f00", "TPCA": "#d55e00",
          "LASSO_RS": "#cc79a7", "Combo_EW": "#2fc79a", "ARMA-GARCH": "#f0e442", "MS-AR": "#a78bfa", "MS-ARX": "#ff7b7b",
          "HistMean": "#7d8590", "AR": "#b0b8c1", "Buy & hold": "#ffffff"}
SIG = {"BUY": "#2fc79a", "SELL": "#ff6b6b", "HOLD": "#8b98a9", "LEAN BUY": "#2fc79a", "LEAN SELL": "#ff6b6b", "NEUTRAL": "#8b98a9"}
TONE = {"HIGH": "#2fc79a", "MEDIUM": "#f5a623", "LOW": "#8b98a9", "n/a": "#5d6b7d"}
BG, CARD, GRID = "#0e1621", "#16212f", "#263447"

HELP = {
    "ratio": "RMSE ratio = the model's typical error divided by the error of 'no change'. Below 1 beats no change, above 1 is worse.",
    "dir": "Direction accuracy: how often the model got up vs down right. 50% is a coin flip.",
    "coverage": "How often the actual move landed inside the model's +-1 sd band. A well-calibrated band covers about 68%.",
    "pup": "Model-implied chance the price is higher at the horizon, from its forecast and its own noise estimate.",
    "z": "Expected move divided by the 1-sd noise. Under 0.1 the signal is HOLD: the move is lost in the noise.",
    "sharpe": "Return per unit of risk (annualised). Above 1 is good; negative means it lost money for the risk taken.",
    "track": "Evidence from the frozen test, not the model's own claim: is the direction right more often than a coin flip "
             "and does it beat 'no change'? Significant / Suggestive / None yet / No call (forecasts exactly no change).",
}

CSS = f"""
<style>
.block-container {{padding-top:1.6rem; max-width:1250px}}
h1,h2,h3 {{letter-spacing:-0.01em}}
.hero {{background:linear-gradient(135deg,#16212f 0%,#1d2f45 100%); border:1px solid {GRID}; border-radius:16px; padding:18px 22px; margin:4px 0 14px}}
.hero .t {{font-size:13px; text-transform:uppercase; letter-spacing:.12em; color:#8fa3bb}}
.hero .s {{font-size:21px; line-height:1.4; margin-top:4px}}
.kpis {{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:6px 0 14px}}
.kpi {{background:{CARD}; border:1px solid {GRID}; border-radius:14px; padding:14px 16px}}
.kpi .l {{font-size:12px; text-transform:uppercase; letter-spacing:.1em; color:#8fa3bb}}
.kpi .v {{font-size:28px; font-weight:650; margin-top:2px}}
.kpi .d {{font-size:13px; color:#a9b7c9; margin-top:2px}}
.chip {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:12.5px; font-weight:650; border:1px solid currentColor}}
.fresh {{background:{CARD}; border:1px solid {GRID}; border-radius:10px; padding:8px 10px; font-size:12.5px; color:#a9b7c9}}
.bar {{height:8px; border-radius:4px; background:{GRID}; position:relative; min-width:90px}}
.bar i {{position:absolute; top:0; bottom:0; border-radius:4px}}
.mt {{width:100%; min-width:640px; border-collapse:collapse; font-size:14px}}
.mt th {{text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#8fa3bb; padding:6px 8px; border-bottom:1px solid {GRID}}}
.mt td {{padding:8px; border-bottom:1px solid {GRID}; vertical-align:middle}}
.dot {{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px}}
.note {{font-size:13px; color:#a9b7c9}}
@media (max-width:700px) {{ .hero .s {{font-size:17px}} .kpi .v {{font-size:23px}} }}
</style>
"""



def setup(title="Oil Forecast Monitor", icon="🛢️"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide", initial_sidebar_state="auto")
    st.markdown(CSS, unsafe_allow_html=True)


def _all_models():
    lt = D.latest()
    return [m for m in D.order(lt["model"].unique()) if m != "RW"] if len(lt) else []


def _seed_from_url():
    if st.session_state.get("_seeded"):
        return
    qp = st.query_params
    if qp.get("models"):
        chosen = set(qp["models"].split(","))
        for m in _all_models():
            st.session_state[f"cb_{m}"] = m in chosen
    if qp.get("h", "").isdigit():
        st.session_state["w_h"] = min(60, max(1, int(qp["h"])))
    st.session_state["_seeded"] = True


def _restore():
    """Widget state is dropped when a page does not draw the widget; keep our own copy and restore it."""
    keep = st.session_state.setdefault("_keep", {})
    for k, v in keep.items():
        st.session_state.setdefault(k, v)


def _remember():
    keep = st.session_state.setdefault("_keep", {})
    for k in list(st.session_state.keys()):
        if k.startswith("cb_") or k in ("w_h", "w_window"):
            keep[k] = st.session_state[k]


def _set_all(on):
    for m in _all_models():
        st.session_state[f"cb_{m}"] = on


def sidebar():
    """Model picker, freshness chip, refresh. Returns (targets, models); both series are always available as tabs."""
    _restore()
    _seed_from_url()
    lt = D.latest()
    allm = _all_models()
    for m in allm:
        st.session_state.setdefault(f"cb_{m}", True)
    with st.sidebar:
        st.markdown("### 🛢️ Oil Forecast Monitor")
        n = sum(st.session_state[f"cb_{m}"] for m in allm)
        with st.expander(f"Models shown ({n}/{len(allm)})", expanded=False):
            c = st.columns(2)
            c[0].button("All", on_click=_set_all, args=(True,), use_container_width=True)
            c[1].button("None", on_click=_set_all, args=(False,), use_container_width=True)
            for m in allm:
                st.checkbox(D.label(m), key=f"cb_{m}")
        last_px, gen, asof = D.freshness()
        txt = "No data yet"
        if asof is not None:
            txt = f"Forecasts for <b>{asof:%d %b %Y}</b>"
            if gen:
                try:
                    g = pd.Timestamp(gen).tz_convert("Asia/Singapore")
                    txt += f"<br>updated {g:%d %b %H:%M} SGT"
                except Exception:
                    pass
            if last_px is not None:
                txt += f"<br>last price {last_px:%d %b %Y}"
        st.markdown(f"<div class='fresh'>{txt}</div>", unsafe_allow_html=True)
        if st.button("Refresh data", use_container_width=True, help="Reload the files from the repo (the site also refreshes itself "
                                                                    "whenever the daily bot commits)."):
            st.cache_data.clear()
            st.rerun()
    _remember()
    models = [m for m in allm if st.session_state[f"cb_{m}"]]
    try:
        st.query_params.update({"models": ",".join(models), "h": str(st.session_state.get("w_h", 5))})
    except Exception:
        pass
    return ["brent_fut", "wti_fut"], models


def chip(text, color):
    return f"<span class='chip' style='color:{color}'>{html.escape(str(text))}</span>"


def kpi(label, value, sub="", color=None):
    st_ = f" style='color:{color}'" if color else ""
    return f"<div class='kpi'><div class='l'>{label}</div><div class='v'{st_}>{value}</div><div class='d'>{sub}</div></div>"


def kpis(items):
    st.markdown("<div class='kpis'>" + "".join(items) + "</div>", unsafe_allow_html=True)


def hero(title, text):
    st.markdown(f"<div class='hero'><div class='t'>{title}</div><div class='s'>{text}</div></div>", unsafe_allow_html=True)


def pbar(p):
    """Centred bar: right of the midline = chance up, left = chance down."""
    if p is None or p != p:
        return "<span class='note'>n/a</span>"
    w = abs(p - 0.5) * 100
    col = SIG["BUY"] if p >= 0.5 else SIG["SELL"]
    pos = f"left:50%;width:{w}%" if p >= 0.5 else f"left:{50 - w}%;width:{w}%"
    return (f"<div style='display:flex;align-items:center;gap:8px'><div class='bar' style='flex:1'>"
            f"<i style='{pos};background:{col}'></i><i style='left:50%;width:1px;background:#8fa3bb'></i></div>"
            f"<span style='font-size:12.5px;min-width:34px'>{100 * p:.0f}%</span></div>")


def model_dot(m):
    return f"<span class='dot' style='background:{COLORS.get(m, '#999')}'></span>{html.escape(D.label(m))}"


def signal_html(df, folded=()):
    rows = []
    for r in df.itertuples():
        col = TONE.get(r.conf, TONE["n/a"])
        hit = "n/a" if r.hit != r.hit else f"{100 * r.hit:.0f}%"
        z = "n/a" if r.z != r.z else f"{abs(r.z):.2f}"
        rows.append(f"<tr><td>{model_dot(r.model)}</td><td>{chip(r.signal, SIG[r.signal])}</td>"
                    f"<td style='text-align:right'>{100 * r.move:+.2f}%</td><td style='text-align:right'>{z}</td>"
                    f"<td style='min-width:140px'>{pbar(r.p_up)}</td><td style='text-align:right'>{hit}</td>"
                    f"<td>{chip(D.SHOW.get(r.conf, r.conf), col)}</td></tr>")
    if len(folded) >= 1:
        names = ", ".join(D.label(m) for m in folded)
        rows.append(f"<tr><td><span class='dot' style='background:{COLORS['LASSO']}'></span>{names}</td>"
                    f"<td>{chip('HOLD', SIG['HOLD'])}</td><td style='text-align:right'>+0.00%</td><td style='text-align:right'>0.00</td>"
                    f"<td>{pbar(0.5)}</td><td style='text-align:right'>n/a</td><td>{chip('No call', TONE['n/a'])}</td></tr>")
    head = ("<tr><th>Model</th><th>Signal</th><th style='text-align:right'>Expected move</th><th style='text-align:right'>Move / noise</th>"
            "<th>Chance up</th><th style='text-align:right'>Hit-rate (test)</th><th>Track record</th></tr>")
    return f"<div style='overflow-x:auto'><table class='mt'>{head}{''.join(rows)}</table></div>"


def plotly_cfg():
    return {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
