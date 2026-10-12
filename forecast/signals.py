"""BUY / SELL / HOLD summary with an evidence-based track record for every model.

Signal      BUY / SELL if the forecast h-day return is larger than `trade_band` x its 1-sd (the same rule as the trading
            test, so the signal and the economic evaluation are consistent); otherwise HOLD.
Track record  NOT the model's own claim. It comes from the model's frozen-test record (Outputs/frozen_eval/metrics.csv):
              HIGH    direction right significantly more often than 50% (one-sided p < 0.05) AND RMSE below no change
              MEDIUM  p < 0.10, or hit-rate above 55% with RMSE below no change
              LOW     otherwise (no demonstrable edge)
            The test uses n / h effective observations, because overlapping h-day returns are not independent.
"""
from math import erf, sqrt

import numpy as np
import pandas as pd

from . import reporting as rp
from .config import OUTPUT_DIR

COLOUR = {"BUY": "#1b7f3b", "SELL": "#b3261e", "HOLD": "#777777"}
RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
TONE = {"HIGH": "#1b7f3b", "MEDIUM": "#b8860b", "LOW": "#8a8a8a", "n/a": "#bdbdbd"}   # row colour by track record
SHOW = {"HIGH": "Significant", "MEDIUM": "Suggestive", "LOW": "None yet", "n/a": "No call", "-": "-"}


def _phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def confidence(hit, n, h, ratio):
    """(label, one-sided p-value of 'direction is no better than a coin flip')."""
    if hit is None or n is None or not np.isfinite(hit) or n <= 0:
        return "n/a", np.nan
    n_eff = max(n / max(h, 1), 1.0)
    p = 1 - _phi((hit - 0.5) / sqrt(0.25 / n_eff))
    if p < 0.05 and ratio < 1:
        return "HIGH", p
    if p < 0.10 or (hit > 0.55 and ratio < 1):
        return "MEDIUM", p
    return "LOW", p


def _evidence():
    f = OUTPUT_DIR / "frozen_eval" / "metrics.csv"
    return pd.read_csv(f) if f.exists() else None


def table(cur, target, h, band):
    """Rows for every non-benchmark model at horizon h from today's forecasts `cur` (one target)."""
    ev = _evidence()
    out = []
    for r in cur[(cur["horizon"] == h) & (cur["model"] != "RW")].itertuples():
        f, s = float(r.forecast_return), float(r.sigma) if pd.notna(r.sigma) else np.nan
        z = f / s if s and np.isfinite(s) else np.nan
        sig = "HOLD" if (not np.isfinite(s) or abs(f) <= band * s) else ("BUY" if f > 0 else "SELL")
        p_up = _phi(z) if np.isfinite(z) else np.nan
        hit = n = ratio = np.nan
        if ev is not None:
            m = ev[(ev["target"] == target) & (ev["model"] == r.model) & (ev["horizon"] == h)]
            if len(m):
                hit, n, ratio = m["dir_acc"].iloc[0], float(m["n"].iloc[0]), m["rmse_ratio"].iloc[0]
        conf, p = confidence(hit, n, h, ratio)                    # a property of the model's record, not of today's call
        out.append({"model": r.model, "signal": sig, "move": f, "z": z, "p_up": p_up, "hit": hit,
                    "n_eff": n / h if n == n else np.nan, "ratio": ratio, "confidence": conf, "p": p})
    df = pd.DataFrame(out)
    if len(df):
        df["_rank"] = df["confidence"].map(RANK).fillna(0)
        df["_p"] = df["p"].fillna(1.0)
        df["_z"] = df["z"].abs().fillna(0)
        df = df.sort_values(["_rank", "_p", "_z"], ascending=[False, True, False]).drop(columns=["_rank", "_p", "_z"])
    return df.reset_index(drop=True)


def consensus(df):
    nb, ns, nh = (df["signal"] == "BUY").sum(), (df["signal"] == "SELL").sum(), (df["signal"] == "HOLD").sum()
    if nb >= 2 and nb > ns:
        label = "LEAN BUY"
    elif ns >= 2 and ns > nb:
        label = "LEAN SELL"
    else:
        label = "NEUTRAL"
    agree = df[df["signal"] == label.split()[-1]] if label != "NEUTRAL" else df.iloc[0:0]
    best = max((RANK.get(c, 0) for c in agree["confidence"]), default=0)
    conf = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(best, "-")
    return label, int(nb), int(ns), int(nh), conf


def block(cfg, latest, target_names):
    """HTML for the top of the e-mail. `latest`: today's forecasts with target / asof_date columns."""
    h, band = cfg.signal_horizon, cfg.trade_band
    parts, any_evidence = [], False
    for t in target_names:
        cur = latest[(latest["target"] == t)]
        cur = cur[cur["asof_date"] == cur["asof_date"].max()]
        df = table(cur, t, h, band)
        if df.empty:
            continue
        label, nb, ns, nh, conf = consensus(df)
        rows = []
        for r in df.itertuples():
            tone = TONE.get(r.confidence, TONE["n/a"])
            weight = "600" if r.confidence == "HIGH" else "400"

            def c(txt):
                return f"<span style='color:{tone};font-weight:{weight}'>{txt}</span>"
            hit = "n/a" if not np.isfinite(r.hit) else f"{100 * r.hit:.0f}% (n={r.n_eff:.0f})"
            if np.isfinite(r.hit):
                any_evidence = True
            rows.append([c(rp.label(r.model)),
                         f"<b style='color:{COLOUR[r.signal]}'>{r.signal}</b>",
                         c(f"{100 * (np.exp(r.move) - 1):+.2f}%"),
                         c("n/a" if not np.isfinite(r.z) else f"{abs(r.z):.2f}"),
                         c("n/a" if not np.isfinite(r.p_up) else f"{100 * r.p_up:.0f}%"),
                         c(hit),
                         f"<b style='color:{tone}'>{SHOW.get(r.confidence, r.confidence)}</b>"])
        col = "#1b7f3b" if "BUY" in label else "#b3261e" if "SELL" in label else "#555"
        parts.append(
            f"<h3 style='margin:14px 0 4px'>{rp.target_label(t)}: next {h} trading day{'s' if h != 1 else ''}</h3>"
            f"<p style='margin:2px 0'>Consensus: <b style='color:{col};font-size:15px'>{label}</b> "
            f"&nbsp;({nb} BUY, {ns} SELL, {nh} HOLD) &nbsp;|&nbsp; best track record among the agreeing models: "
            f"<b>{SHOW.get(conf, conf)}</b></p>"
            + rp._html_table(["Model", "Signal", "Expected move", "Move / noise (sd)", "Model-implied P(up)",
                              "Direction hit-rate in frozen test (effective n)", "Track record"], rows))
    if not parts:
        return ""
    note = ("<p style='font-size:11.5px;color:#666;margin:6px 0 0'>HOLD = the expected move is smaller than "
            f"{band:g} x its 1-sd noise, so there is nothing to act on. <b>Track record</b> = how the model's direction did in the "
            "frozen test (is it right significantly more often than a coin flip, and does it beat \"no change\"?): "
            "<i>Significant</i> = yes on both; <i>Suggestive</i> = some evidence (p &lt; 0.10, or a hit-rate above 55% with a "
            "better error); <i>None yet</i> = no evidence of an edge; <i>No call</i> = the model forecast exactly no change. "
            + ("" if any_evidence else "No track record is available yet. ")
            + "Rows are ordered by track record, strongest first.</p>")
    return ("<div style='border:1px solid #bbb;border-radius:6px;padding:8px 12px;margin:8px 0 14px;background:#fafafa'>"
            "<h2 style='margin:2px 0'>Signal summary</h2>" + "".join(parts) + note + "</div>")
