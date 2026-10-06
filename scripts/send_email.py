"""Email a table and chart of the last 10 trading days of WTI/Brent futures and spot.

Reads Data/raw/yahoo_daily.csv and Data/raw/fred_daily.csv (all series are downloaded by
update_data.py; only these four are shown in the email).

Credentials come from environment variables (never put them in the repo):
    GMAIL_USER          sender Gmail address
    GMAIL_APP_PASSWORD  a Gmail app password (needs 2-step verification)
Optional: EMAIL_TO (default e0725196@u.nus.edu), EMAIL_CC (default yja210199@gmail.com)

Usage:  python scripts/send_email.py             send the email
        python scripts/send_email.py --dry-run   write email_preview/ (html, png, eml) and send nothing
"""
import argparse
import io
import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Data" / "raw"
N_DAYS = 10
TO = os.getenv("EMAIL_TO", "e0725196@u.nus.edu")
CC = os.getenv("EMAIL_CC", "yja210199@gmail.com")

# label -> (file, column)
SERIES = {"WTI futures": ("yahoo_daily.csv", "wti_fut"), "WTI spot": ("fred_daily.csv", "wti_spot"),
          "Brent futures": ("yahoo_daily.csv", "brent_fut"), "Brent spot": ("fred_daily.csv", "brent_spot")}
COLORS = {"WTI futures": "#e69f00", "WTI spot": "#1f4e79", "Brent futures": "#e69f00", "Brent spot": "#1f4e79"}


def load_last_days():
    files = {f: pd.read_csv(RAW / f, index_col=0, parse_dates=True) for f in {v[0] for v in SERIES.values()}}
    df = pd.concat([files[f][c].rename(label) for label, (f, c) in SERIES.items()], axis=1)
    trading_days = files["yahoo_daily.csv"]["wti_fut"].dropna().index        # last 10 futures trading days
    return df.reindex(trading_days).tail(N_DAYS)


def make_chart(df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, name in zip(ax, ["WTI", "Brent"]):
        for kind in ["futures", "spot"]:
            s = df[f"{name} {kind}"]
            a.plot(s.index, s, marker="o", ms=4, lw=1.8, color=COLORS[f"{name} {kind}"], label=kind)
        a.set_title(f"{name}: last {N_DAYS} trading days (USD/bbl)")
        a.legend(); a.grid(alpha=0.3); a.tick_params(axis="x", rotation=40)
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
    return buf.getvalue()


def make_table_html(df):
    shown = df.copy(); shown.index = shown.index.strftime("%Y-%m-%d")
    head = "".join(f"<th style='padding:4px 10px;border-bottom:2px solid #444;text-align:right'>{c}</th>" for c in shown.columns)
    rows = ""
    for date, r in shown.iterrows():
        cells = "".join(
            f"<td style='padding:4px 10px;border-bottom:1px solid #ddd;text-align:right'>"
            f"{'<span style=color:#999>n/a</span>' if pd.isna(v) else f'{v:,.2f}'}</td>" for v in r)
        rows += f"<tr><td style='padding:4px 10px;border-bottom:1px solid #ddd'>{date}</td>{cells}</tr>"
    return (f"<table style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px'>"
            f"<tr><th style='padding:4px 10px;border-bottom:2px solid #444;text-align:left'>Date</th>{head}</tr>{rows}</table>")


def build_message(df, png, status, sender, inline_src="cid:chart"):
    last = df.dropna(how="all").iloc[-1]
    subject = f"Oil data update {df.index[-1].date()}: WTI fut {df['WTI futures'].iloc[-1]:,.2f} | Brent fut {df['Brent futures'].iloc[-1]:,.2f}"
    notes = ["n/a = not yet published. FRED/EIA spot prices lag the Yahoo futures by several days."]
    if status.get("failed_series"):
        notes.append(f"WARNING: these series failed to download today (old values kept): {', '.join(status['failed_series'])}")
    note_html = "".join(f"<p style='font-size:12px;color:#555;font-family:Arial,sans-serif'>{n}</p>" for n in notes)
    html = (f"<html><body style='font-family:Arial,sans-serif'><h3>Oil data update, last {N_DAYS} trading days</h3>"
            f"{make_table_html(df)}<br><img src='{inline_src}' style='max-width:100%'><br>{note_html}"
            f"<p style='font-size:11px;color:#888'>All series were downloaded and merged into Data/raw/; only these four are shown.</p></body></html>")
    text = f"Oil data update, last {N_DAYS} trading days\n\n{df.round(2).to_string()}\n\n" + "\n".join(notes)
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"], msg["Cc"] = subject, sender, TO, CC
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    return msg, html


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); args = ap.parse_args()
    df = load_last_days(); png = make_chart(df)
    sp = ROOT / "update_status.json"; status = json.loads(sp.read_text()) if sp.exists() else {}
    sender = os.getenv("GMAIL_USER", "sender@example.com")
    msg, html = build_message(df, png, status, sender)
    msg.get_payload()[1].add_related(png, "image", "png", cid="<chart>")
    if args.dry_run:
        out = ROOT / "email_preview"; out.mkdir(exist_ok=True)
        (out / "chart.png").write_bytes(png)
        (out / "preview.html").write_text(build_message(df, png, status, sender, inline_src="chart.png")[1])
        (out / "message.eml").write_bytes(bytes(msg))
        print(f"Dry run: wrote {out}/ (preview.html, chart.png, message.eml). Nothing sent.\n\n{df.round(2).to_string()}")
        return 0
    user, pwd = os.getenv("GMAIL_USER"), os.getenv("GMAIL_APP_PASSWORD")
    if not user or not pwd:
        print("ERROR: GMAIL_USER and GMAIL_APP_PASSWORD must be set.", file=sys.stderr); return 1
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
        s.login(user, pwd); s.send_message(msg, to_addrs=[TO, CC])
    print(f"Sent to {TO} (cc {CC}): {msg['Subject']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
