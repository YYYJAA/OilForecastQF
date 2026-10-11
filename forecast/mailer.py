"""Shared e-mail settings and sending (Gmail SMTP with an app password).

Recipients (EMAIL_TO overrides everything and drops the default cc; comma-separated):
    To: the five NUS addresses in DEFAULT_TO below      Cc: yja210199@gmail.com
If EMAIL_SUBJECT_PREFIX is set it is put in front of every subject (the workflows set it for manual runs).
Credentials come from GMAIL_USER and GMAIL_APP_PASSWORD (GitHub secrets); never put them in the repo.
"""
import os
import smtplib
from email.message import EmailMessage

DEFAULT_TO = ",".join([
    "e0725196@u.nus.edu",
    "weizhi.da@u.nus.edu",
    "e0775008@u.nus.edu",
    "e1597843@u.nus.edu",
    "kristin.wei@u.nus.edu",
])
DEFAULT_CC = "yja210199@gmail.com"


def _split(s):
    return [a.strip() for a in s.replace(";", ",").split(",") if a.strip()]


def recipients():
    """Default team list, or an override. If EMAIL_TO is set, only those addresses get the mail and the
    default cc is dropped (so testing never spams the team); EMAIL_CC can still add a cc explicitly."""
    override = os.getenv("EMAIL_TO", "").strip()
    if override:
        return _split(override), _split(os.getenv("EMAIL_CC", ""))
    return _split(DEFAULT_TO), _split(os.getenv("EMAIL_CC") or DEFAULT_CC)


def build_message(subject, plain, html, inline=None):
    """HTML message with optional inline PNGs: inline = {cid: png_bytes}, referenced in the html as cid:<cid>."""
    to, cc = recipients()
    msg = EmailMessage()
    msg["Subject"] = os.getenv("EMAIL_SUBJECT_PREFIX", "") + subject     # "(manual run) " when started by hand
    msg["From"] = os.getenv("GMAIL_USER", "sender@example.com")
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    for cid, png in (inline or {}).items():
        msg.get_payload()[1].add_related(png, "image", "png", cid=f"<{cid}>")
    return msg


def send(msg):
    user, pwd = os.getenv("GMAIL_USER"), os.getenv("GMAIL_APP_PASSWORD")
    if not user or not pwd:
        raise RuntimeError("GMAIL_USER and GMAIL_APP_PASSWORD must be set.")
    to, cc = recipients()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
        s.login(user, pwd)
        s.send_message(msg, to_addrs=to + cc)
    return to, cc
