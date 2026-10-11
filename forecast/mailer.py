"""Shared e-mail settings and sending (Gmail SMTP with an app password).

Recipients (override with comma-separated EMAIL_TO / EMAIL_CC environment variables):
    To: e0725196@u.nus.edu, weizhi.da@u.nus.edu      Cc: yja210199@gmail.com
Credentials come from GMAIL_USER and GMAIL_APP_PASSWORD (GitHub secrets); never put them in the repo.
"""
import os
import smtplib
from email.message import EmailMessage

DEFAULT_TO = "e0725196@u.nus.edu,weizhi.da@u.nus.edu"
DEFAULT_CC = "yja210199@gmail.com"


def _split(s):
    return [a.strip() for a in s.replace(";", ",").split(",") if a.strip()]


def recipients():
    to = _split(os.getenv("EMAIL_TO") or DEFAULT_TO)
    cc = _split(os.getenv("EMAIL_CC") or DEFAULT_CC)
    return to, cc


def build_message(subject, plain, html, inline_png=None):
    """HTML message with an optional inline image referenced in the html as cid:chart."""
    to, cc = recipients()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("GMAIL_USER", "sender@example.com")
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    if inline_png is not None:
        msg.get_payload()[1].add_related(inline_png, "image", "png", cid="<chart>")
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
