"""Email the retraining report (what changed, new training period, impact on forecasts).

Reads models/retrain_report.json, written by scripts/train_models.py.

Usage:  python scripts/send_retrain_email.py             send the email
        python scripts/send_retrain_email.py --dry-run   write email_preview/retrain.html, send nothing
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forecast import mailer, reporting as rp  # noqa: E402
from forecast.config import MODELS_DIR  # noqa: E402


def build(report):
    parts, plain = [], []
    ends = []
    for r in report["reports"]:
        ends.append(f"{r['plugin']} through {r['train_end']}")
        items = "".join(f"<li>{h}</li>" for h in r["headline"])
        tabs = ""
        for t in r["tables"]:
            tabs += f"<h4 style='margin-bottom:4px'>{t['title']}</h4>" + rp._html_table(t["header"], t["rows"])
        prev = (f"previous model trained {r['previous_trained_at'][:10]} (data through {r['previous_train_end']})"
                if r.get("previous_train_end") else "no previous model")
        parts.append(f"<h3>{r['plugin']}</h3><p style='color:#555;margin-top:0'>Trained {r['trained_at'][:16].replace('T', ' ')} UTC "
                     f"in {r['seconds']}s; {prev}.</p><ul>{items}</ul>{tabs}")
        plain += [f"== {r['plugin']} ==", *r["headline"], ""]
    warn = ""
    if report.get("failed"):
        warn = f"<p style='color:#b00'><b>Retraining failed for:</b> {', '.join(report['failed'])} (previous model kept).</p>"
    html = (f"<html><body style='font-family:Arial,sans-serif;max-width:950px'><h2>Models retrained</h2>{warn}"
            f"{''.join(parts)}<hr><p style='font-size:12px;color:#555'>The daily forecast uses these models from the "
            f"next run onward. Hyper-parameters (LASSO penalty, number of PCA factors, regime model) were re-selected "
            f"on past data only. Models are retrained quarterly, or on demand from the GitHub Actions tab.</p></body></html>")
    subject = "Models retrained: " + "; ".join(ends) if ends else "Model retraining failed"
    return subject, "\n".join(plain) or "See HTML version.", html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = MODELS_DIR / "retrain_report.json"
    if not path.exists():
        print("ERROR: models/retrain_report.json not found: run scripts/train_models.py first", file=sys.stderr)
        return 1
    subject, plain, html = build(json.loads(path.read_text()))
    msg = mailer.build_message(subject, plain, html)
    if args.dry_run:
        out = ROOT / "email_preview"
        out.mkdir(exist_ok=True)
        (out / "retrain.html").write_text(html)
        print(f"Dry run: wrote {out / 'retrain.html'}. Nothing sent.\nTo: {msg['To']}  Cc: {msg.get('Cc')}\nSubject: {msg['Subject']}")
        return 0
    try:
        to, cc = mailer.send(msg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Sent to {', '.join(to)} (cc {', '.join(cc)}): {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
