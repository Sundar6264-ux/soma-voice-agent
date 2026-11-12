"""AssemblyAI credit tracker: how much of the free credit the voice agent has used, with email alerts.

    ./venv/bin/python scripts/stt_budget.py                  # print usage
    /home/sundar/tailorbird-web/venv/bin/python scripts/stt_budget.py --alert   # also email at 50/80/95%

AssemblyAI has no API for reading the balance (its dashboard has low-balance alerts, which
should also be switched on), so this adds up what the app itself opened: every streaming session
is billed for as long as it was open, at STT_RATE_PER_HOUR. It cannot see AssemblyAI usage from
anything outside this app, and AssemblyAI's own dashboard stays the source of truth.

--alert sends each threshold once (state in logs/stt_alerts.json) to Soma's Outlook through
Tailorbird's SMTP settings, so it runs with Tailorbird's Python.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voiceagent.config import load_dotenv  # noqa: E402

THRESHOLDS = (50, 80, 95)
ALERT_TO = "owner@example.com"
STATE = ROOT / "logs" / "stt_alerts.json"


def usage() -> dict:
    load_dotenv(ROOT / ".env")
    credit = float(os.environ.get("STT_CREDIT_USD", "50"))
    rate = float(os.environ.get("STT_RATE_PER_HOUR", "0.15"))
    db = Path(os.environ.get("DB_PATH", str(ROOT / "voiceagent.db")))
    if not db.is_absolute():
        db = ROOT / db
    seconds = sessions = 0
    if db.exists():
        with sqlite3.connect(db) as conn:
            try:
                seconds, sessions = conn.execute("SELECT COALESCE(SUM(seconds), 0), COUNT(*) FROM stt_usage").fetchone()
            except sqlite3.OperationalError:
                pass  # table appears with the first recorded session
    spent = seconds / 3600 * rate
    return {"credit": credit, "rate": rate, "seconds": seconds, "sessions": sessions, "spent": spent,
            "left": max(0.0, credit - spent), "percent": 100 * spent / credit if credit else 100.0,
            "minutes_left": max(0.0, credit - spent) / rate * 60 if rate else 0.0}


def send_alert(level: int, u: dict, test: bool = False) -> None:
    sys.path.insert(0, "/home/sundar/tailorbird-web")
    import app  # noqa: F401  (Tailorbird's path bridge for its SMTP config)
    from app.mailer import send
    body = (f"The restaurant voice agent has used about {u['percent']:.0f}% of the AssemblyAI free credit.\n\n"
            f"Used: ${u['spent']:.2f} of ${u['credit']:.2f} ({u['seconds'] / 3600:.1f} hours over {u['sessions']} sessions)\n"
            f"Left: about ${u['left']:.2f}, roughly {u['minutes_left']:.0f} minutes of calls\n\n"
            "When the credit runs out AssemblyAI pauses the API, and callers would hear the agent but it could not "
            "hear them. Add a card or auto-pay at assemblyai.com (Billing) before then.\n\n"
            "This is the voice agent's own count; AssemblyAI's dashboard is the source of truth.")
    if test:
        body = "TEST of the AssemblyAI credit alert. Real alerts look like this, at 50, 80 and 95%.\n\n" + body
    send(ALERT_TO, f"{'[TEST] ' if test else ''}AssemblyAI credit {level}% used (voice agent)", body, brand="flockara")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alert", action="store_true", help="email once when 50, 80 and 95 percent are crossed")
    ap.add_argument("--test-email", action="store_true", help="send one sample alert with the real current numbers")
    args = ap.parse_args()
    u = usage()
    print(f"AssemblyAI credit: ${u['spent']:.2f} of ${u['credit']:.2f} used ({u['percent']:.1f}%), "
          f"${u['left']:.2f} left (about {u['minutes_left']:.0f} call minutes), {u['sessions']} sessions")
    if args.test_email:
        send_alert(50, u, test=True)
        print(f"test alert sent to {ALERT_TO}")
        return 0
    if args.test_email:
        send_alert(50, u, test=True)
        print(f"test alert sent to {ALERT_TO}")
        return 0
    if not args.alert:
        return 0
    STATE.parent.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"alerted": []}
    due = [t for t in THRESHOLDS if u["percent"] >= t and t not in state["alerted"]]
    if due:
        level = max(due)
        send_alert(level, u)
        state["alerted"] = sorted(set(state["alerted"]) | set(due))
        STATE.write_text(json.dumps(state))
        print(f"alert sent for {level}% to {ALERT_TO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
