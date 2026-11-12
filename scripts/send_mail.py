"""Send one email through Tailorbird's mailer. Run with Tailorbird's Python, JSON on stdin:

    echo '{"to": "...", "subject": "...", "body": "..."}' | /home/sundar/tailorbird-web/venv/bin/python scripts/send_mail.py

voiceagent/profile.py starts this as a subprocess for each message a caller leaves, so the voice app
never reads Tailorbird's .env or SMTP settings itself (the same arrangement as scripts/stt_budget.py).
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    req = json.load(sys.stdin)
    sys.path.insert(0, "/home/sundar/tailorbird-web")
    import app  # noqa: F401  (Tailorbird's path bridge for its SMTP config)
    from app.mailer import send
    send(req["to"], req["subject"], req["body"], reply_to=req.get("reply_to") or None, brand=req.get("brand", "flockara"))
    print("sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
