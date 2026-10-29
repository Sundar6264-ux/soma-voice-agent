"""Render asterisk/templates/*.conf into asterisk/etc/ from .env.

    ./venv/bin/python asterisk/render.py          # writes asterisk/etc/, refuses if anything required is missing
    ./venv/bin/python asterisk/render.py --check  # same checks, writes nothing

asterisk/etc/ holds the SIP password, so it is git-ignored and the directory is 0700.
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voiceagent.config import Settings  # noqa: E402
from voiceagent.menu import load_restaurant  # noqa: E402

TEMPLATES = ROOT / "asterisk" / "templates"
OUT = ROOT / "asterisk" / "etc"
REQUIRED = ("SIP_SERVER", "SIP_USERNAME", "SIP_PASSWORD", "SIP_MATCH", "EXTERNAL_IP", "ASTERISK_TOKEN")


def values() -> dict[str, str]:
    settings = Settings.from_env()  # also loads .env into os.environ
    env = os.environ.get
    missing = [k for k in REQUIRED if not env(k)]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)} (see .env.example, self-hosted calls section)")
    for cidr in env("SIP_MATCH").split(","):
        ipaddress.ip_network(cidr.strip(), strict=False)  # raises on a typo
    ipaddress.ip_address(env("EXTERNAL_IP"))
    for k in ("SIP_SERVER", "SIP_USERNAME", "SIP_PASSWORD", "ASTERISK_TOKEN"):
        if any(c in env(k) for c in "\n\r;[]"):
            raise SystemExit(f"{k} contains a character that would break the Asterisk config")
    restaurant = load_restaurant(settings.restaurant_file)
    staff = env("STAFF_DIAL") or restaurant.staff_phone.lstrip("+")
    if staff and not staff.isdigit():
        raise SystemExit("STAFF_DIAL must be digits only")
    return {
        "SIP_SERVER": env("SIP_SERVER"),
        "SIP_USERNAME": env("SIP_USERNAME"),
        "SIP_PASSWORD": env("SIP_PASSWORD"),
        "MATCH_LINES": "\n".join(f"match = {c.strip()}" for c in env("SIP_MATCH").split(",")),
        "EXTERNAL_IP": env("EXTERNAL_IP"),
        "LOCAL_NET": env("LOCAL_NET", "192.168.0.0/16"),
        "SIP_PORT": env("SIP_PORT", "5060"),
        "RTP_START": env("RTP_START", "10000"),
        "RTP_END": env("RTP_END", "10099"),
        "APP_PORT": env("PORT", "8790"),
        "AUDIOSOCKET_PORT": str(settings.audiosocket_port),
        "ASTERISK_TOKEN": env("ASTERISK_TOKEN"),
        "STAFF_DIAL": staff,
    }


def render(vals: dict[str, str]) -> dict[str, str]:
    out = {}
    for tpl in sorted(TEMPLATES.glob("*.conf")):
        text = tpl.read_text()
        for key, val in vals.items():
            text = text.replace(f"@@{key}@@", val)
        if "@@" in text:
            raise SystemExit(f"{tpl.name}: unfilled placeholder left")
        out[tpl.name] = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = render(values())
    if args.check:
        print(f"ok: {', '.join(files)}")
        return
    OUT.mkdir(mode=0o700, exist_ok=True)
    OUT.chmod(0o700)
    for name, text in files.items():
        path = OUT / name
        path.write_text(text)
        path.chmod(0o644)  # read by the container's asterisk user; the 0700 directory keeps other host users out
    print(f"wrote {len(files)} files to {OUT}")


if __name__ == "__main__":
    main()
