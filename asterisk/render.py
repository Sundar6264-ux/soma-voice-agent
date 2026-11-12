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
PARTS = ROOT / "asterisk" / "parts"
TRUNK_KEYS = ("SIP_SERVER", "SIP_USERNAME", "SIP_PASSWORD", "SIP_MATCH", "EXTERNAL_IP")
UNSAFE = "\n\r;[]"


def _clean(key: str, value: str) -> str:
    if any(c in value for c in UNSAFE):
        raise SystemExit(f"{key} contains a character that would break the Asterisk config")
    return value


def _fill(text: str, vals: dict[str, str]) -> str:
    for key, val in vals.items():
        text = text.replace(f"@@{key}@@", val)
    return text


def values() -> dict[str, str]:
    """Placeholders for every template. The carrier line (trunk) and the Tailscale phone app
    (softphone) are each optional, but at least one must be configured."""
    settings = Settings.from_env()  # also loads .env into os.environ
    env = lambda k, d="": os.environ.get(k, d) or d
    if not env("ASTERISK_TOKEN"):
        raise SystemExit("missing in .env: ASTERISK_TOKEN")
    trunk_set = [k for k in TRUNK_KEYS if env(k)]
    trunk_on = bool(trunk_set)
    if trunk_on and len(trunk_set) != len(TRUNK_KEYS):
        missing = [k for k in TRUNK_KEYS if not env(k)]
        raise SystemExit(f"carrier line half configured; missing in .env: {', '.join(missing)}")
    softphone_on = bool(env("SOFTPHONE_PASSWORD"))
    if not trunk_on and not softphone_on:
        raise SystemExit("nothing to answer calls from: set the carrier (SIP_*) or SOFTPHONE_PASSWORD in .env")
    if softphone_on and len(env("SOFTPHONE_PASSWORD")) < 12:
        raise SystemExit("SOFTPHONE_PASSWORD must be at least 12 characters")
    restaurant = load_restaurant(settings.restaurant_file)
    staff = env("STAFF_DIAL") or restaurant.staff_phone.lstrip("+")
    if staff and not staff.isdigit():
        raise SystemExit("STAFF_DIAL must be digits only")
    if not trunk_on:
        staff = ""  # transfers need the carrier line
    vals = {
        "SIP_PORT": env("SIP_PORT", "5060"),
        "LOCAL_NET": env("LOCAL_NET", "192.168.0.0/16"),
        "RTP_START": env("RTP_START", "10000"),
        "RTP_END": env("RTP_END", "10099"),
        "APP_PORT": env("PORT", "8790"),
        "AUDIOSOCKET_PORT": str(settings.audiosocket_port),
        "ASTERISK_TOKEN": _clean("ASTERISK_TOKEN", env("ASTERISK_TOKEN")),
        "STAFF_DIAL": staff,
        "EXTERNAL_LINES": "",
        "TRUNK": "; carrier line not configured (SIP_* in .env)",
        "SOFTPHONE": "; Tailscale phone app not configured (SOFTPHONE_PASSWORD in .env)",
    }
    ipaddress.ip_network(vals["LOCAL_NET"], strict=False)
    if trunk_on:
        for cidr in env("SIP_MATCH").split(","):
            ipaddress.ip_network(cidr.strip(), strict=False)  # raises on a typo
        ipaddress.ip_address(env("EXTERNAL_IP"))
        trunk_vals = {k: _clean(k, env(k)) for k in ("SIP_SERVER", "SIP_USERNAME", "SIP_PASSWORD")}
        trunk_vals["MATCH_LINES"] = "\n".join(f"match = {c.strip()}" for c in env("SIP_MATCH").split(","))
        vals["TRUNK"] = _fill((PARTS / "pjsip_trunk.conf").read_text(), trunk_vals)
        vals["EXTERNAL_LINES"] = (f"external_media_address = {env('EXTERNAL_IP')}\n"
                                  f"external_signaling_address = {env('EXTERNAL_IP')}")
    if softphone_on:
        user = _clean("SOFTPHONE_USER", env("SOFTPHONE_USER", "phone"))
        if not user.isalnum():
            raise SystemExit("SOFTPHONE_USER must be letters and digits only")
        vals["SOFTPHONE"] = _fill((PARTS / "pjsip_softphone.conf").read_text(),
                                  {"SOFTPHONE_USER": user, "SOFTPHONE_PASSWORD": _clean("SOFTPHONE_PASSWORD", env("SOFTPHONE_PASSWORD"))})
    print(f"carrier line: {'on' if trunk_on else 'off'}; Tailscale phone app: {'on' if softphone_on else 'off'}")
    return vals


def render(vals: dict[str, str]) -> dict[str, str]:
    out = {}
    for tpl in sorted(TEMPLATES.glob("*.conf")):
        text = _fill(tpl.read_text(), vals)
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
