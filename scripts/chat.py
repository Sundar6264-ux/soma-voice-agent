"""Talk to the agent in the terminal: the real model, tools and database, no phone.

    ./venv/bin/python scripts/chat.py            # type as the caller, Ctrl-D to hang up
    ./venv/bin/python scripts/chat.py --script   # run a fixed order end to end (the smoke test)

Every turn is a real Claude API call billed to ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voiceagent.agent import AnthropicLLM, Conversation  # noqa: E402
from voiceagent.config import Settings  # noqa: E402
from voiceagent.menu import load_restaurant  # noqa: E402
from voiceagent.store import Store  # noqa: E402
from voiceagent.tools import CallInfo  # noqa: E402

SCRIPT = [
    "Hi, can I get a large margherita with mushrooms?",
    "And a can of diet cola.",
    "That's everything. The name is Jordan, and use this number.",
    "Yes, that's right.",
    "No, that's all. Thanks, bye!",
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", action="store_true", help="run the fixed smoke-test order")
    ap.add_argument("--caller-id", default="+12125550147")
    args = ap.parse_args()

    settings = Settings.from_env()
    restaurant = load_restaurant(settings.restaurant_file)
    store = Store(settings.db_path)
    call = CallInfo(call_sid=f"cli-{int(time.time())}", from_number=args.caller_id, channel="phone")
    conv = Conversation(restaurant, store, AnthropicLLM(settings.model, settings.effort), call)
    store.start_call(call.call_sid, restaurant.id, "cli", call.from_number)
    print(f"[{settings.model}, effort {settings.effort}] {restaurant.name}\nagent> {restaurant.greeting}")

    lines = iter(SCRIPT) if args.script else None
    try:
        while conv.toolbox.control is None:
            if lines is not None:
                text = next(lines, None)
                if text is None:
                    break
                print(f"caller> {text}")
            else:
                try:
                    text = input("caller> ")
                except EOFError:
                    break
            conv.caller_said(text)
            started = time.monotonic()
            first = None
            print("agent> ", end="", flush=True)
            async for chunk in conv.respond():
                first = first or time.monotonic() - started
                print(chunk, end="", flush=True)
            total = time.monotonic() - started
            print(f"   [first words {first or 0:.1f}s, turn {total:.1f}s]")
    finally:
        store.end_call(call.call_sid, conv.outcome(), conv.toolbox.placed_order_id, conv.transcript)

    print(f"\noutcome: {conv.outcome()}")
    for entry in conv.transcript:
        if entry["role"] == "tool":
            flag = " ERROR" if entry["error"] else ""
            print(f"  tool {entry['name']}{flag}: {json.dumps(entry['input'])} -> {entry['result'][:120]}")
    if args.script:
        ok = conv.outcome() == "order_placed"
        order = store.recent_orders(restaurant.id, 1)[0] if ok else None
        print(f"\nSMOKE {'PASS' if ok else 'FAIL'}" + (f": order #{order['id']} total ${order['total_cents'] / 100:.2f}" if ok else ""))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
