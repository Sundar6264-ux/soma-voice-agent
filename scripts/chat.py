"""Talk to the agent in the terminal: the real model, tools and database, no phone.

    ./venv/bin/python scripts/chat.py            # type as the caller, Ctrl-D to hang up
    ./venv/bin/python scripts/chat.py --script   # run a fixed order end to end (the smoke test)
    ./venv/bin/python scripts/chat.py --script --model claude-opus-5-5   # same order on another model

Prints the tokens used and an estimated cost for the call at the end.

Every turn is a real Claude API call billed to ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voiceagent.agent import AnthropicLLM, Conversation  # noqa: E402
from voiceagent.config import Settings  # noqa: E402
from voiceagent.menu import load_restaurant  # noqa: E402
from voiceagent.store import Store  # noqa: E402
from voiceagent.tools import CallInfo  # noqa: E402

# USD per million tokens (input, output), Anthropic's list prices. Cache reads are billed at 0.1x input
# and cache writes at 1.25x input; that split is an estimate for models whose cache prices are not listed here.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}


class CountingLLM(AnthropicLLM):
    """AnthropicLLM that adds up token usage across every model request in the call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "requests": 0}

    @asynccontextmanager
    async def stream(self, system, tools, messages):
        async with super().stream(system, tools, messages) as turn:
            yield _CountingTurn(turn, self.usage)

    def cost(self) -> float | None:
        price = PRICES.get(self.model)
        if not price:
            return None
        u = self.usage
        return (u["input"] * price[0] + u["cache_read"] * price[0] * 0.1 + u["cache_write"] * price[0] * 1.25
                + u["output"] * price[1]) / 1e6


class _CountingTurn:
    def __init__(self, turn, usage):
        self._turn, self._usage = turn, usage

    def text(self):
        return self._turn.text()

    async def final(self):
        msg = await self._turn.final()
        u = msg.usage
        self._usage["requests"] += 1
        self._usage["input"] += u.input_tokens or 0
        self._usage["output"] += u.output_tokens or 0
        self._usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        self._usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        return msg


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
    ap.add_argument("--model", help="override MODEL from .env for this run")
    ap.add_argument("--effort", help="override EFFORT from .env for this run")
    args = ap.parse_args()

    settings = Settings.from_env()
    restaurant = load_restaurant(settings.restaurant_file)
    store = Store(settings.db_path)
    call = CallInfo(call_sid=f"cli-{int(time.time())}", from_number=args.caller_id, channel="phone")
    model, effort = args.model or settings.model, args.effort or settings.effort
    llm = CountingLLM(model, effort)
    conv = Conversation(restaurant, store, llm, call)
    store.start_call(call.call_sid, restaurant.id, "cli", call.from_number)
    print(f"[{model}, effort {effort}] {restaurant.name}\nagent> {restaurant.greeting}")

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

    u = llm.usage
    cost = llm.cost()
    print(f"\noutcome: {conv.outcome()}")
    print(f"model requests: {u['requests']}, tokens in {u['input']} (+{u['cache_read']} cache read, "
          f"+{u['cache_write']} cache write), out {u['output']}"
          + (f", estimated cost ${cost:.4f}" if cost is not None else ""))
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
