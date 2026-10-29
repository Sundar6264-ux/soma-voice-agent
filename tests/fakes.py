"""A scripted stand-in for Claude, so the call flow is tested without API spend."""
from __future__ import annotations

import asyncio
import copy
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from voiceagent.menu import load_restaurant
from voiceagent.store import Store

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "restaurants" / "demo.yaml"
# A Tuesday at 6 PM in New York: the demo restaurant is open.
OPEN_TIME = datetime(2026, 10, 27, 18, 0, tzinfo=ZoneInfo("America/New_York"))
# A Monday: closed all day.
CLOSED_TIME = datetime(2026, 10, 26, 18, 0, tzinfo=ZoneInfo("America/New_York"))


def demo_restaurant(**overrides):
    r = load_restaurant(DEMO)
    if overrides:
        from dataclasses import replace
        r = replace(r, **overrides)
    return r


def temp_store() -> Store:
    tmp = tempfile.mkdtemp(prefix="rva-test-")
    return Store(Path(tmp) / "test.db")


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, args, block_id):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=block_id)


class Step:
    def __init__(self, texts=(), tools=(), stop_reason=None, delay=0.0, raise_exc=None):
        self.texts = list(texts)
        self.tools = list(tools)
        self.stop_reason = stop_reason or ("tool_use" if tools else "end_turn")
        self.delay = delay
        self.raise_exc = raise_exc


def say(*texts, delay=0.0):
    return Step(texts=texts, delay=delay)


def call(*tools, text=None, delay=0.0):
    """tools: (name, args, id) tuples."""
    return Step(texts=[text] if text else [], tools=[tool_block(*t) for t in tools], delay=delay)


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.requests: list[list] = []

    @asynccontextmanager
    async def stream(self, system, tools, messages):
        self.requests.append(copy.deepcopy(messages))
        if not self.script:
            raise AssertionError("FakeLLM ran out of scripted steps")
        yield _FakeTurn(self.script.pop(0))


class _FakeTurn:
    def __init__(self, step: Step):
        self.step = step

    async def text(self):
        for t in self.step.texts:
            if self.step.delay:
                await asyncio.sleep(self.step.delay)
            yield t
        if self.step.raise_exc:
            raise self.step.raise_exc

    async def final(self):
        content = [text_block("".join(self.step.texts))] if self.step.texts else []
        content += self.step.tools
        return SimpleNamespace(content=content, stop_reason=self.step.stop_reason, stop_details=None)


def as_plain(messages):
    """Messages with SimpleNamespace blocks turned into dicts, for prefix comparisons."""
    def conv(v):
        if isinstance(v, SimpleNamespace):
            return {k: conv(x) for k, x in vars(v).items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    return conv(messages)
