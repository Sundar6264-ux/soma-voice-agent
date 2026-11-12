"""One phone call's conversation with Claude: streaming text out, tools in between.

History rules (Claude Opus 5.5 checks that earlier messages are unchanged
whenever it reads back its own thinking blocks, and the prompt cache wants the
same thing):
- The system prompt and tool list never change during a call.
- An assistant turn is appended exactly as the API returned it, or not at all.
- A tool round (assistant tool_use + user tool_result) is appended in one step
  with no await in between, so cancelling a reply can never split the pair.
- When the caller interrupts, the unfinished reply is dropped and what they
  actually heard is passed along in their next turn, never patched into the
  old assistant message.
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Callable

import anthropic

from .menu import Restaurant
from .persona import Persona
from .prompts import build_system_prompt, call_context
from .store import Store
from .tools import TOOLS, CallInfo, Control, ToolBox

log = logging.getLogger(__name__)

MAX_ROUNDS = 8
MAX_JSON_RETRIES = 2
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept server-side `fallbacks: "default"` (Haiku has none).
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")

SORRY_TROUBLE = "Sorry, I'm having trouble on my end right now."
SORRY_REPEAT = "Sorry, I didn't catch that. Could you say it again?"
TRANSFER_LINE = "Okay, connecting you with someone at the restaurant now."

_UNSPEAKABLE = re.compile(r"[*#_`]+")


def _attr(block, name, default=None):
    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)


class AnthropicLLM:
    """Streams one model turn. `stream()` yields an object with `text()` and `final()`."""

    def __init__(self, model: str, effort: str, client: anthropic.AsyncAnthropic | None = None,
                 keepalive_seconds: float = 120.0):
        self.model = model
        self.effort = effort
        if client is None:
            import httpx2
            # Keep the TLS connection between caller turns (the SDK's default drops it after 5 s idle).
            http = anthropic.DefaultAsyncHttpxClient(limits=httpx2.Limits(
                max_connections=1000, max_keepalive_connections=100, keepalive_expiry=keepalive_seconds))
            client = anthropic.AsyncAnthropic(http_client=http)
        self.client = client

    async def warm(self) -> None:
        """Open the connection to the API while the greeting plays (a free GET, no tokens)."""
        try:
            await self.client.models.retrieve(self.model)
        except Exception as exc:  # the first real request simply connects as before
            log.debug("Anthropic warm-up failed: %r", exc)

    @asynccontextmanager
    async def stream(self, system: str, tools: list[dict], messages: list[dict]):
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.model in _FALLBACK_MODELS:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        async with self.client.beta.messages.stream(**kwargs) as s:
            yield _AnthropicTurn(s)


class _AnthropicTurn:
    def __init__(self, stream):
        self._stream = stream
        self.first_event_at: float | None = None  # first stream event of any kind (thinking included)

    async def text(self) -> AsyncIterator[str]:
        async for event in self._stream:
            if self.first_event_at is None:
                self.first_event_at = time.monotonic()
            if event.type == "text":
                yield event.text

    async def final(self):
        return await self._stream.get_final_message()


def restaurant_persona(restaurant: Restaurant, store: Store, clock: Callable[[], datetime] | None = None,
                       system_prompt: str | None = None) -> Persona:
    """The default persona: the phone ordering agent, exactly as before personas existed."""
    kwargs = {"clock": clock} if clock else {}
    return Persona(
        id=restaurant.id,
        system_prompt=system_prompt or build_system_prompt(restaurant),
        tools=TOOLS,
        make_toolbox=lambda call: ToolBox(restaurant, store, call, **kwargs),
        greeting=restaurant.greeting,
        call_context=call_context,
        trouble_line=SORRY_TROUBLE + (" Let me connect you with someone at the restaurant." if restaurant.staff_phone
                                      else " Please call back in a few minutes."),
        refusal_line="Sorry, I can't help with that. Is there anything else I can get for your order?",
        transfer_line=TRANSFER_LINE,
        staff_phone=restaurant.staff_phone,
        dtmf_transfer_line="Connecting you with someone at the restaurant now.",
        restaurant=restaurant,
    )


class Conversation:
    def __init__(self, restaurant: Restaurant | None, store: Store, llm, call: CallInfo,
                 clock: Callable[[], datetime] | None = None, system_prompt: str | None = None,
                 persona: Persona | None = None):
        if persona is None:
            persona = restaurant_persona(restaurant, store, clock, system_prompt)
        self.persona = persona
        self.restaurant = persona.restaurant
        self.llm = llm
        self.call = call
        self.system = persona.system_prompt
        self.tools = persona.tools
        self.toolbox = persona.make_toolbox(call)
        self.messages: list[dict] = []
        self.transcript: list[dict] = []
        if hasattr(self.toolbox, "transcript"):   # a toolbox that emails or shows the conversation so far
            self.toolbox.transcript = self.transcript
        self._notes: list[str] = []
        self._context_sent = False
        # Optional latency hook: called with "llm_sent", "llm_first_event", "llm_first_token" (see media.py).
        self.on_timing: Callable[[str], None] | None = None
        # Optional event hook: tool-raised events for the caller's client (links, for example). See media.py.
        self.on_event: Callable[[dict], None] | None = None

    def _mark(self, name: str) -> None:
        if self.on_timing:
            self.on_timing(name)

    # -- caller side -------------------------------------------------------

    def add_note(self, text: str) -> None:
        """A bracketed note the model sees with the caller's next turn (append-only: never edits history)."""
        self._notes.append(text)

    def caller_said(self, text: str) -> None:
        text = text.strip() or "(silence or unintelligible audio)"
        blocks = [{"type": "text", "text": n} for n in self._notes] + [{"type": "text", "text": text}]
        self._notes = []
        self.transcript.append({"role": "caller", "text": text})
        last = self.messages[-1] if self.messages else None
        if last and last["role"] == "system":
            # The first reply never landed (interrupted or failed), so nothing the
            # model produced follows the context message yet. Fold this turn into
            # the first user turn and keep the context message last, since a
            # mid-conversation system message must be last or precede an assistant turn.
            self.messages.pop()
            self.messages[-1]["content"].extend(blocks)
            self.messages.append(last)
            return
        self.messages.append({"role": "user", "content": blocks})
        if not self._context_sent:
            self.messages.append({"role": "system", "content": self.persona.call_context(self.call.from_number, self.call.channel)})
            self._context_sent = True

    def interrupted(self, heard: str) -> None:
        heard = " ".join(heard.split())
        if heard:
            self._notes.append(f'[The caller interrupted your last reply. They only heard: "{heard}"]')
        else:
            self._notes.append("[The caller started talking before hearing your last reply.]")
        self.transcript.append({"role": "agent", "text": heard, "interrupted": True})

    # -- agent side --------------------------------------------------------

    async def respond(self) -> AsyncIterator[str]:
        """Yield speakable text for one caller turn, running tools as needed."""
        spoken_any = False
        json_retries = 0
        rounds = 0
        while rounds < MAX_ROUNDS:
            round_text: list[str] = []
            try:
                self._mark("llm_sent")
                async with self.llm.stream(self.system, self.tools, self.messages) as turn:
                    async for chunk in turn.text():
                        if getattr(turn, "first_event_at", None) and not round_text:
                            self._mark("llm_first_event")
                        chunk = _UNSPEAKABLE.sub("", chunk)
                        if not chunk:
                            continue
                        self._mark("llm_first_token")
                        if not round_text and spoken_any and not chunk[0].isspace():
                            chunk = " " + chunk
                        round_text.append(chunk)
                        spoken_any = True
                        yield chunk
                    msg = await turn.final()
            except ValueError:
                # Tool input JSON the SDK could not parse at all: there is no
                # tool_use id to answer, so re-issue the turn (bounded).
                json_retries += 1
                log.warning("unparseable tool input on %s (retry %d)", self.call.call_sid, json_retries)
                if json_retries > MAX_JSON_RETRIES:
                    yield (" " if spoken_any else "") + SORRY_REPEAT
                    return
                continue
            except anthropic.APIError as exc:
                log.error("Claude API error on %s: %r", self.call.call_sid, exc)
                yield (" " if spoken_any else "") + self._trouble_line()
                return
            rounds += 1
            json_retries = 0

            text = "".join(round_text).strip()
            stop = _attr(msg, "stop_reason")
            tool_uses = [b for b in _attr(msg, "content", []) if _attr(b, "type") == "tool_use"]
            if stop == "refusal":
                log.warning("refusal on %s: %r", self.call.call_sid, _attr(msg, "stop_details"))
                yield (" " if spoken_any else "") + self.persona.refusal_line
                return
            if stop == "max_tokens":
                log.error("max_tokens on %s", self.call.call_sid)
                yield (" " if spoken_any else "") + SORRY_REPEAT
                return
            if text:
                self.transcript.append({"role": "agent", "text": text})

            if not tool_uses:
                self.messages.append({"role": "assistant", "content": _attr(msg, "content")})
                return

            results = []
            for block in tool_uses:
                name, args = _attr(block, "name"), _attr(block, "input")
                result = self.toolbox.run(name, args)
                self.transcript.append({"role": "tool", "name": name, "input": args,
                                        "result": result.content[:500], "error": result.is_error})
                results.append({"type": "tool_result", "tool_use_id": _attr(block, "id"),
                                "content": result.content, "is_error": result.is_error})
            self.messages.append({"role": "assistant", "content": _attr(msg, "content")})
            self.messages.append({"role": "user", "content": results})
            for event in getattr(self.toolbox, "take_events", lambda: [])():
                if self.on_event:
                    self.on_event(event)

            control = self.toolbox.control
            if control:
                if control.action == "transfer" and self.persona.transfer_line:
                    yield (" " if spoken_any else "") + self.persona.transfer_line
                return
        log.error("tool loop hit %d rounds on %s", MAX_ROUNDS, self.call.call_sid)
        yield (" " if spoken_any else "") + self._trouble_line()

    def _trouble_line(self) -> str:
        if self.persona.staff_phone:
            self.toolbox.control = self.toolbox.control or Control("transfer", "agent error")
        return self.persona.trouble_line

    def outcome(self) -> str:
        return self.toolbox.outcome()
