"""Twilio ConversationRelay session: the WebSocket protocol around a Conversation.

Twilio does speech-to-text and text-to-speech. We receive `setup`, `prompt`,
`interrupt`, `dtmf` and `error` messages and send `text` tokens (spoken as
they arrive) and a final `end`. The browser test page speaks the same
protocol, so one code path serves both.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

from .agent import Conversation
from .tools import CallInfo, Control

log = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]


def speech_seconds(text: str) -> float:
    """Rough time for TTS to finish speaking `text` (about 160 words a minute)."""
    return min(12.0, max(1.5, len(text.split()) / 2.7 + 1.0))


class RelaySession:
    def __init__(self, send: Send, make_conversation: Callable[[CallInfo], Conversation], store,
                 restaurant_id: str, channel: str = "phone", expected_call_sid: str | None = None,
                 sleep: Callable[[float], Awaitable] = asyncio.sleep):
        self.send = send
        self.make_conversation = make_conversation
        self.store = store
        self.restaurant_id = restaurant_id
        self.channel = channel
        self.expected_call_sid = expected_call_sid
        self.sleep = sleep
        self.conv: Conversation | None = None
        self.reply_task: asyncio.Task | None = None
        self.finish_task: asyncio.Task | None = None
        self.ending = False
        self.closed = False
        self._sent: list[str] = []

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "setup":
            await self._setup(msg)
        elif self.conv is None:
            log.warning("%s before setup, ignored", kind)
        elif kind == "prompt":
            if msg.get("last", True):
                await self._prompt(msg.get("voicePrompt", ""))
        elif kind == "interrupt":
            await self._interrupt(msg.get("utteranceUntilInterrupt"))
        elif kind == "dtmf":
            await self._dtmf(str(msg.get("digit", "")))
        elif kind == "error":
            log.error("relay error from Twilio: %s", msg.get("description"))

    async def _setup(self, msg: dict) -> None:
        call_sid = msg.get("callSid", "")
        if self.conv is not None:
            return
        if self.expected_call_sid and call_sid != self.expected_call_sid:
            log.error("setup callSid %s does not match the signed URL (%s); closing", call_sid, self.expected_call_sid)
            self.ending = True
            await self.send({"type": "end", "handoffData": json.dumps({"reasonCode": "error"})})
            return
        call = CallInfo(call_sid=call_sid, from_number=msg.get("from", "") or "", channel=self.channel)
        self.conv = self.make_conversation(call)
        self.store.start_call(call_sid, self.restaurant_id, self.channel, call.from_number)
        log.info("call %s started (%s)", call_sid, self.channel)

    async def _prompt(self, text: str) -> None:
        if self.ending:
            return
        if self.reply_task and not self.reply_task.done():
            # The caller spoke again before we finished replying, without an
            # interrupt message (for example while the model was still thinking).
            await self._cancel_reply()
            self.conv.interrupted(" ".join(self._sent))
        self.conv.caller_said(text)
        self.reply_task = asyncio.create_task(self._reply())

    async def _interrupt(self, heard: str | None) -> None:
        if self.ending:
            return
        was_running = self.reply_task is not None and not self.reply_task.done()
        await self._cancel_reply()
        if was_running or heard:
            self.conv.interrupted(heard if heard is not None else " ".join(self._sent))

    async def _dtmf(self, digit: str) -> None:
        if digit == "0" and not self.ending and self.conv.restaurant.staff_phone:
            await self._cancel_reply()
            self.conv.toolbox.control = Control("transfer", "caller pressed 0")
            await self._say("Connecting you with someone at the restaurant now.")
            self._schedule_finish("Connecting you with someone at the restaurant now.")

    async def _cancel_reply(self) -> None:
        task = self.reply_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _say(self, text: str) -> None:
        await self.send({"type": "text", "token": text, "last": True})

    async def _reply(self) -> None:
        self._sent = []
        try:
            async for chunk in self.conv.respond():
                self._sent.append(chunk)
                await self.send({"type": "text", "token": chunk, "last": False})
            await self.send({"type": "text", "token": "", "last": True})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reply failed on %s", self.conv.call.call_sid)
            await self._say("Sorry, something went wrong on my end. Please call back in a few minutes.")
            self.conv.toolbox.control = self.conv.toolbox.control or Control("end", "internal error")
        if self.conv.toolbox.control:
            self._schedule_finish("".join(self._sent))

    def _schedule_finish(self, last_text: str) -> None:
        # Separate task: a caller talking over the goodbye must not cancel the hang-up.
        self.ending = True
        self.finish_task = asyncio.create_task(self._finish(last_text))

    async def _finish(self, last_text: str) -> None:
        await self.sleep(speech_seconds(last_text))
        control = self.conv.toolbox.control
        handoff = {"reasonCode": control.action, "reason": control.reason,
                   "orderId": self.conv.toolbox.placed_order_id}
        await self.send({"type": "end", "handoffData": json.dumps(handoff)})

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self._cancel_reply()
        if self.finish_task and not self.finish_task.done():
            self.finish_task.cancel()
        if self.conv is not None:
            self.store.end_call(self.conv.call.call_sid, self.conv.outcome(),
                                self.conv.toolbox.placed_order_id, self.conv.transcript)
            log.info("call %s ended: %s", self.conv.call.call_sid, self.conv.outcome())
