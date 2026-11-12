"""Self-hosted call path: a transport (Asterisk AudioSocket or a browser WebSocket)
<-> speech services <-> RelaySession.

The transport delivers the caller's audio and takes back one paced audio frame
at a time. We run speech-to-text on the input, feed finished turns to the same
RelaySession the Twilio path uses (so tools and interruption rules are shared),
turn the replies into speech, and play them back in real time at the
transport's output rate (8 kHz for the phone, 24 kHz for the browser).

Because we pace playback ourselves, we know exactly how much of each sentence
the caller heard when they talk over it, and stopping is immediate.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from . import audiosocket as aus
from .relay import RelaySession

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"([.!?])(\s+)")
BYTES_PER_SECOND = aus.SAMPLE_RATE * 2
BARGE_IN_MIN_WORDS = 2   # a lone "yeah" or a cough while the agent talks is not an interruption
STT_CHUNK_BYTES = BYTES_PER_SECOND // 10  # AssemblyAI wants 50 to 1000 ms per message; send 100 ms
# Measured on Inworld at 8 kHz: about 1070 bytes of audio per character of text, used to estimate how
# far into a sentence playback got before synthesis finished. Scales with the output sample rate.
BYTES_PER_CHAR_8K = 1070
TIME_UP_GOODBYE = "We're out of time for this conversation, so I'll stop here. Thanks for talking with me. Goodbye."


@dataclass
class _Utterance:
    text: str
    is_reply: bool
    audio: bytearray = field(default_factory=bytearray)
    played: int = 0
    done: bool = False
    failed: bool = False
    task: asyncio.Task | None = None
    first_audio_at: float | None = None   # first byte back from TTS
    first_played_at: float | None = None  # first frame written to the transport
    bytes_per_char: int = BYTES_PER_CHAR_8K

    def heard_text(self) -> str:
        if self.done and self.played >= len(self.audio):
            return self.text
        words = self.text.split()
        total = len(self.audio) if self.done else max(len(self.audio), int(len(self.text) * self.bytes_per_char))
        share = min(1.0, self.played / total) if total else 0.0
        return " ".join(words[: int(len(words) * share)])


class Speaker:
    """Real-time playback, one 20 ms frame of raw PCM at a time, silence when idle.

    `write` receives raw 16-bit PCM frames of `frame_bytes` bytes (the transport adds its own framing).
    """

    def __init__(self, write: Callable[[bytes], Awaitable[None]], synthesize, frame_seconds: float = aus.FRAME_MS / 1000,
                 sample_rate: int = aus.SAMPLE_RATE):
        self.write = write
        self.synthesize = synthesize
        self.frame_seconds = frame_seconds
        self.sample_rate = sample_rate
        self.frame_bytes = sample_rate * 2 * aus.FRAME_MS // 1000
        self.silence = b"\x00" * self.frame_bytes
        self.bytes_per_char = BYTES_PER_CHAR_8K * sample_rate // aus.SAMPLE_RATE
        self.queue: list[_Utterance] = []
        self.heard_reply: list[str] = []
        self._drained = asyncio.Event()
        self._drained.set()
        self.on_first_frame: Callable[[_Utterance], None] | None = None

    @property
    def busy(self) -> bool:
        return bool(self.queue)

    @property
    def playing_reply(self) -> bool:
        return any(u.is_reply for u in self.queue)

    def say(self, text: str, is_reply: bool = True) -> _Utterance | None:
        text = text.strip()
        if not text:
            return None
        utt = _Utterance(text, is_reply, bytes_per_char=self.bytes_per_char)
        utt.task = asyncio.create_task(self._fetch(utt))
        self.queue.append(utt)
        self._drained.clear()
        return utt

    async def _fetch(self, utt: _Utterance) -> None:
        try:
            async for chunk in self.synthesize(utt.text):
                if utt.first_audio_at is None and chunk:
                    utt.first_audio_at = time.monotonic()
                utt.audio.extend(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("speech synthesis failed for %r", utt.text[:60])
            utt.failed = True
        finally:
            utt.done = True

    def stop(self) -> str:
        """Cut playback now. Returns the words of the current reply the caller heard."""
        heard = list(self.heard_reply)
        for utt in self.queue:
            if utt.is_reply and utt.played:
                heard.append(utt.heard_text())
            if utt.task and not utt.task.done():
                utt.task.cancel()
        self.queue.clear()
        self.heard_reply = []
        self._drained.set()
        return " ".join(h for h in heard if h)

    async def wait_drained(self) -> None:
        await self._drained.wait()

    async def run(self) -> None:
        next_at = time.monotonic()
        while True:
            frame = self.silence
            while self.queue:
                utt = self.queue[0]
                available = len(utt.audio) - utt.played
                if available >= self.frame_bytes or (utt.done and available > 0):
                    chunk = bytes(utt.audio[utt.played: utt.played + self.frame_bytes])
                    utt.played += len(chunk)
                    frame = chunk.ljust(self.frame_bytes, b"\x00")
                    if utt.first_played_at is None:
                        utt.first_played_at = time.monotonic()
                        if self.on_first_frame:
                            self.on_first_frame(utt)
                    break
                if utt.done:  # finished, or synthesis failed: move on
                    self.queue.pop(0)
                    if utt.is_reply and not utt.failed:
                        self.heard_reply.append(utt.text)
                    if not self.queue:
                        self._drained.set()
                    continue
                break  # still waiting for audio from the TTS service
            await self.write(frame)
            next_at += self.frame_seconds
            delay = next_at - time.monotonic()
            if delay <= 0:
                # Running late: resync the clock, but still yield. A socket write
                # with buffer room does not yield, so without this the loop would
                # starve speech-to-text and the model reply under load.
                next_at = time.monotonic()
                delay = 0
            await asyncio.sleep(delay)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences out of a streaming buffer, plus the unfinished remainder.

    A sentence ends at . ! or ? followed by whitespace, so "$14.50" never splits.
    Each sentence goes to TTS as soon as it is complete, which is what keeps the
    first words fast while the model is still writing the rest.
    """
    out, start = [], 0
    for m in _SENTENCE_END.finditer(buffer):
        sentence = buffer[start:m.end(1)].strip()
        if sentence:
            out.append(sentence)
        start = m.end()
    return out, buffer[start:]


class Transport(Protocol):
    """What MediaCall needs from the wire: caller input in, paced audio frames and JSON events out."""
    in_rate: int    # sample rate of the caller's PCM16 audio
    out_rate: int   # sample rate of the PCM16 frames we send back

    async def read(self) -> tuple[str, bytes | str]:
        """Next input: ("audio", pcm), ("dtmf", digit), ("text", typed caller turn), ("hangup", "") or ("error", detail)."""
        ...

    async def send_audio(self, frame: bytes) -> None: ...
    async def send_event(self, event: dict) -> None: ...
    async def close(self) -> None: ...


class AudioSocketTransport:
    """Asterisk AudioSocket: 8 kHz both ways, no events (the phone has no screen)."""
    in_rate = aus.SAMPLE_RATE
    out_rate = aus.SAMPLE_RATE

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self._write_lock = asyncio.Lock()

    async def read(self) -> tuple[str, bytes | str]:
        try:
            kind, payload = await aus.read_message(self.reader)
        except (asyncio.IncompleteReadError, ConnectionError):
            return "hangup", ""
        if kind == aus.AUDIO_8K:
            return "audio", payload
        if kind == aus.DTMF and payload:
            return "dtmf", payload[:1].decode("ascii", "replace")
        if kind == aus.ERROR:
            return "error", repr(payload)
        if kind == aus.HANGUP:
            return "hangup", ""
        return "other", payload

    async def send_audio(self, frame: bytes) -> None:
        if self.writer.is_closing():
            return
        async with self._write_lock:
            self.writer.write(aus.audio_frame(frame))
            await self.writer.drain()

    async def send_event(self, event: dict) -> None:
        return None

    async def close(self) -> None:
        try:
            if not self.writer.is_closing():
                self.writer.write(aus.pack(aus.HANGUP))
                await self.writer.drain()
            self.writer.close()
        except ConnectionError:
            pass


class WebSocketTransport:
    """A browser over a Starlette WebSocket: binary frames are PCM16 at `in_rate` from the caller and
    at `out_rate` back to them; text frames are JSON events both ways."""
    in_rate = 16000
    out_rate = 24000

    def __init__(self, websocket):
        self.ws = websocket
        self._closed = False

    async def read(self) -> tuple[str, bytes | str]:
        try:
            msg = await self.ws.receive()
        except Exception:
            return "hangup", ""
        if msg.get("type") == "websocket.disconnect":
            return "hangup", ""
        if msg.get("bytes") is not None:
            return "audio", msg["bytes"]
        try:
            data = json.loads(msg.get("text") or "{}")
        except json.JSONDecodeError:
            return "other", ""
        kind = data.get("type")
        if kind == "hangup":
            return "hangup", ""
        if kind == "dtmf":
            return "dtmf", str(data.get("digit", ""))[:1]
        if kind == "text":
            return "text", str(data.get("text", ""))[:2000]
        return "other", ""

    async def send_audio(self, frame: bytes) -> None:
        if self._closed:
            return
        try:
            await self.ws.send_bytes(frame)
        except Exception:
            self._closed = True

    async def send_event(self, event: dict) -> None:
        if self._closed:
            return
        try:
            await self.ws.send_json(event)
        except Exception:
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.ws.close()
        except Exception:
            pass


class MediaCall:
    """One conversation over a Transport: a phone call from Asterisk or a browser session."""

    def __init__(self, transport: Transport, *, call_id: str, from_number: str, greeting: str, stt, synthesize,
                 make_relay: Callable[..., RelaySession], frame_seconds: float = aus.FRAME_MS / 1000,
                 time_limit: float | None = None, warn_at: float | None = None, goodbye: str = TIME_UP_GOODBYE):
        self.transport = transport
        self.call_id = call_id
        self.from_number = from_number
        self.greeting = greeting
        self.stt = stt
        self.speaker = Speaker(transport.send_audio, synthesize, frame_seconds, sample_rate=transport.out_rate)
        self.speaker.on_first_frame = self._on_first_frame
        self.relay = make_relay(send=self._from_relay, sleep=self._wait_spoken)
        self.time_limit = time_limit
        self.warn_at = warn_at
        self.goodbye = goodbye
        self.stt_chunk_bytes = transport.in_rate * 2 // 10
        self.next_action = "hangup"
        self.timings: list[dict] = []   # one entry per caller turn that got a spoken reply (see _on_first_frame)
        self._turn: dict | None = None  # the caller turn being answered right now
        self._sentence_buf = ""
        self._reply_open = False
        self._muted = False
        self._ended = asyncio.Event()
        self._event_tasks: set[asyncio.Task] = set()

    def _emit(self, event: dict) -> None:
        """Send a JSON event to the client without blocking the caller (no-op on the phone)."""
        task = asyncio.create_task(self.transport.send_event(event))
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    # -- RelaySession output ------------------------------------------------

    async def _from_relay(self, msg: dict) -> None:
        if msg.get("type") == "text":
            if self._muted:
                return
            if not self._reply_open:
                self.speaker.heard_reply = []  # a new reply starts
                self._reply_open = True
            token = msg.get("token", "")
            await self.transport.send_event({"type": "agent_text", "token": token, "last": bool(msg.get("last"))})
            self._sentence_buf += token
            sentences, self._sentence_buf = split_sentences(self._sentence_buf)
            if msg.get("last"):
                sentences.append(self._sentence_buf)
                self._sentence_buf = ""
                self._reply_open = False
            for s in sentences:
                utt = self.speaker.say(s)
                if utt is not None and self._turn is not None and "utt" not in self._turn:
                    self._turn["sentence"] = time.monotonic()
                    self._turn["utt"] = utt
        elif msg.get("type") == "end":
            try:
                handoff = json.loads(msg.get("handoffData") or "{}")
            except json.JSONDecodeError:
                handoff = {}
            self.next_action = "transfer" if handoff.get("reasonCode") == "transfer" else "hangup"
            await self.transport.send_event({"type": "end", "reason": handoff.get("reasonCode") or "end"})
            self._ended.set()

    async def _wait_spoken(self, _seconds: float) -> None:
        await self.speaker.wait_drained()

    # -- per-turn latency ---------------------------------------------------
    # Clock points, all on time.monotonic(): speech_end (caller's last word, from AssemblyAI word
    # timestamps) -> stt_turn (final turn received) -> llm_sent -> llm_first_token -> sentence
    # (first complete sentence handed to TTS) -> tts_first_byte -> played (first frame to Asterisk).

    def _mark(self, name: str) -> None:
        if self._turn is not None:
            self._turn.setdefault(name, time.monotonic())

    def _on_first_frame(self, utt: _Utterance) -> None:
        turn = self._turn
        if turn is None or turn.get("utt") is not utt:
            return
        turn["tts_first_byte"] = utt.first_audio_at
        turn["played"] = utt.first_played_at
        self._turn = None
        start = turn.get("speech_end") or turn["stt_turn"]
        stages = [("stt", "speech_end", "stt_turn"), ("queue", "stt_turn", "llm_sent"),
                  ("ttft", "llm_sent", "llm_first_token"), ("sentence", "llm_first_token", "sentence"),
                  ("tts", "sentence", "tts_first_byte"), ("play", "tts_first_byte", "played")]
        ms = {}
        for label, a, b in stages:
            if turn.get(a) is not None and turn.get(b) is not None:
                ms[label] = round((turn[b] - turn[a]) * 1000)
        ms["total"] = round((turn["played"] - start) * 1000)
        entry = {"text": turn["text"], "reply": utt.text, "ms": ms,
                 "at": {k: v for k, v in turn.items() if k not in ("text", "utt")}}
        self.timings.append(entry)
        log.info("turn latency %s: %s (caller %r, reply %r)", self.call_id,
                 " ".join(f"{k}={v}" for k, v in ms.items()), turn["text"][:60], utt.text[:60])

    # -- caller input -------------------------------------------------------

    async def _barge_in(self) -> None:
        if not self.speaker.busy:
            return
        was_reply = self.speaker.playing_reply
        heard = self.speaker.stop()
        self._sentence_buf = ""
        self._reply_open = False
        self._emit({"type": "clear"})  # the client drops any audio it has buffered, right now
        if was_reply:
            self._muted = True  # drop tokens the cancelled reply may still emit
            try:
                await self.relay.handle({"type": "interrupt", "utteranceUntilInterrupt": heard})
            finally:
                self._muted = False

    async def _caller_turn(self, text: str, speech_end: float | None = None) -> None:
        await self._barge_in()
        self._turn = {"text": text, "speech_end": speech_end, "stt_turn": time.monotonic()}
        self._emit({"type": "caller_text", "text": text})
        await self.relay.handle({"type": "prompt", "voicePrompt": text, "last": True})

    async def _stt_loop(self) -> None:
        async for event in self.stt.events():
            if self.relay.ending:
                continue
            if event.kind == "partial" and len(event.text.split()) >= BARGE_IN_MIN_WORDS:
                await self._barge_in()
            elif event.kind == "turn" and event.text:
                await self._caller_turn(event.text, event.speech_end)

    async def _audio_loop(self) -> None:
        pending = bytearray()
        while True:
            kind, payload = await self.transport.read()
            if kind == "audio":
                pending.extend(payload)
                if len(pending) >= self.stt_chunk_bytes:
                    await self.stt.send_audio(bytes(pending))
                    pending.clear()
            elif kind == "dtmf" and payload:
                await self.relay.handle({"type": "dtmf", "digit": payload})
            elif kind == "text" and payload.strip() and not self.relay.ending:
                await self._caller_turn(payload.strip())
            elif kind == "hangup":
                return
            elif kind == "error":
                log.error("transport error on %s: %s", self.call_id, payload)
                return

    # -- hard time limit ----------------------------------------------------

    async def _timer_loop(self) -> None:
        started = time.monotonic()
        if self.warn_at is not None and self.warn_at < self.time_limit:
            await asyncio.sleep(self.warn_at)
            if self.relay.conv is not None and not self.relay.ending:
                left = round(self.time_limit - self.warn_at)
                self.relay.conv.add_note(f"[About {left} seconds of conversation time are left. In your next reply, "
                                         "mention once, briefly, that about a minute is left and offer to take a message.]")
                self._emit({"type": "time", "seconds_left": left})
            await asyncio.sleep(self.time_limit - (time.monotonic() - started))
        else:
            await asyncio.sleep(self.time_limit)
        log.info("call %s reached its %ss time limit", self.call_id, self.time_limit)
        self.relay.ending = True
        await self.relay._cancel_reply()
        self.speaker.stop()
        self._sentence_buf = ""
        self._reply_open = False
        self._emit({"type": "clear"})
        await self.transport.send_event({"type": "agent_text", "token": self.goodbye, "last": True})
        self.speaker.say(self.goodbye, is_reply=False)
        await self.speaker.wait_drained()
        self.next_action = "hangup"
        await self.transport.send_event({"type": "end", "reason": "time_limit"})
        self._ended.set()

    async def run(self) -> str:
        """Run the call until either side hangs up. Returns "hangup" or "transfer"."""
        tasks: list[asyncio.Task] = []
        try:
            await self.stt.connect()
            await self.relay.handle({"type": "setup", "callSid": self.call_id, "from": self.from_number})
            if self.relay.conv is not None:
                self.relay.conv.on_timing = self._mark
            if self.relay.conv is not None:
                self.relay.conv.on_event = self._emit
            tasks = [asyncio.create_task(self.speaker.run(), name="speaker"),
                     asyncio.create_task(self._stt_loop(), name="stt"),
                     asyncio.create_task(self._audio_loop(), name="audio"),
                     asyncio.create_task(self._ended.wait(), name="ended")]
            if self.time_limit is not None:
                tasks.append(asyncio.create_task(self._timer_loop(), name="timer"))
            await self.transport.send_event({"type": "agent_text", "token": self.greeting, "last": True})
            self.speaker.say(self.greeting, is_reply=False)
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.get_name() not in ("ended", "timer") and t.exception():
                    log.error("call %s: %s task failed: %r", self.call_id, t.get_name(), t.exception())
        except Exception:
            log.exception("call %s failed", self.call_id)
            self.next_action = "hangup"
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.speaker.stop()
            for t in list(self._event_tasks):
                t.cancel()
            await self.relay.close()
            await self.stt.close()
            await self.transport.close()
        return self.next_action


class AudioSocketServer:
    """Accepts AudioSocket connections from Asterisk for calls registered via /asterisk/start."""

    def __init__(self, host: str, port: int, start_call: Callable[[str, asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]):
        self.host, self.port = host, port
        self.start_call = start_call
        self.server: asyncio.base_events.Server | None = None
        self._calls: set[asyncio.Task] = set()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, self.host, self.port)
        log.info("AudioSocket listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        """Stop accepting calls and end live ones (wait_closed waits for every open connection)."""
        if not self.server:
            return
        self.server.close()
        for task in list(self._calls):
            task.cancel()
        await asyncio.gather(*self._calls, return_exceptions=True)
        await self.server.wait_closed()
        self.server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        self._calls.add(task)
        try:
            await self._serve(reader, writer)
        finally:
            self._calls.discard(task)
            if not writer.is_closing():
                writer.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            kind, payload = await asyncio.wait_for(aus.read_message(reader), timeout=5)
            if kind != aus.UUID or len(payload) != 16:
                raise ValueError(f"first message was type {kind:#x}, not a UUID")
            await self.start_call(aus.parse_uuid(payload), reader, writer)
        except Exception as exc:
            log.warning("AudioSocket connection rejected: %r", exc)
            try:
                writer.write(aus.pack(aus.HANGUP))
                writer.close()
            except ConnectionError:
                pass
