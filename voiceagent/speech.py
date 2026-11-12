"""Speech services for the self-hosted call path: AssemblyAI in, Inworld out.

Both sit behind tiny interfaces (`open_stt`, `synthesize`) so tests run
against fakes and either vendor can be swapped without touching the call logic.
Audio everywhere is 8 kHz 16-bit mono PCM, the format Asterisk hands us.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urlencode

import httpx
import websockets

log = logging.getLogger(__name__)

ASSEMBLYAI_URL = "wss://streaming.assemblyai.com/v3/ws"
INWORLD_URL = "https://api.inworld.ai/tts/v1/voice:stream"


@dataclass
class SpeechEvent:
    kind: str          # "speech_started", "partial" or "turn"
    text: str = ""
    speech_end: float | None = None   # turn events: when the caller's last word ended (monotonic clock, estimated)


# Idle TCP connections to the speech and model APIs are kept for this long. httpx closes them after
# 5 s by default, which is shorter than the gap between two caller turns, so every turn paid a new
# TLS handshake before its first byte.
KEEPALIVE_SECONDS = 120.0


class AssemblyAISTT:
    """One streaming session per call. Billed for as long as the socket is open."""

    def __init__(self, api_key: str, *, sample_rate: int = 8000, min_turn_silence_ms: int = 400,
                 max_turn_silence_ms: int = 1280, end_of_turn_confidence: float | None = None,
                 wait_for_formatting: bool = False):
        self.api_key = api_key
        self.params = {
            "speech_model": "universal-streaming-english",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
            "format_turns": "true",
            "min_turn_silence": min_turn_silence_ms,
            "max_turn_silence": max_turn_silence_ms,
        }
        if end_of_turn_confidence is not None:
            self.params["end_of_turn_confidence_threshold"] = end_of_turn_confidence
        # AssemblyAI sends each finished turn twice: first as soon as it decides the turn ended
        # (lowercase, no punctuation), then again formatted a little later. The model copes fine
        # with the unformatted text, so by default the first one is acted on and the second ignored.
        self.wait_for_formatting = wait_for_formatting
        self.ws = None
        self._opened_at: float | None = None
        self._bytes_per_ms = sample_rate * 2 / 1000
        self._audio_ms_sent = 0.0
        self._last_send_at: float | None = None
        self._last_turn_order = None
        self.billed_seconds = 0.0  # session length: AssemblyAI bills from socket open to close, idle time included

    async def connect(self) -> None:
        self._opened_at = time.monotonic()
        self.ws = await websockets.connect(f"{ASSEMBLYAI_URL}?{urlencode(self.params)}",
                                           additional_headers={"Authorization": self.api_key})

    async def send_audio(self, pcm: bytes) -> None:
        await self.ws.send(pcm)
        self._audio_ms_sent += len(pcm) / self._bytes_per_ms
        self._last_send_at = time.monotonic()

    def _speech_end(self, msg: dict) -> float | None:
        """When the last word of a turn ended, on this machine's monotonic clock.

        Word timestamps are milliseconds into the audio stream; we know how much audio had been
        sent at what time, so the difference places the end of speech on our clock.
        """
        ends = [w.get("end") for w in msg.get("words") or [] if isinstance(w, dict) and w.get("end") is not None]
        if not ends or self._last_send_at is None:
            return None
        return self._last_send_at - max(0.0, self._audio_ms_sent - max(ends)) / 1000

    async def events(self) -> AsyncIterator[SpeechEvent]:
        async for raw in self.ws:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "Begin":
                model = (msg.get("configuration") or {}).get("model")
                log.info("AssemblyAI session %s (model %s)", msg.get("id"), model)
            elif kind == "SpeechStarted":
                yield SpeechEvent("speech_started")
            elif kind == "Turn":
                text = (msg.get("transcript") or "").strip()
                if msg.get("end_of_turn"):
                    if self.wait_for_formatting and not msg.get("turn_is_formatted"):
                        continue
                    order = msg.get("turn_order")
                    if order is not None and order == self._last_turn_order:
                        continue  # the formatted copy of a turn already acted on
                    self._last_turn_order = order
                    yield SpeechEvent("turn", text, speech_end=self._speech_end(msg))
                elif text:
                    yield SpeechEvent("partial", text)
            elif kind == "Termination":
                self.billed_seconds = max(self.billed_seconds, float(msg.get("session_duration_seconds") or 0))
                log.info("AssemblyAI session ended: %.0fs audio", msg.get("audio_duration_seconds") or 0)
                return

    async def close(self) -> None:
        if self._opened_at is not None:
            self.billed_seconds = max(self.billed_seconds, time.monotonic() - self._opened_at)
        if self.ws is None:
            return
        try:
            await self.ws.send(json.dumps({"type": "Terminate"}))
        except Exception:
            pass
        await self.ws.close()


class InworldTTS:
    """Inworld streaming TTS with a cap on simultaneous requests.

    Each sentence is synthesized as soon as the model finishes it, so a long reply would otherwise
    open several requests at once and hit Inworld's per-account concurrency limit (low on the free
    tier; over-limit requests are rejected with HTTP 429). Requests queue on a semaphore in the
    order they were made, so sentences still play in order and the next one is prepared while the
    current one plays. A 429 is retried with backoff instead of dropping the sentence.
    """

    def __init__(self, api_key: str, *, voice: str, model: str, speaking_rate: float = 1.0,
                 client: httpx.AsyncClient | None = None, max_concurrent: int = 2, retry_delays=(0.3, 0.8, 1.6)):
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.speaking_rate = speaking_rate
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0),
                                                  limits=httpx.Limits(keepalive_expiry=KEEPALIVE_SECONDS))
        self.max_concurrent = max_concurrent
        self.retry_delays = tuple(retry_delays)
        self._slots: asyncio.Semaphore | None = None

    async def warm(self) -> None:
        """Open the TLS connection to Inworld before the first sentence needs it (free: no synthesis)."""
        try:
            await self.client.get(INWORLD_URL, headers={"Authorization": f"Basic {self.api_key}"})
        except Exception as exc:  # a failed warm-up changes nothing: the first sentence connects as before
            log.debug("Inworld warm-up failed: %r", exc)

    async def synthesize(self, text: str, *, sample_rate: int = 8000, voice: str | None = None,
                         speaking_rate: float | None = None) -> AsyncIterator[bytes]:
        """Yield raw PCM chunks at `sample_rate` as Inworld generates them, in `voice` (default: self.voice)
        at `speaking_rate` (default: self.speaking_rate; Inworld accepts 0.5 to 1.5)."""
        if self._slots is None:
            self._slots = asyncio.Semaphore(self.max_concurrent)
        body = {
            "text": text,
            "voiceId": voice or self.voice,
            "modelId": self.model,
            "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": sample_rate, "speakingRate": speaking_rate or self.speaking_rate},
        }
        headers = {"Authorization": f"Basic {self.api_key}", "Content-Type": "application/json"}
        async with self._slots:
            for attempt in range(len(self.retry_delays) + 1):
                async with self.client.stream("POST", INWORLD_URL, json=body, headers=headers) as resp:
                    if resp.status_code == 429 and attempt < len(self.retry_delays):
                        await resp.aread()
                        log.warning("Inworld concurrency limit hit; retrying in %.1fs", self.retry_delays[attempt])
                        await asyncio.sleep(self.retry_delays[attempt])
                        continue
                    if resp.status_code != 200:
                        detail = (await resp.aread())[:300]
                        raise RuntimeError(f"Inworld TTS HTTP {resp.status_code}: {detail!r}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        msg = json.loads(line)
                        if "error" in msg:
                            raise RuntimeError(f"Inworld TTS error: {msg['error']}")
                        audio = (msg.get("result") or {}).get("audioContent")
                        if audio:
                            yield base64.b64decode(audio)
                    return


def with_fallback_voice(synthesize, *, voice: str, fallback: str, sample_rate: int, speaking_rate: float | None = None):
    """A synthesize(text) for one persona: `voice` first (at `speaking_rate`: a cloned voice can be
    slower than the stock ones), and if synthesis fails before any audio arrived (a cloned voice can be
    rejected or rate limited), the same sentence once more in `fallback` at its normal rate."""
    async def run(text: str) -> AsyncIterator[bytes]:
        started = False
        try:
            async for chunk in synthesize(text, sample_rate=sample_rate, voice=voice, speaking_rate=speaking_rate):
                started = True
                yield chunk
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if started or voice == fallback:
                raise
            log.warning("voice %r failed (%r); retrying the sentence with %r", voice, exc, fallback)
        async for chunk in synthesize(text, sample_rate=sample_rate, voice=fallback):
            yield chunk
    return run
