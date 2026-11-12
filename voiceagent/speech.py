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


class AssemblyAISTT:
    """One streaming session per call. Billed for as long as the socket is open."""

    def __init__(self, api_key: str, *, sample_rate: int = 8000, min_turn_silence_ms: int = 400,
                 max_turn_silence_ms: int = 1280):
        self.api_key = api_key
        self.params = {
            "speech_model": "universal-streaming-english",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
            "format_turns": "true",
            "min_turn_silence": min_turn_silence_ms,
            "max_turn_silence": max_turn_silence_ms,
        }
        self.ws = None
        self._opened_at: float | None = None
        self.billed_seconds = 0.0  # session length: AssemblyAI bills from socket open to close, idle time included

    async def connect(self) -> None:
        self._opened_at = time.monotonic()
        self.ws = await websockets.connect(f"{ASSEMBLYAI_URL}?{urlencode(self.params)}",
                                           additional_headers={"Authorization": self.api_key})

    async def send_audio(self, pcm: bytes) -> None:
        await self.ws.send(pcm)

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
                if msg.get("end_of_turn") and msg.get("turn_is_formatted"):
                    yield SpeechEvent("turn", text)
                elif text and not msg.get("end_of_turn"):
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
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0))
        self.max_concurrent = max_concurrent
        self.retry_delays = tuple(retry_delays)
        self._slots: asyncio.Semaphore | None = None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw 8 kHz PCM chunks as Inworld generates them."""
        if self._slots is None:
            self._slots = asyncio.Semaphore(self.max_concurrent)
        body = {
            "text": text,
            "voiceId": self.voice,
            "modelId": self.model,
            "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": 8000, "speakingRate": self.speaking_rate},
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
