"""Smoke test of the real Asterisk container against the app, with no carrier and no API keys.

Runs the app on 127.0.0.1:$PORT with scripted speech services and a scripted
model, waits for one call, and prints what flowed. Place the call with:

    docker exec rva-asterisk asterisk -rx "channel originate Local/s@rva-call application Playback tt-monkeys"

The Playback side plays a sound file into the call, so real caller audio reaches
the app; once a second of it has arrived, the fake speech-to-text "hears" a
goodbye, the scripted model ends the call, and Asterisk hangs up.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from tests.fakes import FakeLLM, call  # noqa: E402
from voiceagent.config import Settings  # noqa: E402
from voiceagent.main import create_app  # noqa: E402
from voiceagent.speech import SpeechEvent  # noqa: E402

stats = {"caller_audio_bytes": 0, "tts_texts": []}
done = asyncio.Event()


class SmokeSTT:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.said_bye = False

    async def connect(self):
        pass

    async def send_audio(self, pcm):
        stats["caller_audio_bytes"] += len(pcm)
        nonzero = any(pcm)
        if stats["caller_audio_bytes"] >= 16000 and not self.said_bye:
            self.said_bye = True
            stats["caller_audio_nonzero"] = nonzero
            await self.queue.put(SpeechEvent("turn", "That's all, thanks, bye."))

    async def events(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event

    async def close(self):
        done.set()


async def tone(text):
    stats["tts_texts"].append(text)
    sample = (3000).to_bytes(2, "little", signed=True) + (-3000).to_bytes(2, "little", signed=True)
    data = sample * (len(text) * 100)
    for i in range(0, len(data), 3200):
        await asyncio.sleep(0)
        yield data[i:i + 3200]


async def main() -> int:
    token = os.environ["ASTERISK_TOKEN"]
    tmp = Path(tempfile.mkdtemp(prefix="rva-smoke-"))
    settings = replace(Settings.from_env(), db_path=tmp / "smoke.db", asterisk_token=token)
    llm = FakeLLM([call(("end_call", {}, "t1"), text="Thanks for calling, goodbye!")])
    app = create_app(settings, llm=llm, stt_factory=SmokeSTT, synthesize=tone)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8790")),
                                           log_level="warning"))
    serve = asyncio.create_task(server.serve())
    print("ready: place the call now", flush=True)
    try:
        await asyncio.wait_for(done.wait(), timeout=90)
    except asyncio.TimeoutError:
        print("SMOKE FAIL: no call completed in 90 s")
        return 1
    await asyncio.sleep(1.5)  # let the dialplan fetch /asterisk/next
    calls = app.state.store.recent_calls("demo")
    server.should_exit = True
    await serve
    print(f"caller audio received: {stats['caller_audio_bytes']} bytes "
          f"({stats['caller_audio_bytes'] / 16000:.1f} s), non-silent: {stats.get('caller_audio_nonzero')}")
    print(f"spoken by agent: {stats['tts_texts']}")
    print(f"call record: {calls[0]['outcome'] if calls else None}, transcript: {calls[0]['transcript'] if calls else None}")
    ok = bool(calls) and stats["caller_audio_bytes"] >= 16000 and "Thanks for calling, goodbye!" in stats["tts_texts"]
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
