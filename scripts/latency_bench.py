"""End-to-end reply latency with no phone: a scripted caller talks to the real call pipeline.

    ./venv/bin/python scripts/latency_bench.py                 # current code and settings
    ./venv/bin/python scripts/latency_bench.py --baseline      # the pre-tuning behaviour, for comparison
    ./venv/bin/python scripts/latency_bench.py --min-silence 250 --max-silence 1280

Inworld speaks a short ordering conversation as the caller (cached in logs/bench/ so later runs
reuse the audio), each utterance is streamed over a socket pair to a real MediaCall exactly as
Asterisk would send it (20 ms frames, real-time pace), and the reply goes through the real
AssemblyAI session, Conversation (real Claude), sentence splitter, Inworld and Speaker. One
utterance has a 600 ms pause in the middle, to show whether turn detection cuts callers off.

Prints per-turn stage timings (ms), median and p90 per stage, the transcripts AssemblyAI heard,
and an estimate of what the run cost. Orders go to a temporary database, not the kitchen board.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import httpx  # noqa: E402

from chat import PRICES, CountingLLM  # noqa: E402
from voiceagent import audiosocket as aus  # noqa: E402
from voiceagent.agent import Conversation  # noqa: E402
from voiceagent.config import Settings  # noqa: E402
from voiceagent.media import MediaCall  # noqa: E402
from voiceagent.menu import load_restaurant  # noqa: E402
from voiceagent.relay import RelaySession  # noqa: E402
from voiceagent.speech import AssemblyAISTT, InworldTTS  # noqa: E402
from voiceagent.store import Store  # noqa: E402
from voiceagent.tools import CallInfo  # noqa: E402

CACHE = ROOT / "logs" / "bench"
CALLER_VOICE = "Dennis"
STT_RATE_PER_HOUR = 0.15
INWORLD_PER_CHAR = 5.0 / 1_000_000
PAUSE = "<pause>"
# A short pickup order. The second line pauses mid-sentence (600 ms) like a caller thinking.
SCRIPT = [
    "Can I get a large margherita with mushrooms?",
    f"And I'd also like {PAUSE} a can of diet cola.",
    "That's everything. The name is Jordan, and use this number.",
    "Yes, that's right.",
]
STAGES = ["stt", "queue", "ttft", "sentence", "tts", "play", "wire", "total"]


async def caller_audio(tts: InworldTTS, text: str, spent: dict) -> bytes:
    """8 kHz PCM for one caller line, with PAUSE rendered as 600 ms of silence. Cached on disk."""
    parts = []
    for piece in text.split(PAUSE):
        piece = piece.strip()
        if not piece:
            continue
        key = hashlib.sha1(f"{CALLER_VOICE}|{tts.model}|{piece}".encode()).hexdigest()[:16]
        path = CACHE / f"{key}.pcm"
        if not path.exists():
            audio = bytearray()
            async for chunk in tts.synthesize(piece):
                audio.extend(chunk)
            spent["inworld_chars"] += len(piece)
            path.write_bytes(bytes(audio[: len(audio) - len(audio) % 2]))
        parts.append(path.read_bytes())
    silence = b"\x00" * (aus.SAMPLE_RATE * 2 * 600 // 1000)
    return silence.join(parts)


class Line:
    """Our end of the socket pair: plays caller audio at real-time pace, watches what comes back."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader, self.writer = reader, writer
        self.pending = bytearray()
        self.speech_end_sent: float | None = None
        self.first_speech_at: float | None = None   # first non-silent frame since the last utterance
        self.last_speech_at: float | None = None
        self.hung_up = False

    async def send_frames(self) -> None:
        next_at = time.monotonic()
        while not self.hung_up:
            if self.pending:
                frame = bytes(self.pending[: aus.FRAME_BYTES]).ljust(aus.FRAME_BYTES, b"\x00")
                del self.pending[: aus.FRAME_BYTES]
                if not self.pending:
                    self.speech_end_sent = time.monotonic()
            else:
                frame = aus.SILENCE_FRAME
            self.writer.write(aus.audio_frame(frame))
            await self.writer.drain()
            next_at += aus.FRAME_MS / 1000
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))

    async def listen(self) -> None:
        try:
            while True:
                kind, payload = await aus.read_message(self.reader)
                if kind == aus.HANGUP:
                    break
                if kind == aus.AUDIO_8K and payload.strip(b"\x00"):
                    now = time.monotonic()
                    if self.first_speech_at is None:
                        self.first_speech_at = now
                    self.last_speech_at = now
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        self.hung_up = True

    def say(self, pcm: bytes) -> None:
        self.first_speech_at = None
        self.speech_end_sent = None
        self.pending.extend(pcm)

    def agent_quiet_for(self, seconds: float) -> bool:
        return self.last_speech_at is not None and time.monotonic() - self.last_speech_at >= seconds


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true",
                    help="pre-tuning behaviour: wait for formatted turns, 5 s keepalive, no warm-up, 400/1280 ms")
    ap.add_argument("--min-silence", type=int, help="AssemblyAI min_turn_silence in ms")
    ap.add_argument("--max-silence", type=int, help="AssemblyAI max_turn_silence in ms")
    ap.add_argument("--eot-confidence", type=float, help="AssemblyAI end_of_turn_confidence_threshold")
    ap.add_argument("--json", help="also write the per-turn results to this file")
    args = ap.parse_args()

    s = Settings.from_env()
    missing = [n for n, v in (("ANTHROPIC_API_KEY", True), ("ASSEMBLYAI_API_KEY", s.assemblyai_api_key),
                              ("INWORLD_API_KEY", s.inworld_api_key)) if not v]
    if missing:
        print(f"missing in .env: {', '.join(missing)}")
        return 2
    CACHE.mkdir(parents=True, exist_ok=True)
    restaurant = load_restaurant(s.restaurant_file)
    spent = {"inworld_chars": 0}

    min_sil = args.min_silence or (400 if args.baseline else s.stt_min_turn_silence_ms)
    max_sil = args.max_silence or (1280 if args.baseline else s.stt_max_turn_silence_ms)
    eot = args.eot_confidence if args.eot_confidence is not None else (None if args.baseline else s.stt_end_of_turn_confidence)
    wait_fmt = True if args.baseline else s.stt_wait_for_formatting
    keepalive = 5.0 if args.baseline else None

    tts_kwargs = {}
    if keepalive is not None:
        tts_kwargs["client"] = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0),
                                                 limits=httpx.Limits(keepalive_expiry=keepalive))
    agent_tts = InworldTTS(s.inworld_api_key, voice=s.inworld_voice, model=s.inworld_model,
                           speaking_rate=s.inworld_speaking_rate, max_concurrent=s.inworld_max_concurrent, **tts_kwargs)
    caller_tts = InworldTTS(s.inworld_api_key, voice=CALLER_VOICE, model=s.inworld_model)
    lines = [await caller_audio(caller_tts, text, spent) for text in SCRIPT]

    llm = CountingLLM(s.model, s.effort, **({"keepalive_seconds": keepalive} if keepalive else {}))
    spoken_chars = {"n": 0}
    real_synth = agent_tts.synthesize

    async def counting_synth(text):
        spoken_chars["n"] += len(text)
        async for chunk in real_synth(text):
            yield chunk

    from datetime import datetime, time as dtime
    hh, mm = (int(x) for x in (s.demo_time or "18:00").split(":"))
    fixed = datetime.combine(restaurant.local_now().date(), dtime(hh, mm), tzinfo=restaurant.tz)
    store = Store(Path(tempfile.mkdtemp(prefix="rva-bench-")) / "bench.db")
    make_conv = lambda call: Conversation(restaurant, store, llm, call, clock=lambda: fixed)
    make_relay = lambda send, sleep: RelaySession(send, make_conv, store, restaurant.id, channel="phone", sleep=sleep)
    stt = AssemblyAISTT(s.assemblyai_api_key, min_turn_silence_ms=min_sil, max_turn_silence_ms=max_sil,
                        end_of_turn_confidence=eot, wait_for_formatting=wait_fmt)

    a, b = socket.socketpair()
    a.setblocking(False), b.setblocking(False)
    our_reader, our_writer = await asyncio.open_connection(sock=a)
    call_reader, call_writer = await asyncio.open_connection(sock=b)
    call = MediaCall(call_reader, call_writer, call_id=f"bench-{int(time.time())}", from_number="+12125550147",
                     greeting=restaurant.greeting, stt=stt, synthesize=counting_synth, make_relay=make_relay)
    if not args.baseline:
        asyncio.create_task(llm.warm())
        asyncio.create_task(agent_tts.warm())
    line = Line(our_reader, our_writer)
    started = time.monotonic()
    call_task = asyncio.create_task(call.run())
    tasks = [asyncio.create_task(line.send_frames()), asyncio.create_task(line.listen())]
    mode = "baseline" if args.baseline else "current"
    print(f"[{mode}] model {s.model}/{s.effort}, STT min {min_sil} max {max_sil} eot {eot or 'default'}, "
          f"formatted turns {'awaited' if wait_fmt else 'not awaited'}, warm-up {'off' if args.baseline else 'on'}")

    async def wait_until(pred, timeout=45.0):
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline or line.hung_up or call_task.done():
                return False
            await asyncio.sleep(0.02)
        return True

    results = []
    cut_offs: list[int] = []
    try:
        await wait_until(lambda: line.first_speech_at and line.agent_quiet_for(1.0) and not call.speaker.busy)
        for i, (text, pcm) in enumerate(zip(SCRIPT, lines)):
            n_before = len(call.timings)
            line.say(pcm)
            ok = await wait_until(lambda: len(call.timings) > n_before and line.first_speech_at is not None
                                  and line.agent_quiet_for(1.2) and not call.speaker.busy
                                  and call.relay.reply_task is not None and call.relay.reply_task.done())
            if not ok:
                print(f"turn {i + 1}: no reply within the time limit")
                break
            entry = call.timings[n_before]
            ms, at = dict(entry["ms"]), entry["at"]
            if line.speech_end_sent is None or at["stt_turn"] < line.speech_end_sent:
                early = round(((line.speech_end_sent or time.monotonic()) - at["stt_turn"]) * 1000)
                print(f"turn {i + 1}: CUT OFF, the turn ended {early} ms before the caller finished "
                      f"(heard {entry['text']!r}); not counted")
                cut_offs.append(i + 1)
                await wait_until(lambda: line.agent_quiet_for(1.5) and not call.speaker.busy)
                continue
            # Measured from our side of the wire: the caller's last audio frame to the first agent audio frame.
            ms["stt"] = round((at["stt_turn"] - line.speech_end_sent) * 1000)
            ms["wire"] = round((line.first_speech_at - at["played"]) * 1000)
            ms["total"] = round((line.first_speech_at - line.speech_end_sent) * 1000)
            if at.get("speech_end"):
                ms["stt_est_err"] = round((at["speech_end"] - line.speech_end_sent) * 1000)
            results.append({"turn": i + 1, "said": text.replace(PAUSE, "...").strip(), "heard": entry["text"],
                            "reply": entry["reply"], "ms": ms})
            print(f"turn {i + 1}: " + " ".join(f"{k}={ms.get(k)}" for k in STAGES))
            print(f"   said:  {results[-1]['said']}\n   heard: {entry['text']}\n   reply: {entry['reply'][:90]}")
    finally:
        line.hung_up = True
        our_writer.close()
        for t in tasks:
            t.cancel()
        try:
            await asyncio.wait_for(call_task, 10)
        except (asyncio.TimeoutError, Exception):
            call_task.cancel()
        Store(s.db_path).record_stt("latency-bench", stt.billed_seconds)

    if cut_offs:
        print(f"\nturns cut off mid-sentence: {cut_offs}")
    if results:
        print("\nstage            median     p90   (ms, speech end to first agent audio; n=%d turns)" % len(results))
        for k in STAGES:
            vals = [r["ms"][k] for r in results if r["ms"].get(k) is not None]
            if vals:
                print(f"{k:12} {statistics.median(vals):10.0f} {percentile(vals, 0.9):8.0f}")
    u = llm.usage
    claude = llm.cost() or 0.0
    stt_cost = stt.billed_seconds / 3600 * STT_RATE_PER_HOUR
    inworld = (spent["inworld_chars"] + spoken_chars["n"]) * INWORLD_PER_CHAR
    print(f"\nrun time {time.monotonic() - started:.0f}s, outcome {call.relay.conv.outcome() if call.relay.conv else '?'}")
    print(f"cost estimate: Claude ${claude:.4f} ({u['requests']} requests), AssemblyAI ${stt_cost:.4f} "
          f"({stt.billed_seconds:.0f}s), Inworld ${inworld:.4f} ({spent['inworld_chars']} caller + {spoken_chars['n']} agent chars)"
          f" = ${claude + stt_cost + inworld:.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"mode": mode, "min_silence": min_sil, "max_silence": max_sil,
                                               "turns": results}, indent=2))
    return 0 if len(results) == len(SCRIPT) and not cut_offs else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
