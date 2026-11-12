"""Check both speech services with no phone line: Inworld speaks a sentence, AssemblyAI hears it back.

    ./venv/bin/python scripts/speech_check.py
    ./venv/bin/python scripts/speech_check.py --text "Two large pepperoni pizzas, please." --voice Dennis

Needs INWORLD_API_KEY and ASSEMBLYAI_API_KEY in .env. Costs a fraction of a cent. Saves the
synthesized audio as a WAV in ~/Files/voice-agent/ so you can listen to it in the Files app, and
prints the timings that matter on a call: how long until the voice starts, and how long after the
speaker stops until the transcript is ready.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voiceagent.config import Settings  # noqa: E402
from voiceagent.speech import AssemblyAISTT, InworldTTS  # noqa: E402

RATE = 8000
BYTES_PER_SEC = RATE * 2
OUT_DIR = Path.home() / "Files" / "voice-agent"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hi, can I get a large margherita with mushrooms and a can of diet cola for pickup?")
    ap.add_argument("--voice", help="Inworld voice id (default: INWORLD_VOICE from .env)")
    args = ap.parse_args()
    s = Settings.from_env()
    missing = [n for n, v in (("INWORLD_API_KEY", s.inworld_api_key), ("ASSEMBLYAI_API_KEY", s.assemblyai_api_key)) if not v]
    if missing:
        print(f"missing in .env: {', '.join(missing)}")
        return 2
    voice = args.voice or s.inworld_voice

    # 1) Inworld: text to 8 kHz PCM, the format Asterisk plays.
    tts = InworldTTS(s.inworld_api_key, voice=voice, model=s.inworld_model, speaking_rate=s.inworld_speaking_rate)
    audio = bytearray()
    started = time.monotonic()
    first = None
    async for chunk in tts.synthesize(args.text):
        first = first or time.monotonic() - started
        audio.extend(chunk)
    tts_total = time.monotonic() - started
    if len(audio) % 2:
        audio = audio[:-1]
    seconds = len(audio) / BYTES_PER_SEC
    print(f"Inworld ({s.inworld_model}, voice {voice}): first audio after {first:.2f}s, all {seconds:.1f}s of speech "
          f"in {tts_total:.2f}s")
    if seconds < 0.5:
        print("FAIL: almost no audio came back; check the voice id and model")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = OUT_DIR / f"inworld_{voice}_{time.strftime('%Y%m%d-%H%M%S')}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(audio))
    print(f"saved {wav_path} (open it in files.flockara.com > Files > voice-agent)")

    # 2) AssemblyAI: stream that audio back at real-time pace, then silence, like a caller finishing a sentence.
    stt = AssemblyAISTT(s.assemblyai_api_key)
    await stt.connect()
    turns: list[tuple[float, str]] = []
    spoke_until = 0.0

    async def listen():
        async for ev in stt.events():
            if ev.kind == "turn":
                turns.append((time.monotonic(), ev.text))
                return

    listener = asyncio.create_task(listen())
    chunk = BYTES_PER_SEC // 10
    for i in range(0, len(audio), chunk):
        await stt.send_audio(bytes(audio[i:i + chunk]).ljust(chunk, b"\x00"))
        await asyncio.sleep(0.1)
    spoke_until = time.monotonic()
    silence = b"\x00" * chunk
    for _ in range(40):  # up to 4 s of silence
        if listener.done():
            break
        await stt.send_audio(silence)
        await asyncio.sleep(0.1)
    try:
        await asyncio.wait_for(listener, timeout=3)
    except asyncio.TimeoutError:
        pass
    await stt.close()
    from voiceagent.store import Store
    Store(s.db_path).record_stt("speech-check", stt.billed_seconds)
    if not turns:
        print("FAIL: AssemblyAI returned no finished turn")
        return 1
    at, heard = turns[0]
    print(f"AssemblyAI: turn ready {at - spoke_until:.2f}s after the speaker stopped")
    print(f"  said:  {args.text}")
    print(f"  heard: {heard}")
    norm = lambda t: "".join(c for c in t.lower() if c.isalnum() or c == " ").split()
    said, got = norm(args.text), norm(heard)
    matched = sum(1 for w in said if w in got)
    print(f"words matched: {matched}/{len(said)}")
    print("SPEECH CHECK PASS" if matched >= 0.8 * len(said) else "SPEECH CHECK: transcript differs, listen to the WAV")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
