import asyncio
import socket
import tempfile
import unittest
import uuid as uuidlib
from dataclasses import replace
from pathlib import Path

import httpx

from voiceagent import audiosocket as aus
from voiceagent.main import create_app
from voiceagent.media import Speaker, _Utterance, split_sentences
from voiceagent.speech import SpeechEvent

from .fakes import DEMO, FakeLLM, as_plain, call, say


class WarmableLLM(FakeLLM):
    def __init__(self, script):
        super().__init__(script)
        self.warmed = 0

    async def warm(self):
        self.warmed += 1
from .test_app_twilio import settings as base_settings

TOKEN = "asterisk-test-token"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSTT:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.audio = bytearray()
        self.closed = False
        self.loop = None

    async def connect(self):
        self.loop = asyncio.get_running_loop()

    async def send_audio(self, pcm):
        self.audio.extend(pcm)

    async def events(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            yield event

    async def close(self):
        self.closed = True


def fake_tts(bytes_per_char=40, chunk=640):
    spoken: list[str] = []

    async def synthesize(text):
        spoken.append(text)
        data = b"\x01\x00" * (len(text) * bytes_per_char // 2)
        for i in range(0, len(data), chunk):
            await asyncio.sleep(0)
            yield data[i:i + chunk]
    synthesize.spoken = spoken
    return synthesize


async def eventually(predicate, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


class FakeAsterisk:
    """Plays Asterisk's side of AudioSocket: sends the UUID and caller audio, records what comes back."""

    def __init__(self, port: int, call_id: str):
        self.port, self.call_id = port, call_id
        self.frames: list[bytes] = []
        self.hung_up = False

    @property
    def speech_frames(self) -> int:
        return sum(1 for f in self.frames if f.strip(b"\x00"))

    async def run(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(aus.pack(aus.UUID, uuidlib.UUID(self.call_id).bytes))
        for _ in range(10):
            writer.write(aus.audio_frame(b"\x00" * aus.FRAME_BYTES))
        await writer.drain()
        self.writer = writer
        try:
            while True:
                kind, payload = await aus.read_message(reader)
                if kind == aus.HANGUP:
                    self.hung_up = True
                    return
                if kind == aus.AUDIO_8K:
                    self.frames.append(payload)
        except asyncio.IncompleteReadError:
            return


class FramingTests(unittest.TestCase):
    def test_pack_layout(self):
        self.assertEqual(aus.pack(aus.HANGUP), b"\x00\x00\x00")
        frame = aus.audio_frame(b"\x01\x02" * 160)
        self.assertEqual(frame[:3], b"\x10\x01\x40")  # type 0x10, length 320 big-endian
        self.assertEqual(len(frame), 323)

    def test_uuid_round_trip(self):
        u = uuidlib.uuid4()
        self.assertEqual(aus.parse_uuid(u.bytes), str(u))

    def test_split_sentences_keeps_prices_and_remainder(self):
        sentences, rest = split_sentences("That's $14.50 total. Anything else? Your order")
        self.assertEqual(sentences, ["That's $14.50 total.", "Anything else?"])
        self.assertEqual(rest, " Your order".lstrip() if rest.startswith("Y") else rest)
        self.assertEqual(rest.strip(), "Your order")

    def test_heard_text_proportional(self):
        utt = _Utterance("one two three four", True)
        utt.audio.extend(b"\x00" * 1000)
        utt.done = True
        utt.played = 500
        self.assertEqual(utt.heard_text(), "one two")
        utt.played = 1000
        self.assertEqual(utt.heard_text(), "one two three four")


class SpeakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_plays_in_order_then_silence_and_drains(self):
        written = []

        async def write(frame):
            written.append(frame)  # raw PCM; the transport adds the AudioSocket header

        speaker = Speaker(write, fake_tts(), frame_seconds=0.001)
        task = asyncio.create_task(speaker.run())
        speaker.say("Hello there.")
        speaker.say("Second.")
        await asyncio.wait_for(speaker.wait_drained(), 3)
        task.cancel()
        self.assertEqual(speaker.heard_reply, ["Hello there.", "Second."])
        self.assertTrue(all(len(f) == aus.FRAME_BYTES for f in written))

    async def test_stop_returns_partial_heard(self):
        async def write(frame):
            pass

        speaker = Speaker(write, fake_tts(bytes_per_char=4000), frame_seconds=0.001)
        task = asyncio.create_task(speaker.run())
        speaker.say("Our pizzas are Margherita and Wild Mushroom.")
        await eventually(lambda: speaker.queue and speaker.queue[0].played > 100000)
        heard = speaker.stop()
        task.cancel()
        self.assertTrue(heard and "Our pizzas are Margherita and Wild Mushroom.".startswith(heard))
        self.assertFalse(speaker.busy)


class SelfHostedCallTests(unittest.IsolatedAsyncioTestCase):
    async def make(self, script, tts=None, **overrides):
        self.port = free_port()
        self.stt = FakeSTT()
        self.llm = WarmableLLM(script)
        self.tts = tts or fake_tts()
        s = replace(base_settings(**overrides), asterisk_token=TOKEN, audiosocket_port=self.port)
        self.app = create_app(s, llm=self.llm, stt_factory=lambda: self.stt, synthesize=self.tts, frame_seconds=0.001)
        await self.app.state.audiosocket.start()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.app.state.audiosocket.stop()

    async def start_call(self, caller="+12125550100"):
        r = await self.http.get("/asterisk/start", params={"token": TOKEN, "from": caller, "to": "+18005550100"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.text), 36)
        ast = FakeAsterisk(self.port, r.text)
        return r.text, ast, asyncio.create_task(ast.run())

    async def test_endpoints_need_token(self):
        await self.make([])
        self.assertEqual((await self.http.get("/asterisk/start")).status_code, 403)
        self.assertEqual((await self.http.get("/asterisk/next", params={"token": "nope"})).status_code, 403)
        self.assertEqual((await self.http.get("/asterisk/next", params={"token": TOKEN, "call": "x"})).text, "hangup")

    async def test_busy_when_full(self):
        await self.make([], max_sessions=1)
        first = await self.http.get("/asterisk/start", params={"token": TOKEN})
        self.assertEqual(len(first.text), 36)
        second = await self.http.get("/asterisk/start", params={"token": TOKEN})
        self.assertEqual(second.text, "busy")

    async def test_unknown_call_id_is_hung_up(self):
        await self.make([])
        ast = FakeAsterisk(self.port, str(uuidlib.uuid4()))
        await asyncio.wait_for(ast.run(), 3)
        self.assertTrue(ast.hung_up)
        self.assertEqual(ast.frames, [])

    async def test_greeting_turn_reply_and_hangup(self):
        await self.make([call(("end_call", {}, "t1"), text="Thanks for calling, goodbye!")])
        call_id, ast, task = await self.start_call()
        await eventually(lambda: len(self.tts.spoken) >= 1)
        self.assertTrue(self.tts.spoken[0].startswith("Thanks for calling Juniper Street Kitchen"))  # greeting
        await eventually(lambda: len(self.stt.audio) >= 1600)  # caller audio reaches STT in >= 100 ms chunks
        await self.stt.queue.put(SpeechEvent("turn", "That's all, bye."))
        await asyncio.wait_for(task, 5)
        self.assertTrue(ast.hung_up)
        self.assertIn("Thanks for calling, goodbye!", self.tts.spoken)
        self.assertGreater(ast.speech_frames, 0)
        self.assertEqual((await self.http.get("/asterisk/next", params={"token": TOKEN, "call": call_id})).text, "hangup")
        calls = self.app.state.store.recent_calls("demo")
        self.assertEqual(calls[0]["from_number"], "+12125550100")
        self.assertEqual(calls[0]["transcript"][0], {"role": "caller", "text": "That's all, bye."})
        self.assertTrue(self.stt.closed)
        # Caller ID reached the model.
        self.assertIn("+12125550100", as_plain(self.llm.requests[0])[1]["content"])
        # Both services were warmed while the greeting played.
        self.assertEqual(self.llm.warmed, 1)

    async def test_turn_latency_is_measured_per_caller_turn(self):
        await self.make([say("One moment."), say("Done.")])
        call_id, ast, task = await self.start_call()
        await eventually(lambda: len(self.tts.spoken) >= 1)
        with self.assertLogs("voiceagent.media", level="INFO") as logs:
            await self.stt.queue.put(SpeechEvent("turn", "Hello?", speech_end=asyncio.get_running_loop().time() - 0.3))
            await eventually(lambda: self.app.state.audiosocket and len(self.tts.spoken) >= 2 and ast.speech_frames > 0)
            await eventually(lambda: len(logs.output) > 0 and any("turn latency" in l for l in logs.output))
        line = next(l for l in logs.output if "turn latency" in l)
        for stage in ("stt=", "ttft=", "sentence=", "tts=", "play=", "total="):
            self.assertIn(stage, line)
        # The stt stage runs from the caller's last word (300 ms before the turn arrived) to the turn.
        self.assertRegex(line, r"stt=(2[5-9]\d|3\d\d|4[0-4]\d) ")
        ast.writer.write(aus.pack(aus.HANGUP))
        await ast.writer.drain()
        await asyncio.wait_for(task, 5)

    async def test_barge_in_stops_audio_and_tells_model_what_was_heard(self):
        long_reply = "We have three pizzas tonight. The Margherita is classic. The Wild Mushroom has no sauce."
        await self.make([say(long_reply), say("Sure, a Margherita. What size?")],
                        tts=fake_tts(bytes_per_char=3000))
        call_id, ast, task = await self.start_call()
        # Let the greeting finish, then ask.
        await eventually(lambda: self.app.state.audiosocket is not None and len(self.tts.spoken) >= 1)
        await self.stt.queue.put(SpeechEvent("turn", "What pizzas do you have?"))
        await eventually(lambda: long_reply.split(". ")[0] + "." in self.tts.spoken)
        frames_before = ast.speech_frames
        # First sentence is 29 chars * 3000 bytes = 272 frames; interrupt partway through the second.
        await eventually(lambda: ast.speech_frames > frames_before + 400)
        await self.stt.queue.put(SpeechEvent("partial", "just a"))
        await self.stt.queue.put(SpeechEvent("turn", "Just a margherita."))
        await eventually(lambda: len(self.llm.requests) == 2)
        users = [m for m in as_plain(self.llm.requests[1]) if m["role"] == "user"]
        texts = [b["text"] for b in users[-1]["content"]]
        self.assertTrue(any("interrupted your last reply" in t for t in texts), texts)
        note = next(t for t in texts if "interrupted" in t)
        heard = note.split('heard: "', 1)[1].rstrip('"]') if 'heard: "' in note else ""
        self.assertTrue(heard.startswith("We have three pizzas tonight. The"), heard)
        self.assertTrue(long_reply.startswith(heard), heard)
        self.assertNotEqual(heard, long_reply)
        self.assertEqual(texts[-1], "Just a margherita.")
        ast.writer.write(aus.pack(aus.HANGUP))
        await ast.writer.drain()
        await asyncio.wait_for(task, 5)

    async def test_transfer_reported_to_dialplan(self):
        path = Path(tempfile.mkdtemp()) / "r.yaml"
        path.write_text(DEMO.read_text().replace('staff_phone: ""', 'staff_phone: "+12125550000"'))
        await self.make([call(("transfer_to_staff", {"reason": "catering"}, "t1"))], restaurant_file=path)
        call_id, ast, task = await self.start_call()
        await eventually(lambda: len(self.tts.spoken) >= 1)
        await self.stt.queue.put(SpeechEvent("turn", "I need catering for fifty people."))
        await asyncio.wait_for(task, 5)
        self.assertIn("Okay, connecting you with someone at the restaurant now.", self.tts.spoken)
        self.assertEqual((await self.http.get("/asterisk/next", params={"token": TOKEN, "call": call_id})).text, "transfer")

    async def test_caller_hangup_cleans_up(self):
        await self.make([])
        call_id, ast, task = await self.start_call()
        await eventually(lambda: len(self.tts.spoken) >= 1)
        ast.writer.write(aus.pack(aus.HANGUP))
        await ast.writer.drain()
        await asyncio.wait_for(task, 5)
        await eventually(lambda: self.stt.closed)
        await eventually(lambda: self.app.state.store.recent_calls("demo") and
                         self.app.state.store.recent_calls("demo")[0]["outcome"] == "no_order")


if __name__ == "__main__":
    unittest.main()
