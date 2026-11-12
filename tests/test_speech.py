import asyncio
import base64
import json
import unittest

import httpx

from voiceagent.speech import InworldTTS


def ndjson(audio: bytes) -> bytes:
    return (json.dumps({"result": {"audioContent": base64.b64encode(audio).decode()}}) + "\n").encode()


class InworldConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_never_more_than_two_requests_at_once_and_order_kept(self):
        live = {"now": 0, "peak": 0}

        async def handler(request: httpx.Request):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.02)
            live["now"] -= 1
            text = json.loads(request.content)["text"]
            return httpx.Response(200, content=ndjson(text.encode()))

        tts = InworldTTS("k", voice="v", model="m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

        async def collect(text):
            return b"".join([c async for c in tts.synthesize(text)])

        results = await asyncio.gather(*(collect(f"sentence {i}.") for i in range(6)))
        self.assertEqual(results, [f"sentence {i}.".encode() for i in range(6)])
        self.assertLessEqual(live["peak"], 2)

    async def test_429_is_retried_not_dropped(self):
        calls = {"n": 0}

        async def handler(request):
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, json={"error": "concurrency"})
            return httpx.Response(200, content=ndjson(b"ok"))

        tts = InworldTTS("k", voice="v", model="m", retry_delays=(0.01, 0.01, 0.01),
                         client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        self.assertEqual(b"".join([c async for c in tts.synthesize("hi")]), b"ok")
        self.assertEqual(calls["n"], 3)

    async def test_gives_up_after_retries(self):
        async def handler(request):
            return httpx.Response(429, json={"error": "concurrency"})

        tts = InworldTTS("k", voice="v", model="m", retry_delays=(0.01,),
                         client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with self.assertRaisesRegex(RuntimeError, "429"):
            _ = [c async for c in tts.synthesize("hi")]


if __name__ == "__main__":
    unittest.main()


class FakeSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return json.dumps(self.messages.pop(0))

    async def send(self, data):
        self.sent.append(data)


def turn(text, *, eot, formatted, order=0, words=()):
    return {"type": "Turn", "transcript": text, "end_of_turn": eot, "turn_is_formatted": formatted,
            "turn_order": order, "words": [{"text": t, "start": a, "end": b} for t, a, b in words]}


class AssemblyAITurnTests(unittest.IsolatedAsyncioTestCase):
    MESSAGES = [turn("hi there", eot=False, formatted=False),
                turn("hi there can i order", eot=True, formatted=False, words=[("hi", 100, 300), ("order", 900, 1200)]),
                turn("Hi there, can I order?", eot=True, formatted=True, words=[("Hi", 100, 300), ("order?", 900, 1200)]),
                turn("yes", eot=True, formatted=True, order=1)]

    async def collect(self, stt):
        stt.ws = FakeSocket(self.MESSAGES)
        await stt.send_audio(b"\x00" * 16000 * 2)   # 2 s of 8 kHz audio
        return [e async for e in stt.events()]

    async def test_acts_on_end_of_turn_once_and_ignores_the_formatted_copy(self):
        from voiceagent.speech import AssemblyAISTT
        events = await self.collect(AssemblyAISTT("k"))
        self.assertEqual([(e.kind, e.text) for e in events],
                         [("partial", "hi there"), ("turn", "hi there can i order"), ("turn", "yes")])
        # The last word ended at 1.2 s into 2 s of sent audio, so speech ended about 0.8 s before the send.
        import time
        self.assertAlmostEqual(time.monotonic() - events[1].speech_end, 0.8, delta=0.1)
        self.assertIsNone(events[2].speech_end)

    async def test_can_wait_for_the_formatted_turn(self):
        from voiceagent.speech import AssemblyAISTT
        events = await self.collect(AssemblyAISTT("k", wait_for_formatting=True))
        self.assertEqual([(e.kind, e.text) for e in events],
                         [("partial", "hi there"), ("turn", "Hi there, can I order?"), ("turn", "yes")])

    async def test_speech_end_is_placed_on_our_clock(self):
        import time
        from voiceagent.speech import AssemblyAISTT
        stt = AssemblyAISTT("k")
        stt.ws = FakeSocket([])
        await stt.send_audio(b"\x00" * 16000 * 2)
        sent_at = time.monotonic()
        end = stt._speech_end({"words": [{"end": 1200}]})
        self.assertAlmostEqual(sent_at - end, 0.8, delta=0.05)
        self.assertIsNone(stt._speech_end({"words": []}))


class InworldWarmTests(unittest.IsolatedAsyncioTestCase):
    async def test_warm_opens_the_connection_without_synthesizing(self):
        seen = []

        async def handler(request):
            seen.append((request.method, request.url.path))
            return httpx.Response(405)

        tts = InworldTTS("k", voice="v", model="m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        await tts.warm()   # a non-200 answer is fine; only the connection matters
        self.assertEqual(seen, [("GET", "/tts/v1/voice:stream")])
