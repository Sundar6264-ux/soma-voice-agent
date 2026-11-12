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
