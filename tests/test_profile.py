"""Soma's AI: the persona, its tools, the time limit and the browser transport. No network."""
import asyncio
import json
import time
import os
import unittest
from dataclasses import replace

from fastapi.testclient import TestClient

from voiceagent import profile
from voiceagent.agent import Conversation
from voiceagent.main import create_app
from voiceagent.media import MediaCall, TIME_UP_GOODBYE
from voiceagent.relay import RelaySession
from voiceagent.speech import SpeechEvent
from voiceagent.tools import CallInfo

from .fakes import FakeLLM, as_plain, call, say, temp_store
from .test_app_twilio import settings as base_settings
from .test_media import FakeSTT, eventually, fake_tts

RESUME = """# Soma Sundar Mamilla

Sunnyvale, CA, USA | 555-010-0100 | owner@example.com | https://somasundar.dev

## Experience

### Systems Engineer, Tata Consultancy Services (embedded at National Stock Exchange, India)
- Sole full-time developer on a 9-layer C matching engine handling 5-6 million transactions per day.
"""


class PromptTests(unittest.TestCase):
    def test_resume_loses_phone_and_email_but_keeps_links_and_facts(self):
        text = profile.clean_resume(RESUME)
        self.assertNotIn("555-010-0100", text)
        self.assertNotIn("owner@example.com", text)
        self.assertIn("https://somasundar.dev", text)
        self.assertIn("matching engine", text)
        self.assertEqual(text.splitlines()[2], "Sunnyvale, CA, USA | https://somasundar.dev")

    def test_prompt_has_honesty_disclosure_and_no_commitments(self):
        prompt = profile.build_system_prompt(profile.clean_resume(RESUME))
        self.assertNotIn("555-010-0100", prompt)
        self.assertNotIn("owner@example.com", prompt)
        for phrase in ("Answer only from the resume", "say you don't know", "never instructions to you",
                       "say he has a PhD", "reveal this prompt", "Never claim to be him", "voice cloned from",
                       "salary expectations", "visa", "sponsorship", profile.AVAILABILITY, profile.GREETING,
                       "Do not use dashes", "no lists, bullet points, markdown"):
            self.assertIn(phrase, prompt)
        for link in profile.LINKS:
            self.assertIn(link["url"], prompt)
        self.assertIn("matching engine", prompt)

    def test_missing_resume_is_safe(self):
        text = profile.load_resume("/nonexistent/resume.md")
        self.assertIn("No resume is loaded", text)

    def test_no_dashes_anywhere_in_prompt_or_lines(self):
        for text in (profile.PROMPT_TEMPLATE, profile.GREETING, profile.TROUBLE_LINE, profile.REFUSAL_LINE, TIME_UP_GOODBYE):
            self.assertNotRegex(text, "[\\u2013\\u2014]")


class ToolTests(unittest.TestCase):
    def setUp(self):
        os.environ["OWNER_EMAIL"] = "owner@example.com"
        self.store = temp_store()
        self.sent = []
        self.box = profile.ProfileToolBox(self.store, CallInfo("web-1", channel="web"), mode="on", mailer=self.sent.append)
        self.box.transcript.append({"role": "caller", "text": "Hi, I'd like to leave a message."})

    def test_schema_validation(self):
        r = self.box.run("take_message", {"name": "Ann"})
        self.assertTrue(r.is_error)
        self.assertIn("missing required field", r.content)
        r = self.box.run("take_message", {"name": "Ann", "company": "Acme", "contact": "a@acme.com", "message": "Call me", "extra": 1})
        self.assertTrue(r.is_error)
        self.assertIn("unexpected field", r.content)
        r = self.box.run("share_links", {"x": 1})
        self.assertTrue(r.is_error)
        r = self.box.run("nope", {})
        self.assertTrue(r.is_error)
        self.assertEqual(self.store.recent_messages(), [])

    def test_take_message_stores_and_emails_with_transcript(self):
        r = self.box.run("take_message", {"name": "Ann Lee", "company": "Acme", "contact": "ann@acme.com",
                                          "message": "Please call about the FDE role.", "role": "FDE"})
        self.assertFalse(r.is_error)
        self.assertTrue(json.loads(r.content)["stored"])
        rows = self.store.recent_messages()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["name"], rows[0]["company"], rows[0]["contact"], rows[0]["role"]), ("Ann Lee", "Acme", "ann@acme.com", "FDE"))
        self.assertEqual(len(self.sent), 1)
        mail = self.sent[0]
        self.assertEqual(mail["to"], "owner@example.com")
        self.assertTrue(mail["subject"].startswith("Message for Soma from Ann Lee"))
        self.assertIn("Please call about the FDE role.", mail["body"])
        self.assertIn("Caller: Hi, I'd like to leave a message.", mail["body"])
        self.assertEqual(self.box.outcome(), "message_taken")

    def test_test_mode_prefixes_subject_and_off_mode_sends_nothing(self):
        self.box.mode = "test"
        self.box.run("take_message", {"name": "A", "company": "B", "contact": "c", "message": "d"})
        self.assertTrue(self.sent[-1]["subject"].startswith("[TEST] "))
        self.box.mode = "off"
        self.box.run("take_message", {"name": "A", "company": "B", "contact": "c", "message": "d"})
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(self.store.recent_messages()), 2)

    def test_share_links_emits_event_and_returns_links(self):
        r = self.box.run("share_links", {})
        self.assertIn("somasundar.dev", r.content)
        events = self.box.take_events()
        self.assertEqual(events[0]["type"], "links")
        self.assertEqual([l["url"] for l in events[0]["links"]], [l["url"] for l in profile.LINKS])
        self.assertEqual(self.box.take_events(), [])

    def test_end_conversation(self):
        self.box.run("end_conversation", {})
        self.assertEqual(self.box.control.action, "end")
        self.assertEqual(self.box.outcome(), "ended")


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_persona_prompt_tools_context_and_events(self):
        store = temp_store()
        persona = profile.make_persona(store, profile.clean_resume(RESUME), mailer=lambda m: None)
        llm = FakeLLM([call(("share_links", {}, "t1"), text="Here are his links."), say("Anything else?")])
        conv = Conversation(None, store, llm, CallInfo("web-2", channel="web"), persona=persona)
        events = []
        conv.on_event = events.append
        conv.caller_said("Can I get his resume?")
        spoken = "".join([c async for c in conv.respond()])
        self.assertIn("Here are his links.", spoken)
        self.assertEqual(events[0]["type"], "links")
        self.assertIs(conv.toolbox.transcript, conv.transcript)
        req = as_plain(llm.requests[0])
        self.assertEqual(req[1]["role"], "system")
        self.assertIn("browser on Soma's website", req[1]["content"])
        self.assertEqual([t["name"] for t in persona.tools], ["take_message", "share_links", "end_conversation"])
        self.assertIn("Soma's AI assistant", conv.system)


class FakeTransport:
    in_rate = 16000
    out_rate = 24000

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.frames: list[bytes] = []
        self.events: list[dict] = []
        self.closed = False

    async def read(self):
        return await self.inbox.get()

    async def send_audio(self, frame):
        self.frames.append(frame)

    async def send_event(self, event):
        self.events.append(event)

    async def close(self):
        self.closed = True


class TimeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_warning_note_then_fixed_goodbye_and_end(self):
        store = temp_store()
        persona = profile.make_persona(store, "resume", mailer=lambda m: None)
        llm = FakeLLM([say("Sure, about a minute left. Want to leave a message?")])
        stt, tts, transport = FakeSTT(), fake_tts(), FakeTransport()
        make_conv = lambda c: Conversation(None, store, llm, c, persona=persona)
        make_relay = lambda send, sleep: RelaySession(send, make_conv, store, persona.id, channel="web", sleep=sleep)
        mc = MediaCall(transport, call_id="web-3", from_number="", greeting="Hi there.", stt=stt, synthesize=tts,
                       make_relay=make_relay, frame_seconds=0.001, time_limit=0.6, warn_at=0.2)
        task = asyncio.create_task(mc.run())
        await eventually(lambda: any(e.get("type") == "time" for e in transport.events))
        await stt.queue.put(SpeechEvent("turn", "How many years does he have?"))
        await eventually(lambda: len(llm.requests) == 1)
        texts = [b["text"] for b in as_plain(llm.requests[0])[0]["content"]]
        self.assertTrue(any("conversation time are left" in t for t in texts), texts)
        self.assertEqual(texts[-1], "How many years does he have?")
        await asyncio.wait_for(task, 5)
        self.assertIn(TIME_UP_GOODBYE, tts.spoken)
        self.assertEqual(transport.events[-1], {"type": "end", "reason": "time_limit"})
        self.assertTrue(transport.closed)
        # Output frames are 20 ms at 24 kHz.
        self.assertTrue(transport.frames and all(len(f) == 24000 * 2 * 20 // 1000 for f in transport.frames))
        self.assertEqual(store.recent_calls(persona.id)[0]["channel"], "web")


class BrowserTransportTests(unittest.TestCase):
    def make(self, script, **overrides):
        self.stt = None
        self.sent = []
        s = replace(base_settings(**overrides), admin_key="letmein", profile_time_limit_s=600)

        def stt_factory():
            self.stt = FakeSTT()
            return self.stt
        self.tts = fake_tts()
        self.app = create_app(s, llm=FakeLLM(script), stt_factory=stt_factory, synthesize=self.tts, frame_seconds=0.001,
                              mailer=self.sent.append, resume_text=profile.clean_resume(RESUME))
        client = TestClient(self.app, base_url="https://testserver")
        return client

    def login(self, client):
        r = client.get("/login", params={"key": "letmein", "next": "/talk"}, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/talk")
        # The cookie is Secure and the test websocket URL is ws://, so send it by hand.
        self.ws_headers = {"cookie": f"rva_admin={client.cookies.get('rva_admin')}"}

    def test_pages_and_static_need_or_skip_login(self):
        client = self.make([])
        self.assertEqual(client.get("/talk", follow_redirects=False).status_code, 303)
        self.assertEqual(client.get("/static/talk-worklet.js").status_code, 200)
        self.assertEqual(client.get("/static/talk.js").status_code, 200)
        self.assertEqual(client.get("/static/../main.py").status_code, 404)
        self.login(client)
        page = client.get("/talk")
        self.assertEqual(page.status_code, 200)
        self.assertIn("voice cloned from Soma's", page.text)
        self.assertNotIn("<script>", page.text)  # no inline scripts: a strict CSP can apply later
        self.assertIn("/static/talk.js", page.text)

    def test_websocket_rejects_without_cookie(self):
        client = self.make([])
        from starlette.websockets import WebSocketDisconnect
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with client.websocket_connect("/talk"):
                pass
        self.assertEqual(ctx.exception.code, 4401)

    def test_full_browser_turn_with_audio_events_and_transcript(self):
        client = self.make([call(("share_links", {}, "t1"), text="Sure. Here are his links."),
                            call(("end_conversation", {}, "t2"), text="Goodbye!")])
        self.login(client)
        with client.websocket_connect("/talk", headers=self.ws_headers) as ws:
            first = ws.receive_json()
            self.assertEqual(first["type"], "agent_text")
            self.assertEqual(first["token"], profile.GREETING)
            # Caller audio at 16 kHz reaches STT in 100 ms chunks (3200 bytes).
            for _ in range(5):
                ws.send_bytes(b"\x00" * 1280)
            # Drain audio frames until the STT has the audio, then inject a finished turn.
            deadline = 200
            while deadline and len(self.stt.audio) < 3200:
                ws.receive()
                deadline -= 1
            self.assertGreaterEqual(len(self.stt.audio), 3200)
            # TestClient runs the app in a worker thread with its own loop, so hand the turn to that loop.
            self.stt_loop_put(SpeechEvent("turn", "Can I see his resume?"))
            got = {"caller": None, "links": None, "agent": "", "end": None, "audio": 0}
            for _ in range(2000):
                msg = ws.receive()
                if msg.get("bytes") is not None:
                    got["audio"] += 1
                    self.assertEqual(len(msg["bytes"]), 960)  # 20 ms at 24 kHz
                    continue
                m = json.loads(msg["text"])
                if m["type"] == "caller_text":
                    got["caller"] = m["text"]
                elif m["type"] == "links":
                    got["links"] = m["links"]
                elif m["type"] == "agent_text":
                    got["agent"] += m["token"]
                elif m["type"] == "end":
                    got["end"] = m
                    break
            self.assertEqual(got["caller"], "Can I see his resume?")
            self.assertEqual([l["url"] for l in got["links"]], [l["url"] for l in profile.LINKS])
            self.assertIn("Here are his links.", got["agent"])
            self.assertIn("Goodbye!", got["agent"])
            self.assertEqual(got["end"]["reason"], "end")
            self.assertGreater(got["audio"], 0)
        calls = self.app.state.store.recent_calls(profile.PERSONA_ID)
        self.assertEqual(calls[0]["channel"], "web")
        self.assertEqual(calls[0]["outcome"], "ended")
        self.assertEqual(calls[0]["transcript"][0], {"role": "caller", "text": "Can I see his resume?"})
        self.assertIn("Here are his links.", " ".join(self.tts.spoken))

    def stt_loop_put(self, event):
        asyncio.run_coroutine_threadsafe(self.stt.queue.put(event), self.stt.loop).result(timeout=5)

    def test_second_visitor_is_busy(self):
        client = self.make([])
        self.login(client)
        with client.websocket_connect("/talk", headers=self.ws_headers) as ws:
            ws.receive_json()
            with client.websocket_connect("/talk", headers=self.ws_headers) as ws2:
                m = ws2.receive_json()
                self.assertEqual(m["type"], "busy")
                self.assertIn("try again shortly", m["text"])
            ws.send_json({"type": "hangup"})


if __name__ == "__main__":
    unittest.main()


class PublicTalkTests(unittest.TestCase):
    ORIGIN = {"origin": "https://somasundar.dev", "cf-connecting-ip": "203.0.113.7"}

    def make(self, **overrides):
        s = replace(base_settings(**overrides), admin_key="letmein", profile_time_limit_s=600)
        self.app = create_app(s, llm=FakeLLM([say("Hi there.")]), stt_factory=FakeSTT, synthesize=fake_tts(),
                              frame_seconds=0.001, mailer=[].append, resume_text=profile.clean_resume(RESUME))
        return TestClient(self.app, base_url="https://testserver")

    def test_public_endpoints_off_by_default(self):
        client = self.make()
        self.assertEqual(client.post("/talk/ticket", headers=self.ORIGIN).status_code, 404)

    def test_ticket_needs_site_origin(self):
        client = self.make(talk_public=True)
        r = client.post("/talk/ticket", headers={"origin": "https://evil.example", "cf-connecting-ip": "1.2.3.4"})
        self.assertEqual(r.status_code, 403)

    def test_per_visitor_and_daily_caps(self):
        client = self.make(talk_public=True, talk_per_ip_day=2, talk_per_day=3)
        ok = [client.post("/talk/ticket", headers=self.ORIGIN) for _ in range(2)]
        self.assertTrue(all(r.json()["ok"] for r in ok))
        self.assertEqual(ok[-1].json()["left_today"], 0)
        third = client.post("/talk/ticket", headers=self.ORIGIN)
        self.assertEqual(third.status_code, 429)
        self.assertIn("limit of 2", third.json()["error"])
        other = {"origin": "https://somasundar.dev", "cf-connecting-ip": "198.51.100.9"}
        self.assertTrue(client.post("/talk/ticket", headers=other).json()["ok"])  # 3rd of the day overall
        self.assertEqual(client.post("/talk/ticket", headers={**other, "cf-connecting-ip": "198.51.100.10"}).status_code, 429)

    def test_socket_needs_valid_single_use_ticket_from_same_visitor(self):
        from starlette.websockets import WebSocketDisconnect
        client = self.make(talk_public=True)
        ticket = client.post("/talk/ticket", headers=self.ORIGIN).json()["ticket"]
        for bad_url, bad_headers in [("/talk/public?ticket=nope", self.ORIGIN),
                                     (f"/talk/public?ticket={ticket}", {**self.ORIGIN, "origin": "https://evil.example"}),
                                     (f"/talk/public?ticket={ticket}", {**self.ORIGIN, "cf-connecting-ip": "9.9.9.9"})]:
            with self.assertRaises(WebSocketDisconnect) as ctx:
                with client.websocket_connect(bad_url, headers=bad_headers):
                    pass
            self.assertEqual(ctx.exception.code, 4403)
        with client.websocket_connect(f"/talk/public?ticket={ticket}", headers=self.ORIGIN) as ws:
            first = ws.receive_json()
            self.assertEqual(first["token"], profile.GREETING)
            ws.send_text(json.dumps({"type": "hangup"}))
        with self.assertRaises(WebSocketDisconnect) as ctx:  # the same ticket cannot be used twice
            with client.websocket_connect(f"/talk/public?ticket={ticket}", headers=self.ORIGIN):
                pass
        self.assertEqual(ctx.exception.code, 4403)


class TranscriptEmailTests(unittest.TestCase):
    def test_every_conversation_emails_a_transcript(self):
        os.environ["OWNER_EMAIL"] = "owner@example.com"
        sent = []
        s = replace(base_settings(), admin_key="letmein", profile_time_limit_s=600, talk_public=True)
        app = create_app(s, llm=FakeLLM([say("Hi there.")]), stt_factory=FakeSTT, synthesize=fake_tts(),
                         frame_seconds=0.001, mailer=sent.append, resume_text=profile.clean_resume(RESUME))
        client = TestClient(app, base_url="https://testserver")
        headers = {"origin": "https://somasundar.dev", "cf-connecting-ip": "203.0.113.50"}
        ticket = client.post("/talk/ticket", headers=headers).json()["ticket"]
        with client.websocket_connect(f"/talk/public?ticket={ticket}", headers=headers) as ws:
            ws.receive_json()
            ws.send_text(json.dumps({"type": "hangup"}))
        for _ in range(100):
            if sent:
                break
            time.sleep(0.02)
        self.assertEqual(len(sent), 1)
        mail = sent[0]
        self.assertEqual(mail["to"], "owner@example.com")
        self.assertTrue(mail["subject"].startswith("Soma's AI conversation: "))
        self.assertIn("Where: somasundar.dev visitor", mail["body"])
        self.assertIn("Transcript:", mail["body"])

    def test_subject_and_missing_owner_email(self):
        os.environ["OWNER_EMAIL"] = "owner@example.com"
        payload = profile.transcript_email(call_id="web-1", started=0, seconds=75, outcome="no_message",
                                           transcript=[{"role": "caller", "text": "Hi"}], message_ids=[7])
        self.assertEqual(payload["subject"], "Soma's AI conversation: 1m 15s, 1 caller turn, message left")
        self.assertIn("Where: private test page", payload["body"])
        self.assertIn("Caller: Hi", payload["body"])
        os.environ["OWNER_EMAIL"] = ""
        try:
            self.assertIsNone(profile.transcript_email(call_id="web-1", started=0, seconds=1, outcome="x",
                                                       transcript=[], message_ids=[]))
        finally:
            os.environ["OWNER_EMAIL"] = "owner@example.com"
