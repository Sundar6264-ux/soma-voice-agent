import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from voiceagent import twilio
from voiceagent.config import Settings
from voiceagent.main import create_app

from .fakes import DEMO, FakeLLM, demo_restaurant, say

BASE = "https://voice.example.test"


def settings(**overrides) -> Settings:
    tmp = Path(tempfile.mkdtemp(prefix="rva-app-"))
    s = Settings(
        restaurant_file=DEMO, db_path=tmp / "app.db", model="claude-opus-5-5", effort="low",
        public_base_url=BASE, twilio_auth_token="test-auth-token", allow_unsigned_twilio=False,
        admin_key="letmein", relay_secret="relay-secret", max_sessions=4,
        relay_attrs={"language": "en-US", "ttsProvider": "", "voice": "", "transcriptionProvider": "", "speechModel": ""},
    )
    return replace(s, **overrides)


class TwilioHelperTests(unittest.TestCase):
    def test_signature_round_trip_and_tamper(self):
        params = {"CallSid": "CA1", "From": "+12125550100", "To": "+18005550100"}
        sig = twilio.request_signature("tok", BASE + "/twilio/voice", params)
        self.assertTrue(twilio.valid_request("tok", BASE + "/twilio/voice", params, sig))
        self.assertFalse(twilio.valid_request("tok", BASE + "/twilio/voice", {**params, "From": "+1999"}, sig))
        self.assertFalse(twilio.valid_request("other", BASE + "/twilio/voice", params, sig))
        self.assertFalse(twilio.valid_request("", BASE + "/twilio/voice", params, sig))

    def test_twiml_is_valid_xml_with_escaped_url(self):
        body = twilio.connect_twiml(ws_url="wss://h/relay?call=CA1&token=abc", action_url=BASE + "/twilio/relay-done",
                                    restaurant=demo_restaurant(), relay_attrs={"language": "en-US", "voice": ""})
        root = ET.fromstring(body)
        relay = root.find("Connect/ConversationRelay")
        self.assertEqual(relay.get("url"), "wss://h/relay?call=CA1&token=abc")
        self.assertEqual(relay.get("language"), "en-US")
        self.assertIsNone(relay.get("voice"))  # empty settings are left to Twilio's defaults
        self.assertIn("Margherita Pizza", relay.get("hints"))
        self.assertTrue(relay.get("welcomeGreeting").startswith("Thanks for calling Juniper Street Kitchen"))


class AppTests(unittest.TestCase):
    def client(self, llm=None, **overrides):
        self.app = create_app(settings(**overrides), llm=llm or FakeLLM([]))
        return TestClient(self.app, base_url="https://testserver")

    def signed_post(self, client, path, params, token="test-auth-token"):
        sig = twilio.request_signature(token, BASE + path, params)
        return client.post(path, data=params, headers={"X-Twilio-Signature": sig})

    def test_voice_webhook_requires_signature(self):
        c = self.client()
        self.assertEqual(c.post("/twilio/voice", data={"CallSid": "CA1"}).status_code, 403)
        self.assertEqual(self.signed_post(c, "/twilio/voice", {"CallSid": "CA1"}, token="wrong").status_code, 403)
        r = self.signed_post(c, "/twilio/voice", {"CallSid": "CA1", "From": "+12125550100"})
        self.assertEqual(r.status_code, 200)
        url = ET.fromstring(r.text).find("Connect/ConversationRelay").get("url")
        self.assertEqual(url, f"wss://voice.example.test/relay?call=CA1&token={twilio.relay_token('relay-secret', 'CA1')}")

    def test_relay_done_routes(self):
        c = self.client()
        hang = self.signed_post(c, "/twilio/relay-done", {"CallSid": "CA1", "HandoffData": '{"reasonCode": "end"}'})
        self.assertIn("<Hangup/>", hang.text)
        dropped = self.signed_post(c, "/twilio/relay-done", {"CallSid": "CA1"})
        self.assertIn("disconnected", dropped.text)

    def test_relay_done_dials_staff_on_transfer(self):
        path = Path(tempfile.mkdtemp()) / "r.yaml"
        path.write_text(DEMO.read_text().replace('staff_phone: ""', 'staff_phone: "+12125550000"'))
        c = self.client(restaurant_file=path)
        r = self.signed_post(c, "/twilio/relay-done", {"CallSid": "CA1", "HandoffData": '{"reasonCode": "transfer"}'})
        self.assertIn("<Dial", r.text)
        self.assertIn("+12125550000", r.text)

    def test_relay_socket_rejects_bad_token(self):
        c = self.client()
        with self.assertRaises(WebSocketDisconnect):
            with c.websocket_connect("/relay?call=CA1&token=nope") as ws:
                ws.receive_json()

    def test_relay_socket_full_exchange(self):
        c = self.client(llm=FakeLLM([say("Hi, what can I get you?")]))
        token = twilio.relay_token("relay-secret", "CA7")
        with c.websocket_connect(f"/relay?call=CA7&token={token}") as ws:
            ws.send_json({"type": "setup", "callSid": "CA7", "from": "+12125550100"})
            ws.send_json({"type": "prompt", "voicePrompt": "hello", "last": True})
            self.assertEqual(ws.receive_json()["token"], "Hi, what can I get you?")
            self.assertTrue(ws.receive_json()["last"])
        calls = self.app.state.store.recent_calls("demo")
        self.assertEqual(calls[0]["call_sid"], "CA7")

    def test_admin_pages_need_login(self):
        c = self.client()
        self.assertEqual(c.get("/api/orders").status_code, 401)
        self.assertEqual(c.get("/board", follow_redirects=False).status_code, 303)
        self.assertEqual(c.get("/login?key=wrong").status_code, 401)
        c.get("/login?key=letmein", follow_redirects=False)
        r = c.get("/api/orders")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["orders"], [])
        self.assertEqual(c.post("/api/orders/99/status", json={"status": "ready"}).status_code, 404)
        self.assertEqual(c.post("/api/orders/1/status", json={"status": "eaten"}).status_code, 400)

    def test_login_disabled_without_admin_key(self):
        c = self.client(admin_key="")
        self.assertEqual(c.get("/login?key=").status_code, 401)

    def test_sim_socket_needs_login(self):
        c = self.client()
        with self.assertRaises(WebSocketDisconnect):
            with c.websocket_connect("/sim") as ws:
                ws.receive_json()


if __name__ == "__main__":
    unittest.main()
