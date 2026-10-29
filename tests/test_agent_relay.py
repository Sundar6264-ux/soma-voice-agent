import asyncio
import json
import unittest

import anthropic
import httpx

from voiceagent.agent import TRANSFER_LINE, Conversation
from voiceagent.relay import RelaySession
from voiceagent.tools import CallInfo

from .fakes import OPEN_TIME, FakeLLM, Step, as_plain, call, demo_restaurant, say, temp_store

ORDER_SCRIPT = [
    call(("get_store_status", {}, "t1"), text="Sure, one moment."),
    call(("add_item", {"item_id": "margherita", "quantity": 1, "option_ids": ["size_14"]}, "t2")),
    say("Got it, a 14-inch Margherita. Anything else?"),
    # caller: "that's all, name is Dana"
    call(("set_customer", {"name": "Dana", "phone": ""}, "t3"), ("review_order", {}, "t4")),
    say("That's one 14-inch Margherita, total $19.44. Is that right?"),
    # caller: "yes"
    call(("place_order", {}, "t5")),
    say("Your order number is 1, ready at 6:20 PM. Anything else?"),
    # caller: "no, bye"
    call(("end_call", {}, "t6"), text="Thanks for calling, goodbye!"),
]


def assert_append_only(test, requests):
    """Each request's history must start with the previous request's history, unchanged."""
    plain = [as_plain(r) for r in requests]
    for before, after in zip(plain, plain[1:]):
        test.assertEqual(after[:len(before)], before, "history was edited between requests")


def assert_well_formed(test, messages):
    plain = as_plain(messages)
    test.assertEqual(plain[0]["role"], "user")
    for i, m in enumerate(plain):
        if m["role"] == "system":
            test.assertTrue(i > 0 and plain[i - 1]["role"] == "user")
            test.assertTrue(i == len(plain) - 1 or plain[i + 1]["role"] == "assistant")
        if m["role"] == "assistant":
            uses = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if uses:
                nxt = plain[i + 1]
                test.assertEqual(nxt["role"], "user")
                test.assertEqual([b["tool_use_id"] for b in nxt["content"]], uses)


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    def make(self, script, **overrides):
        self.store = temp_store()
        self.llm = FakeLLM(script)
        return Conversation(demo_restaurant(**overrides), self.store, self.llm,
                            CallInfo("CA123", "+12125550100"), clock=lambda: OPEN_TIME)

    async def speak(self, conv, text):
        conv.caller_said(text)
        return "".join([c async for c in conv.respond()])

    async def test_full_order(self):
        conv = self.make(ORDER_SCRIPT)
        r1 = await self.speak(conv, "Hi, can I get a large margherita?")
        self.assertEqual(r1, "Sure, one moment. Got it, a 14-inch Margherita. Anything else?")
        await self.speak(conv, "That's all. The name is Dana.")
        await self.speak(conv, "Yes.")
        r4 = await self.speak(conv, "No, bye.")
        self.assertEqual(r4, "Thanks for calling, goodbye!")
        self.assertEqual(conv.toolbox.control.action, "end")
        self.assertEqual(conv.outcome(), "order_placed")
        orders = self.store.recent_orders("demo")
        self.assertEqual(orders[0]["customer_name"], "Dana")
        self.assertEqual(orders[0]["customer_phone"], "+12125550100")
        self.assertEqual(orders[0]["total_cents"], 1944)
        assert_append_only(self, self.llm.requests)
        assert_well_formed(self, conv.messages)
        # Caller ID reaches the model once, as a mid-conversation system message.
        first = as_plain(self.llm.requests[0])
        self.assertEqual(first[1]["role"], "system")
        self.assertIn("+12125550100", first[1]["content"])

    async def test_interrupted_reply_is_dropped_and_noted(self):
        conv = self.make([
            say("Sure."),
            say("We have three pizzas: ", "Margherita, ", "Hot Honey Pepperoni, ", "and Wild Mushroom.", delay=0.05),
            say("Okay, a Margherita. What size?"),
        ])
        await self.speak(conv, "Hello")
        conv.caller_said("What pizzas do you have?")
        heard = []

        async def consume():
            async for chunk in conv.respond():
                heard.append(chunk)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.08)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        conv.interrupted("".join(heard))
        await self.speak(conv, "Just a margherita")
        last_user = as_plain(self.llm.requests[-1])[-1]
        self.assertEqual(last_user["role"], "user")
        self.assertIn("They only heard", last_user["content"][0]["text"])
        self.assertEqual(last_user["content"][1]["text"], "Just a margherita")
        # The cut-off reply never entered history.
        assistant_texts = [b["text"] for m in as_plain(conv.messages) if m["role"] == "assistant"
                           for b in m["content"] if b["type"] == "text"]
        self.assertNotIn("We have three pizzas: Margherita, Hot Honey Pepperoni, and Wild Mushroom.", assistant_texts)
        assert_append_only(self, self.llm.requests)
        assert_well_formed(self, conv.messages)

    async def test_first_reply_interrupted_keeps_system_message_valid(self):
        conv = self.make([say("Hi there, ", "what can I get you?", delay=0.05), say("Sure, a Caesar. Add chicken?")])
        conv.caller_said("Hi")
        task = asyncio.create_task(anext(aiter(conv.respond())))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        conv.interrupted("")
        await self.speak(conv, "a caesar salad please")
        assert_well_formed(self, conv.messages)
        msgs = as_plain(conv.messages)
        self.assertEqual([m["role"] for m in msgs], ["user", "system", "assistant"])
        self.assertEqual(msgs[0]["content"][-1]["text"], "a caesar salad please")

    async def test_bad_tool_input_becomes_error_result(self):
        conv = self.make([
            call(("add_item", {"item_id": "margherita", "quantity": 1, "option_ids": []}, "t1")),
            say("What size would you like, 10 or 14 inch?"),
        ])
        r = await self.speak(conv, "a margherita")
        self.assertEqual(r, "What size would you like, 10 or 14 inch?")
        result = as_plain(conv.messages)[3]["content"][0]
        self.assertTrue(result["is_error"])
        self.assertIn("size", result["content"])

    async def test_unparseable_tool_json_retries_the_turn(self):
        conv = self.make([Step(raise_exc=ValueError("bad json")), say("Sure, what can I get you?")])
        self.assertEqual(await self.speak(conv, "hi"), "Sure, what can I get you?")
        self.assertEqual(len(self.llm.requests), 2)

    async def test_api_error_is_spoken_not_raised(self):
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        err = anthropic.APIConnectionError(request=req)
        conv = self.make([Step(raise_exc=err)])
        r = await self.speak(conv, "hi")
        self.assertIn("trouble", r)
        self.assertIsNone(conv.toolbox.control)  # no staff line in the demo config

    async def test_refusal_is_not_added_to_history(self):
        conv = self.make([Step(stop_reason="refusal")])
        r = await self.speak(conv, "something odd")
        self.assertIn("can't help with that", r)
        self.assertNotIn("assistant", [m["role"] for m in conv.messages])

    async def test_transfer_speaks_fixed_line(self):
        conv = self.make([call(("transfer_to_staff", {"reason": "complaint"}, "t1"))], staff_phone="+12125550000")
        self.assertEqual(await self.speak(conv, "I want a manager"), TRANSFER_LINE)
        self.assertEqual(conv.outcome(), "transferred")


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sent = []
        self.store = temp_store()
        self.slept = []

    def relay(self, script, expected=None, **overrides):
        restaurant = demo_restaurant(**overrides)
        self.llm = FakeLLM(script)

        async def send(msg):
            self.sent.append(msg)

        async def fake_sleep(s):
            self.slept.append(s)

        make = lambda c: Conversation(restaurant, self.store, self.llm, c, clock=lambda: OPEN_TIME)
        return RelaySession(send, make, self.store, restaurant.id, expected_call_sid=expected, sleep=fake_sleep)

    async def wait_reply(self, relay):
        await relay.reply_task
        if relay.finish_task:
            await relay.finish_task

    async def test_tokens_then_last(self):
        relay = self.relay([say("Hi! ", "What can I get you?")], expected="CA9")
        await relay.handle({"type": "setup", "callSid": "CA9", "from": "+12125550100"})
        await relay.handle({"type": "prompt", "voicePrompt": "hello", "last": True})
        await self.wait_reply(relay)
        self.assertEqual(self.sent, [
            {"type": "text", "token": "Hi! ", "last": False},
            {"type": "text", "token": "What can I get you?", "last": False},
            {"type": "text", "token": "", "last": True},
        ])
        await relay.close()
        calls = self.store.recent_calls("demo")
        self.assertEqual(calls[0]["outcome"], "no_order")
        self.assertEqual(calls[0]["transcript"][0], {"role": "caller", "text": "hello"})

    async def test_mismatched_call_sid_ends(self):
        relay = self.relay([], expected="CA9")
        await relay.handle({"type": "setup", "callSid": "CAother"})
        self.assertEqual(self.sent[0]["type"], "end")
        self.assertIsNone(relay.conv)

    async def test_interrupt_cancels_and_records_what_was_heard(self):
        relay = self.relay([say("Our pizzas are ", "Margherita, ", "and more.", delay=0.05), say("Sure, a Margherita.")])
        await relay.handle({"type": "setup", "callSid": "CA1", "from": ""})
        await relay.handle({"type": "prompt", "voicePrompt": "what pizzas?"})
        await asyncio.sleep(0.07)
        await relay.handle({"type": "interrupt", "utteranceUntilInterrupt": "Our pizzas are"})
        self.assertTrue(relay.reply_task.done())
        await relay.handle({"type": "prompt", "voicePrompt": "a margherita"})
        await self.wait_reply(relay)
        # The first reply never landed, so this turn folds into the first user message.
        users = [m for m in as_plain(self.llm.requests[-1]) if m["role"] == "user"]
        texts = [b["text"] for b in users[-1]["content"]]
        self.assertEqual(texts, ["what pizzas?", '[The caller interrupted your last reply. They only heard: "Our pizzas are"]',
                                 "a margherita"])
        assert_well_formed(self, relay.conv.messages)

    async def test_end_call_sends_end_after_speech(self):
        relay = self.relay([call(("end_call", {}, "t1"), text="Thanks, goodbye!")])
        await relay.handle({"type": "setup", "callSid": "CA2", "from": ""})
        await relay.handle({"type": "prompt", "voicePrompt": "that's all, bye"})
        await self.wait_reply(relay)
        end = self.sent[-1]
        self.assertEqual(end["type"], "end")
        self.assertEqual(json.loads(end["handoffData"])["reasonCode"], "end")
        self.assertEqual(len(self.slept), 1)
        # Talking over the goodbye does not start a new reply.
        await relay.handle({"type": "prompt", "voicePrompt": "wait"})
        self.assertEqual(self.sent[-1]["type"], "end")

    async def test_dtmf_zero_transfers_when_staff_line_exists(self):
        relay = self.relay([], staff_phone="+12125550000")
        await relay.handle({"type": "setup", "callSid": "CA3", "from": ""})
        await relay.handle({"type": "dtmf", "digit": "0"})
        await relay.finish_task
        self.assertEqual(json.loads(self.sent[-1]["handoffData"])["reasonCode"], "transfer")

    async def test_dtmf_zero_ignored_without_staff_line(self):
        relay = self.relay([])
        await relay.handle({"type": "setup", "callSid": "CA4", "from": ""})
        await relay.handle({"type": "dtmf", "digit": "0"})
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
