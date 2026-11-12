import json
import unittest
from decimal import Decimal

from voiceagent.order import Order, OrderError, normalize_phone
from voiceagent.tools import TOOLS, CallInfo, ToolBox, validate_input

from .fakes import CLOSED_TIME, OPEN_TIME, demo_restaurant, temp_store


class OrderTests(unittest.TestCase):
    def setUp(self):
        self.order = Order(demo_restaurant())

    def test_prices_tax_and_total_in_cents(self):
        self.order.add("margherita", 2, ["size_14", "top_mushroom"])   # (13 + 5 + 1.5) * 2 = 39.00
        self.order.add("soda", 1, ["soda_cola"])                       # 2.50
        s = self.order.summary()
        self.assertEqual(s["subtotal"], "$41.50")
        self.assertEqual(s["tax"], "$3.32")                            # 8% of 41.50 = 3.32
        self.assertEqual(s["total"], "$44.82")
        self.assertEqual(s["item_count"], 3)

    def test_tax_rounds_half_up(self):
        # 8% never lands on half a cent, so use a rate that does: 1000 * 8.75% = 87.5.
        o = Order(demo_restaurant(tax_rate=Decimal("0.0875")))
        o.add("sparkling_water", 1, [])
        o.add("tiramisu", 1, [])
        o.add("soda", 1, ["soda_cola"])  # 300 + 800 + 250 = 1350 -> 118.125 -> 118
        self.assertEqual(o.tax_cents, 118)
        o.remove(3)                      # 1100 -> 96.25 -> 96
        self.assertEqual(o.tax_cents, 96)
        self.assertEqual(o.restaurant.tax_cents(1000), 88)

    def test_required_choice_missing(self):
        with self.assertRaisesRegex(OrderError, "size"):
            self.order.add("margherita", 1, [])

    def test_unknown_item_and_option(self):
        with self.assertRaisesRegex(OrderError, "not on the menu"):
            self.order.add("lasagna", 1, [])
        with self.assertRaisesRegex(OrderError, "not an option"):
            self.order.add("margherita", 1, ["size_10", "sauce_buffalo"])

    def test_option_requires_other_option(self):
        with self.assertRaisesRegex(OrderError, "only available with small 10-inch"):
            self.order.add("margherita", 1, ["size_14", "crust_gf"])
        line = self.order.add("margherita", 1, ["crust_gf", "size_10"])
        self.assertEqual(line.option_ids, ("size_10", "crust_gf"))  # menu order, not caller order

    def test_group_max_and_duplicates(self):
        with self.assertRaisesRegex(OrderError, "at most 1"):
            self.order.add("margherita", 1, ["size_10", "size_14"])
        with self.assertRaisesRegex(OrderError, "twice"):
            self.order.add("margherita", 1, ["size_10", "top_olive", "top_olive"])

    def test_quantity_limits(self):
        with self.assertRaises(OrderError):
            self.order.add("tiramisu", 0, [])
        with self.assertRaises(OrderError):
            self.order.add("tiramisu", 21, [])

    def test_update_and_remove(self):
        line = self.order.add("margherita", 1, ["size_10"])
        self.order.update(line.line_id, quantity=3, option_ids=["size_14"])
        self.assertEqual(self.order.summary()["subtotal"], "$54.00")
        self.order.remove(line.line_id)
        self.assertEqual(self.order.lines, [])
        with self.assertRaises(OrderError):
            self.order.remove(line.line_id)

    def test_review_gate(self):
        self.order.add("tiramisu", 1, [])
        self.order.set_customer("Sam", "(212) 555-0100")
        with self.assertRaisesRegex(OrderError, "changed since"):
            self.order.check_ready()
        self.order.review()
        self.order.check_ready()
        self.order.add("soda", 1, ["soda_cola"])
        with self.assertRaisesRegex(OrderError, "changed since"):
            self.order.check_ready()

    def test_phone_normalization(self):
        self.assertEqual(normalize_phone("212-555-0100"), "+12125550100")
        self.assertEqual(normalize_phone("1 212 555 0100"), "+12125550100")
        self.assertEqual(normalize_phone("+44 20 7946 0958"), "+442079460958")
        with self.assertRaises(OrderError):
            normalize_phone("five five five")


class ToolTests(unittest.TestCase):
    def make(self, when=OPEN_TIME, **overrides):
        self.store = temp_store()
        return ToolBox(demo_restaurant(**overrides), self.store, CallInfo("CA1", "+12125550100"), clock=lambda: when)

    def test_schemas_are_deterministic_and_streaming(self):
        names = [t["name"] for t in TOOLS]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(t["eager_input_streaming"] for t in TOOLS))
        self.assertTrue(all(t["input_schema"]["additionalProperties"] is False for t in TOOLS))

    def test_validate_input(self):
        self.assertIsNone(validate_input("add_item", {"item_id": "soda", "quantity": 1, "option_ids": []}))
        self.assertIn("missing", validate_input("add_item", {"item_id": "soda"}))
        self.assertIn("integer", validate_input("add_item", {"item_id": "soda", "quantity": "2", "option_ids": []}))
        self.assertIn("integer", validate_input("add_item", {"item_id": "soda", "quantity": True, "option_ids": []}))
        self.assertIn("at least", validate_input("add_item", {"item_id": "soda", "quantity": 0, "option_ids": []}))
        self.assertIn("unexpected", validate_input("review_order", {"x": 1}))
        self.assertIn("object", validate_input("review_order", "not json"))
        self.assertIn("unknown tool", validate_input("delete_db", {}))

    def test_invalid_input_is_an_error_result_not_a_crash(self):
        box = self.make()
        res = box.run("add_item", {"item_id": "soda", "quantity": "lots", "option_ids": []})
        self.assertTrue(res.is_error)
        self.assertIn("INVALID_INPUT", res.content)
        self.assertEqual(box.order.lines, [])

    def test_full_order_flow_writes_to_store(self):
        box = self.make()
        status = json.loads(box.run("get_store_status", {}).content)
        self.assertTrue(status["accepting_orders"])
        self.assertFalse(box.run("add_item", {"item_id": "wings", "quantity": 1, "option_ids": ["sauce_hot_honey"]}).is_error)
        res = box.run("place_order", {})
        self.assertTrue(res.is_error)  # no customer, not reviewed
        box.run("set_customer", {"name": "Priya", "phone": ""})  # falls back to caller ID
        self.assertEqual(box.order.customer_phone, "+12125550100")
        review = json.loads(box.run("review_order", {}).content)
        self.assertEqual(review["total"], "$15.12")
        placed = json.loads(box.run("place_order", {}).content)
        self.assertEqual(placed["total"], "$15.12")
        self.assertEqual(placed["ready_at"], "6:20 PM")
        orders = self.store.recent_orders("demo")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["total_cents"], 1512)
        self.assertEqual(orders[0]["lines"][0]["options"][0]["name"], "hot honey")
        # Placed orders are frozen.
        self.assertTrue(box.run("add_item", {"item_id": "soda", "quantity": 1, "option_ids": ["soda_cola"]}).is_error)
        self.assertTrue(box.run("place_order", {}).is_error)
        self.assertEqual(len(self.store.recent_orders("demo")), 1)

    def test_closed_restaurant_refuses_to_place(self):
        box = self.make(when=CLOSED_TIME)
        box.run("add_item", {"item_id": "tiramisu", "quantity": 1, "option_ids": []})
        box.run("set_customer", {"name": "Lee", "phone": "2125550199"})
        box.run("review_order", {})
        res = box.run("place_order", {})
        self.assertTrue(res.is_error)
        self.assertIn("not accepting", res.content)
        self.assertEqual(self.store.recent_orders("demo"), [])

    def test_last_order_window(self):
        r = demo_restaurant()
        late = OPEN_TIME.replace(hour=20, minute=50)  # Tuesday closes at 9 PM, last orders 8:45
        self.assertFalse(r.accepting_orders(late))
        self.assertTrue(r.accepting_orders(late.replace(minute=40)))

    def test_transfer_without_staff_line_records_callback(self):
        box = self.make()
        res = json.loads(box.run("transfer_to_staff", {"reason": "catering for 40"}).content)
        self.assertFalse(res["transferred"])
        self.assertIsNone(box.control)
        self.assertEqual(box.callback_reason, "catering for 40")

    def test_transfer_with_staff_line_sets_control(self):
        box = self.make(staff_phone="+12125550000")
        json.loads(box.run("transfer_to_staff", {"reason": "complaint"}).content)
        self.assertEqual(box.control.action, "transfer")

    def test_end_call(self):
        box = self.make()
        box.run("end_call", {})
        self.assertEqual(box.control.action, "end")


if __name__ == "__main__":
    unittest.main()
