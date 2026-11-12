"""Tools the model calls during a call, and the code that runs them.

Tool inputs are validated here against the same schemas sent to the API:
with eager input streaming the API does not validate them, so a malformed
input becomes an error result the model can retry, never a crash or a
half-applied change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from .menu import Restaurant
from .order import Order, OrderError
from .store import Store


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "eager_input_streaming": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


_OPTIONS = {"type": "array", "items": {"type": "string"},
            "description": "Option ids from the menu, every required choice included."}

# Order and content are fixed so the request prefix stays byte-identical and cacheable.
TOOLS = [
    _tool("get_store_status",
          "Current local date and time, whether orders are being accepted right now, today's hours and the pickup wait. "
          "Call before promising a pickup time or when the caller asks about hours.",
          {}, []),
    _tool("add_item",
          "Add a menu item to the order. Ask for any required choice (size, sauce, flavor) before calling.",
          {"item_id": {"type": "string"},
           "quantity": {"type": "integer", "minimum": 1},
           "option_ids": _OPTIONS,
           "special_instructions": {"type": "string", "description": "Short kitchen note, e.g. 'well done'. Empty if none."}},
          ["item_id", "quantity", "option_ids"]),
    _tool("update_item",
          "Change the quantity, options or note of a line already in the order. option_ids replaces the line's whole option list.",
          {"line_id": {"type": "integer"},
           "quantity": {"type": "integer", "minimum": 1},
           "option_ids": _OPTIONS,
           "special_instructions": {"type": "string"}},
          ["line_id"]),
    _tool("remove_item", "Remove a line from the order.", {"line_id": {"type": "integer"}}, ["line_id"]),
    _tool("set_customer",
          "Record the name and callback phone number for the order. Use the caller ID number unless the caller gives another.",
          {"name": {"type": "string"}, "phone": {"type": "string"}},
          ["name", "phone"]),
    _tool("review_order",
          "Get the authoritative order with prices and total, to read back to the caller before placing it. "
          "Must be called again after any change.",
          {}, []),
    _tool("place_order",
          "Send the order to the kitchen. Only after review_order was read back and the caller clearly said yes.",
          {}, []),
    _tool("transfer_to_staff",
          "Hand the call to a person: the caller asks for one, has a complaint, a catering or large order, "
          "an allergy question the menu does not answer, or you are stuck.",
          {"reason": {"type": "string"}}, ["reason"]),
    _tool("end_call",
          "Hang up after you have said goodbye in this same response. Use when the caller is done.",
          {}, []),
]

_SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}
_TYPES = {"string": str, "integer": int, "array": list, "object": dict}


def schemas_of(tools: list[dict]) -> dict[str, dict]:
    return {t["name"]: t["input_schema"] for t in tools}


def validate_input(name: str, args, schemas: dict[str, dict] | None = None) -> str | None:
    """Return an error message, or None when args match the tool's schema.

    `schemas` defaults to the restaurant tools; other personas pass schemas_of(their TOOLS).
    """
    schema = (schemas if schemas is not None else _SCHEMAS).get(name)
    if schema is None:
        return f"unknown tool {name!r}"
    if not isinstance(args, dict):
        return "input must be a JSON object"
    props = schema["properties"]
    for key in schema["required"]:
        if key not in args:
            return f"missing required field {key!r}"
    for key, value in args.items():
        if key not in props:
            return f"unexpected field {key!r}"
        spec = props[key]
        expected = _TYPES[spec["type"]]
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            return f"{key!r} must be a {spec['type']}"
        if spec["type"] == "integer" and value < spec.get("minimum", value):
            return f"{key!r} must be at least {spec['minimum']}"
        if spec["type"] == "array" and not all(isinstance(v, str) for v in value):
            return f"{key!r} must be a list of strings"
    return None


@dataclass
class CallInfo:
    call_sid: str
    from_number: str = ""
    channel: str = "phone"


@dataclass
class Control:
    """What the call should do after the current response is spoken."""
    action: str  # "end" or "transfer"
    reason: str = ""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolBox:
    restaurant: Restaurant
    store: Store
    call: CallInfo
    clock: Callable[[], datetime] = field(default=lambda: datetime.now().astimezone())
    order: Order = None
    placed_order_id: int | None = None
    control: Control | None = None
    callback_reason: str = ""

    def __post_init__(self):
        if self.order is None:
            self.order = Order(self.restaurant)

    def run(self, name: str, args) -> ToolResult:
        problem = validate_input(name, args)
        if problem:
            return ToolResult(json.dumps({"INVALID_INPUT": problem, "received": args}), is_error=True)
        if self.placed_order_id is not None and name in {"add_item", "update_item", "remove_item", "set_customer", "place_order"}:
            return ToolResult(
                f"order #{self.placed_order_id} was already sent to the kitchen and cannot be changed by phone agent; "
                "offer transfer_to_staff for changes", is_error=True)
        try:
            return ToolResult(json.dumps(getattr(self, f"_t_{name}")(**args)))
        except OrderError as exc:
            return ToolResult(str(exc), is_error=True)

    def _t_get_store_status(self) -> dict:
        r = self.restaurant
        now = r.local_now(self.clock())
        spans = r.todays_hours(now)
        return {
            "local_time": now.strftime("%A %I:%M %p").replace(" 0", " "),
            "accepting_orders": r.accepting_orders(now),
            "today_hours": [f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}" for s, e in spans] or "closed today",
            "last_order_minutes_before_close": r.last_order_minutes,
            "pickup_wait_minutes": r.pickup_wait_minutes,
            "weekly_hours": r.hours_text(),
        }

    def _t_add_item(self, item_id: str, quantity: int, option_ids: list[str], special_instructions: str = "") -> dict:
        line = self.order.add(item_id, quantity, option_ids, special_instructions)
        return {"added": line.as_dict(), "order_subtotal": self.order.summary()["subtotal"]}

    def _t_update_item(self, line_id: int, quantity: int | None = None, option_ids: list[str] | None = None,
                       special_instructions: str | None = None) -> dict:
        line = self.order.update(line_id, quantity, option_ids, special_instructions)
        return {"updated": line.as_dict(), "order_subtotal": self.order.summary()["subtotal"]}

    def _t_remove_item(self, line_id: int) -> dict:
        line = self.order.remove(line_id)
        return {"removed": line.describe(), "order_subtotal": self.order.summary()["subtotal"]}

    def _t_set_customer(self, name: str, phone: str) -> dict:
        self.order.set_customer(name, phone or self.call.from_number)
        return {"customer_name": self.order.customer_name, "customer_phone": self.order.customer_phone}

    def _t_review_order(self) -> dict:
        return self.order.review()

    def _t_place_order(self) -> dict:
        r = self.restaurant
        now = r.local_now(self.clock())
        if not r.accepting_orders(now):
            raise OrderError("the restaurant is not accepting orders right now (check get_store_status); do not place it")
        self.order.check_ready()
        ready = now + timedelta(minutes=r.pickup_wait_minutes)
        summary = self.order.summary()
        self.placed_order_id = self.store.create_order(
            restaurant_id=r.id, call_sid=self.call.call_sid,
            customer_name=self.order.customer_name, customer_phone=self.order.customer_phone,
            lines=summary["lines"], subtotal_cents=self.order.subtotal_cents, tax_cents=self.order.tax_cents,
            total_cents=self.order.total_cents, ready_at=ready.isoformat(timespec="minutes"),
        )
        return {
            "order_number": self.placed_order_id,
            "total": summary["total"],
            "ready_at": ready.strftime("%I:%M %p").lstrip("0"),
            "payment": r.payment_note,
            "pickup_address": r.address,
        }

    def _t_transfer_to_staff(self, reason: str) -> dict:
        if not self.restaurant.staff_phone:
            self.callback_reason = reason[:200]
            return {"transferred": False,
                    "note": "No staff line is set up for live transfer. Apologize, say the team will call back on "
                            "the number we have, and offer to keep helping with the order."}
        self.control = Control("transfer", reason[:200])
        return {"transferred": True}

    def _t_end_call(self) -> dict:
        self.control = Control("end")
        return {"ended": True}

    def outcome(self) -> str:
        if self.placed_order_id is not None:
            return "order_placed"
        if self.control and self.control.action == "transfer":
            return "transferred"
        if self.callback_reason:
            return "callback_requested"
        return "no_order"
