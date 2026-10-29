"""The cart for one call. All prices and totals are computed here, never by the model.

Every change bumps `version`. `review()` records the version the caller heard
read back, and `check_ready()` refuses to place an order whose contents
changed after that read-back, so "yes, that's right" always refers to what
actually gets sent to the kitchen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .menu import Item, Restaurant, money


class OrderError(ValueError):
    """A problem the caller can fix; the message is shown to the model."""


@dataclass
class Line:
    line_id: int
    item: Item
    quantity: int
    option_ids: tuple[str, ...]
    instructions: str

    @property
    def unit_cents(self) -> int:
        return self.item.price_cents + sum(self.item.option(o)[1].price_cents for o in self.option_ids)

    @property
    def total_cents(self) -> int:
        return self.unit_cents * self.quantity

    def describe(self) -> str:
        names = [self.item.option(o)[1].name for o in self.option_ids]
        text = f"{self.quantity} x {self.item.name}"
        if names:
            text += f" ({', '.join(names)})"
        if self.instructions:
            text += f", note: {self.instructions}"
        return text

    def as_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "item_id": self.item.id,
            "name": self.item.name,
            "quantity": self.quantity,
            "options": [{"id": o, "name": self.item.option(o)[1].name} for o in self.option_ids],
            "instructions": self.instructions,
            "unit_price": money(self.unit_cents),
            "line_total": money(self.total_cents),
            "line_total_cents": self.total_cents,
        }


_PHONE_DIGITS = re.compile(r"\d")


def normalize_phone(raw: str) -> str:
    """US numbers to E.164; anything else kept as digits with a leading +."""
    digits = "".join(_PHONE_DIGITS.findall(raw or ""))
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if 8 <= len(digits) <= 15 and (raw or "").strip().startswith("+"):
        return "+" + digits
    raise OrderError(f"{raw!r} does not look like a phone number; ask the caller to repeat it digit by digit")


@dataclass
class Order:
    restaurant: Restaurant
    lines: list[Line] = field(default_factory=list)
    customer_name: str = ""
    customer_phone: str = ""
    version: int = 0
    reviewed_version: int = -1
    _next_line: int = 1

    def _validate_options(self, item: Item, option_ids: list[str]) -> tuple[str, ...]:
        if len(set(option_ids)) != len(option_ids):
            raise OrderError("an option was given twice; extra portions are not supported, list each option once")
        chosen: dict[str, list[str]] = {g.id: [] for g in item.option_groups}
        for opt_id in option_ids:
            found = item.option(opt_id)
            if not found:
                valid = [o.id for g in item.option_groups for o in g.options]
                raise OrderError(f"{opt_id!r} is not an option for {item.name}; valid options: {valid or 'none'}")
            chosen[found[0].id].append(opt_id)
        for group in item.option_groups:
            n = len(chosen[group.id])
            if n < group.min:
                names = ", ".join(o.name for o in group.options)
                raise OrderError(f"{item.name} needs a choice of {group.name.lower()} ({names}); ask the caller")
            if n > group.max:
                raise OrderError(f"{item.name} allows at most {group.max} {group.name.lower()}")
        for opt_id in option_ids:
            _, opt = item.option(opt_id)
            for needed in opt.requires:
                if needed not in option_ids:
                    raise OrderError(f"{opt.name} is only available with {item.option(needed)[1].name}")
        # Keep the menu's own order so read-backs sound consistent.
        ordering = [o.id for g in item.option_groups for o in g.options]
        return tuple(sorted(option_ids, key=ordering.index))

    def _check_quantity(self, quantity: int) -> None:
        limit = self.restaurant.max_quantity_per_line
        if not 1 <= quantity <= limit:
            raise OrderError(f"quantity must be 1 to {limit} per line; for larger orders offer a staff callback")

    def _line(self, line_id: int) -> Line:
        for line in self.lines:
            if line.line_id == line_id:
                return line
        raise OrderError(f"there is no line {line_id}; current lines: {[l.line_id for l in self.lines]}")

    def add(self, item_id: str, quantity: int, option_ids: list[str], instructions: str = "") -> Line:
        item = self.restaurant.items.get(item_id)
        if not item:
            raise OrderError(f"{item_id!r} is not on the menu; use an id from the menu exactly")
        self._check_quantity(quantity)
        line = Line(self._next_line, item, quantity, self._validate_options(item, option_ids), instructions.strip()[:200])
        self._next_line += 1
        self.lines.append(line)
        self.version += 1
        return line

    def update(self, line_id: int, quantity: int | None = None, option_ids: list[str] | None = None,
               instructions: str | None = None) -> Line:
        line = self._line(line_id)
        if quantity is not None:
            self._check_quantity(quantity)
        new_options = line.option_ids if option_ids is None else self._validate_options(line.item, option_ids)
        if quantity is not None:
            line.quantity = quantity
        line.option_ids = new_options
        if instructions is not None:
            line.instructions = instructions.strip()[:200]
        self.version += 1
        return line

    def remove(self, line_id: int) -> Line:
        line = self._line(line_id)
        self.lines.remove(line)
        self.version += 1
        return line

    def set_customer(self, name: str, phone: str) -> None:
        name = " ".join((name or "").split())[:60]
        if not name:
            raise OrderError("a name for the order is required")
        self.customer_name = name
        self.customer_phone = normalize_phone(phone)
        self.version += 1

    @property
    def subtotal_cents(self) -> int:
        return sum(l.total_cents for l in self.lines)

    @property
    def tax_cents(self) -> int:
        return self.restaurant.tax_cents(self.subtotal_cents)

    @property
    def total_cents(self) -> int:
        return self.subtotal_cents + self.tax_cents

    def summary(self) -> dict:
        return {
            "lines": [l.as_dict() for l in self.lines],
            "item_count": sum(l.quantity for l in self.lines),
            "subtotal": money(self.subtotal_cents),
            "tax": money(self.tax_cents),
            "total": money(self.total_cents),
            "customer_name": self.customer_name or None,
            "customer_phone": self.customer_phone or None,
            "reviewed": self.reviewed_version == self.version,
        }

    def review(self) -> dict:
        self.reviewed_version = self.version
        return self.summary()

    def check_ready(self) -> None:
        problems = []
        if not self.lines:
            problems.append("the order is empty")
        if not self.customer_name or not self.customer_phone:
            problems.append("no name and phone yet (call set_customer)")
        if self.reviewed_version != self.version:
            problems.append("the order changed since it was last read back; call review_order, read it to the caller and get a yes")
        if problems:
            raise OrderError("cannot place the order: " + "; ".join(problems))
