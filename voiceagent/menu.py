"""Restaurant config: menu, hours, facts. Loaded from YAML and validated up front.

Prices are integer cents everywhere past this module, so the agent's totals
never drift by a float rounding error.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class ConfigError(ValueError):
    pass


def to_cents(value) -> int:
    cents = Decimal(str(value)) * 100
    if cents != cents.to_integral_value() or cents < 0:
        raise ConfigError(f"price {value!r} is not a non-negative amount in whole cents")
    return int(cents)


def money(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


@dataclass(frozen=True)
class Option:
    id: str
    name: str
    price_cents: int
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptionGroup:
    id: str
    name: str
    min: int
    max: int
    options: tuple[Option, ...]


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    category: str
    price_cents: int
    description: str = ""
    aliases: tuple[str, ...] = ()
    dietary: tuple[str, ...] = ()
    allergens: tuple[str, ...] | None = None  # None means "not recorded", not "none"
    option_groups: tuple[OptionGroup, ...] = ()

    def option(self, option_id: str) -> tuple[OptionGroup, Option] | None:
        for group in self.option_groups:
            for opt in group.options:
                if opt.id == option_id:
                    return group, opt
        return None


@dataclass(frozen=True)
class Restaurant:
    id: str
    name: str
    address: str
    timezone: str
    greeting: str
    staff_phone: str
    tax_rate: Decimal
    pickup_wait_minutes: int
    last_order_minutes: int
    max_quantity_per_line: int
    payment_note: str
    fulfillment: str
    hours: dict[str, tuple[tuple[time, time], ...]]
    facts: tuple[str, ...]
    items: dict[str, Item] = field(default_factory=dict)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def local_now(self, now: datetime | None = None) -> datetime:
        return (now or datetime.now(self.tz)).astimezone(self.tz)

    def todays_hours(self, now: datetime | None = None) -> tuple[tuple[time, time], ...]:
        return self.hours[DAYS[self.local_now(now).weekday()]]

    def accepting_orders(self, now: datetime | None = None) -> bool:
        """Open, and not inside the last-order window before closing."""
        local = self.local_now(now)
        for start, end in self.todays_hours(local):
            cutoff = (datetime.combine(local.date(), end) - timedelta(minutes=self.last_order_minutes)).time()
            if start <= local.time() < cutoff:
                return True
        return False

    def hours_text(self) -> str:
        lines = []
        for day in DAYS:
            spans = self.hours[day]
            if not spans:
                lines.append(f"{day}: closed")
            else:
                lines.append(f"{day}: " + ", ".join(f"{_clock(s)} to {_clock(e)}" for s, e in spans))
        return "\n".join(lines)

    def tax_cents(self, subtotal_cents: int) -> int:
        return int((Decimal(subtotal_cents) * self.tax_rate).quantize(Decimal(1), rounding="ROUND_HALF_UP"))


def _clock(t: time) -> str:
    hour = t.hour % 12 or 12
    suffix = "AM" if t.hour < 12 else "PM"
    return f"{hour}:{t.minute:02d} {suffix}" if t.minute else f"{hour} {suffix}"


def _parse_time(value: str, where: str) -> time:
    try:
        return time.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: {value!r} is not HH:MM") from exc


def _require(mapping: dict, key: str, where: str):
    if key not in mapping or mapping[key] in (None, ""):
        raise ConfigError(f"{where}: missing {key!r}")
    return mapping[key]


def load_restaurant(path: str | Path) -> Restaurant:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    tz = _require(raw, "timezone", "restaurant")
    try:
        ZoneInfo(tz)
    except Exception as exc:
        raise ConfigError(f"unknown timezone {tz!r}") from exc

    hours: dict[str, tuple[tuple[time, time], ...]] = {}
    raw_hours = _require(raw, "hours", "restaurant")
    for day in DAYS:
        spans = []
        for span in raw_hours.get(day) or []:
            start, end = (_parse_time(v, f"hours.{day}") for v in span)
            if end <= start:
                raise ConfigError(f"hours.{day}: close {end} is not after open {start} (past-midnight spans are not supported)")
            spans.append((start, end))
        hours[day] = tuple(sorted(spans))

    items: dict[str, Item] = {}
    for section in _require(raw, "menu", "restaurant"):
        category = _require(section, "category", "menu section")
        for raw_item in _require(section, "items", f"menu.{category}"):
            item_id = _require(raw_item, "id", f"menu.{category} item")
            where = f"item {item_id}"
            if item_id in items:
                raise ConfigError(f"{where}: duplicate id")
            groups = []
            option_ids: set[str] = set()
            for raw_group in raw_item.get("option_groups") or []:
                group_id = _require(raw_group, "id", where)
                options = []
                for raw_opt in _require(raw_group, "options", f"{where}.{group_id}"):
                    opt_id = _require(raw_opt, "id", f"{where}.{group_id} option")
                    if opt_id in option_ids:
                        raise ConfigError(f"{where}: duplicate option id {opt_id}")
                    option_ids.add(opt_id)
                    options.append(Option(
                        id=opt_id,
                        name=_require(raw_opt, "name", f"{where}.{opt_id}"),
                        price_cents=to_cents(raw_opt.get("price", 0)),
                        requires=tuple(raw_opt.get("requires") or ()),
                    ))
                lo, hi = int(raw_group.get("min", 0)), int(raw_group.get("max", len(options)))
                if not 0 <= lo <= hi or hi < 1:
                    raise ConfigError(f"{where}.{group_id}: bad min/max {lo}/{hi}")
                groups.append(OptionGroup(group_id, _require(raw_group, "name", f"{where}.{group_id}"), lo, hi, tuple(options)))
            for group in groups:
                for opt in group.options:
                    missing = [r for r in opt.requires if r not in option_ids]
                    if missing:
                        raise ConfigError(f"{where}.{opt.id}: requires unknown option(s) {missing}")
            allergens = raw_item.get("allergens")
            items[item_id] = Item(
                id=item_id,
                name=_require(raw_item, "name", where),
                category=category,
                price_cents=to_cents(_require(raw_item, "price", where)),
                description=raw_item.get("description", ""),
                aliases=tuple(raw_item.get("aliases") or ()),
                dietary=tuple(raw_item.get("dietary") or ()),
                allergens=None if allergens is None else tuple(allergens),
                option_groups=tuple(groups),
            )

    fulfillment = raw.get("fulfillment", "pickup")
    if fulfillment != "pickup":
        raise ConfigError("only fulfillment: pickup is supported so far")

    return Restaurant(
        id=_require(raw, "id", "restaurant"),
        name=_require(raw, "name", "restaurant"),
        address=raw.get("address", ""),
        timezone=tz,
        greeting=" ".join(_require(raw, "greeting", "restaurant").split()),
        staff_phone=str(raw.get("staff_phone") or ""),
        tax_rate=Decimal(str(raw.get("tax_rate", 0))),
        pickup_wait_minutes=int(raw.get("pickup_wait_minutes", 20)),
        last_order_minutes=int(raw.get("last_order_minutes", 0)),
        max_quantity_per_line=int(raw.get("max_quantity_per_line", 20)),
        payment_note=raw.get("payment_note", ""),
        fulfillment=fulfillment,
        hours=hours,
        facts=tuple(raw.get("facts") or ()),
        items=items,
    )


def menu_text(restaurant: Restaurant) -> str:
    """The menu as the model sees it: ids for tool calls, names for speaking."""
    out = []
    category = None
    for item in restaurant.items.values():
        if item.category != category:
            category = item.category
            out.append(f"\n## {category}")
        line = f"- {item.name} [id: {item.id}] {money(item.price_cents)}"
        if item.description:
            line += f". {item.description}"
        out.append(line)
        if item.aliases:
            out.append(f"  callers may say: {', '.join(item.aliases)}")
        if item.dietary:
            out.append(f"  dietary: {', '.join(item.dietary)}")
        if item.allergens is None:
            out.append("  allergens: not recorded")
        else:
            out.append(f"  allergens: {', '.join(item.allergens) or 'none listed'}")
        for group in item.option_groups:
            rule = "required, choose 1" if group.min == group.max == 1 else f"choose {group.min} to {group.max}"
            opts = []
            for opt in group.options:
                text = f"{opt.name} [{opt.id}]"
                text += f" +{money(opt.price_cents)}" if opt.price_cents else ""
                if opt.requires:
                    text += f" (only with {', '.join(opt.requires)})"
                opts.append(text)
            out.append(f"  {group.name} ({rule}): {'; '.join(opts)}")
    return "\n".join(out).strip()


if __name__ == "__main__":
    r = load_restaurant(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "restaurants" / "demo.yaml")
    print(f"{r.name}: {len(r.items)} items, tax {r.tax_rate}, accepting orders now: {r.accepting_orders()}\n")
    print(r.hours_text(), "\n")
    print(menu_text(r))
