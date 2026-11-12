"""SQLite persistence: placed orders (the kitchen board reads these) and call records."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    restaurant_id TEXT NOT NULL,
    call_sid TEXT,
    customer_name TEXT NOT NULL,
    customer_phone TEXT NOT NULL,
    lines_json TEXT NOT NULL,
    subtotal_cents INTEGER NOT NULL,
    tax_cents INTEGER NOT NULL,
    total_cents INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    ready_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_created ON orders(created_at);
CREATE TABLE IF NOT EXISTS stt_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_sid TEXT,
    seconds REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_sid TEXT,
    name TEXT NOT NULL,
    company TEXT NOT NULL,
    contact TEXT NOT NULL,
    role TEXT,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    call_sid TEXT PRIMARY KEY,
    restaurant_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    from_number TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    outcome TEXT,
    order_id INTEGER,
    transcript_json TEXT
);
"""

ORDER_STATUSES = ("new", "accepted", "ready", "picked_up", "cancelled")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit or roll back
                yield conn
        finally:
            conn.close()

    def _write(self, sql: str, params: tuple) -> int:
        with self._lock, self._conn() as conn:
            return conn.execute(sql, params).lastrowid

    def create_order(self, *, restaurant_id: str, call_sid: str, customer_name: str, customer_phone: str,
                     lines: list[dict], subtotal_cents: int, tax_cents: int, total_cents: int, ready_at: str) -> int:
        return self._write(
            "INSERT INTO orders (restaurant_id, call_sid, customer_name, customer_phone, lines_json,"
            " subtotal_cents, tax_cents, total_cents, ready_at, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (restaurant_id, call_sid, customer_name, customer_phone, json.dumps(lines),
             subtotal_cents, tax_cents, total_cents, ready_at, utcnow()),
        )

    def recent_orders(self, restaurant_id: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM orders WHERE restaurant_id = ? ORDER BY id DESC LIMIT ?", (restaurant_id, limit)
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["lines"] = json.loads(d.pop("lines_json"))
            out.append(d)
        return out

    def set_status(self, order_id: int, status: str) -> bool:
        if status not in ORDER_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._lock, self._conn() as conn:
            return conn.execute("UPDATE orders SET status = ? WHERE id = ?", (status, order_id)).rowcount == 1

    def start_call(self, call_sid: str, restaurant_id: str, channel: str, from_number: str) -> None:
        self._write(
            "INSERT OR IGNORE INTO calls (call_sid, restaurant_id, channel, from_number, started_at) VALUES (?,?,?,?,?)",
            (call_sid, restaurant_id, channel, from_number, utcnow()),
        )

    def end_call(self, call_sid: str, outcome: str, order_id: int | None, transcript: list[dict]) -> None:
        self._write(
            "UPDATE calls SET ended_at = ?, outcome = ?, order_id = ?, transcript_json = ? WHERE call_sid = ?",
            (utcnow(), outcome, order_id, json.dumps(transcript), call_sid),
        )

    def save_message(self, *, call_sid: str, name: str, company: str, contact: str, message: str, role: str = "") -> int:
        return self._write(
            "INSERT INTO messages (call_sid, name, company, contact, role, message, created_at) VALUES (?,?,?,?,?,?,?)",
            (call_sid, name, company, contact, role, message, utcnow()),
        )

    def recent_messages(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def record_stt(self, call_sid: str, seconds: float) -> None:
        if seconds > 0:
            self._write("INSERT INTO stt_usage (call_sid, seconds, created_at) VALUES (?,?,?)",
                        (call_sid, round(seconds, 2), utcnow()))

    def stt_seconds(self) -> float:
        with self._conn() as conn:
            return float(conn.execute("SELECT COALESCE(SUM(seconds), 0) FROM stt_usage").fetchone()[0])

    def recent_calls(self, restaurant_id: str, limit: int = 30) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT call_sid, channel, from_number, started_at, ended_at, outcome, order_id, transcript_json"
                " FROM calls WHERE restaurant_id = ? ORDER BY started_at DESC LIMIT ?", (restaurant_id, limit)
            ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["transcript"] = json.loads(d.pop("transcript_json") or "[]")
            out.append(d)
        return out
