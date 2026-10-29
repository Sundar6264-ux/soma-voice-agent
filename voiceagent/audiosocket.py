"""Asterisk AudioSocket framing (docs.asterisk.org, Channel Drivers > AudioSocket).

Every message is a 1-byte type, a 2-byte big-endian payload length, then the
payload. Audio is 16-bit signed little-endian mono PCM; type 0x10 is 8 kHz,
which is what phone calls carry, so no resampling happens anywhere.
"""
from __future__ import annotations

import asyncio
import struct
import uuid as uuidlib

HANGUP = 0x00
UUID = 0x01
DTMF = 0x03
AUDIO_8K = 0x10
ERROR = 0xFF

SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 320 bytes = 20 ms of 8 kHz slin
SILENCE_FRAME = b"\x00" * FRAME_BYTES


def pack(kind: int, payload: bytes = b"") -> bytes:
    return struct.pack(">BH", kind, len(payload)) + payload


def audio_frame(pcm: bytes) -> bytes:
    return pack(AUDIO_8K, pcm)


async def read_message(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read one message. Raises asyncio.IncompleteReadError when the socket closes."""
    header = await reader.readexactly(3)
    kind, length = struct.unpack(">BH", header)
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


def parse_uuid(payload: bytes) -> str:
    return str(uuidlib.UUID(bytes=payload))
