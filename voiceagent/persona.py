"""A persona is everything that makes one agent different from another: the system
prompt, the tool list and the toolbox that runs them, the greeting, the per-call
context message and the fixed lines spoken when something goes wrong.

The restaurant is the default persona (agent.restaurant_persona); "Soma's AI" is
voiceagent/profile.py. Conversation and RelaySession only talk to this interface,
so neither assumes a restaurant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .tools import CallInfo, Control, ToolResult


class ToolBoxLike(Protocol):
    control: Control | None

    def run(self, name: str, args) -> ToolResult: ...
    def outcome(self) -> str: ...


@dataclass
class Persona:
    id: str
    system_prompt: str
    tools: list[dict]
    make_toolbox: Callable[[CallInfo], ToolBoxLike]
    greeting: str
    call_context: Callable[[str, str], str]   # (from_number, channel) -> mid-conversation system message
    # Fixed lines, never written by the model.
    trouble_line: str = "Sorry, I'm having trouble on my end right now. Please try again in a few minutes."
    refusal_line: str = "Sorry, I can't help with that. Is there anything else I can help with?"
    transfer_line: str = ""
    staff_phone: str = ""        # non-empty: the trouble line transfers, and DTMF 0 transfers
    dtmf_transfer_line: str = ""
    restaurant: object = None    # the restaurant persona keeps its Restaurant here (tests read conv.restaurant)
    extra: dict = field(default_factory=dict)
