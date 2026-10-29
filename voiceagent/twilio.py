"""Twilio helpers: request signature check, per-call relay tokens, TwiML."""
from __future__ import annotations

import base64
import hashlib
import hmac
from xml.sax.saxutils import quoteattr, escape

from .menu import Restaurant


def request_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Twilio's X-Twilio-Signature: base64 HMAC-SHA1 of the URL plus sorted key+value pairs."""
    payload = url + "".join(k + v for k, v in sorted(params.items()))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def valid_request(auth_token: str, url: str, params: dict[str, str], signature: str) -> bool:
    if not auth_token or not signature:
        return False
    return hmac.compare_digest(request_signature(auth_token, url, params), signature)


def relay_token(secret: str, call_sid: str) -> str:
    """Ties a relay WebSocket to the call our signed webhook answered."""
    return hmac.new(secret.encode(), call_sid.encode(), hashlib.sha256).hexdigest()[:32]


def valid_relay_token(secret: str, call_sid: str, token: str) -> bool:
    return bool(call_sid) and hmac.compare_digest(relay_token(secret, call_sid), token or "")


def speech_hints(restaurant: Restaurant) -> str:
    """Menu words that help speech recognition ('margherita', 'tiramisu')."""
    words: list[str] = []
    for item in restaurant.items.values():
        words.append(item.name)
        words.extend(item.aliases)
        for group in item.option_groups:
            words.extend(o.name for o in group.options)
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w.replace(",", " "))
    return ",".join(out)[:2000]


def _xml(body: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>{body}</Response>'


def connect_twiml(*, ws_url: str, action_url: str, restaurant: Restaurant, relay_attrs: dict[str, str]) -> str:
    attrs = {
        "url": ws_url,
        "welcomeGreeting": restaurant.greeting,
        "dtmfDetection": "true",
        "hints": speech_hints(restaurant),
        **{k: v for k, v in relay_attrs.items() if v},
    }
    rendered = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items())
    return _xml(f"<Connect action={quoteattr(action_url)}><ConversationRelay {rendered} /></Connect>")


def say_hangup_twiml(text: str) -> str:
    return _xml(f"<Say>{escape(text)}</Say><Hangup/>")


def dial_twiml(number: str, fallback_text: str) -> str:
    return _xml(f"<Dial timeout=\"25\">{escape(number)}</Dial><Say>{escape(fallback_text)}</Say><Hangup/>")


def hangup_twiml() -> str:
    return _xml("<Hangup/>")
