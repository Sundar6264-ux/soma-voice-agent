"""FastAPI app: Twilio webhooks, the relay WebSocket, the kitchen board, the browser test call."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid as uuidlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response

from . import twilio
from .agent import AnthropicLLM, Conversation
from .config import Settings
from .media import AudioSocketServer, MediaCall
from .menu import load_restaurant
from .prompts import build_system_prompt
from .relay import RelaySession
from .speech import AssemblyAISTT, InworldTTS
from .store import ORDER_STATUSES, Store

log = logging.getLogger(__name__)
PAGES = Path(__file__).parent / "pages"
ADMIN_COOKIE = "rva_admin"


PENDING_TTL_SECONDS = 60


def create_app(settings: Settings | None = None, llm=None, stt_factory=None, synthesize=None,
               frame_seconds: float | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    restaurant = load_restaurant(settings.restaurant_file)
    store = Store(settings.db_path)
    system_prompt = build_system_prompt(restaurant)
    sessions = {"live": 0}
    llm_holder = {"llm": llm}
    has_credentials = llm is not None or bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if not has_credentials:
        log.warning("no ANTHROPIC_API_KEY: every call will hear an apology and hang up")

    # -- self-hosted call path (Asterisk AudioSocket) -------------------------
    pending_calls: dict[str, tuple[str, float]] = {}   # call id -> (caller number, registered at)
    call_results: dict[str, tuple[str, float]] = {}    # call id -> ("hangup" | "transfer", finished at)

    def make_stt():
        if stt_factory:
            return stt_factory()
        return AssemblyAISTT(settings.assemblyai_api_key)

    tts_holder = {"synth": synthesize}

    def get_synth():
        if tts_holder["synth"] is None:
            tts = InworldTTS(settings.inworld_api_key, voice=settings.inworld_voice, model=settings.inworld_model,
                             speaking_rate=settings.inworld_speaking_rate)
            tts_holder["synth"] = tts.synthesize
        return tts_holder["synth"]

    async def start_media_call(call_id: str, reader, writer) -> None:
        entry = pending_calls.pop(call_id, None)
        if entry is None or time.monotonic() - entry[1] > PENDING_TTL_SECONDS:
            raise ValueError(f"unknown or expired call id {call_id}")
        sessions["live"] += 1
        try:
            make_relay = lambda send, sleep: RelaySession(send, make_conversation, store, restaurant.id,
                                                          channel="phone", sleep=sleep)
            kwargs = {"frame_seconds": frame_seconds} if frame_seconds else {}
            stt = make_stt()
            call = MediaCall(reader, writer, call_id=call_id, from_number=entry[0], greeting=restaurant.greeting,
                             stt=stt, synthesize=get_synth(), make_relay=make_relay, **kwargs)
            try:
                action = await call.run()
            finally:
                store.record_stt(call_id, getattr(stt, "billed_seconds", 0.0))
        finally:
            sessions["live"] -= 1
        now = time.monotonic()
        call_results[call_id] = (action, now)
        for key in [k for k, (_, at) in call_results.items() if now - at > 300]:
            call_results.pop(key, None)

    audiosocket = AudioSocketServer("127.0.0.1", settings.audiosocket_port, start_media_call)

    @asynccontextmanager
    async def lifespan(_app):
        if settings.selfhost_enabled:
            missing = [n for n, v in (("ASSEMBLYAI_API_KEY", settings.assemblyai_api_key or stt_factory),
                                      ("INWORLD_API_KEY", settings.inworld_api_key or synthesize)) if not v]
            if missing:
                log.warning("self-hosted calls enabled but %s missing: callers will hear nothing", ", ".join(missing))
            await audiosocket.start()
        try:
            yield
        finally:
            await audiosocket.stop()

    app = FastAPI(title="Restaurant voice agent", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings, app.state.restaurant, app.state.store = settings, restaurant, store
    app.state.audiosocket = audiosocket

    def require_asterisk(token: str) -> None:
        if not settings.selfhost_enabled or not hmac.compare_digest(token, settings.asterisk_token):
            raise HTTPException(status_code=403, detail="bad token")

    @app.get("/asterisk/start", response_class=PlainTextResponse)
    async def asterisk_start(request: Request):
        q = request.query_params
        require_asterisk(q.get("token", ""))
        if sessions["live"] + len(pending_calls) >= settings.max_sessions:
            return PlainTextResponse("busy")
        now = time.monotonic()
        for key in [k for k, (_, at) in pending_calls.items() if now - at > PENDING_TTL_SECONDS]:
            pending_calls.pop(key, None)
        call_id = str(uuidlib.uuid4())
        pending_calls[call_id] = (q.get("from", "")[:32], now)
        log.info("asterisk call %s from %s to %s", call_id, q.get("from", ""), q.get("to", ""))
        return PlainTextResponse(call_id)

    @app.get("/asterisk/next", response_class=PlainTextResponse)
    async def asterisk_next(token: str = "", call: str = ""):
        require_asterisk(token)
        action, _ = call_results.pop(call, ("hangup", 0))
        return PlainTextResponse(action)

    def get_llm():
        if llm_holder["llm"] is None:
            llm_holder["llm"] = AnthropicLLM(settings.model, settings.effort)
        return llm_holder["llm"]

    demo_clock = None
    if settings.demo_time:
        from datetime import datetime as _dt, time as _time
        try:
            hh, mm = (int(x) for x in settings.demo_time.split(":"))
            demo_at = _time(hh, mm)
        except ValueError:
            raise SystemExit(f"DEMO_TIME must be HH:MM, got {settings.demo_time!r}")
        demo_clock = lambda: _dt.combine(restaurant.local_now().date(), demo_at, tzinfo=restaurant.tz)
        log.warning("DEMO_TIME=%s: the agent's clock is fixed for testing; remove it before real use",
                    settings.demo_time)

    def make_conversation(call):
        return Conversation(restaurant, store, get_llm(), call, clock=demo_clock, system_prompt=system_prompt)

    # -- admin auth (board, browser test call) ----------------------------

    def admin_cookie_value() -> str:
        return hmac.new(settings.admin_key.encode(), b"rva-admin", hashlib.sha256).hexdigest()

    def is_admin(cookies) -> bool:
        return bool(settings.admin_key) and hmac.compare_digest(cookies.get(ADMIN_COOKIE, ""), admin_cookie_value())

    def require_admin(request: Request) -> None:
        if not is_admin(request.cookies):
            raise HTTPException(status_code=401, detail="open /login?key=... first")

    @app.get("/login")
    async def login(key: str = "", next: str = "/board"):
        if not settings.admin_key or not hmac.compare_digest(key, settings.admin_key):
            raise HTTPException(status_code=401, detail="wrong or missing key")
        target = next if next in {"/board", "/try"} else "/board"
        resp = RedirectResponse(target, status_code=303)
        resp.set_cookie(ADMIN_COOKIE, admin_cookie_value(), httponly=True, samesite="strict",
                        secure=settings.public_base_url.startswith("https://"), max_age=30 * 86400)
        return resp

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "restaurant": restaurant.id, "model": settings.model, "claude_credentials": has_credentials,
                "twilio_signing": bool(settings.twilio_auth_token) and not settings.allow_unsigned_twilio,
                "selfhost_calls": settings.selfhost_enabled, "demo_time": settings.demo_time or None,
                "assemblyai_key": bool(settings.assemblyai_api_key), "inworld_key": bool(settings.inworld_api_key),
                "live_sessions": sessions["live"]}

    # -- Twilio ------------------------------------------------------------

    async def twilio_form(request: Request) -> dict[str, str]:
        form = {k: str(v) for k, v in (await request.form()).items()}
        if settings.allow_unsigned_twilio:
            return form
        if not settings.public_base_url:
            raise HTTPException(status_code=503, detail="PUBLIC_BASE_URL is not set")
        url = settings.public_base_url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        if not twilio.valid_request(settings.twilio_auth_token, url, form, request.headers.get("X-Twilio-Signature", "")):
            log.warning("rejected unsigned or mis-signed Twilio request to %s", request.url.path)
            raise HTTPException(status_code=403, detail="bad signature")
        return form

    def xml(body: str) -> Response:
        return Response(body, media_type="text/xml")

    @app.post("/twilio/voice")
    async def twilio_voice(request: Request):
        form = await twilio_form(request)
        call_sid = form.get("CallSid", "")
        if not call_sid:
            raise HTTPException(status_code=400, detail="no CallSid")
        if sessions["live"] >= settings.max_sessions:
            return xml(twilio.say_hangup_twiml(
                f"Sorry, all of our lines at {restaurant.name} are busy. Please call back in a few minutes."))
        token = twilio.relay_token(settings.relay_secret, call_sid)
        ws_url = f"{settings.ws_base_url}/relay?call={call_sid}&token={token}"
        return xml(twilio.connect_twiml(ws_url=ws_url, action_url=f"{settings.public_base_url}/twilio/relay-done",
                                        restaurant=restaurant, relay_attrs=settings.relay_attrs))

    @app.post("/twilio/relay-done")
    async def twilio_relay_done(request: Request):
        """Twilio calls this when the relay ends: our `end` message, a dropped socket, or a hang-up."""
        form = await twilio_form(request)
        try:
            handoff = json.loads(form.get("HandoffData") or "{}")
        except json.JSONDecodeError:
            handoff = {}
        reason = handoff.get("reasonCode")
        if reason == "transfer" and restaurant.staff_phone:
            return xml(twilio.dial_twiml(restaurant.staff_phone,
                                         "Sorry, nobody could pick up. Please call back in a few minutes. Goodbye."))
        if reason in {"end", "transfer"}:
            return xml(twilio.hangup_twiml())
        return xml(twilio.say_hangup_twiml("Sorry, we got disconnected. Please call back. Goodbye."))

    async def run_relay(websocket: WebSocket, relay: RelaySession, first: dict | None = None) -> None:
        sessions["live"] += 1
        try:
            if first:
                await relay.handle(first)
            while True:
                msg = await websocket.receive_json()
                await relay.handle(msg)
        except WebSocketDisconnect:
            pass
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("bad relay message: %r", exc)
        finally:
            sessions["live"] -= 1
            await relay.close()

    @app.websocket("/relay")
    async def relay_ws(websocket: WebSocket, call: str = "", token: str = ""):
        if not twilio.valid_relay_token(settings.relay_secret, call, token):
            await websocket.close(code=4403)
            return
        await websocket.accept()
        relay = RelaySession(websocket.send_json, make_conversation, store, restaurant.id,
                             channel="phone", expected_call_sid=call)
        await run_relay(websocket, relay)

    # -- browser test call (same protocol as Twilio) -----------------------

    @app.websocket("/sim")
    async def sim_ws(websocket: WebSocket):
        if not is_admin(websocket.cookies):
            await websocket.close(code=4401)
            return
        if sessions["live"] >= settings.max_sessions:
            await websocket.close(code=4429)
            return
        await websocket.accept()
        await websocket.send_json({"type": "text", "token": restaurant.greeting, "last": True})
        relay = RelaySession(websocket.send_json, make_conversation, store, restaurant.id, channel="browser")
        await run_relay(websocket, relay, first={"type": "setup", "callSid": "web-" + secrets.token_hex(6), "from": ""})

    @app.get("/try")
    async def try_page(request: Request):
        if not is_admin(request.cookies):
            return RedirectResponse("/login-required", status_code=303)
        return FileResponse(PAGES / "try.html")

    # -- kitchen board -----------------------------------------------------

    @app.get("/board")
    async def board(request: Request):
        if not is_admin(request.cookies):
            return RedirectResponse("/login-required", status_code=303)
        return FileResponse(PAGES / "board.html")

    @app.get("/login-required", response_class=HTMLResponse)
    async def login_required():
        return HTMLResponse("<p style='font-family:sans-serif'>Open <code>/login?key=YOUR_ADMIN_KEY</code> first.</p>",
                            status_code=401)

    @app.get("/api/orders")
    async def api_orders(request: Request):
        require_admin(request)
        return {"restaurant": restaurant.name, "orders": store.recent_orders(restaurant.id)}

    @app.post("/api/orders/{order_id}/status")
    async def api_order_status(order_id: int, request: Request):
        require_admin(request)
        status = (await request.json()).get("status", "")
        if status not in ORDER_STATUSES:
            raise HTTPException(status_code=400, detail=f"status must be one of {ORDER_STATUSES}")
        if not store.set_status(order_id, status):
            raise HTTPException(status_code=404, detail="no such order")
        return {"ok": True}

    @app.get("/api/calls")
    async def api_calls(request: Request):
        require_admin(request)
        return {"calls": store.recent_calls(restaurant.id)}

    return app


def _app_factory() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app()
