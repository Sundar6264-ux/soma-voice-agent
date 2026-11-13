"""FastAPI app: Twilio webhooks, the relay WebSocket, the kitchen board, the browser test call."""
from __future__ import annotations

import asyncio
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
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response

from . import profile, twilio
from .agent import AnthropicLLM, Conversation
from .config import Settings
from .media import AudioSocketServer, AudioSocketTransport, MediaCall, WebSocketTransport
from .menu import load_restaurant
from .prompts import build_system_prompt
from .relay import RelaySession
from .speech import AssemblyAISTT, InworldTTS, with_fallback_voice
from .store import ORDER_STATUSES, Store

log = logging.getLogger(__name__)
PAGES = Path(__file__).parent / "pages"
STATIC = Path(__file__).parent / "static"
ADMIN_COOKIE = "rva_admin"
PROFILE_WARN_BEFORE_END_S = 60


PENDING_TTL_SECONDS = 60


def create_app(settings: Settings | None = None, llm=None, stt_factory=None, synthesize=None,
               frame_seconds: float | None = None, mailer=None, resume_text: str | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    restaurant = load_restaurant(settings.restaurant_file)
    store = Store(settings.db_path)
    system_prompt = build_system_prompt(restaurant)
    # "Soma's AI": the resume is read once here, never modified, and lives in the cached system prompt.
    persona = profile.make_persona(store, resume_text if resume_text is not None else profile.load_resume(settings.profile_resume_path),
                                   message_email=settings.message_email, mailer=mailer)
    sessions = {"live": 0, "talk": 0}
    llm_holder = {"llm": llm}
    has_credentials = llm is not None or bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if not has_credentials:
        log.warning("no ANTHROPIC_API_KEY: every call will hear an apology and hang up")

    # -- self-hosted call path (Asterisk AudioSocket) -------------------------
    pending_calls: dict[str, tuple[str, float]] = {}   # call id -> (caller number, registered at)
    call_results: dict[str, tuple[str, float]] = {}    # call id -> ("hangup" | "transfer", finished at)

    def make_stt(sample_rate: int = 8000):
        if stt_factory:
            return stt_factory()
        return AssemblyAISTT(settings.assemblyai_api_key, sample_rate=sample_rate,
                             min_turn_silence_ms=settings.stt_min_turn_silence_ms,
                             max_turn_silence_ms=settings.stt_max_turn_silence_ms,
                             end_of_turn_confidence=settings.stt_end_of_turn_confidence,
                             wait_for_formatting=settings.stt_wait_for_formatting)

    tts_holder = {"synth": synthesize, "tts": None, "profile_synth": None}

    def get_tts():
        if tts_holder["tts"] is None and synthesize is None:
            tts_holder["tts"] = InworldTTS(settings.inworld_api_key, voice=settings.inworld_voice, model=settings.inworld_model,
                                           speaking_rate=settings.inworld_speaking_rate,
                                           max_concurrent=settings.inworld_max_concurrent)
        return tts_holder["tts"]

    def get_synth():
        """The phone voice: 8 kHz in the stock voice."""
        if tts_holder["synth"] is None:
            tts_holder["synth"] = get_tts().synthesize
        return tts_holder["synth"]

    def get_profile_synth():
        """The browser voice: 24 kHz in the cloned voice, falling back to the stock voice per sentence."""
        if tts_holder["profile_synth"] is None:
            if synthesize is not None:   # tests: the fake ignores rate and voice
                tts_holder["profile_synth"] = synthesize
            else:
                tts_holder["profile_synth"] = with_fallback_voice(
                    get_tts().synthesize, voice=settings.profile_inworld_voice or settings.inworld_voice,
                    fallback=settings.inworld_voice, sample_rate=WebSocketTransport.out_rate,
                    speaking_rate=settings.profile_speaking_rate)
        return tts_holder["profile_synth"]

    def warm_services() -> None:
        """Open the Anthropic and Inworld connections while the greeting plays, so the first reply
        does not pay two TLS handshakes. Fire and forget; failures are harmless."""
        for svc in (get_llm(), get_tts()):
            warm = getattr(svc, "warm", None)
            if warm:
                asyncio.create_task(warm())

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
            call = MediaCall(AudioSocketTransport(reader, writer), call_id=call_id, from_number=entry[0],
                             greeting=restaurant.greeting, stt=stt, synthesize=get_synth(), make_relay=make_relay, **kwargs)
            warm_services()
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

    def make_profile_conversation(call):
        return Conversation(None, store, get_llm(), call, persona=persona)

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
        target = next if next in {"/board", "/try", "/talk"} else "/board"
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
                "live_sessions": sessions["live"], "talk_sessions": sessions["talk"],
                "profile_voice": settings.profile_inworld_voice or settings.inworld_voice, "message_email": settings.message_email}

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

    # -- Soma's AI: browser voice conversation over our own audio pipeline ----

    def talk_busy() -> bool:
        return sessions["talk"] >= 1 or sessions["live"] >= settings.max_sessions

    # Public tickets (Phase 2). somasundar.dev's server forwards /api/talk/ticket and the /api/talk/ws upgrade
    # here with the visitor's cf-connecting-ip and Origin. A ticket is single use, valid for 60 s, bound to the
    # visitor, and only issued inside the per-visitor and per-day caps, so every public conversation is counted.
    ticket_secret = secrets.token_bytes(32)
    used_tickets: dict[str, float] = {}

    def visitor_of(headers) -> str:
        ip = headers.get("cf-connecting-ip") or headers.get("x-real-ip") or "local"
        # Keyed with the persistent relay secret so a restart does not reset today's per-visitor counts.
        return hmac.new(settings.relay_secret.encode(), b"talk-visitor:" + ip.encode(), hashlib.sha256).hexdigest()[:16]

    def origin_ok(headers) -> bool:
        return headers.get("origin", "") in settings.talk_allowed_origins

    def make_ticket(visitor: str) -> str:
        body = f"{int(time.time()) + 60}.{secrets.token_hex(8)}.{visitor}"
        sig = hmac.new(ticket_secret, body.encode(), hashlib.sha256).hexdigest()[:32]
        return f"{body}.{sig}"

    def take_ticket(ticket: str, visitor: str) -> bool:
        try:
            exp, nonce, owner, sig = ticket.split(".")
            good = hmac.new(ticket_secret, f"{exp}.{nonce}.{owner}".encode(), hashlib.sha256).hexdigest()[:32]
            now = time.time()
            if not hmac.compare_digest(sig, good) or int(exp) < now or owner != visitor or nonce in used_tickets:
                return False
        except ValueError:
            return False
        for key in [k for k, at in used_tickets.items() if at < now - 120]:
            used_tickets.pop(key, None)
        used_tickets[nonce] = now
        return True

    @app.post("/talk/ticket")
    async def talk_ticket(request: Request):
        if not settings.talk_public:
            raise HTTPException(status_code=404, detail="not found")
        if not origin_ok(request.headers):
            return JSONResponse({"ok": False, "error": "Not allowed from this site."}, status_code=403)
        if talk_busy():
            return JSONResponse({"ok": False, "error": "Soma's AI is talking with someone else right now. Please try again in a few minutes."}, status_code=429)
        visitor = visitor_of(request.headers)
        day = time.strftime("%Y-%m-%d", time.gmtime())
        mine, total = store.talk_counts(day, visitor)
        if mine >= settings.talk_per_ip_day:
            return JSONResponse({"ok": False, "error": f"That's the limit of {settings.talk_per_ip_day} conversations for today. The contact page reaches Soma directly."}, status_code=429)
        if total >= settings.talk_per_day:
            return JSONResponse({"ok": False, "error": "Soma's AI has reached today's limit. Please try again tomorrow, or use the contact page."}, status_code=429)
        store.record_talk_ticket(day, visitor)
        return {"ok": True, "ticket": make_ticket(visitor), "left_today": settings.talk_per_ip_day - mine - 1}

    @app.websocket("/talk/public")
    async def talk_public_ws(websocket: WebSocket, ticket: str = ""):
        if not settings.talk_public or not origin_ok(websocket.headers) or not take_ticket(ticket, visitor_of(websocket.headers)):
            await websocket.close(code=4403)
            return
        await run_talk(websocket, "pub-")

    @app.websocket("/talk")
    async def talk_ws(websocket: WebSocket):
        if not is_admin(websocket.cookies):
            await websocket.close(code=4401)
            return
        await run_talk(websocket, "web-")

    def send_transcript(call, call_id: str, started_wall: float, seconds: float) -> None:
        """Email Soma the transcript of every Soma's AI conversation (TRANSCRIPT_EMAIL=off turns it off)."""
        conv = getattr(call.relay, "conv", None)
        if not settings.transcript_email or conv is None:
            return
        try:
            payload = profile.transcript_email(
                call_id=call_id, started=started_wall, seconds=seconds, outcome=conv.outcome(),
                transcript=conv.transcript, message_ids=list(getattr(conv.toolbox, "message_ids", [])),
                test=settings.message_email == "test")
            if payload is None:
                log.warning("transcript for %s not emailed: OWNER_EMAIL is not set", call_id)
                return
            (mailer or profile.send_mail_subprocess)(payload)
        except Exception:
            log.exception("could not email the transcript for %s", call_id)

    async def run_talk(websocket: WebSocket, prefix: str):
        if talk_busy():
            # One browser conversation at a time: the cloned voice runs on Inworld's free-tier concurrency.
            await websocket.accept()
            await websocket.send_json({"type": "busy", "text": "Soma's AI is talking with someone else right now. Please try again shortly."})
            await websocket.close(code=4429)
            return
        await websocket.accept()
        sessions["talk"] += 1
        sessions["live"] += 1
        call_id = prefix + secrets.token_hex(6)
        try:
            transport = WebSocketTransport(websocket)
            stt = make_stt(transport.in_rate)
            make_relay = lambda send, sleep: RelaySession(send, make_profile_conversation, store, persona.id,
                                                          channel="web", sleep=sleep)
            kwargs = {"frame_seconds": frame_seconds} if frame_seconds else {}
            limit = settings.profile_time_limit_s
            call = MediaCall(transport, call_id=call_id, from_number="", greeting=persona.greeting, stt=stt,
                             synthesize=get_profile_synth(), make_relay=make_relay, time_limit=limit,
                             warn_at=max(0, limit - PROFILE_WARN_BEFORE_END_S), **kwargs)
            warm_services()
            started_wall, started = time.time(), time.monotonic()
            try:
                await call.run()
            finally:
                store.record_stt(call_id, getattr(stt, "billed_seconds", 0.0))
                send_transcript(call, call_id, started_wall, time.monotonic() - started)
        except Exception:
            log.exception("talk session %s failed", call_id)
        finally:
            sessions["talk"] -= 1
            sessions["live"] -= 1

    @app.get("/talk")
    async def talk_page(request: Request):
        if not is_admin(request.cookies):
            return RedirectResponse("/login-required", status_code=303)
        return FileResponse(PAGES / "talk.html")

    @app.get("/static/{name}")
    async def static_file(name: str):
        path = STATIC / name
        if "/" in name or not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.get("/api/messages")
    async def api_messages(request: Request):
        require_admin(request)
        return {"messages": store.recent_messages()}

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
