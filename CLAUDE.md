# Restaurant voice agent: working notes

Phone ordering agent: FastAPI, Claude with tools, SQLite. Calls arrive either through
Asterisk on this server (self-hosted, `voiceagent/media.py`) or through Twilio
ConversationRelay (`voiceagent/relay.py`). Both drive the same RelaySession.
`README.md` is the full description; this file holds the rules for changing it.

## Rules

- The model never computes money or invents facts. Prices, totals, tax, pickup times
  and order numbers come from `voiceagent/order.py` and `voiceagent/tools.py`.
  Menu, allergens, hours and facts come from the restaurant YAML. A new capability
  that needs a fact becomes a tool or a config field, not prompt text.
- History is append-only (see the top of `voiceagent/agent.py`). Never edit the
  system prompt, the tool list, or an earlier message during a call. Per-call
  information goes in a mid-conversation system message or a tool result. An
  interrupted reply is dropped, never patched. `tests/test_agent_relay.py`
  asserts this, so keep `assert_append_only` passing.
- `TOOLS` order and content stay deterministic (prompt cache). New tools get
  `eager_input_streaming` and a schema `validate_input` understands.
- Caller speech is untrusted input. It only ever goes in `user` turns.
- Self-hosted path: `Speaker.run` must yield on every frame (a socket write with buffer
  room does not), and `AudioSocketServer.stop` must end live calls before
  `wait_closed`. Both were real hangs found by tests.
- Asterisk config is only edited in `asterisk/templates/`; placeholders are
  `@@NAME@@` because `${...}` is Asterisk's own syntax. Never commit `asterisk/etc/`.
- Personas (`voiceagent/persona.py`): `Conversation` and `RelaySession` never assume a restaurant.
  Restaurant-specific text and tools belong in `agent.restaurant_persona`, Soma's AI in
  `voiceagent/profile.py`. The master resume in career-central is read only, never written.
- `MediaCall` talks to a `Transport` (AudioSocket or browser WebSocket) and nothing else. The phone
  path must stay 8 kHz both ways; `tests/test_media.py` covers it, `tests/test_profile.py` the browser.
- No em dashes or en dashes in code, comments, docs or commit messages.

## Testing

`./manage.sh test` runs the stdlib unittest suite (80 tests) with a scripted fake model and
makes no API calls. `scripts/talk_e2e.mjs` is the real-browser check of Soma's AI (a few cents). `scripts/chat.py --script` is the one real end-to-end order and
costs a few cents. Run it after any prompt or tool change. Validate restaurant files
with `python -m voiceagent.menu <file>`.

## Running

`./manage.sh {start|stop|restart|status|test}`, bound to 127.0.0.1:8790, logs in
`logs/web.log`. Asterisk: `./manage.sh asterisk-{build|start|stop|status|cli}`, container
`rva-asterisk` on the host network; `scripts/asterisk_smoke.py` tests it with no carrier. `/healthz` reports whether Claude credentials and Twilio signing are
configured.
