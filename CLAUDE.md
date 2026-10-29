# Restaurant voice agent: working notes

Phone ordering agent: Twilio ConversationRelay, FastAPI, Claude with tools, SQLite.
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
- No em dashes or en dashes in code, comments, docs or commit messages.

## Testing

`./manage.sh test` runs the stdlib unittest suite with a scripted fake model and
makes no API calls. `scripts/chat.py --script` is the one real end-to-end order and
costs a few cents. Run it after any prompt or tool change. Validate restaurant files
with `python -m voiceagent.menu <file>`.

## Running

`./manage.sh {start|stop|restart|status|test}`, bound to 127.0.0.1:8790, logs in
`logs/web.log`. `/healthz` reports whether Claude credentials and Twilio signing are
configured.
