# Restaurant voice agent

A phone line that takes pickup orders for a restaurant. A caller dials a Twilio
number, talks to Claude like they would to a person at the counter, and the
confirmed order shows up on a kitchen board with an order number and pickup time.

The demo restaurant is fictional (Juniper Street Kitchen, `restaurants/demo.yaml`).
A real restaurant is one YAML file: menu, options, prices, hours, tax, facts and a
staff number for live transfer.

## How a call works

```
Caller ─ phone ─ Twilio ConversationRelay ─ WebSocket /relay ─ RelaySession ─ Conversation ─ Claude
                 (speech to text, text to speech)               (protocol)    (tools, history)
                                                                                    │
                                                              Order cart + SQLite ──┴── /board
```

1. Twilio hits `POST /twilio/voice`. The signature is checked, then TwiML connects
   the call to ConversationRelay with a WebSocket URL signed for that one CallSid.
2. Twilio speaks the greeting, transcribes the caller and sends each utterance as a
   `prompt`. The agent streams its reply back as `text` tokens, which Twilio starts
   speaking before the reply is finished.
3. The model works the order through tools (`add_item`, `review_order`,
   `place_order`, ...). Code, not the model, owns every price, total, pickup time
   and order number.
4. `end_call` or `transfer_to_staff` sends `end` once the last sentence has had
   time to play. Twilio then calls `/twilio/relay-done`, which hangs up or dials the
   staff number.

## Design decisions

- **The model never does math or invents facts.** The menu, allergens, hours and
  facts come from the YAML file. Totals come from `order.py` in integer cents.
  Allergens that aren't recorded are labelled "not recorded", and the prompt says
  to offer a person rather than guess.
- **Read back before it counts.** `place_order` is refused unless `review_order` ran
  after the last change to the cart, so the caller's "yes" always matches what the
  kitchen gets. Once an order is placed it is frozen.
- **Interruptions are real.** When the caller talks over the agent, the reply in
  progress is cancelled. The unfinished assistant message is dropped instead of
  saved half done. The next turn tells the model what the caller actually heard
  (Twilio's `utteranceUntilInterrupt`).
- **Append-only history.** The system prompt and tool list are frozen per call, and
  a tool call and its result are always appended together. Claude Opus 5.5 checks
  that the earlier conversation is unchanged whenever it reads back its own
  thinking, and the prompt cache relies on the same thing. The caller ID arrives as
  a mid-conversation system message, not as an edit to the system prompt. Tests
  assert that every request's history extends the previous one exactly.
- **Malformed tool input is an error the model can fix,** not a crash. Tool inputs
  are validated against their schemas because eager input streaming skips the
  server-side check.
- **No card numbers on the phone.** Payment is at pickup, and the prompt stops a
  caller who starts reading a card.
- **Safe failure.** If an API error, a refusal, a tool loop or a missing key gets in
  the way, the caller hears a short apology, and is transferred when a staff line
  is configured.

## Run it

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env          # set ANTHROPIC_API_KEY and ADMIN_KEY at least
./manage.sh test              # 43 tests, no API calls
./venv/bin/python scripts/chat.py --script   # one real order end to end (a few cents)
./venv/bin/python scripts/chat.py            # talk to it in the terminal
./manage.sh start             # 127.0.0.1:8790
```

Then open `http://127.0.0.1:8790/login?key=<ADMIN_KEY>`. You get:

- `/board`: the kitchen board, with live orders and new, accepted, ready and
  picked-up buttons.
- `/try`: a browser test call over the same WebSocket protocol Twilio uses, by
  typing or with Chrome's microphone. Replies are spoken by the browser.

## Put it on a real phone number

1. Twilio account, a voice number, and ConversationRelay onboarding
   (Console > Voice > ConversationRelay; per Twilio, access is not instant).
2. A public HTTPS hostname for this app, for example a new Cloudflare Tunnel route
   to `127.0.0.1:8790`. WebSockets must pass through (`wss://`).
3. `.env`: `PUBLIC_BASE_URL`, `TWILIO_AUTH_TOKEN`, `ADMIN_KEY`, a fixed `RELAY_SECRET`.
4. On the number: "A call comes in" set to Webhook, `POST https://<host>/twilio/voice`.
5. `curl https://<host>/healthz` should show `claude_credentials: true` and
   `twilio_signing: true`.

## Onboarding a real restaurant

Copy `restaurants/demo.yaml`, edit it, then run
`./venv/bin/python -m voiceagent.menu restaurants/<name>.yaml`. That validates the
file (ids, prices in whole cents, option rules, hours) and prints the menu exactly
as the model sees it. Set `RESTAURANT_FILE` and restart. Things to collect from the
restaurant: every modifier and its price, which allergens are confirmed, the
last-order cutoff, and a staff number for transfers.

## Not verified yet

These were written from Twilio's documentation, not tested on a live call:

- The ConversationRelay attribute names `hints` and `dtmfDetection`, and how
  `utteranceUntilInterrupt` behaves.
- The 160-words-a-minute estimate used to wait before hanging up.
- Latency at `EFFORT=low` on `claude-opus-5-5`. `scripts/chat.py` prints the time
  to first words and the total per turn. If it feels slow on a phone, measure
  `claude-sonnet-5-5` or `claude-haiku-5-5` (set `MODEL`) before changing anything else.

## Roadmap

Delivery (addresses and zones), scheduled pickup times, SMS confirmation with the
order number, POS or printer handoff instead of the board, menu items marked out of
stock from the board, more than one restaurant per deployment, and a transcript
review page for tuning the prompt.
