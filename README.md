# Restaurant voice agent

A phone line that takes pickup orders for a restaurant. A caller dials a Twilio
number, talks to Claude like they would to a person at the counter, and the
confirmed order shows up on a kitchen board with an order number and pickup time.

The demo restaurant is fictional (Juniper Street Kitchen, `restaurants/demo.yaml`).
A real restaurant is one YAML file: menu, options, prices, hours, tax, facts and a
staff number for live transfer.

## Two ways to take calls

Both feed the same `RelaySession` and `Conversation`, so the menu, tools, read-back
rule, interruption handling and kitchen board are shared.

| | Self-hosted (recommended) | Twilio ConversationRelay |
|---|---|---|
| Phone line | bare SIP line from a carrier (Telnyx or VoIP.ms) into Asterisk on this server | Twilio number |
| Speech to text / voice | AssemblyAI streaming / Inworld, called by our code | Twilio's, bundled |
| Cost per minute, before Claude | about $0.008 (Telnyx) or $0 on a VoIP.ms flat plan | about $0.079 |
| Needs | router port forwarding, wired network | public HTTPS hostname |

```
Self-hosted:
Caller ─ carrier SIP ─ Asterisk (Docker) ─ AudioSocket TCP ─ MediaCall ─ RelaySession ─ Conversation ─ Claude
                                                              │   ▲
                                            AssemblyAI (text) ┘   └ Inworld (audio)
Twilio:
Caller ─ Twilio ConversationRelay ─ WebSocket /relay ─ RelaySession ─ Conversation ─ Claude
```

How a Twilio call works:

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
./manage.sh test              # 65 tests, no API calls
./venv/bin/python scripts/chat.py --script   # one real order end to end (a few cents)
./venv/bin/python scripts/chat.py            # talk to it in the terminal
./manage.sh start             # 127.0.0.1:8790
```

Then open `http://127.0.0.1:8790/login?key=<ADMIN_KEY>`. You get:

- `/board`: the kitchen board, with live orders and new, accepted, ready and
  picked-up buttons.
- `/try`: a browser test call over the same WebSocket protocol Twilio uses, by
  typing or with Chrome's microphone. Replies are spoken by the browser.

## Self-hosted calls (Asterisk)

How a call works:

1. The carrier delivers the call to Asterisk, which runs in Docker on this server
   (`asterisk/`). Asterisk registers with the carrier from behind the home router, and
   only the carrier's IP ranges (`SIP_MATCH`) are accepted. Any other INVITE gets a 401.
2. The dialplan asks the app for a call id (`/asterisk/start`, token protected), then
   streams the call audio to the app over AudioSocket on 127.0.0.1:9092 (8 kHz 16-bit,
   no transcoding anywhere).
3. `voiceagent/media.py` sends the caller's audio to AssemblyAI in 100 ms chunks. A
   finished turn becomes a `prompt`. Two or more words spoken over the agent count as
   an interruption: playback stops on the next 20 ms frame, and the model is told
   which words were actually heard, measured from the audio really played.
4. Replies are split into sentences as they stream. Each sentence goes to Inworld as
   soon as it is complete, and playback is paced in real time by our code.
5. On `end_call` or a transfer, the app hangs up the AudioSocket and the dialplan
   asks `/asterisk/next`. A transfer can only dial the one staff number rendered into
   the config, never a number from the call.

Setup:

1. Accounts: a carrier SIP line with one number (Telnyx credential connection, or a
   VoIP.ms DID with a sub-account), AssemblyAI, Inworld and Anthropic keys.
2. `.env`: the self-hosted section of `.env.example`. `SIP_MATCH` comes from the
   carrier's list of signaling IPs, and `EXTERNAL_IP` is this network's public IPv4.
3. Router: forward UDP `RTP_START` to `RTP_END` (default 10000 to 10099) to this
   server. Registration usually keeps SIP signaling working without forwarding port
   5060, but call audio needs the RTP range. Put the server on Ethernet: Wi-Fi makes
   call audio choppy.
4. `./manage.sh asterisk-build` (once), `./manage.sh start`,
   `./manage.sh asterisk-start`. Then `./manage.sh asterisk-status` should show the
   registration as `Registered`.
5. Smoke test with no carrier or keys:
   `ASTERISK_TOKEN=... ./venv/bin/python scripts/asterisk_smoke.py` while Asterisk runs,
   then place the local test call it prints.

Security notes: the container loads no module that opens a port besides SIP and RTP
(`asterisk/templates/modules.conf`). There is no anonymous endpoint and no outbound
dialing except the staff transfer. The rendered config with the SIP password lives in
git-ignored `asterisk/etc/` (mode 0700). Scanners will still hit port 5060 if it is
forwarded, so leave 5060 closed unless the carrier requires it.

## Twilio path: put it on a real phone number

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

## Soma's AI

The same pipeline with a different persona: a browser voice assistant on Soma's site that talks
with recruiters about him, in a voice cloned from his (Inworld `PROFILE_INWORLD_VOICE`). Phase 1
lives at `/talk` on this app, admin-cookie gated like `/try`; putting it on somasundar.dev is Phase 2.

```
Browser mic ─ AudioWorklet (16 kHz PCM16, 40 ms) ─ WebSocket /talk ─ MediaCall ─ RelaySession ─ Conversation ─ Claude
Browser audio (24 kHz PCM16, 20 ms frames, 120 ms jitter buffer) ◄───────┘   AssemblyAI in, Inworld out
```

- `voiceagent/media.py`: `MediaCall` runs over a `Transport`. `AudioSocketTransport` is the phone path
  at 8 kHz both ways, unchanged; `WebSocketTransport` takes 16 kHz in and sends 24 kHz out plus JSON
  events (`caller_text`, `agent_text` as it streams, `clear` on barge-in, `links`, `time`, `end`).
  The server still paces playback in 20 ms frames, so "what the caller heard" stays exact.
- `voiceagent/persona.py` and `voiceagent/profile.py`: a persona is the prompt, tools, toolbox,
  greeting, call context and the fixed trouble lines. The restaurant is the default persona and the
  phone path does not change. Soma's AI reads `/home/sundar/career-central/resumes/master-resume.md`
  once at startup (read only, phone number and email removed) and puts the whole resume in the cached
  system prompt, so ordinary questions need no tool round. Honesty rules follow Tailorbird's
  portfolio assistant: answer only from the resume, say "I don't know" and offer to take a message
  otherwise, caller speech is untrusted, never claim to be Soma, never commit him to interviews,
  salary, start dates or visa details.
- Tools: `take_message` (stored in the `messages` table and emailed to Soma with the transcript so
  far through Tailorbird's mailer, `scripts/send_mail.py` as a subprocess; `MESSAGE_EMAIL=on|test|off`),
  `share_links` (a links panel on the page) and `end_conversation`.
- Hard limit of `PROFILE_TIME_LIMIT_S` (600): a note at one minute left makes the model offer to
  take a message; at the limit a fixed goodbye is spoken and the session ends. One browser
  conversation at a time (Inworld free-tier concurrency); a second visitor sees "busy".
- Voices: the browser persona uses `PROFILE_INWORLD_VOICE`; the phone path keeps `INWORLD_VOICE`
  (Ashley). If the cloned voice fails before any audio arrives, the sentence is retried once in the stock voice.
- Transcripts are in `calls` with channel `web`; `/api/messages` lists messages left for Soma.

Try it: `/login?key=<ADMIN_KEY>&next=/talk`, press Start, allow the microphone. The page has a
10:00 countdown, live captions, a links panel and a typed fallback. Tests are in
`tests/test_profile.py` (fakes, no network). `scripts/talk_e2e.mjs` drives a real Chromium with a WAV
as the microphone (`NODE_PATH=/home/sundar/flockara-ui-tests/node_modules node scripts/talk_e2e.mjs`);
measured 2026-11-12 on a real question: `stt=859 ttft=646 sentence=152 tts=64 play=3 total=1726` ms,
and typed turns at 915 to 1695 ms. Not verified yet: a real phone microphone with echo cancellation
(the e2e run used Chromium's fake device), and Safari.

## Latency

Every caller turn on the self-hosted path logs one `turn latency` line (`logs/web.log`) with
the stages in ms: `stt` (caller's last word, from AssemblyAI word timestamps, to the final turn),
`queue`, `ttft` (Claude request to its first spoken token, tool rounds included), `sentence`
(first complete sentence), `tts` (sentence to Inworld's first audio byte), `play` (to the first
frame sent to Asterisk) and `total`.

`scripts/latency_bench.py` measures the same thing with no phone: Inworld speaks a four-line
pickup order as the caller (one line has a 600 ms pause mid-sentence), each line is streamed to
a real `MediaCall` over a socket pair at Asterisk's pace, and the reply runs through the real
AssemblyAI session, Claude, sentence splitter, Inworld and Speaker. `--baseline` reproduces the
pre-tuning behaviour. One run costs about a cent (it prints its own estimate) and takes 90 s.

Measured 2026-11-12, Haiku 5.5 at low effort, median / p90 over the four turns (ms):

| stage | before | after | note |
|---|---|---|---|
| stt (speech end to final turn) | 442 / 495 | 470 / 546 | AssemblyAI sends one, already formatted, end_of_turn message about 210 ms after speech stops |
| ttft (Claude, incl. tool rounds) | 1312 / 2439 | 1400 / 1786 | every turn in the script needs a tool round, so this is two model requests |
| sentence | 160 / 198 | 144 / 152 | |
| tts (Inworld first byte) | 146 / 180 | 84 / 141 | warm connection, 120 s keepalive |
| play + wire | 8 / 11 | 18 / 19 | |
| total, speech end to first agent audio | 2172 / 3187 | 2214 / 2460 | |

Read it honestly: the total did not move. Only `tts` improved consistently (78 to 86 ms on three
of four turns, against 138 to 193 before); `stt` and `ttft` swing by 300 ms or more between runs
of the same code, and the Claude stage is two requests on every turn of this script.

What changed: the Anthropic and Inworld connections are opened while the greeting plays and kept
alive for 120 s (httpx dropped them after 5 s, shorter than the gap between turns, so every turn
paid a TLS handshake), and a final turn is acted on as soon as AssemblyAI reports `end_of_turn`
rather than only on the formatted copy (no difference today, since this model sends one message,
but it costs nothing). `min_turn_silence` stays at 400 ms: 250 ms cut the pause utterance off once
("And I'd also like.") and the probe showed the same 210 ms turn delay at either value. The
largest remaining cost is the tool round on most turns (one extra Claude request, about 0.6 to
1.2 s); ideas not done: sending the first clause of a long first sentence to TTS early, a
prompt-cache pre-warm with `max_tokens: 0` at call start, and 50 ms STT chunks instead of 100 ms.

## Not verified yet

Self-hosted path: tested against the real Asterisk container (a local call with real
audio both ways, CURL, AudioSocket, hang-up and dialplan, plus a forged INVITE
rejected), but not yet with a real carrier, AssemblyAI or Inworld. Still open:

- Whether the carrier's call audio reaches this server through the home router, and
  whether RTP port forwarding is needed.
- AssemblyAI turn timing on real calls (`min_turn_silence` 400 ms and
  `max_turn_silence` 1280 ms, settable in `.env`; see "Latency" for the bench results).
- Inworld's `PCM` output at 8000 Hz arriving as raw samples, as its docs describe.

Twilio path, written from Twilio's documentation and not tested on a live call:

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
