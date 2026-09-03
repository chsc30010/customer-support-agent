# Omnichannel Customer Support Agent

A support agent that takes phone calls, texts, web chat and email through one
pipeline, answers from a knowledge base with citations, and hands over to a
human when it should not be the one answering.

The handover is the point. Plenty of support bots can answer a question. This
one can tell you that across 60 hand-labelled contacts, **every contact that
needed a person reached a person**, and it resolved 88% of the rest from the
right help centre article.

```
$ support-agent ask "my kitchen camera keeps going offline" --channel voice

customer: my kitchen camera keeps going offline
  [technical_issue 0.74 | neutral | heuristic]
  agent: Unplug the camera for ten seconds and plug it back in. Wait two full
         minutes for it to rejoin the network. Does that sort it, or is there
         something else?
  source: Camera shows offline or will not connect - First steps (4.90)

$ support-agent ask "this is the third time and nobody has called me back" --channel voice

customer: this is the third time and nobody has called me back
  [complaint 0.82 | frustrated | heuristic]
  agent: I want this logged properly rather than handled by me. Let me put you
         through to someone now.
  --> HANDOFF to escalations [high] because intent_requires_a_human
```

## How it works

```
                 ┌───────────── one pipeline ──────────────┐
 voice  ─┐       │                                         │       ┌─ SSML + <Gather>
 sms    ─┤       │  classify ─► retrieve ─► answer ─► policy       ├─ SMS segments
 chat   ─┼─► normalize   │          │           │        │  ─► render ─┼─ markdown + link
 email  ─┘       │       │          │           │        │       └─ subject + body
                 │       │          │           │        └─ hand over, or keep talking
                 │       │          │           └─ grounded in passages, or nothing
                 │       │          └─ BM25 over 52 help centre passages
                 │       └─ lexicon, or Claude with the lexicon behind it
                 └─────────────────────────────────────────┘
                                     │
                                  evals/ ─► 60 labelled contacts, scored three ways
```

A phone call, a text and an email are four different transports and one
conversation. Everything channel-shaped happens at the two ends -- `telephony/`
on the way in, `render/` on the way out -- so the middle serves a call and an
email with no branch between them.

| Module | Job |
|---|---|
| `models.py` | The normalized types. Every channel becomes an `InboundMessage`; the agent only ever produces an `AgentReply`. |
| `classify/` | Intent and sentiment. A deterministic lexicon, or Claude with the lexicon behind it. |
| `kb/` | The help centre on disk, indexed with BM25. Pure standard library, no vector database. |
| `answer/` | Drafts a reply **only** from retrieved passages. "I cannot answer this" is a valid outcome. |
| `policy.py` | Seven ordered rules deciding whether a human takes this one, which queue, and how urgently. |
| `render/` | SSML for the phone, 153-character segments for SMS, markdown for chat, a subject line for email. |
| `telephony/` | Twilio signature verification, TwiML, and the speech seam. |
| `server.py` | FastAPI: two Twilio webhooks, two JSON endpoints, one health check. |
| `evals/` | The golden set and the harness. See [evals/README.md](evals/README.md). |

## When it hands over

Seven rules, in order, first match wins. They live in one readable file
(`policy.py`) so a support lead can audit them without reading Python well.

| Rule | Why |
|---|---|
| The customer asked for a person | Nothing outranks this, including a perfectly good answer. |
| The customer is angry | An angry customer does not want a better article. |
| It is a complaint | Complaints need a named owner, not a resolution. |
| Intent is unknown or below the confidence floor | Guessing is worse than admitting it. |
| No knowledge base passage supports an answer | The only alternative is inventing one. |
| The help centre says a human must do this step | Some processes are human-only by design, and the article says so. |
| Several turns without resolving it | Going round in circles is its own failure. |

The first four are checked before any lookup happens, so an angry customer who
asked for a person is not made to wait through a retrieval and a model call
whose result gets thrown away.

The handover carries a note the receiving agent can read in two seconds --
intent, sentiment, why the bot gave up, what the customer opened with and last
said -- plus the full transcript, so nobody is asked to explain twice. Routing
uses what the *conversation* was about rather than the last turn, because the
turn that triggers a handoff is usually the least informative one.

## Voice is not text read out

Voice reaches the agent as a transcript and leaves as speech markup. The
differences that bite are handled in `render/voice.py`:

- **URLs are dropped.** A spoken URL is noise, so the agent offers to text it instead.
- **References are spelled.** `KH-482913` becomes `<say-as interpret-as="characters">`, because a caller has to write it down.
- **Sentences get breaks.** `350ms` between sentences, `200ms` at the commas of a long one. Callers barge in at pauses.
- **Replies are shorter.** Two sentences on voice and SMS, three in chat, four by email -- the same passage becomes a different length of answer.
- **Twilio's own confidence is used.** A turn the recogniser was unsure of lowers classification confidence, which pushes a misheard "cancel" towards a human rather than towards a cancellation.
- **Pressing zero works.** `<Gather input="speech dtmf">`, so the escape hatch every caller already knows is there even on a line too noisy to hear.

Long sentences are *not* chopped into shorter ones. An earlier version split
them at their commas; it read badly in the transcript and changed what the
customer was told. The sentence is fine, it just needs breathing room.

## The classifier and the answerer

Both have two implementations behind one interface, so "does the model help?"
is a question with an answer.

**Deterministic** -- the default. No credentials, no network. Weighted phrase
matching for intent and sentiment, and an answerer that quotes the knowledge
base verbatim. It cannot hallucinate, because it only ever emits sentences that
exist in an article. It scores 96.7% intent accuracy and 100% escalation recall
on the golden set. This is not a placeholder: it is the baseline the model has
to beat, and it is what answers when the model is unreachable.

**Claude** -- `LLM_PROVIDER=anthropic`. One schema-constrained call per turn;
`output_config.format` guarantees the JSON parses, which removes the "the model
wrapped its JSON in prose" failure entirely. Classification runs at `low`
effort because it is high volume and latency-sensitive; drafting runs at
`medium`, because a wrong answer to a customer costs more than the tokens. The
answerer is instructed to return `answerable: false` rather than improvise.
Every failure path -- rate limit, connection error, refusal, unparseable output
-- falls back to the deterministic implementation, so the line degrades in
quality rather than stopping.

## Quick start

```bash
cd customer-support-agent
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

support-agent ask "how do I get a return label" --channel chat
support-agent simulate fixtures/transcripts/call-camera-offline.jsonl --channel voice
support-agent kb "my camera will not connect"

python evals/run_evals.py --show-failures --sweep
python -m pytest
```

None of that needs a Twilio account, an API key, or a network connection.

## Taking real calls

```bash
cp .env.example .env      # fill in TWILIO_AUTH_TOKEN and PUBLIC_BASE_URL
support-agent serve
```

Point a Twilio number at the webhooks:

| Twilio setting | URL |
|---|---|
| Voice, "A call comes in" | `POST https://<your-host>/twilio/voice` |
| Messaging, "A message comes in" | `POST https://<your-host>/twilio/sms` |

Every webhook is verified against `X-Twilio-Signature` before it is read.
Unsigned requests are refused, and when `TWILIO_AUTH_TOKEN` and
`PUBLIC_BASE_URL` are not both set the endpoints return 503 rather than quietly
accepting anything. These endpoints accept a phone call and can transfer it;
left open they are a free way for anyone to drive the agent and fill the
escalation queue.

`PUBLIC_BASE_URL` must match the URL configured in Twilio exactly -- scheme,
host, port and path -- because that string is part of what is signed, and
behind a tunnel the URL the app sees is not the URL Twilio called.
`ALLOW_UNSIGNED_WEBHOOKS=1` exists for local curl testing and belongs nowhere
else.

`/chat` and `/email` take JSON and are not signed. Put your own session auth in
front of them.

## Configuration

Every setting has a default that works offline; missing credentials degrade a
capability rather than crashing the agent. The three that change behaviour:

| Variable | Default | Effect |
|---|---|---|
| `MIN_INTENT_CONFIDENCE` | `0.35` | Below this the agent hands over instead of guessing. |
| `MIN_RETRIEVAL_SCORE` | `2.5` | Below this there is nothing to ground an answer in. `--sweep` shows the tradeoff. |
| `MAX_TURNS_BEFORE_HANDOFF` | `4` | Customer turns before it stops trying. |

## What this does not do

- **No tool actions.** It cannot look up an order, issue a refund or open a ticket. It answers from the help centre and routes everything else. Adding actions means adding a permission policy and an audit log, which is a larger piece of work than it looks.
- **Conversations live in one process.** `ConversationStore` is an in-memory dict with a two hour TTL. Running more than one worker means replacing that one file with Redis, and nothing else.
- **The handoff is a transfer, not an integration.** An escalated call is dialled or queued; no ticket is created in a CRM. The `Escalation` object carries everything such an integration would need.
- **The knowledge base is 15 articles for one fictional company.** Swapping in a real one means replacing `kb/articles/` and relabelling the golden set.
- **Multi-turn behaviour is not scored.** The golden set is single-turn. The agent has specific behaviour for later turns -- suppressing passages it has already used, carrying intent across a short follow-up -- and `fixtures/transcripts/` is for looking at that by hand.
- **English only.** The lexicon, the stemmer and the voice settings all assume it.
