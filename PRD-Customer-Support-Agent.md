# PRD: Omnichannel Customer Support Agent

## 1. Summary

A support agent that answers customer contacts across four channels -- phone calls, SMS, web chat and email -- from a knowledge base, and hands the contact to a human whenever it should not be the one answering. The interesting half is the second one. Deciding to answer is easy; deciding *not* to, reliably, is the part that determines whether a support org can actually put this in front of customers.

## 2. Contacts

| Name | Role | Comment |
|---|---|---|
| Sarath Chandu Chitti (chsc30010) | Owner / sole builder | Builds and maintains the LLM-backed projects in this workspace (AI News Digest, Job Signal Agent, LLM Eval Pipeline, personal RAG, text-to-SQL clarifier, this agent) |

## 3. Background

Every support-bot demo answers the happy path. The failure that matters in production is the other one: a customer with a problem the bot half-recognises, who gets a confident, wrong, well-written answer and has to start over with a human twenty minutes later. That single interaction costs more goodwill than the fifty easy deflections earned.

Two things in this workspace led here. The LLM Eval Pipeline established that a component nobody measures is a component nobody can improve -- it exists specifically so a prompt change can be scored instead of eyeballed. The Job Signal Agent then applied that to a real ranker and found two bugs the eval caught and manual testing had not. This project takes the same discipline into a domain where being wrong has an actual cost to somebody other than me.

Why now, and why voice specifically: most of the "AI support agent" work I have seen treats voice as text with a phone number attached. It isn't. A URL is useless out loud, an order reference has to be spelled rather than pronounced, a caller needs a pause to interrupt at, and a speech recogniser hands you a transcript with a confidence score that a text channel never has. I wanted to find out how much of that is real engineering and how much is hand-waving. It is mostly real, and it all lives in about 120 lines.

## 4. Objective

Deflect the routine tier-1 contact volume that does not need a person, while getting every contact that *does* need a person to one -- without the customer having to ask twice or repeat themselves.

The second clause is the constraint the whole design serves. A deflection rate is easy to move by answering more things; it only means something alongside a handoff rate you trust.

**Key Results (achieved on the 60-case labelled set, deterministic path, no model calls):**
- Escalation recall of 100% (18/18). Every contact a support lead labelled as needing a human reached a human.
- Intent accuracy of 96.7% (58/60) across nine intents and four channels.
- 88.1% of answerable contacts resolved from the correct help centre article (37/42).
- Escalation precision of 85.7% -- three unnecessary transfers out of 42 answerable contacts. This is the number to improve; recall is the number to hold.
- The full pipeline runs with no credentials, no network and no vendor account, so the baseline is reproducible by anyone who clones it.

**Key Results (targets for a real deployment, not yet measured):**
- Under 2% of resolved contacts reopened or re-contacted within 24 hours.
- Median handle time for escalated contacts lower than the same intent handled cold, because the agent already collected the context.
- Escalation recall stays at or above 98% on a held-out set the agent was never tuned against.

## 5. Market Segment(s)

The intended buyer is a support organisation of roughly 20-200 agents, at a company with a real help centre and a contact mix that is mostly repeat questions: where is my order, how do I return this, why was I charged twice, my device will not connect. Consumer hardware, subscription services, e-commerce.

Constraints that shape the design:
- They already have a help centre. They do not want to rewrite it as intents, decision trees or training data. The knowledge base here is ordinary markdown articles with headings.
- They already have a phone number, and it is on Twilio or something like it. Voice cannot be a later phase.
- Their tolerance for a wrong answer is close to zero, and their tolerance for an unnecessary transfer is fairly high. Those two facts should be visible in the code, not just in a slide.
- Someone who is not an engineer -- a support lead -- needs to be able to read and challenge the escalation rules.

Being honest about scope: this has not run a single contact of real traffic. It is a working system with a measured baseline, not a deployed one.

## 6. Value Proposition(s)

**Job to be done:** "When a customer contacts us on any channel, answer it from what we have already written if we genuinely can, and get it to the right person with the full context if we cannot -- and tell me which of those you did and why."

**Gains:**
- One pipeline across four channels. Adding email did not mean a second agent, a second set of rules or a second thing to keep in sync; a channel is an adapter at each end.
- A handoff that arrives with a summary the receiving agent reads in two seconds -- intent, sentiment, why the bot gave up, what the customer opened with and last said -- plus the transcript. Nobody is asked to explain twice.
- Every answer cites the article and section it came from, so a wrong answer is traceable to a content problem rather than to a black box.
- An escalation policy of seven ordered rules in one readable file, which a support lead can audit without being able to write Python.
- A number for the claim. "It escalates appropriately" is an opinion; 100% recall at 85.7% precision on 60 labelled contacts is a measurement, and the harness shows the five cases it got wrong.

**Pains avoided:**
- The confident wrong answer. The answerer can only emit sentences that exist in an article, and when nothing supports an answer the contact goes to a person rather than to an improvised one.
- The customer who has to repeat the whole story to the human they get transferred to.
- A voice experience that reads URLs aloud, pronounces "KH-482913" as a number, and talks over the caller with no pause to interrupt at.
- Being locked to one model vendor, or to any vendor: the deterministic path is a complete working system, and the language model is an upgrade to it rather than a dependency of it.

**Why this over the alternatives:** Intercom Fin, Zendesk AI and Ada all do a version of this and do it well. What I could not find in any of them is a legible answer to "under exactly what conditions will you not answer, and what is your measured miss rate on that?" -- the escalation logic is a confidence slider and a promise. That question is the product here. This is also self-hosted, which matters for a support org that would rather not send every customer transcript to a third party, and it is roughly 2,500 lines rather than a platform.

## 7. Solution

### 7.1 UX / Prototypes

There is no UI to build. There are four entry points and one conversation.

**Voice.** Twilio calls `/twilio/voice`; the caller hears a greeting *inside* a `<Gather>`, so someone who already knows what they want can talk over it. Each turn posts a transcript and a confidence score to `/twilio/voice/turn`. The agent replies as SSML and re-opens the `<Gather>`, or says why it is transferring and dials out. Pressing zero works at every point, including on a line too noisy to recognise speech.

**SMS.** Twilio posts to `/twilio/sms`; the reply comes back as one or more `<Message>` elements, segmented at 153 characters and capped at three parts.

**Chat and email.** JSON in, JSON out, with the rendered reply plus the intent, the confidence, the citations and the escalation decision, so a widget or a mailbox integration can do whatever it likes with them.

For development there is a CLI, because iterating on a support agent through an actual phone is unreasonable:

```
support-agent ask "my kitchen camera keeps going offline" --channel voice
support-agent simulate fixtures/transcripts/call-angry-billing.jsonl --channel voice
support-agent kb "my camera will not connect"
```

`simulate` replays a scripted call turn by turn, including keypresses and low-confidence transcripts, through exactly the same code the phone leg uses.

### 7.2 Key Features

- **One normalised pipeline.** Every channel becomes an `InboundMessage`; the agent only ever produces an `AgentReply`. Classify, retrieve, answer, decide. Nothing in the middle branches on channel.
- **Seven ordered escalation rules**, first match wins: the customer asked for a person; the customer is angry; it is a complaint; intent is unknown or below the confidence floor; no passage supports an answer; the help centre itself says the step is human-only; too many turns without resolving it. The first four are evaluated *before* any lookup, so an angry caller who asked for a person is not made to wait through a retrieval and a model call whose result gets thrown away.
- **Human-only content, marked in the content.** An article section can declare `handoff_sections` in its front matter -- changing the email address on an account, recovering a lost second factor. The agent still gives the grounded answer, then hands over, because those steps genuinely cannot be finished without a person.
- **Two implementations behind each interface.** A deterministic classifier (weighted phrase lexicon, negation folding so "cannot log in" and "can't log in" are one thing) and extractive answerer that quotes articles verbatim and therefore cannot hallucinate; and a Claude-backed pair that falls back to them on rate limits, connection errors, refusals and unparseable output. The deterministic pair is the baseline the model has to beat, not a placeholder.
- **BM25 retrieval over the help centre**, section by section, in pure standard library. Section headings are weighted above article titles, because the heading is what says which part of the article answers the question.
- **Voice-specific rendering.** URLs dropped and offered as a text instead; references spelled with `<say-as interpret-as="characters">`; 350ms between sentences and 200ms at the commas of a long one; two sentences on voice and SMS against four by email. Twilio's own recognition confidence discounts classification confidence, so a misheard "cancel" drifts towards a human rather than towards a cancellation.
- **No repeating itself.** Passages already used in a conversation are excluded from later turns, so "I already tried that" gets the next step in the article, and running out of article produces an honest handoff instead of the same paragraph again.
- **Routing on the conversation, not the turn.** The turn that triggers a handoff is usually the least informative one -- a keypress, or "just get me a person" -- so the queue is chosen from what the conversation established.
- **Webhooks that fail closed.** Twilio signatures verified in fifteen lines of standard library; unsigned requests refused; a 503 rather than silent acceptance when the token and public URL are not both configured. These endpoints accept a phone call and can transfer it.
- **An eval harness**, scoring routing, escalation and retrieval separately, with a confusion matrix, a failure list, a threshold sweep and non-zero exit codes for CI.

### 7.3 Technology

Python 3.10+, FastAPI and uvicorn, the official `anthropic` SDK when a model is configured, `requests` and `python-dotenv`. No Twilio SDK -- TwiML is four XML tags and signature verification is `hmac` -- and no vector database, because 52 passages do not need one. Conversation state is an in-memory dict with a two-hour TTL. `pytest` for the 67 unit tests.

### 7.4 Assumptions

- The help centre is good enough to answer from. Two failures during development were content problems, not code problems: the sign-in article said "sign in" everywhere and never "log in", and the pricing section was headed "The plans" rather than anything containing "how much". Both were fixed in the article. A real deployment will find more of these, and the harness is how it will find them.
- A support lead is willing to own the escalation rules and the labelled set. Without that, the thresholds drift into whatever the last person guessed.
- Single process is enough to start. Swapping the conversation store for Redis is a one-file change, and the file exists mainly to make that obvious.
- The intent taxonomy of nine intents transfers to another company with renaming rather than restructuring. Untested.
- Twilio's speech recognition is good enough on a normal phone line. The agent discounts low-confidence turns, but there is a quality floor below which nothing above the recogniser helps.
- The 96.7% intent accuracy is an upper bound, not a generalisation estimate. The lexicon gaps were found *by* the golden set and closed by editing the lexicon, so the set was used to tune the thing it measures. A held-out second set is the honest next step and is listed as such below.

## 8. Release

**Now (shipped):** All four channels end to end. Twilio voice and SMS webhooks with signature verification, barge-in, DTMF, silence re-prompting, and transfer or enqueue. JSON endpoints for chat and email. Nine intents, four-level sentiment, BM25 retrieval over 15 articles, extractive and Claude-backed answerers, seven escalation rules with queue and priority routing, per-channel rendering, multi-turn state with repetition suppression. 60-case golden set, three-axis eval harness with CI gates, 67 unit tests, CLI with `ask`, `simulate`, `kb` and `serve`. Pushed as its own private repo.

**Next:** A held-out labelled set that the lexicon was never tuned against, to convert the 96.7% from an upper bound into an estimate. Then run the same 60 cases through the Claude path and put the two side by side -- the whole point of building both was to be able to answer whether the model earns its cost, and that comparison has not been run yet.

**Later:** Multi-turn evaluation, because the golden set is single-turn while the agent has behaviour specifically for later turns and none of it is scored. A tool layer for order lookup and refunds under a policy cap, which needs a permission model and an audit log before it needs any code. A real handoff integration that opens a ticket carrying the `Escalation` object rather than only transferring the call. Redis-backed conversation state for more than one worker.
