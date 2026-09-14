# Code review

**Date:** 2026-09-14
**Commit reviewed:** [`b2c2adc`](https://github.com/chsc30010/customer-support-agent/tree/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47)
**Scope:** all of `src/support_agent/` -- not a diff. The request path, conversation state, classification, answering, rendering and configuration were read in full.

Line links below point at the commit that was reviewed, so they stay accurate after fixes move the code.

## Summary

Ten bugs. None of them is caught by the 110 unit tests or by the eval gate in CI -- the golden set is single-turn and single-process, and one existing test (`test_poor_speech_recognition_lowers_confidence`) actively hides finding 5.

Six affect the agent as it runs today, on the deterministic path. Four stay dormant until `LLM_PROVIDER=anthropic` is switched on, and should be fixed before it is.

Nine were confirmed by reading the code. Finding 6 is reasoned from how Twilio resolves relative URLs and has not been reproduced against a live deployment.

| # | Finding | Affects |
|---|---|---|
| 1 | Anonymous chats share one conversation | today |
| 2 | Escalated conversations keep running through the bot | today |
| 3 | Mentioning "agent" or "customer service" forces a handoff | today |
| 4 | Conversations expire two hours after they start, not after they go quiet | today |
| 5 | Poor speech recognition can *raise* classifier confidence | today, voice |
| 6 | Voice calls fail under a path prefix | today, some deployments |
| 7 | One slow model call blocks the whole server | model path |
| 8 | The model answerer ignores the retrieval floor | model path |
| 9 | The model stays off without an API key, even with valid credentials | model path |
| 10 | Pronunciation fixes corrupt words that merely contain them | model path |

---

## Affects the agent today

### 1. Anonymous chats share one conversation

[`server.py:210`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/server.py#L210)

`/chat` falls back to the fixed id `chat:anonymous` whenever a request has no `conversation_id`, so every such customer is written into the same conversation.

**Scenario.** A widget forgets to send `conversation_id`. Customer A asks how to get a return label; customer B says they were charged twice. Both land in one conversation. B's retrieval skips passages that were served to A. Turns add up across everyone, so the looping rule hands every anonymous customer to a human after four messages in total. And B's escalation summary and transcript contain A's messages -- one customer's words shown to the agent picking up another's contact.

`/email` has the same flaw when `from` is missing: every such message lands in `email:unknown` ([line 224](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/server.py#L224)).

**Fix.** Refuse requests without an identifier (400), or mint a fresh id per session and return it in the response so the client can send it back.

### 2. Escalated conversations keep running through the bot

[`agent.py:97`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/agent.py#L97)

`conversation.escalated` is set on handoff and read nowhere, so a handoff changes nothing about how later messages are handled.

**Scenario.** An SMS thread is escalated ("I have passed this to our team"). The customer texts again -- "also, where is my order?" The whole pipeline runs and the bot answers into a thread a person now owns. From the fourth customer message onwards the looping rule fires on every further text, so each one files another escalation into the same queue.

**Fix.** At the top of `handle`, short-circuit when the conversation is already escalated: acknowledge that a person has it, attach the message to the existing handoff, and do not classify, answer or escalate again.

### 3. Mentioning "agent" or "customer service" forces a handoff

[`lexicon.py:243`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/classify/lexicon.py#L243)

`agent` and `customer service` ([line 254](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/classify/lexicon.py#L254)) are treated as requests for a person, matched by prefix ([`heuristic.py:63`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/classify/heuristic.py#L63)). That flag outranks every other policy rule, so saying the word is enough.

**Scenario.** "The agent I spoke to yesterday said my refund was processed -- when will it show up?" is handed to a human instead of being answered from the returns article. So is "I emailed customer service last week about my order". The prefix match also fires on "agents" and "agentic".

Bare `agent` was added to recover golden-set case g54, where a caller says only "agent". That case is real; the fix went too wide.

**Fix.** Drop the bare entries and keep the phrasal ones ("speak to an agent", "live agent"). Treat a lone word as a request for a person only when it is essentially the whole message -- "agent", "a person please".

### 4. Conversations expire two hours after they start, not after they go quiet

[`conversations.py:56`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/conversations.py#L56)

`_expire` measures age from `started_at`. There is no last-activity field, so an active conversation is dropped two hours after its first message however recently the customer wrote.

**Scenario.** An SMS thread starts at 10:00. At 12:05, still in the middle of the problem, the customer replies "yes, the black one". The conversation is thrown away and a new one begins. The carried-over intent is gone, so the short reply classifies as unknown and goes to a human. The record of what was already sent is gone, so the bot can repeat itself. And the turn count restarts, so the looping rule can never catch a long, unresolved thread.

**Fix.** Track the time of the last turn and expire on that.

### 5. Poor speech recognition can raise classifier confidence

[`heuristic.py:201`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/classify/heuristic.py#L201)

The discount multiplies confidence by `0.5 + heard` whenever Twilio's recognition confidence is below 0.6. For values between 0.5 and 0.6 that factor is greater than 1. It is also applied *after* confidence is clamped to 1.0 ([line 149](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/classify/heuristic.py#L149)).

**Scenario.** At a recognition confidence of 0.55 the factor is 1.05, so a borderline-misheard phrase scores higher than the same phrase heard clearly -- the opposite of the intent. At 0.59, a confidence of 0.98 becomes 1.068. The factor also jumps from 1.09 to 1.0 at the 0.6 boundary.

The existing test only tries 0.3, where the factor is 0.8, so it passes.

**Fix.** Use a factor that can only shrink confidence and is continuous at the threshold -- for example `max(0.5, heard / 0.6)` -- apply it before the clamp, and add test cases at 0.55 and 0.59.

### 6. Voice calls fail under a path prefix

[`server.py:108`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/server.py#L108)

The URLs Twilio calls back during a call are root-relative (`/twilio/voice/turn`, also [line 137](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/server.py#L137) and [line 157](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/server.py#L157)). Signatures are checked against `PUBLIC_BASE_URL`, path included.

**Scenario.** The app runs at `https://example.com/support`, and `PUBLIC_BASE_URL` says so, as the README instructs. The first webhook validates. Twilio then resolves `/twilio/voice/turn` against the host root and calls `https://example.com/twilio/voice/turn`, without `/support`. That either 404s at the proxy or reaches the app and fails signature verification with a 403. Either way the call ends straight after the greeting.

Not reproduced against a live deployment.

**Fix.** Build absolute callback URLs from `PUBLIC_BASE_URL`, so Twilio calls exactly the URL the app will verify.

---

## Dormant until the model path is on

### 7. One slow model call blocks the whole server

[`server.py:142`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/server.py#L142)

Every route is `async def`, but each calls the synchronous `agent.handle()`, which on the model path makes blocking Anthropic SDK calls. Nothing else is served while it waits.

**Scenario.** An email turn spends its 30-second budget waiting on Claude. For those 30 seconds the event loop is blocked, so a customer on a live call gets no response to their next turn. Twilio gives up on the webhook after about 15 seconds and the call drops. The per-channel latency budget cannot prevent this -- the voice request never gets to run at all.

On the deterministic path each turn takes milliseconds, which is why this is invisible today.

**Fix.** Declare the routes with plain `def` so FastAPI runs them in its threadpool, or call `await run_in_threadpool(agent.handle, message)`. If requests then run concurrently, `ConversationStore` needs a lock.

### 8. The model answerer ignores the retrieval floor

[`answer/llm.py:71`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/answer/llm.py#L71)

The extractive answerer refuses to answer when the best passage scores below `MIN_RETRIEVAL_SCORE`. The model answerer has no such check: any non-empty list of passages goes to Claude, and if Claude calls it answerable the reply is marked grounded.

**Scenario.** A message clears the intent floor, but the best passage it retrieves scores 1.8 against a floor of 2.5 -- a coincidence, not a match. On the deterministic path that is a no-grounding handoff. On the model path Claude may still produce an answer from it, which is then treated as grounded, and no handoff happens. The floor, and the tuning `--sweep` did on it, stop applying -- and this is precisely the confident wrong answer the escalation policy exists to prevent.

**Fix.** Before calling the model, return an ungrounded answer when `passages[0].score` is below the floor, as the extractive engine does.

### 9. The model stays off without an API key, even with valid credentials

[`config.py:91`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/config.py#L91)

`llm_enabled` returns true only when `ANTHROPIC_API_KEY` is set. The client itself is built with a bare `anthropic.Anthropic()`, which also accepts `ANTHROPIC_AUTH_TOKEN` and `ant auth login` profiles -- as the comment at [`llm.py:92`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/llm.py#L92) says.

**Scenario.** `LLM_PROVIDER=anthropic`, with credentials from `ant auth login`. The agent quietly builds the deterministic classifier and answerer, `/health` reports `"classifier": "heuristic"`, and `support-agent shadow` refuses to run for lack of credentials. The operator believes the model is on.

**Fix.** Decide availability from whether the client can actually be constructed, rather than from one environment variable.

### 10. Pronunciation fixes corrupt words that merely contain them

[`render/voice.py:51`](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/render/voice.py#L51)

`_despell` replaces pronunciation keys case-insensitively and without word boundaries, and it runs on every voice reply ([line 98](https://github.com/chsc30010/customer-support-agent/blob/b2c2adc2f74a96c8c2f2b71ada365bb5f6b6cc47/src/support_agent/render/voice.py#L98)).

**Scenario.** A model-written reply, "You can activate night vision under Settings", is read out as "acti V A T e night vision". "Private" becomes "pri V A T e"; "12K" becomes "1two K". None of the current help-centre articles or fixed agent phrases contain an affected word, so today this only hits text the model writes -- and any future article edit.

**Fix.** Anchor each pattern on word boundaries (`\b...\b`).

---

## Suggested order

1. **Finding 1** first. It shows one customer's messages to the person handling another, and the fix is a few lines.
2. **Findings 2 and 3.** Between them they account for most wrong handoffs: duplicate escalations on threads a person already owns, and handoffs for customers who only mentioned an agent.
3. **Finding 4**, then **5** with its missing test cases, then **6** before deploying under a path prefix.
4. **Findings 7, 8 and 9** before the model path is enabled anywhere real. Number 7 in particular turns the latency budget into a guarantee that does not hold.
5. **Finding 10** whenever the voice renderer is next touched.

Several of these want tests that do not exist yet: two customers on `/chat` without ids, a message arriving after escalation, a follow-up after two hours of activity, and recognition confidence at 0.55.
