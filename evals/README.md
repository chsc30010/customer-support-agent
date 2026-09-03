# Evaluating the support agent

A support agent that answers everything looks good in a demo and is bad in
production. What matters is whether it hands over exactly the contacts a human
should get. That is a measurable claim, so this directory measures it.

```
python evals/run_evals.py
python evals/run_evals.py --show-failures --sweep
python evals/run_evals.py --min-escalation-recall 1.0 --min-intent-accuracy 0.90
```

The last form exits non-zero when a threshold is missed, so it works as a gate
on a pull request rather than as something someone remembers to run.

## The golden set

`golden_set.jsonl` holds 60 hand-labelled contacts spread across all four
channels and all nine intents. Each one carries three labels:

| Field | What it means |
|---|---|
| `intent` | What the customer actually wants. |
| `should_escalate` | Whether a support lead would want a human on this one. |
| `expected_article` | Which help centre article answers it, where one does. |

`should_escalate` is labelled from the customer's side, not from the code's
behaviour -- it is true when the customer asked for a person, when they are
angry, when it is a complaint, when the help centre says only a human can do
the step, and when the request is simply outside what Kestrel support covers.

## What is measured

**Intent accuracy** with a full confusion matrix and per-intent precision and
recall, so a regression points at the intent that broke rather than at a
single number going down.

**Escalation precision and recall**, with escalation as the positive class.
These are not symmetric and should not be traded off as though they were. A
false positive costs a few minutes of an agent's time. A false negative is a
customer who needed a person and got a confident wrong answer instead. Recall
is the number to hold at 1.0; precision is the one to improve.

**Retrieval hit@1 and hit@3**, computed by handing the retriever the *labelled*
intent rather than the predicted one. Scoring it that way keeps a routing
regression from reading as a retrieval regression.

**Resolution rate** end to end: of the contacts a human would have let the bot
handle, how many did it answer, ground in the knowledge base, and cite the
right article for.

## Current results

Heuristic classifier, extractive answers, no model calls:

```
Intent accuracy       96.7%  (58/60)
Escalation recall    100.0%  (18/18 contacts that needed a human got one)
Escalation precision  85.7%  (3 handed over unnecessarily)
Retrieval hit@1       93.2%  hit@3  97.7%  (over 44 labelled cases)
Resolved correctly    88.1%  (42 answerable contacts)
```

Read escalation recall first. Every contact that a person should have handled
reached a person. The cost is three unnecessary transfers out of 42, which is
the right side of that trade to be wrong on.

## Honest limits

**This set was used to tune the thing it measures.** The lexicon gaps it found
were closed by editing the lexicon, so 96.7% is an upper bound on the accuracy,
not an estimate of accuracy on contacts nobody has seen. A held-out second set
is the obvious next piece of work.

**Sixty cases is small.** A single label change moves accuracy by 1.7 points,
so treat differences under about 3 points as noise.

**Every case is one turn.** Multi-turn failures -- drift, repetition, a
follow-up that retrieves against the wrong context -- are not scored here, even
though the agent has behaviour specifically for them. `fixtures/transcripts/`
exists for looking at those by hand, and scoring them is not yet automated.

## What the harness has caught

The three fixes below came from running it, not from reading the code:

- **Seventeen contacts falling to `unknown`**, which the escalation policy
  correctly turned into handoffs -- so the top-line behaviour looked fine while
  the agent was resolving nothing. Intent accuracy was 66.7% before the lexicon
  gaps were closed, and escalation precision was 53.1%.
- **A retrieval floor that was doing nothing useful below 2.5.** `--sweep`
  scores every floor from 1.5 to 4.0: dropping to 2.0 buys four points of
  escalation precision and *zero* extra correctly resolved contacts, which
  means the extra answers it lets through are answers from the wrong article.
- **A stemmer that mapped "business" and "businesses" to different tokens**,
  because the doubled-consonant rule meant for "cancelling" was being applied
  to words that legitimately end in a double letter.

## Files

| File | Job |
|---|---|
| `golden_set.jsonl` | The labelled contacts. |
| `harness.py` | Runs each case through the real agent and records what happened. |
| `metrics.py` | Scoring, kept separate so the numbers can be checked by hand. |
| `run_evals.py` | The report, the failure list, the sweep, and the CI gates. |
