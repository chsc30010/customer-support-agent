"""When the agent should stop talking and fetch a person.

This is the part that decides whether the whole system is trustworthy. An
agent that answers everything is not better than one that answers most things
and knows which ones it got wrong -- it is worse, because nobody can tell the
difference until a customer is angry.

The rules are ordered and the first match wins. They are deliberately readable
rather than clever: a support lead should be able to audit this file.
"""

from __future__ import annotations

from .config import Settings
from .models import (
    Answer,
    Channel,
    Classification,
    Conversation,
    Escalation,
    EscalationReason,
    Intent,
    Queue,
    Sentiment,
)

#: Intents a human owns outright, however confident the classifier is.
SENSITIVE_INTENTS: frozenset[Intent] = frozenset({Intent.COMPLAINT})

QUEUE_FOR_INTENT: dict[Intent, Queue] = {
    Intent.BILLING: Queue.BILLING,
    Intent.CANCELLATION: Queue.RETENTION,
    Intent.TECHNICAL_ISSUE: Queue.TECHNICAL,
    Intent.COMPLAINT: Queue.ESCALATIONS,
}


def queue_for(classification: Classification, conversation: Conversation) -> Queue:
    """Route on what the conversation is about, not only on the last turn.

    The turn that triggers a handoff is often the least informative one -- a
    keypress, or "just get me a person". Falling back to the intent the
    conversation established stops a billing dispute from landing in general
    support because the customer gave up mid-sentence.
    """
    if classification.sentiment is Sentiment.ANGRY:
        return Queue.ESCALATIONS
    intent = classification.intent
    if intent is Intent.UNKNOWN:
        intent = conversation.last_intent
    return QUEUE_FOR_INTENT.get(intent, Queue.GENERAL)


def priority_for(classification: Classification, channel: Channel) -> str:
    """A caller waiting on hold is a worse experience than a queued email."""
    if classification.sentiment is Sentiment.ANGRY:
        return "urgent" if channel.is_realtime else "high"
    if classification.intent in SENSITIVE_INTENTS:
        return "high"
    if classification.sentiment is Sentiment.FRUSTRATED and channel.is_realtime:
        return "high"
    return "normal"


def pre_answer_reason(
    classification: Classification, settings: Settings | None = None
) -> EscalationReason:
    """The reasons that are knowable before we look anything up.

    Checking these first is not just tidiness: it means an angry customer who
    asked for a person is not made to wait through a retrieval and a model call
    whose result will be thrown away.
    """
    settings = settings or Settings.from_env()
    if classification.wants_human:
        return EscalationReason.CUSTOMER_ASKED
    if classification.sentiment is Sentiment.ANGRY:
        return EscalationReason.ANGRY_CUSTOMER
    if classification.intent in SENSITIVE_INTENTS:
        return EscalationReason.SENSITIVE_INTENT
    if (
        classification.intent is Intent.UNKNOWN
        or classification.confidence < settings.min_intent_confidence
    ):
        return EscalationReason.LOW_CONFIDENCE
    return EscalationReason.NONE


def escalation_for(
    reason: EscalationReason,
    classification: Classification,
    conversation: Conversation,
) -> Escalation:
    """Wrap a reason in everything the receiving human needs."""
    if reason is EscalationReason.NONE:
        return Escalation(escalate=False)
    return Escalation(
        escalate=True,
        reason=reason,
        queue=queue_for(classification, conversation),
        priority=priority_for(classification, conversation.channel),
        summary=summarize(classification, reason, conversation),
        transcript=conversation.transcript(),
    )


def decide(
    classification: Classification,
    answer: Answer | None,
    conversation: Conversation,
    settings: Settings | None = None,
) -> Escalation:
    """Decide whether this turn goes to a human, and to which queue."""
    settings = settings or Settings.from_env()
    reason = _reason(classification, answer, conversation, settings)
    return escalation_for(reason, classification, conversation)


def _reason(
    classification: Classification,
    answer: Answer | None,
    conversation: Conversation,
    settings: Settings,
) -> EscalationReason:
    early = pre_answer_reason(classification, settings)
    if early is not EscalationReason.NONE:
        return early
    if answer is None or not answer.grounded:
        return EscalationReason.NO_GROUNDING
    if any(passage.requires_human for passage in answer.citations):
        return EscalationReason.HUMAN_ONLY
    # Counted after the grounding check so a conversation that is going fine is
    # never cut off just for being long.
    if len(conversation.customer_turns) >= settings.max_turns_before_handoff:
        return EscalationReason.LOOPING
    return EscalationReason.NONE


READABLE_REASON = {
    EscalationReason.CUSTOMER_ASKED: "the customer asked for a person",
    EscalationReason.LOW_CONFIDENCE: "the agent could not identify the problem",
    EscalationReason.NO_GROUNDING: "no help centre article covers this",
    EscalationReason.ANGRY_CUSTOMER: "the customer is angry",
    EscalationReason.SENSITIVE_INTENT: "this is a complaint and needs an owner",
    EscalationReason.LOOPING: "several turns without resolving it",
    EscalationReason.HUMAN_ONLY: "this step can only be done by a person",
}


def summarize(
    classification: Classification,
    reason: EscalationReason,
    conversation: Conversation,
) -> str:
    """The note the receiving agent reads before saying hello.

    It answers the three things a human needs in the first two seconds: what
    the customer wants, how they feel, and why the bot gave up.
    """
    turns = conversation.customer_turns
    opening = turns[0].text.strip() if turns else ""
    latest = turns[-1].text.strip() if turns else ""

    lines = [
        "Channel: {} | Intent: {} ({:.2f}) | Sentiment: {}".format(
            conversation.channel.value,
            classification.intent.value,
            classification.confidence,
            classification.sentiment.value,
        ),
        "Handing over because {}.".format(READABLE_REASON[reason]),
        "Opened with: {}".format(_clip(opening)),
    ]
    if latest and latest != opening:
        lines.append("Most recently: {}".format(_clip(latest)))
    lines.append("Turns so far: {}".format(len(turns)))
    if classification.evidence:
        lines.append("Signals: {}".format(", ".join(classification.evidence[:5])))
    return "\n".join(lines)


def _clip(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"
