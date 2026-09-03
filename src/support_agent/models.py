"""The normalized data model.

A phone call, a text message, a web chat and an email arrive over four very
different transports. They all become an ``InboundMessage`` before the agent
sees them, and the agent only ever produces an ``AgentReply``. Channel-specific
behaviour lives at the two ends -- in ``telephony/`` on the way in and
``render/`` on the way out -- never in the middle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Channel(str, Enum):
    VOICE = "voice"
    SMS = "sms"
    CHAT = "chat"
    EMAIL = "email"

    @property
    def is_spoken(self) -> bool:
        """Will the reply be heard rather than read?"""
        return self is Channel.VOICE

    @property
    def is_realtime(self) -> bool:
        """Is a human waiting on the other end right now?

        Drives escalation urgency: a caller on hold is a worse experience than
        an email that gets queued, so voice and chat escalate sooner.
        """
        return self in (Channel.VOICE, Channel.CHAT)

    @property
    def supports_links(self) -> bool:
        """Can the customer click a URL in the reply?"""
        return self is not Channel.VOICE


class Intent(str, Enum):
    ORDER_STATUS = "order_status"
    RETURNS_REFUND = "returns_refund"
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    TECHNICAL_ISSUE = "technical_issue"
    CANCELLATION = "cancellation"
    PRODUCT_INFO = "product_info"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"

    @property
    def is_negative(self) -> bool:
        return self in (Sentiment.FRUSTRATED, Sentiment.ANGRY)


class EscalationReason(str, Enum):
    NONE = "none"
    CUSTOMER_ASKED = "customer_asked_for_human"
    LOW_CONFIDENCE = "intent_below_confidence_floor"
    NO_GROUNDING = "no_supporting_knowledge_base_content"
    ANGRY_CUSTOMER = "customer_is_angry"
    SENSITIVE_INTENT = "intent_requires_a_human"
    LOOPING = "too_many_turns_without_resolution"
    HUMAN_ONLY = "the_help_centre_says_a_human_must_do_this"


class Queue(str, Enum):
    GENERAL = "general_support"
    BILLING = "billing_disputes"
    RETENTION = "retention"
    TECHNICAL = "technical_support"
    ESCALATIONS = "escalations"


@dataclass
class InboundMessage:
    """One customer utterance, whatever transport carried it."""

    conversation_id: str
    channel: Channel
    text: str
    sender: str = ""
    received_at: datetime = field(default_factory=_now)
    #: DTMF keypresses on a voice call, e.g. "1" or "0".
    digits: str = ""
    #: Twilio's speech recognition confidence, 0.0-1.0. None off voice.
    speech_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    role: str  # "customer" or "agent"
    text: str
    at: datetime = field(default_factory=_now)


@dataclass
class Conversation:
    id: str
    channel: Channel
    customer_ref: str = ""
    turns: list[Turn] = field(default_factory=list)
    started_at: datetime = field(default_factory=_now)
    escalated: bool = False
    closed: bool = False
    #: Intent carried over from the previous turn, so a bare "yes" or
    #: "the second one" is still understood in context.
    last_intent: "Intent" = Intent.UNKNOWN
    #: Passages already used to answer in this conversation. Saying the same
    #: paragraph twice is how a customer learns the agent is not listening.
    served: set[str] = field(default_factory=set)

    def add(self, role: str, text: str) -> None:
        self.turns.append(Turn(role=role, text=text))

    @property
    def customer_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "customer"]

    def transcript(self) -> str:
        label = {"customer": "Customer", "agent": "Agent"}
        return "\n".join(f"{label.get(t.role, t.role)}: {t.text}" for t in self.turns)


@dataclass
class Passage:
    """A retrievable section of a knowledge base article."""

    article_id: str
    article_title: str
    section: str
    text: str
    url: str = ""
    intents: tuple[Intent, ...] = ()
    #: The help centre itself says this step needs a person -- changing the
    #: email on an account, recovering a lost second factor. Answering is
    #: still useful; finishing without a human is not possible.
    requires_human: bool = False
    score: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.article_id}#{self.section}"

    @property
    def citation(self) -> str:
        return f"{self.article_title} - {self.section}"


@dataclass
class Classification:
    intent: Intent
    confidence: float
    sentiment: Sentiment = Sentiment.NEUTRAL
    #: Terms that drove the decision. Kept for the eval harness and handoff notes.
    evidence: tuple[str, ...] = ()
    wants_human: bool = False
    source: str = "heuristic"


@dataclass
class Answer:
    text: str
    citations: list[Passage] = field(default_factory=list)
    grounded: bool = False
    source: str = "extractive"


@dataclass
class Escalation:
    escalate: bool
    reason: EscalationReason = EscalationReason.NONE
    queue: Queue = Queue.GENERAL
    priority: str = "normal"  # normal | high | urgent
    summary: str = ""
    transcript: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "escalate": self.escalate,
            "reason": self.reason.value,
            "queue": self.queue.value,
            "priority": self.priority,
            "summary": self.summary,
            "transcript": self.transcript,
        }


@dataclass
class AgentReply:
    """What the agent decided, before rendering.

    ``text`` is channel-neutral prose. ``render/`` turns it into SSML, SMS
    segments, markdown or an email body -- the agent itself never writes markup.
    """

    conversation_id: str
    channel: Channel
    text: str
    classification: Classification
    answer: Answer | None = None
    escalation: Escalation = field(default_factory=lambda: Escalation(escalate=False))
    expects_reply: bool = True

    @property
    def escalated(self) -> bool:
        return self.escalation.escalate
