"""A record of every decision the agent made.

Until now nothing survived the process. That single gap blocked three separate
things: there was no way to find out which questions the knowledge base fails
to answer, no way to build an evaluation set out of real contacts rather than
invented ones, and no way to compare the model path against the deterministic
one on traffic that actually happened.

One append-only JSONL file fixes all three. It is deliberately not a database:
the file is diffable, greppable, trivially shipped somewhere else, and costs
nothing when the feature is switched off.

Writing is best-effort. A support line must not stop answering customers
because a disk is full, so every failure here is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import AgentReply, Conversation, InboundMessage

log = logging.getLogger(__name__)

# Redaction patterns. These are not a compliance control -- they are a way to
# keep the obvious identifiers out of a file that gets copied around, read in
# a terminal, and eventually attached to a ticket. Anything genuinely
# sensitive should not reach this file in the first place.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_LONG_DIGITS = re.compile(r"\b\d{9,}\b")
#: 13-19 digits in groups, i.e. a payment card. Checked before the phone
#: pattern, which would otherwise swallow it and label it a phone number.
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def redact(text: str) -> str:
    """Mask emails, phone numbers and card-shaped digit runs.

    Order matters: cards first, because the phone pattern also matches them
    and would label a card number as a phone.
    """
    text = _CARD.sub("[card]", text)
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONG_DIGITS.sub("[number]", text)


@dataclass
class Decision:
    """One turn, as it was actually decided.

    Flat and stringly-typed on purpose. This file will be read by things that
    are not this program -- a notebook, a jq one-liner, a spreadsheet -- and
    the enums mean nothing to them.
    """

    # Every field has a default so that a journal written by one version can
    # be read by another. A log format outlives the code that wrote it, and a
    # reader that raises on a field added last month is a reader that loses
    # last month's data.
    at: str = ""
    conversation_id: str = ""
    channel: str = ""
    turn: int = 0
    text: str = ""
    intent: str = ""
    confidence: float = 0.0
    sentiment: str = ""
    classifier: str = ""
    escalated: bool = False
    reason: str = ""
    queue: str = ""
    priority: str = ""
    grounded: bool = False
    answer_source: str = ""
    citation: str = ""
    retrieval_score: float = 0.0
    latency_ms: int = 0
    #: Article ids the retriever offered, best first, whether or not any of
    #: them cleared the grounding floor. The near misses are the interesting
    #: part when working out why a contact went unanswered.
    retrieved: list[str] = field(default_factory=list)

    @property
    def unanswered(self) -> bool:
        """Did the agent fail to answer something it arguably should have?

        Deliberately excludes the escalations that are correct by design -- a
        complaint, an angry customer, someone asking for a person. Those are
        the system working. These are the ones where it did not know.
        """
        return self.escalated and self.reason in (
            "no_supporting_knowledge_base_content",
            "intent_below_confidence_floor",
        )


class DecisionJournal:
    """Appends one line per turn. Disabled when no path is configured."""

    def __init__(self, path: str | Path | None = None, redact_text: bool = True) -> None:
        self.path = Path(path) if path else None
        self.redact_text = redact_text
        # Uvicorn serves concurrently; two turns finishing together must not
        # interleave half-written lines.
        self._lock = threading.Lock()
        self._broken = False

    @property
    def enabled(self) -> bool:
        return self.path is not None and not self._broken

    def record(
        self,
        message: InboundMessage,
        reply: AgentReply,
        conversation: Conversation,
        latency_ms: int,
        retrieved: list[str] | None = None,
    ) -> Decision | None:
        if not self.enabled:
            return None

        answer = reply.answer
        citation = ""
        score = 0.0
        if answer is not None and answer.citations:
            best = answer.citations[0]
            citation = best.key
            score = best.score

        text = message.text or ""
        decision = Decision(
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            conversation_id=conversation.id,
            channel=reply.channel.value,
            turn=len(conversation.customer_turns),
            text=redact(text) if self.redact_text else text,
            intent=reply.classification.intent.value,
            confidence=reply.classification.confidence,
            sentiment=reply.classification.sentiment.value,
            classifier=reply.classification.source,
            escalated=reply.escalated,
            reason=reply.escalation.reason.value,
            queue=reply.escalation.queue.value if reply.escalated else "",
            priority=reply.escalation.priority if reply.escalated else "",
            grounded=bool(answer and answer.grounded),
            answer_source=answer.source if answer else "",
            citation=citation,
            retrieval_score=score,
            latency_ms=latency_ms,
            retrieved=retrieved or [],
        )
        self._append(decision)
        return decision

    def _append(self, decision: Decision) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(asdict(decision), ensure_ascii=False)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError as exc:
            # Log once, then stay quiet. A support line does not stop taking
            # calls because it cannot write its own diary.
            log.warning("decision journal disabled after write failure: %s", exc)
            self._broken = True


def read_decisions(path: str | Path) -> Iterator[Decision]:
    """Read a journal back, skipping anything malformed.

    A half-written last line is normal if the process was killed mid-append,
    and is not a reason to refuse to read the other 40,000 records.
    """
    known = {f for f in Decision.__dataclass_fields__}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed journal line %d", number)
                continue
            yield Decision(**{k: v for k, v in row.items() if k in known})
