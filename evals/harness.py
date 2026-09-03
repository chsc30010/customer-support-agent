"""Running the golden set through the agent and scoring the result.

Three things are measured, because the agent can fail at three different
places and the fixes are not the same:

* **Routing** -- did it work out what the customer wanted?
* **Escalation** -- did it hand over exactly the contacts a human should get?
  Recall matters more than precision here. An unnecessary transfer costs a few
  minutes of agent time; a missed one is a customer given a wrong answer.
* **Retrieval** -- given the right intent, does the right article come back?
  Scored separately so a routing regression does not read as a retrieval one.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.agent import SupportAgent  # noqa: E402
from support_agent.config import Settings  # noqa: E402
from support_agent.kb import BM25Retriever  # noqa: E402
from support_agent.models import Channel, InboundMessage, Intent  # noqa: E402

GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"


@dataclass
class Case:
    id: str
    channel: Channel
    text: str
    intent: Intent
    should_escalate: bool
    expected_article: str = ""


@dataclass
class Outcome:
    case: Case
    predicted_intent: Intent
    confidence: float
    escalated: bool
    reason: str
    queue: str
    grounded: bool
    cited_article: str
    #: Article ids the retriever returns when handed the *labelled* intent, so
    #: retrieval can be scored without classification in the way.
    retrieved: list[str] = field(default_factory=list)

    @property
    def intent_correct(self) -> bool:
        return self.predicted_intent is self.case.intent

    @property
    def escalation_correct(self) -> bool:
        return self.escalated == self.case.should_escalate

    @property
    def resolved_correctly(self) -> bool:
        """Answered, not escalated, and citing the article a human would."""
        if self.case.should_escalate or not self.case.expected_article:
            return False
        return (
            not self.escalated
            and self.grounded
            and self.cited_article == self.case.expected_article
        )


def load_cases(path: Path = GOLDEN_SET) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        cases.append(
            Case(
                id=row["id"],
                channel=Channel(row["channel"]),
                text=row["text"],
                intent=Intent(row["intent"]),
                should_escalate=bool(row["should_escalate"]),
                expected_article=row.get("expected_article", ""),
            )
        )
    return cases


def run(cases: list[Case], settings: Settings | None = None) -> list[Outcome]:
    settings = settings or Settings.from_env()
    retriever = BM25Retriever()
    agent = SupportAgent(settings=settings, retriever=retriever)

    outcomes: list[Outcome] = []
    for case in cases:
        reply = agent.handle(
            InboundMessage(
                # A fresh id per case: the golden set is single-turn, and
                # leaking state between cases would make scores unreproducible.
                conversation_id=f"eval:{case.id}",
                channel=case.channel,
                text=case.text,
            )
        )
        answer = reply.answer
        cited = ""
        if answer is not None and answer.grounded and answer.citations:
            cited = answer.citations[0].article_id
        retrieved = [
            p.article_id
            for p in retriever.search(case.text, top_k=3, intent=case.intent)
        ]
        outcomes.append(
            Outcome(
                case=case,
                predicted_intent=reply.classification.intent,
                confidence=reply.classification.confidence,
                escalated=reply.escalated,
                reason=reply.escalation.reason.value,
                queue=reply.escalation.queue.value if reply.escalated else "",
                grounded=bool(answer and answer.grounded),
                cited_article=cited,
                retrieved=retrieved,
            )
        )
    return outcomes
