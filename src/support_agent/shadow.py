"""Replaying recorded traffic through the model path, to see if it disagrees.

The obvious way to compare the two implementations is to run both on every
live turn and log the difference. That is also the wrong way: it doubles the
latency of a phone call to collect data, and the customer pays for the
experiment.

Replaying the journal costs the customer nothing, because the customer has
already hung up. It can be run at any hour, re-run against a different model,
and interrupted without consequence. The one thing it cannot capture is
whether the model would have been fast enough live -- which is what the turn
budget in ``llm.py`` is there to enforce anyway.

What matters is not the agreement rate. It is the shape of the disagreements:

* the agent escalated and the model would have answered -- contacts a human
  handled that maybe nobody needed to;
* the agent answered and the model would have escalated -- possible wrong
  answers that went out.

The second list is the one to read first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .agent import SupportAgent
from .config import Settings
from .conversations import ConversationStore
from .journal import Decision, DecisionJournal, read_decisions
from .models import AgentReply, Channel, InboundMessage


@dataclass
class Divergence:
    recorded: Decision
    shadow: AgentReply

    @property
    def text(self) -> str:
        return self.recorded.text


@dataclass
class ShadowReport:
    replayed: int = 0
    same_intent: int = 0
    same_escalation: int = 0
    #: Served by a human, but the model would have answered it.
    would_have_answered: list[Divergence] = field(default_factory=list)
    #: Answered by the agent, but the model would have fetched a person.
    would_have_escalated: list[Divergence] = field(default_factory=list)
    #: Same escalation decision, different intent -- routing changes, so the
    #: contact reaches a different queue.
    rerouted: list[Divergence] = field(default_factory=list)
    #: Turns where the model actually ran, rather than falling back.
    model_turns: int = 0

    @property
    def intent_agreement(self) -> float:
        return self.same_intent / self.replayed if self.replayed else 0.0

    @property
    def escalation_agreement(self) -> float:
        return self.same_escalation / self.replayed if self.replayed else 0.0


def replay(
    path: str | Path,
    settings: Settings | None = None,
    limit: int | None = None,
) -> ShadowReport:
    """Run the journal back through a model-backed agent and diff the outcome."""
    settings = settings or Settings.from_env()

    agent = SupportAgent(
        settings=settings,
        store=ConversationStore(),
        # Explicitly journal-less. A replay writing into the file it is
        # reading would poison the next replay, and the run after that.
        journal=DecisionJournal(None),
    )

    report = ShadowReport()
    for recorded in read_decisions(path):
        if limit is not None and report.replayed >= limit:
            break
        if not recorded.text.strip():
            continue

        shadow = agent.handle(
            InboundMessage(
                # Namespaced so a replay cannot collide with live state if
                # someone points this at a running process by mistake.
                conversation_id=f"shadow:{recorded.conversation_id}",
                channel=Channel(recorded.channel),
                text=recorded.text,
            )
        )

        report.replayed += 1
        if shadow.classification.source != "heuristic":
            report.model_turns += 1

        if shadow.classification.intent.value == recorded.intent:
            report.same_intent += 1

        if shadow.escalated == recorded.escalated:
            report.same_escalation += 1
            if (
                shadow.classification.intent.value != recorded.intent
                and recorded.escalated
            ):
                report.rerouted.append(Divergence(recorded, shadow))
        elif recorded.escalated:
            report.would_have_answered.append(Divergence(recorded, shadow))
        else:
            report.would_have_escalated.append(Divergence(recorded, shadow))

    return report


def render(report: ShadowReport, settings: Settings) -> str:
    if not report.replayed:
        return "Nothing in the journal to replay.\n"

    model = settings.llm_model if settings.llm_enabled else "heuristic (no model configured)"
    lines = [
        "Shadow replay",
        "=" * 72,
        f"turns replayed     {report.replayed}",
        f"model              {model}",
        f"turns the model actually decided  {report.model_turns}"
        f"  ({report.replayed - report.model_turns} fell back)",
        "",
        f"Intent agreement      {report.intent_agreement:6.1%}",
        f"Escalation agreement  {report.escalation_agreement:6.1%}",
        "",
    ]

    if report.model_turns == 0:
        lines.append(
            "The model never ran, so this compares the deterministic path with "
            "itself. Set LLM_PROVIDER=anthropic and provide a key."
        )
        return "\n".join(lines) + "\n"

    lines.append(
        f"Would have escalated instead of answering: {len(report.would_have_escalated)}"
    )
    lines.append(
        "  Read these first. Each one is a reply that went to a customer and "
        "that the model thought a person should have handled."
    )
    for item in report.would_have_escalated[:10]:
        lines.append(
            f'  - "{item.text}"'
        )
        lines.append(
            f"      served: answered from {item.recorded.citation or 'nothing'} | "
            f"shadow: {item.shadow.escalation.reason.value}"
        )
    lines.append("")

    lines.append(
        f"Would have answered instead of escalating: {len(report.would_have_answered)}"
    )
    lines.append(
        "  Potential deflection, but only where the answer is right -- an "
        "escalation the model is willing to replace is not automatically a "
        "contact it should have handled."
    )
    for item in report.would_have_answered[:10]:
        citation = ""
        if item.shadow.answer and item.shadow.answer.citations:
            citation = item.shadow.answer.citations[0].citation
        lines.append(f'  - "{item.text}"')
        lines.append(
            f"      served: {item.recorded.reason} | shadow: answers from {citation}"
        )
    lines.append("")

    if report.rerouted:
        lines.append(f"Same decision, different queue: {len(report.rerouted)}")
        for item in report.rerouted[:5]:
            lines.append(
                f'  - "{item.text}"  {item.recorded.intent} -> '
                f"{item.shadow.classification.intent.value}"
            )
    return "\n".join(lines).rstrip() + "\n"
