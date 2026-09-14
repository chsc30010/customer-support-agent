import json

from support_agent.config import Settings
from support_agent.journal import Decision
from support_agent.shadow import render, replay

SETTINGS = Settings()


def decision(text, escalated, reason, intent="unknown", citation=""):
    return Decision(
        at="2026-09-09T12:00:00+00:00", conversation_id="c", channel="chat",
        turn=1, text=text, intent=intent, confidence=0.0, sentiment="neutral",
        classifier="heuristic", escalated=escalated, reason=reason,
        queue="general_support" if escalated else "", priority="normal" if escalated else "",
        grounded=not escalated, answer_source="extractive" if not escalated else "",
        citation=citation, retrieval_score=0.0, latency_ms=1,
    )


def write(tmp_path, decisions, name="decisions.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(d.__dict__) for d in decisions) + "\n", encoding="utf-8"
    )
    return path


def test_a_replay_never_writes_back_into_the_journal(tmp_path):
    """A replay that appended to the file it reads would poison every run after."""
    path = write(tmp_path, [decision("how do I get a return label", True, "intent_below_confidence_floor")])
    before = path.read_text(encoding="utf-8")
    replay(path, SETTINGS)
    assert path.read_text(encoding="utf-8") == before


def test_a_recorded_escalation_the_agent_now_answers_is_flagged(tmp_path):
    # Recorded as escalated; the deterministic path answers it, so replaying
    # surfaces it as a contact that may not have needed a human.
    path = write(
        tmp_path,
        [decision("how do I get a return label", True, "intent_below_confidence_floor")],
    )
    report = replay(path, SETTINGS)
    assert report.replayed == 1
    assert len(report.would_have_answered) == 1
    assert report.would_have_answered[0].text == "how do I get a return label"


def test_a_recorded_answer_the_agent_now_escalates_is_flagged(tmp_path):
    path = write(
        tmp_path,
        [decision("this is absolutely ridiculous, I have had enough", False, "none")],
    )
    report = replay(path, SETTINGS)
    assert len(report.would_have_escalated) == 1


def test_agreement_is_counted(tmp_path):
    path = write(
        tmp_path,
        [
            decision("how do I get a return label", False, "none", intent="returns_refund"),
            decision("I forgot my password", False, "none", intent="account_access"),
        ],
    )
    report = replay(path, SETTINGS)
    assert report.replayed == 2
    assert report.escalation_agreement == 1.0
    assert report.intent_agreement == 1.0


def test_blank_turns_are_skipped(tmp_path):
    path = write(tmp_path, [decision("   ", False, "none")])
    assert replay(path, SETTINGS).replayed == 0


def test_the_limit_is_honoured(tmp_path):
    rows = [decision(f"question number {n} about my order", False, "none") for n in range(5)]
    assert replay(write(tmp_path, rows), SETTINGS, limit=2).replayed == 2


def test_a_replay_without_a_model_says_so_instead_of_implying_a_result(tmp_path):
    path = write(tmp_path, [decision("how do I get a return label", False, "none")])
    report = replay(path, SETTINGS)
    assert report.model_turns == 0
    assert "model never ran" in render(report, SETTINGS)


def test_an_empty_journal_renders_without_dividing_by_zero(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    report = replay(path, SETTINGS)
    assert report.intent_agreement == 0.0
    assert "Nothing in the journal" in render(report, SETTINGS)


def test_turns_held_for_a_person_are_not_replayed(tmp_path):
    # The agent made no decision on these, so there is nothing to compare.
    held = decision("any news on my order", False, "none")
    held.classifier = "held_for_person"
    assert replay(write(tmp_path, [held]), SETTINGS).replayed == 0
