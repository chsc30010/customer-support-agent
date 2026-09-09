import json

from support_agent.agent import SupportAgent
from support_agent.config import Settings
from support_agent.journal import DecisionJournal, Decision, read_decisions, redact
from support_agent.models import Channel, InboundMessage


def test_redaction_masks_the_obvious_identifiers():
    masked = redact("email me at sam.jones+kestrel@example.co.uk or 07700 900123")
    assert "sam.jones" not in masked and "@" not in masked
    assert "900123" not in masked
    assert "[email]" in masked and "[phone]" in masked


def test_a_card_number_is_not_labelled_a_phone_number():
    # The phone pattern also matches a card, so cards have to be masked first.
    assert redact("my card is 4111 1111 1111 1111") == "my card is [card]"


def test_redaction_leaves_the_question_intact():
    masked = redact("where is order 4471, call me on 07700 900123")
    assert masked.startswith("where is order 4471,")


def test_a_journal_with_no_path_is_off():
    journal = DecisionJournal(None)
    assert journal.enabled is False


def test_a_turn_is_recorded(tmp_path):
    path = tmp_path / "decisions.jsonl"
    agent = SupportAgent(settings=Settings(), journal=DecisionJournal(path))
    agent.handle(
        InboundMessage(
            conversation_id="c1", channel=Channel.CHAT, text="how do I get a return label"
        )
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["intent"] == "returns_refund"
    assert row["escalated"] is False
    assert row["grounded"] is True
    assert row["citation"].startswith("returns-and-refunds#")
    assert row["channel"] == "chat"
    assert row["turn"] == 1
    assert row["latency_ms"] >= 0


def test_the_articles_offered_are_recorded_even_when_none_were_good_enough(tmp_path):
    path = tmp_path / "decisions.jsonl"
    agent = SupportAgent(settings=Settings(), journal=DecisionJournal(path))
    agent.handle(
        InboundMessage(
            conversation_id="c1",
            channel=Channel.CHAT,
            text="my camera keeps going offline",
        )
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["retrieved"]  # the near misses are the useful part later


def test_customer_text_is_redacted_on_the_way_in(tmp_path):
    path = tmp_path / "decisions.jsonl"
    agent = SupportAgent(settings=Settings(), journal=DecisionJournal(path))
    agent.handle(
        InboundMessage(
            conversation_id="c1",
            channel=Channel.SMS,
            text="call me back on 07700 900123",
        )
    )
    assert "900123" not in path.read_text(encoding="utf-8")


def test_redaction_can_be_turned_off(tmp_path):
    path = tmp_path / "decisions.jsonl"
    agent = SupportAgent(
        settings=Settings(), journal=DecisionJournal(path, redact_text=False)
    )
    agent.handle(
        InboundMessage(
            conversation_id="c1", channel=Channel.SMS, text="call me on 07700 900123"
        )
    )
    assert "900123" in path.read_text(encoding="utf-8")


def test_a_write_failure_disables_the_journal_rather_than_the_agent(tmp_path):
    # A directory where a file should be: every write raises.
    blocked = tmp_path / "decisions.jsonl"
    blocked.mkdir()
    agent = SupportAgent(settings=Settings(), journal=DecisionJournal(blocked))

    reply = agent.handle(
        InboundMessage(conversation_id="c1", channel=Channel.CHAT, text="I forgot my password")
    )
    assert reply.answer.grounded  # the customer still got an answer
    assert agent.journal.enabled is False  # and it stopped trying


def test_reading_back_skips_a_half_written_line(tmp_path):
    path = tmp_path / "decisions.jsonl"
    good = json.dumps(
        {
            "at": "2026-09-09T12:00:00+00:00",
            "conversation_id": "c1",
            "channel": "sms",
            "turn": 1,
            "text": "where is my order",
            "intent": "order_status",
            "confidence": 0.6,
            "sentiment": "neutral",
            "classifier": "heuristic",
            "escalated": False,
            "reason": "none",
            "queue": "",
            "priority": "",
            "grounded": True,
            "answer_source": "extractive",
            "citation": "order-tracking#Finding your tracking number",
            "retrieval_score": 5.1,
            "latency_ms": 2,
            "retrieved": ["order-tracking"],
        }
    )
    path.write_text(good + "\n" + '{"conversation_id": "trunc', encoding="utf-8")

    rows = list(read_decisions(path))
    assert len(rows) == 1
    assert rows[0].intent == "order_status"


def test_unknown_fields_do_not_break_an_older_reader(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({"conversation_id": "c1", "text": "hi", "invented_field": 1}) + "\n",
        encoding="utf-8",
    )
    assert list(read_decisions(path))[0].conversation_id == "c1"


def test_only_the_did_not_know_escalations_count_as_unanswered():
    def decision(reason, escalated=True):
        return Decision(
            at="", conversation_id="c", channel="chat", turn=1, text="t",
            intent="unknown", confidence=0.0, sentiment="neutral",
            classifier="heuristic", escalated=escalated, reason=reason,
            queue="", priority="", grounded=False, answer_source="",
            citation="", retrieval_score=0.0, latency_ms=0,
        )

    assert decision("no_supporting_knowledge_base_content").unanswered
    assert decision("intent_below_confidence_floor").unanswered
    # These are the system working as designed, not a knowledge gap.
    assert not decision("customer_asked_for_human").unanswered
    assert not decision("customer_is_angry").unanswered
    assert not decision("intent_requires_a_human").unanswered
    assert not decision("none", escalated=False).unanswered
