from support_agent.config import Settings
from support_agent.models import (
    Answer,
    Channel,
    Classification,
    Conversation,
    EscalationReason,
    Intent,
    Passage,
    Queue,
    Sentiment,
)
from support_agent.policy import decide, priority_for, queue_for

SETTINGS = Settings()


def conversation(channel=Channel.CHAT, turns=1, last_intent=Intent.UNKNOWN):
    c = Conversation(id="t", channel=channel, last_intent=last_intent)
    for _ in range(turns):
        c.add("customer", "hello")
    return c


def classification(**kwargs):
    base = dict(intent=Intent.ORDER_STATUS, confidence=0.8, sentiment=Sentiment.NEUTRAL)
    base.update(kwargs)
    return Classification(**base)


def grounded(requires_human=False):
    passage = Passage(
        article_id="a", article_title="A", section="S", text="text",
        requires_human=requires_human, score=5.0,
    )
    return Answer(text="here you go", citations=[passage], grounded=True)


def test_a_confident_grounded_answer_is_not_escalated():
    result = decide(classification(), grounded(), conversation(), SETTINGS)
    assert result.escalate is False
    assert result.reason is EscalationReason.NONE


def test_asking_for_a_human_wins_over_everything():
    result = decide(classification(wants_human=True), grounded(), conversation(), SETTINGS)
    assert result.reason is EscalationReason.CUSTOMER_ASKED


def test_anger_escalates_even_with_a_good_answer():
    result = decide(
        classification(sentiment=Sentiment.ANGRY), grounded(), conversation(), SETTINGS
    )
    assert result.reason is EscalationReason.ANGRY_CUSTOMER
    assert result.queue is Queue.ESCALATIONS


def test_complaints_always_reach_a_person():
    result = decide(classification(intent=Intent.COMPLAINT), grounded(), conversation(), SETTINGS)
    assert result.reason is EscalationReason.SENSITIVE_INTENT


def test_low_confidence_escalates():
    result = decide(classification(confidence=0.1), grounded(), conversation(), SETTINGS)
    assert result.reason is EscalationReason.LOW_CONFIDENCE


def test_an_ungrounded_answer_escalates():
    result = decide(classification(), Answer(text="", grounded=False), conversation(), SETTINGS)
    assert result.reason is EscalationReason.NO_GROUNDING


def test_a_human_only_article_section_escalates():
    result = decide(classification(), grounded(requires_human=True), conversation(), SETTINGS)
    assert result.reason is EscalationReason.HUMAN_ONLY


def test_a_conversation_that_keeps_going_is_handed_over():
    long_chat = conversation(turns=SETTINGS.max_turns_before_handoff)
    assert decide(classification(), grounded(), long_chat, SETTINGS).reason is EscalationReason.LOOPING


def test_routing_falls_back_to_the_conversation_intent():
    # The turn that triggers a handoff is often a bare "get me a person".
    unknown_turn = classification(intent=Intent.UNKNOWN, confidence=0.0, wants_human=True)
    billing_chat = conversation(last_intent=Intent.BILLING)
    assert queue_for(unknown_turn, billing_chat) is Queue.BILLING


def test_cancellations_go_to_retention():
    assert queue_for(classification(intent=Intent.CANCELLATION), conversation()) is Queue.RETENTION


def test_a_caller_on_hold_outranks_an_email():
    angry = classification(sentiment=Sentiment.ANGRY)
    assert priority_for(angry, Channel.VOICE) == "urgent"
    assert priority_for(angry, Channel.EMAIL) == "high"
    assert priority_for(classification(), Channel.VOICE) == "normal"


def test_the_handover_note_names_the_reason_and_quotes_the_customer():
    chat = Conversation(id="t", channel=Channel.CHAT)
    chat.add("customer", "I have been charged twice and I want it fixed")
    result = decide(classification(sentiment=Sentiment.ANGRY), grounded(), chat, SETTINGS)
    assert "customer is angry" in result.summary
    assert "charged twice" in result.summary
    assert result.transcript
