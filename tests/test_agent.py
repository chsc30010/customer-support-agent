from support_agent.agent import SupportAgent
from support_agent.config import Settings
from support_agent.models import Channel, EscalationReason, InboundMessage, Intent, Queue


def agent(**overrides):
    return SupportAgent(settings=Settings(**overrides))


def say(bot, text, conversation_id="c", channel=Channel.CHAT, **kwargs):
    return bot.handle(
        InboundMessage(
            conversation_id=conversation_id, channel=channel, text=text, **kwargs
        )
    )


def test_a_straightforward_question_is_answered_and_cited():
    reply = say(agent(), "how do I get a return label")
    assert not reply.escalated
    assert reply.answer.grounded
    assert reply.answer.citations[0].article_id == "returns-and-refunds"


def test_the_agent_does_not_repeat_itself():
    bot = agent()
    first = say(bot, "my camera keeps going offline")
    second = say(bot, "I already tried that")
    assert first.answer.grounded and second.answer.grounded
    assert first.answer.text != second.answer.text


def test_running_out_of_article_hands_over_rather_than_repeating():
    bot = agent()
    replies = [
        say(bot, "my camera keeps going offline"),
        say(bot, "I already tried that"),
        say(bot, "still the same"),
        say(bot, "and again"),
    ]
    assert replies[-1].escalated
    assert replies[-1].escalation.queue is Queue.TECHNICAL


def test_context_carries_across_turns():
    bot = agent()
    say(bot, "I want to cancel my subscription")
    follow_up = say(bot, "will I lose my recordings")
    assert follow_up.classification.intent is Intent.CANCELLATION


def test_a_handoff_carries_the_transcript_and_a_reason():
    bot = agent()
    say(bot, "I have been charged twice this month")
    reply = say(bot, "just get me a person please")
    assert reply.escalated
    assert reply.escalation.reason is EscalationReason.CUSTOMER_ASKED
    # Routed on what the conversation was about, not on the last turn.
    assert reply.escalation.queue is Queue.BILLING
    assert "charged twice" in reply.escalation.transcript
    assert reply.expects_reply is False


def test_an_angry_customer_is_not_made_to_wait_for_a_lookup():
    reply = say(agent(), "this is absolutely ridiculous, I have had enough")
    assert reply.escalated
    assert reply.answer is None  # no retrieval, no model call


def test_a_voice_turn_the_recogniser_barely_heard_is_treated_as_weaker():
    clear = say(agent(), "I forgot my password", channel=Channel.VOICE, speech_confidence=0.95)
    muddy = say(
        agent(), "I forgot my password", channel=Channel.VOICE, speech_confidence=0.25
    )
    assert muddy.classification.confidence < clear.classification.confidence


def test_conversations_are_kept_apart():
    bot = agent()
    say(bot, "my camera keeps going offline", conversation_id="a")
    other = say(bot, "how do I get a return label", conversation_id="b")
    assert other.answer.citations[0].article_id == "returns-and-refunds"
    assert len(bot.store) == 2


def test_every_channel_has_a_greeting():
    bot = agent()
    for channel in Channel:
        assert bot.greeting(channel)
