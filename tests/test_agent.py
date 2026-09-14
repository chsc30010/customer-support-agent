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
    answered, tried, exhausted, after = replies
    assert not answered.escalated and not tried.escalated
    # The handoff happens the moment the article runs out.
    assert exhausted.escalated
    assert exhausted.escalation.reason is EscalationReason.NO_GROUNDING
    assert exhausted.escalation.queue is Queue.TECHNICAL
    # And the conversation then stays with that person. This test used to
    # assert that the fourth turn escalated too, which only passed because an
    # escalated conversation kept running through the bot and escalated again.
    assert not after.escalated
    assert after.classification.source == "held_for_person"


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


def test_a_message_after_a_handoff_is_held_for_the_person():
    bot = agent()
    handoff = say(bot, "just get me a person please", channel=Channel.SMS)
    assert handoff.escalated

    follow_up = say(bot, "also where is my order", channel=Channel.SMS)
    assert not follow_up.escalated  # no second escalation
    assert follow_up.answer is None  # nothing was looked up or answered
    assert follow_up.classification.source == "held_for_person"
    assert "someone from our team has this conversation" in follow_up.text.lower()


def test_a_handed_over_conversation_never_files_another_escalation():
    bot = agent()
    say(bot, "just get me a person please", channel=Channel.SMS)
    # Before the fix the looping rule re-escalated every text from the fourth
    # customer turn onwards, flooding the queue with duplicates.
    later = [say(bot, f"any news on this {n}", channel=Channel.SMS) for n in range(6)]
    assert not any(reply.escalated for reply in later)


def test_held_messages_stay_on_the_transcript_for_the_person():
    bot = agent()
    say(bot, "just get me a person please", channel=Channel.SMS)
    say(bot, "my order number is 4471", channel=Channel.SMS)
    assert "my order number is 4471" in bot.store.get("c").transcript()


def test_a_caller_already_being_transferred_is_told_so():
    bot = agent()
    say(bot, "put me through to a person", channel=Channel.VOICE, conversation_id="call")
    reply = say(bot, "hello?", channel=Channel.VOICE, conversation_id="call")
    assert "already being connected" in reply.text


def test_a_held_message_is_journaled_as_held(tmp_path):
    from support_agent.journal import DecisionJournal, read_decisions

    path = tmp_path / "decisions.jsonl"
    bot = SupportAgent(settings=Settings(), journal=DecisionJournal(path))
    say(bot, "just get me a person please", channel=Channel.SMS)
    say(bot, "any update", channel=Channel.SMS)
    rows = list(read_decisions(path))
    assert [row.classifier for row in rows] == ["heuristic", "held_for_person"]
    assert rows[1].escalated is False
