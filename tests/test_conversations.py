"""Conversation expiry. Timestamps are set directly, so nothing waits on a clock."""

from datetime import datetime, timedelta, timezone

from support_agent.agent import SupportAgent
from support_agent.config import Settings
from support_agent.conversations import ConversationStore
from support_agent.models import Channel, InboundMessage, Intent

TTL = timedelta(hours=2)
THREAD = "sms:+15551234567"


def ago(**kwargs):
    return datetime.now(timezone.utc) - timedelta(**kwargs)


def test_a_long_running_conversation_that_is_still_active_is_kept():
    store = ConversationStore(ttl=TTL)
    conversation = store.get_or_create(THREAD, Channel.SMS)
    # Started three hours ago, but the customer wrote ten minutes ago. Expiry
    # used to count from the start, so this was thrown away mid-problem.
    conversation.started_at = ago(hours=3)
    conversation.last_activity = ago(minutes=10)
    assert store.get_or_create(THREAD, Channel.SMS) is conversation


def test_a_conversation_silent_past_the_ttl_is_dropped():
    store = ConversationStore(ttl=TTL)
    conversation = store.get_or_create(THREAD, Channel.SMS)
    conversation.started_at = ago(hours=3)
    conversation.last_activity = ago(hours=2, minutes=5)
    fresh = store.get_or_create(THREAD, Channel.SMS)
    assert fresh is not conversation
    assert fresh.turns == []


def test_adding_a_turn_refreshes_last_activity():
    store = ConversationStore(ttl=TTL)
    conversation = store.get_or_create("c", Channel.CHAT)
    conversation.last_activity = ago(hours=1)
    before = conversation.last_activity
    conversation.add("customer", "still waiting")
    assert conversation.last_activity > before
    assert conversation.last_activity == conversation.turns[-1].at


def test_an_active_thread_past_two_hours_keeps_its_context():
    bot = SupportAgent(settings=Settings())

    def text(message):
        return bot.handle(
            InboundMessage(conversation_id=THREAD, channel=Channel.SMS, text=message)
        )

    text("I want to cancel my subscription")
    conversation = bot.store.get(THREAD)
    # The thread began over two hours ago but has not gone quiet. Before the
    # fix it was dropped here, the carried-over intent went with it, and the
    # short follow-up below classified as unknown and went to a person.
    conversation.started_at = ago(hours=2, minutes=5)

    follow_up = text("will I lose my recordings")
    assert bot.store.get(THREAD) is conversation
    assert follow_up.classification.intent is Intent.CANCELLATION
