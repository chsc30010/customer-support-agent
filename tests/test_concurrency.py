"""Turns run off the event loop, and concurrent turns on one conversation are safe.

A deliberately slow classifier stands in for a slow model call: it blocks the
thread it runs on, which is exactly what a synchronous Anthropic SDK call does.
"""

import threading
import time

from fastapi.testclient import TestClient

from support_agent.agent import SupportAgent
from support_agent.classify.heuristic import HeuristicClassifier
from support_agent.config import Settings
from support_agent.conversations import ConversationStore
from support_agent.models import Channel, InboundMessage
from support_agent.server import create_app

SLOW_SECONDS = 1.0


class SlowClassifier(HeuristicClassifier):
    """Blocks its thread before classifying, like a slow model call would."""

    def __init__(self, started: threading.Event) -> None:
        self.started = started

    def classify(self, message, conversation=None, deadline=None):
        self.started.set()
        time.sleep(SLOW_SECONDS)
        return super().classify(message, conversation, deadline)


def test_a_slow_turn_does_not_block_other_requests():
    started = threading.Event()
    agent = SupportAgent(settings=Settings(), classifier=SlowClassifier(started))
    app = create_app(agent=agent, settings=Settings(allow_unsigned_webhooks=True))

    # One client, one event loop, shared by both requests. Without the context
    # manager each request gets its own loop, and a blocked loop could never
    # show up in this test.
    with TestClient(app) as client:
        slow = threading.Thread(
            target=lambda: client.post(
                "/chat", json={"conversation_id": "slow", "text": "where is my order"}
            )
        )
        slow.start()
        assert started.wait(5), "the slow turn never started"

        began = time.monotonic()
        health = client.get("/health")
        waited = time.monotonic() - began
        slow.join()

    assert health.status_code == 200
    # Before the fix, /health could not be served until the slow turn finished.
    assert waited < SLOW_SECONDS / 2, f"/health waited {waited:.2f}s behind a slow turn"


def test_turns_on_one_conversation_do_not_interleave():
    """Two messages on one conversation arriving together are handled in order."""
    started = threading.Event()
    agent = SupportAgent(settings=Settings(), classifier=SlowClassifier(started))

    def turn(text):
        agent.handle(InboundMessage(conversation_id="same", channel=Channel.CHAT, text=text))

    first = threading.Thread(target=turn, args=("where is my order",))
    second = threading.Thread(target=turn, args=("how do I get a return label",))
    first.start()
    assert started.wait(5), "the first turn never started"
    second.start()
    first.join()
    second.join()

    roles = [t.role for t in agent.store.get("same").turns]
    # Interleaved turns would read customer, customer, agent, agent: each reply
    # built against a conversation the other turn was changing underneath it.
    assert roles == ["customer", "agent", "customer", "agent"]


def test_concurrent_lookups_of_one_conversation_share_a_single_object():
    store = ConversationStore()
    seen = []
    barrier = threading.Barrier(16)

    def lookup():
        barrier.wait()
        seen.append(store.get_or_create("sms:+15550001111", Channel.SMS))

    threads = [threading.Thread(target=lookup) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(conversation) for conversation in seen}) == 1
    assert len(store) == 1


def test_each_conversation_gets_its_own_lock():
    store = ConversationStore()
    assert store.lock_for("a") is store.lock_for("a")
    assert store.lock_for("a") is not store.lock_for("b")
