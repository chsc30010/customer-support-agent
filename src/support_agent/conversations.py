"""In-process conversation state.

A phone call, an SMS thread and a chat session all need the same thing: the
turns so far, keyed by something the transport gives us (a Twilio CallSid, a
phone number, a session id). This keeps them in memory with a time to live,
which is the right shape for one process and the wrong shape for several --
swapping in Redis means replacing this file and nothing else.

Turns run on worker threads, so everything here is safe to call concurrently.
The store guards its own dictionary, and ``turn()`` runs the turns of one
conversation one after the other without holding up any other conversation.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

from .models import Channel, Conversation

DEFAULT_TTL = timedelta(hours=2)


class _TurnLock:
    """A conversation's lock, with a count of the turns using or waiting on it.

    The count is what makes it safe to forget a lock when its conversation
    expires: a lock is only dropped when no turn holds it or is about to. A
    lock forgotten while a turn was about to take it would let the next message
    on that conversation create a second lock, and run alongside the first.
    """

    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


class ConversationStore:
    def __init__(self, ttl: timedelta = DEFAULT_TTL) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._turn_locks: dict[str, _TurnLock] = {}
        self._ttl = ttl
        # Reentrant because get_or_create expires stale conversations while it
        # already holds this lock.
        self._lock = threading.RLock()

    def get_or_create(
        self, conversation_id: str, channel: Channel, customer_ref: str = ""
    ) -> Conversation:
        with self._lock:
            self._expire()
            existing = self._conversations.get(conversation_id)
            if existing is not None:
                return existing
            created = Conversation(
                id=conversation_id, channel=channel, customer_ref=customer_ref
            )
            self._conversations[conversation_id] = created
            return created

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def close(self, conversation_id: str) -> None:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is not None:
                conversation.closed = True

    def drop(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)
            self._forget_lock(conversation_id)

    def lock_for(self, conversation_id: str) -> threading.Lock:
        """The lock that keeps one conversation's turns from overlapping."""
        with self._lock:
            return self._turn_lock(conversation_id).lock

    @contextmanager
    def turn(self, conversation_id: str) -> Iterator[None]:
        """Hold a conversation's lock for the length of one turn.

        Now that turns run on threads, two messages on the same conversation
        can arrive together. Each turn reads and changes the conversation's
        turns, served passages, carried-over intent and escalation state, so
        they must run one after the other -- while turns on other conversations
        carry on unaffected.
        """
        with self._lock:
            entry = self._turn_lock(conversation_id)
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._lock:
                entry.users -= 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._conversations)

    def _turn_lock(self, conversation_id: str) -> _TurnLock:
        entry = self._turn_locks.get(conversation_id)
        if entry is None:
            entry = self._turn_locks[conversation_id] = _TurnLock()
        return entry

    def _forget_lock(self, conversation_id: str) -> None:
        entry = self._turn_locks.get(conversation_id)
        if entry is not None and entry.users == 0:
            del self._turn_locks[conversation_id]

    def _expire(self) -> None:
        """Drop conversations that have been silent for longer than the TTL.

        Counted from the last activity, not the start. Measuring from
        started_at threw away every conversation two hours after its first
        message, however recently the customer had written -- and with it the
        carried-over intent, the record of what had already been sent, and the
        turn count the looping rule depends on.
        """
        cutoff = datetime.now(timezone.utc) - self._ttl
        stale = [
            key
            for key, conversation in self._conversations.items()
            if conversation.last_activity < cutoff
        ]
        for key in stale:
            del self._conversations[key]
            self._forget_lock(key)
