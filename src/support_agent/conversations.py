"""In-process conversation state.

A phone call, an SMS thread and a chat session all need the same thing: the
turns so far, keyed by something the transport gives us (a Twilio CallSid, a
phone number, a session id). This keeps them in memory with a time to live,
which is the right shape for one process and the wrong shape for several --
swapping in Redis means replacing this file and nothing else.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import Channel, Conversation

DEFAULT_TTL = timedelta(hours=2)


class ConversationStore:
    def __init__(self, ttl: timedelta = DEFAULT_TTL) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._ttl = ttl

    def get_or_create(
        self, conversation_id: str, channel: Channel, customer_ref: str = ""
    ) -> Conversation:
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
        return self._conversations.get(conversation_id)

    def close(self, conversation_id: str) -> None:
        conversation = self._conversations.get(conversation_id)
        if conversation is not None:
            conversation.closed = True

    def drop(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)

    def __len__(self) -> int:
        return len(self._conversations)

    def _expire(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._ttl
        stale = [
            key
            for key, conversation in self._conversations.items()
            if conversation.started_at < cutoff
        ]
        for key in stale:
            del self._conversations[key]
