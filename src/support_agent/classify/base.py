"""The classifier interface.

Two implementations sit behind it -- a deterministic lexicon and a language
model -- so "does the LLM actually classify better?" is a question the eval
harness can answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Classification, Conversation, InboundMessage


class Classifier(ABC):
    """Turns one customer utterance into an intent, a sentiment and a confidence."""

    name: str = "classifier"

    @abstractmethod
    def classify(
        self, message: InboundMessage, conversation: Conversation | None = None
    ) -> Classification:
        """Classify ``message``. Must never raise: an unclassifiable message is
        ``Intent.UNKNOWN`` with confidence 0.0, which the policy layer turns
        into a human handoff."""
