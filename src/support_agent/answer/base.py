"""The answer interface.

An answer is only ever built from retrieved passages. Nothing in this package
is allowed to introduce a fact the knowledge base does not contain -- an
ungrounded answer becomes a handoff, which is the whole point of the design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Answer, Classification, Conversation, Passage


class AnswerEngine(ABC):
    name: str = "answer"

    @abstractmethod
    def answer(
        self,
        question: str,
        passages: list[Passage],
        classification: Classification,
        conversation: Conversation | None = None,
    ) -> Answer:
        """Draft a reply from ``passages``.

        Returning ``Answer(grounded=False)`` is a valid and expected outcome.
        It is the signal that a human should take this one.
        """
