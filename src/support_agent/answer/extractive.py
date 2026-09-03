"""The deterministic answerer: quote the knowledge base, do not paraphrase it.

Because it only ever emits sentences that exist verbatim in an article, it
cannot hallucinate. That is a real ceiling on quality -- the wording is
sometimes stilted for the question actually asked -- and it is the baseline the
language model has to beat.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..models import Answer, Channel, Classification, Conversation, Passage
from .base import AnswerEngine

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: How much of an article section to quote, per channel. A spoken reply is
#: heard at about three words a second and cannot be skimmed, and a text that
#: runs to three segments gets read as spam -- so the same passage becomes a
#: different length of answer depending on where it is going.
MAX_SENTENCES = {
    Channel.VOICE: 2,
    Channel.SMS: 2,
    Channel.CHAT: 3,
    Channel.EMAIL: 4,
}
DEFAULT_MAX_SENTENCES = 3


def sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]


class ExtractiveAnswerEngine(AnswerEngine):
    name = "extractive"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def answer(
        self,
        question: str,
        passages: list[Passage],
        classification: Classification,
        conversation: Conversation | None = None,
    ) -> Answer:
        if not passages:
            return Answer(text="", grounded=False, source=self.name)

        best = passages[0]
        if best.score < self.settings.min_retrieval_score:
            return Answer(text="", citations=[best], grounded=False, source=self.name)

        limit = DEFAULT_MAX_SENTENCES
        if conversation is not None:
            limit = MAX_SENTENCES.get(conversation.channel, DEFAULT_MAX_SENTENCES)
        body = " ".join(sentences(best.text)[:limit])
        return Answer(
            text=body,
            citations=[best],
            grounded=True,
            source=self.name,
        )
