"""A Claude-backed answerer, grounded in retrieved passages."""

from __future__ import annotations

from ..config import Settings
from ..llm import ClaudeClient
from ..models import Answer, Channel, Classification, Conversation, Passage
from .base import AnswerEngine
from .extractive import ExtractiveAnswerEngine

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answerable", "answer", "sources"],
    "additionalProperties": False,
}

SYSTEM = """You write replies for the Kestrel Home customer support line.

You are given numbered excerpts from the help centre. Every factual claim in
your reply must come from those excerpts.

- If the excerpts do not answer the question, set answerable to false and leave
  answer empty. Do not improvise, do not offer a general suggestion, and do not
  say what is "usually" true. An unanswerable question goes to a human, which
  is a good outcome; a confident wrong answer is not.
- When you can answer, be direct and short. Lead with the thing the customer
  has to do. No greeting, no sign-off, no apology unless the excerpts describe
  an actual failure on our side.
- sources lists the numbers of the excerpts you used.
- Never invent an order number, a date, an amount, a phone number or a URL.
- Write plain sentences with no markdown, bullets or emphasis. The reply may be
  read aloud over a phone line."""

#: Voice replies are spoken, so they are held to a tighter length than text.
CHANNEL_GUIDANCE = {
    Channel.VOICE: "This will be spoken aloud on a phone call. Keep it under 45 words and avoid URLs.",
    Channel.SMS: "This is a text message. Keep it under 45 words.",
    Channel.CHAT: "This is a web chat message. Keep it under 90 words.",
    Channel.EMAIL: "This is an email reply. Up to 150 words is fine.",
}


class LLMAnswerEngine(AnswerEngine):
    """One grounded generation per turn, with the extractive engine behind it."""

    name = "llm"

    def __init__(
        self,
        settings: Settings | None = None,
        client: ClaudeClient | None = None,
        fallback: AnswerEngine | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.client = client or ClaudeClient(self.settings)
        self.fallback = fallback or ExtractiveAnswerEngine(self.settings)

    def answer(
        self,
        question: str,
        passages: list[Passage],
        classification: Classification,
        conversation: Conversation | None = None,
    ) -> Answer:
        if not passages:
            return Answer(text="", grounded=False, source=self.name)

        data = self.client.json_call(
            system=SYSTEM,
            prompt=self._prompt(question, passages, conversation),
            schema=SCHEMA,
            effort="medium",
            max_tokens=1024,
        )
        if data is None:
            return self.fallback.answer(question, passages, classification, conversation)

        text = str(data.get("answer", "")).strip()
        if not data.get("answerable") or not text:
            return Answer(text="", citations=passages[:1], grounded=False, source=self.name)

        return Answer(
            text=text,
            citations=self._cited(data.get("sources", []), passages),
            grounded=True,
            source=self.name,
        )

    @staticmethod
    def _cited(sources: list, passages: list[Passage]) -> list[Passage]:
        chosen: list[Passage] = []
        for raw in sources:
            try:
                index = int(str(raw).strip().lstrip("#")) - 1
            except ValueError:
                continue
            if 0 <= index < len(passages) and passages[index] not in chosen:
                chosen.append(passages[index])
        # A reply the model did not attribute is still grounded in what it was
        # shown, so fall back to the top passage rather than dropping citations.
        return chosen or passages[:1]

    def _prompt(
        self, question: str, passages: list[Passage], conversation: Conversation | None
    ) -> str:
        excerpts = "\n\n".join(
            "[{}] {} -- {}\n{}".format(
                i, p.article_title, p.section, " ".join(p.text.split())
            )
            for i, p in enumerate(passages, start=1)
        )
        parts = [f"Help centre excerpts:\n{excerpts}"]
        if conversation is not None:
            parts.append(CHANNEL_GUIDANCE.get(conversation.channel, ""))
            if conversation.turns:
                parts.append("Conversation so far:\n" + conversation.transcript())
        parts.append(f"Customer question:\n{question}")
        return "\n\n".join(p for p in parts if p)
