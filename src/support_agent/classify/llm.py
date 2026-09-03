"""A Claude-backed classifier, with the heuristic as its safety net."""

from __future__ import annotations

from ..config import Settings
from ..llm import ClaudeClient
from ..models import Classification, Conversation, InboundMessage, Intent, Sentiment
from .base import Classifier
from .heuristic import HeuristicClassifier

_INTENTS = [i.value for i in Intent]
_SENTIMENTS = [s.value for s in Sentiment]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": _INTENTS},
        "confidence": {"type": "number"},
        "sentiment": {"type": "string", "enum": _SENTIMENTS},
        "wants_human": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "confidence", "sentiment", "wants_human", "evidence"],
    "additionalProperties": False,
}

SYSTEM = """You triage inbound customer support contacts for a smart home camera company.

Classify the customer's LATEST message. Rules:
- Pick exactly one intent from the allowed list. Use "unknown" when the message
  is chit-chat, unintelligible, or genuinely does not fit -- guessing is worse
  than admitting you don't know, because "unknown" routes to a human.
- "cancellation" means ending a subscription or account. Cancelling an *order*
  is "returns_refund".
- "complaint" is for customers escalating about how they have been treated, not
  for any message that happens to sound annoyed. An angry customer with a clear
  technical problem is "technical_issue" with sentiment "angry".
- confidence is 0.0-1.0 and should reflect real uncertainty, not enthusiasm.
- wants_human is true only if the customer asked for a person.
- evidence is up to 3 short quoted spans from the message that drove your choice.
Some messages are phone call transcripts and will contain recognition errors."""


class LLMClassifier(Classifier):
    """One low-effort Claude call per turn, schema-constrained.

    Falls back to the heuristic when the model is unavailable, declines, or
    returns something unusable -- so the agent's behaviour degrades in quality
    rather than stopping.
    """

    name = "llm"

    def __init__(
        self,
        settings: Settings | None = None,
        client: ClaudeClient | None = None,
        fallback: Classifier | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.client = client or ClaudeClient(self.settings)
        self.fallback = fallback or HeuristicClassifier()

    def classify(
        self, message: InboundMessage, conversation: Conversation | None = None
    ) -> Classification:
        data = self.client.json_call(
            system=SYSTEM,
            prompt=self._prompt(message, conversation),
            schema=SCHEMA,
            effort="low",
            max_tokens=512,
        )
        if data is None:
            return self.fallback.classify(message, conversation)

        try:
            intent = Intent(data["intent"])
            sentiment = Sentiment(data["sentiment"])
            confidence = max(0.0, min(1.0, float(data["confidence"])))
        except (KeyError, ValueError, TypeError):
            return self.fallback.classify(message, conversation)

        evidence = tuple(str(e) for e in data.get("evidence", [])[:3])
        return Classification(
            intent=intent,
            confidence=round(confidence, 3),
            sentiment=sentiment,
            evidence=evidence,
            # The heuristic's phrase list is exhaustive for this and cheap to
            # run, so trust either signal rather than only the model's.
            wants_human=bool(data.get("wants_human"))
            or self.fallback.classify(message, conversation).wants_human,
            source=self.name,
        )

    def _prompt(
        self, message: InboundMessage, conversation: Conversation | None
    ) -> str:
        parts = [f"Channel: {message.channel.value}"]
        if message.speech_confidence is not None:
            parts.append(
                f"Speech recognition confidence: {message.speech_confidence:.2f}"
            )
        if conversation and conversation.turns:
            parts.append("Conversation so far:\n" + conversation.transcript())
        parts.append(f"Latest customer message:\n{message.text}")
        return "\n\n".join(parts)
