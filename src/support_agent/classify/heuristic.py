"""A deterministic classifier: weighted phrases, no network, no credentials.

This is not a placeholder for the language model. It is the baseline the model
has to beat, and having it means the agent still answers when the LLM is
unreachable, out of quota, or simply not configured.
"""

from __future__ import annotations

import re

from ..models import Classification, Conversation, InboundMessage, Intent, Sentiment
from .base import Classifier
from .lexicon import (
    ANGRY_TERMS,
    FRUSTRATED_TERMS,
    HUMAN_REQUEST_PHRASES,
    INTENT_PHRASES,
    POSITIVE_TERMS,
)

_APOSTROPHES = str.maketrans("", "", "'\u2019\u02bc")
_NON_WORD = re.compile(r"[^a-z0-9]+")

#: Speech recognisers write "cannot" and "will not"; people type "can't" and
#: "won't". Folding them together means the lexicon needs one spelling rather
#: than two, and it is the difference between recognising a phone call and not.
_NEGATIONS = (
    ("can not", "cant"),
    ("cannot", "cant"),
    ("will not", "wont"),
    ("do not", "dont"),
    ("does not", "doesnt"),
    ("did not", "didnt"),
    ("is not", "isnt"),
    ("was not", "wasnt"),
    ("have not", "havent"),
    ("has not", "hasnt"),
    ("would not", "wouldnt"),
    ("could not", "couldnt"),
    ("am unable to", "cant"),
    ("unable to", "cant"),
)


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, fold negations, collapse the rest to spaces.

    Both ``"I can't log in!!"`` and ``"I cannot log in"`` come out as
    ``"cant log in"``. Lexicon phrases are written in this form.
    """
    lowered = text.lower().translate(_APOSTROPHES)
    flat = _NON_WORD.sub(" ", lowered).strip()
    for written, folded in _NEGATIONS:
        flat = flat.replace(written, folded)
    return flat


def _matcher(phrase: str) -> re.Pattern[str]:
    # Anchored at a word start but open at the end, so "refund" also matches
    # "refunded" and "refunds" without matching "unrefundable".
    return re.compile(r"\b" + re.escape(phrase))


_INTENT_MATCHERS: dict[Intent, tuple[tuple[re.Pattern[str], str, float], ...]] = {
    intent: tuple((_matcher(p), p, w) for p, w in phrases.items())
    for intent, phrases in INTENT_PHRASES.items()
}
_HUMAN_MATCHERS = tuple((_matcher(p), p) for p in HUMAN_REQUEST_PHRASES)
_ANGRY_MATCHERS = tuple((_matcher(p), p) for p in ANGRY_TERMS)
_FRUSTRATED_MATCHERS = tuple((_matcher(p), p) for p in FRUSTRATED_TERMS)
_POSITIVE_MATCHERS = tuple((_matcher(p), p) for p in POSITIVE_TERMS)

#: A reply this short is usually an answer to the agent's own question
#: ("yes", "the black one"), not a new topic.
_FOLLOW_UP_WORD_LIMIT = 6


def _shouting(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 12:
        caps = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps > 0.6:
            return True
    return text.count("!") >= 2


def score_sentiment(text: str) -> tuple[Sentiment, tuple[str, ...]]:
    normalized = normalize(text)
    angry = [p for m, p in _ANGRY_MATCHERS if m.search(normalized)]
    frustrated = [p for m, p in _FRUSTRATED_MATCHERS if m.search(normalized)]
    positive = [p for m, p in _POSITIVE_MATCHERS if m.search(normalized)]
    shouting = _shouting(text)

    heat = 2.5 * len(angry) + 1.0 * len(frustrated) + (1.0 if shouting else 0.0)
    evidence = tuple(angry + frustrated)
    if heat >= 2.5:
        return Sentiment.ANGRY, evidence
    if heat >= 1.0:
        return Sentiment.FRUSTRATED, evidence
    if positive:
        return Sentiment.POSITIVE, tuple(positive)
    return Sentiment.NEUTRAL, ()


class HeuristicClassifier(Classifier):
    name = "heuristic"

    def classify(
        self, message: InboundMessage, conversation: Conversation | None = None
    ) -> Classification:
        text = message.text or ""
        normalized = normalize(text)
        sentiment, sentiment_evidence = score_sentiment(text)
        wants_human = message.digits.strip() == "0" or any(
            m.search(normalized) for m, _ in _HUMAN_MATCHERS
        )

        scores: dict[Intent, float] = {}
        hits: dict[Intent, list[str]] = {}
        for intent, matchers in _INTENT_MATCHERS.items():
            total = 0.0
            matched: list[str] = []
            for matcher, phrase, weight in matchers:
                if matcher.search(normalized):
                    total += weight
                    matched.append(phrase)
            if total:
                scores[intent] = total
                hits[intent] = matched

        if not scores:
            return self._no_signal(
                message, conversation, sentiment, sentiment_evidence, wants_human
            )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        intent, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        # Rewards both mass and margin: a lone strong phrase is confident, two
        # intents tied on strong phrases is not.
        confidence = max(0.0, min(1.0, (top - 0.5 * runner_up) / (top + 1.0)))
        confidence = self._discount_for_bad_audio(confidence, message)

        return Classification(
            intent=intent,
            confidence=round(confidence, 3),
            sentiment=sentiment,
            evidence=tuple(hits[intent] + list(sentiment_evidence)),
            wants_human=wants_human,
            source=self.name,
        )

    def _no_signal(
        self,
        message: InboundMessage,
        conversation: Conversation | None,
        sentiment: Sentiment,
        sentiment_evidence: tuple[str, ...],
        wants_human: bool,
    ) -> Classification:
        """No phrase matched. Either it is a short follow-up to a topic already
        on the table, or the agent genuinely does not know."""
        words = len(normalize(message.text or "").split())
        carried = conversation.last_intent if conversation else Intent.UNKNOWN
        if carried is not Intent.UNKNOWN and 0 < words <= _FOLLOW_UP_WORD_LIMIT:
            return Classification(
                intent=carried,
                confidence=0.5,
                sentiment=sentiment,
                evidence=("carried over from previous turn",) + sentiment_evidence,
                wants_human=wants_human,
                source=self.name,
            )
        return Classification(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            sentiment=sentiment,
            evidence=sentiment_evidence,
            wants_human=wants_human,
            source=self.name,
        )

    @staticmethod
    def _discount_for_bad_audio(confidence: float, message: InboundMessage) -> float:
        """A phone transcript the recogniser was unsure of is weaker evidence.

        Twilio reports its own confidence per utterance; ignoring it means
        treating a misheard "cancel" exactly like a clearly spoken one.
        """
        heard = message.speech_confidence
        if heard is None or heard >= 0.6:
            return confidence
        return confidence * (0.5 + heard)
