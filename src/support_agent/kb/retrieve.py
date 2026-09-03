"""BM25 retrieval over the knowledge base.

Pure Python, pure standard library. The corpus is a few dozen passages, so an
index rebuilt at startup costs milliseconds and there is no vector database to
run, version or pay for. If the corpus grew past a few thousand passages this
is the piece to replace -- nothing else depends on how retrieval works.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

from ..models import Intent, Passage
from .store import load_passages

K1 = 1.2
B = 0.75

#: Passages tagged with the intent the classifier picked get a nudge, not an
#: override: a strong lexical match in another article still wins, which
#: matters because the classifier is sometimes wrong.
INTENT_BOOST = 1.2

#: How much an expansion term counts next to a word the customer typed.
EXPANSION_WEIGHT = 0.7
#: Expansion only helps a query too short to disambiguate itself. Applied to a
#: longer query it does damage: "I cannot log in to the app" retrieved the
#: section on changing the email address on an account, because the intent
#: label account_access matched that heading word for word.
EXPANSION_MAX_TERMS = 2

STOPWORDS = frozenset(
    """a an the and or but if then than that this these those is are was were be been
    being am do does did doing have has had having i me my we our you your it its of
    to in on at for with from by as about into over after before under can could
    will would should shall may might must not no so very just get got there here
    what when where which who whom how why any some all each other more most
    please thanks thank hi hello hey ok okay yes yeah""".split()
)

_WORD = re.compile(r"[a-z0-9]+")
_DOUBLED = re.compile(r"([bcdfghjklmnpqrstvwz])\1$")


def stem(word: str) -> str:
    """A light stemmer: enough to match "refunds" to "refund", nothing more.

    Porter would be more principled, but the failure that matters here is
    over-stemming two different words together, and a short rule list is
    easier to reason about than a full algorithm.

    The doubled-consonant reduction is applied only after an -ing or -ed strip
    ("cancelling" -> "cancell" -> "cancel"). Applied unconditionally it also
    eats the second l of a word that legitimately ends in one, which made
    "business" and "businesses" stem to two different things.
    """
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    if word.endswith("ing") and len(word) > 5:
        word = _DOUBLED.sub(r"\1", word[:-3])
    elif word.endswith("ed") and len(word) > 4:
        word = _DOUBLED.sub(r"\1", word[:-2])
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [
        stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1
    ]


class BM25Retriever:
    """Ranks knowledge base passages against the words a customer used."""

    def __init__(self, passages: list[Passage] | None = None) -> None:
        self.passages = passages if passages is not None else load_passages()
        self._docs: list[dict[str, int]] = []
        self._lengths: list[int] = []
        self._df: dict[str, int] = {}

        for passage in self.passages:
            # Title and heading are repeated so a query matching the heading
            # ("return label") outranks one that only brushes the body. The
            # heading gets more weight than the article title because it is the
            # heading that says which part of the article answers the question.
            tokens = (
                tokenize(passage.article_title) * 2
                + tokenize(passage.section) * 3
                + tokenize(passage.text)
            )
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._docs.append(counts)
            self._lengths.append(len(tokens))
            for token in counts:
                self._df[token] = self._df.get(token, 0) + 1

        self._n = len(self._docs)
        self._avgdl = (sum(self._lengths) / self._n) if self._n else 0.0

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, top_k: int = 3, intent: Intent | None = None
    ) -> list[Passage]:
        """Return the ``top_k`` best passages, each carrying its score.

        Scores are divided by the square root of the query length so a one-line
        text and a rambling voicemail land on the same scale. The policy layer
        compares against a single threshold, and that threshold has to mean the
        same thing on every channel.
        """
        weights: dict[str, float] = {}
        for token in tokenize(query):
            weights[token] = 1.0
        if (
            intent is not None
            and intent is not Intent.UNKNOWN
            and len(weights) <= EXPANSION_MAX_TERMS
        ):
            # Expand the query with the intent label itself. Short messages
            # ("where is my order") carry one content word, which is not enough
            # to separate "Tracking your order" from "Cancelling an order";
            # the classifier already worked out which one the customer meant.
            # Reduced weight, because the label is a hint about the article and
            # not a word the customer said -- at full weight a section whose
            # heading happens to repeat the label outranks the one that answers
            # the question.
            for token in tokenize(intent.value.replace("_", " ")):
                weights[token] = max(weights.get(token, 0.0), EXPANSION_WEIGHT)
        if not weights or not self._n:
            return []

        norm = math.sqrt(sum(weights.values()))
        results: list[Passage] = []
        for index, counts in enumerate(self._docs):
            length = self._lengths[index]
            score = 0.0
            for token, weight in weights.items():
                tf = counts.get(token, 0)
                if not tf:
                    continue
                denom = tf + K1 * (1 - B + B * length / self._avgdl)
                score += weight * self._idf(token) * (tf * (K1 + 1)) / denom
            if score <= 0:
                continue
            passage = self.passages[index]
            if intent is not None and intent in passage.intents:
                score *= INTENT_BOOST
            results.append(replace(passage, score=round(score / norm, 4)))

        results.sort(key=lambda p: (-p.score, p.article_id, p.section))
        return results[:top_k]
