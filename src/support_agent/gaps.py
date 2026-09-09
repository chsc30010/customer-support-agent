"""What the knowledge base cannot answer, ranked by how often it comes up.

Every contact the agent could not handle is a question somebody expected an
answer to. Individually they are noise; grouped and counted they are a work
list for whoever writes the help centre, in priority order, derived from real
customers rather than from guesses.

Two kinds of failure look identical from the outside and need opposite fixes,
so this separates them:

* **Content gap** -- nothing in the help centre covers the question. Somebody
  has to write an article.
* **Recognition gap** -- an article covers it perfectly well, but the
  classifier did not work out what was being asked, so retrieval never got a
  chance. Somebody has to add phrasing to the lexicon.

Telling a content writer to write an article that already exists is a good way
to have them stop reading your reports.

**Grouping is lexical, and that is a real limit.** "Do you have a shop" and
"is there a store near me" are the same question and share no words, so they
land in separate groups. Rare shared words are weighted above common ones,
which recovers some of it, but nothing here understands synonyms. This is the
piece to replace with embeddings once the volume justifies it; until then the
report lists what it could not group rather than pretending it grouped it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .journal import Decision, read_decisions
from .kb import BM25Retriever, tokenize

#: How much weighted overlap makes two contacts the same question. Low,
#: because support contacts are short: four content words each means a single
#: shared word is already a quarter of the vocabulary. The IDF weighting is
#: what stops that from merging everything -- a shared "swap" counts, a shared
#: "one" barely does.
SIMILARITY = 0.14
#: Below this a "cluster" is one person asking one odd question, which is not
#: a signal about the knowledge base.
MIN_CLUSTER = 2
#: A near miss is worth naming; a coincidence is not. Something always scores
#: highest, and "do you sponsor local football teams" matches the
#: compatibility article at 1.8 out of a 2.5 floor purely by chance. Within a
#: fifth of the floor is a near miss worth an editor's time; further away and
#: naming the article sends them to the wrong file.
NEAREST_IS_MEANINGFUL = 0.8
#: Word rarity is meaningless across a handful of documents -- with two
#: contacts every word is equally "rare", which distorts the comparison rather
#: than sharpening it. Below this, fall back to plain overlap.
MIN_DOCS_FOR_WEIGHTING = 10

CONTENT_GAP = "content gap"
RECOGNITION_GAP = "recognition gap"


def inverse_frequency(documents: list[set[str]]) -> dict[str, float]:
    """Weight each word by how rare it is among the unanswered contacts.

    Computed over this set rather than over the knowledge base on purpose: the
    question is which words distinguish one unanswered contact from another,
    and "order" is uninformative here precisely because so many of them
    mention it.
    """
    total = len(documents)
    seen = Counter(token for document in documents for token in document)
    return {
        token: math.log(1 + total / (1 + count)) for token, count in seen.items()
    }


def similarity(
    left: set[str], right: set[str], weights: dict[str, float] | None = None
) -> float:
    """Jaccard, weighted by word rarity when weights are supplied."""
    if not left or not right:
        return 0.0
    if weights is None:
        return len(left & right) / len(left | right)
    shared = sum(weights.get(t, 1.0) for t in left & right)
    combined = sum(weights.get(t, 1.0) for t in left | right)
    return shared / combined if combined else 0.0


@dataclass
class Gap:
    """A group of contacts that failed for what looks like the same reason."""

    texts: list[str] = field(default_factory=list)
    channels: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    tokens: list[set[str]] = field(default_factory=list)
    nearest_article: str = ""
    nearest_score: float = 0.0
    verdict: str = CONTENT_GAP

    @property
    def size(self) -> int:
        return len(self.texts)

    @property
    def example(self) -> str:
        """The shortest phrasing, which is usually the clearest."""
        return min(self.texts, key=len) if self.texts else ""

    def closest(
        self, tokens: set[str], weights: dict[str, float] | None
    ) -> float:
        # Single linkage: close to any member is close enough. Contacts about
        # one topic arrive worded a dozen ways, and insisting each is close to
        # the group's centroid splits them back apart.
        return max(
            (similarity(tokens, existing, weights) for existing in self.tokens),
            default=0.0,
        )


def cluster(
    decisions: list[Decision], threshold: float = SIMILARITY
) -> list[Gap]:
    tokenized = [(d, set(tokenize(d.text))) for d in decisions]
    tokenized = [(d, t) for d, t in tokenized if t]
    documents = [t for _, t in tokenized]
    weights = (
        inverse_frequency(documents)
        if len(documents) >= MIN_DOCS_FOR_WEIGHTING
        else None
    )

    gaps: list[Gap] = []
    for decision, tokens in tokenized:
        # Best match rather than first match: with single linkage the order
        # contacts arrive in should not decide which group they land in.
        best, score = None, 0.0
        for gap in gaps:
            found = gap.closest(tokens, weights)
            if found >= threshold and found > score:
                best, score = gap, found
        if best is None:
            best = Gap()
            gaps.append(best)
        best.texts.append(decision.text)
        best.tokens.append(tokens)
        best.channels[decision.channel] += 1
        best.reasons[decision.reason] += 1
    return gaps


def diagnose(
    gap: Gap, retriever: BM25Retriever, floor: float
) -> None:
    """Work out which article is nearest, and whether one exists at all.

    Retrieval is re-run here rather than read from the journal on purpose: a
    contact rejected on low confidence never reached the retriever, and those
    are exactly the ones where knowing the nearest article settles whether
    this is a missing article or a missing phrase.
    """
    votes: Counter = Counter()
    scores: dict[str, list[float]] = {}
    for text in gap.texts:
        hits = retriever.search(text, top_k=1)
        if not hits:
            continue
        votes[hits[0].article_id] += 1
        scores.setdefault(hits[0].article_id, []).append(hits[0].score)

    if votes:
        article, _ = votes.most_common(1)[0]
        found = scores[article]
        mean = round(sum(found) / len(found), 3)
        # A weak match is a coincidence, not a near miss. Naming it would send
        # a content writer to open an article about password resets because a
        # question about swapping a camera happened to share two words with it.
        if mean >= floor * NEAREST_IS_MEANINGFUL:
            gap.nearest_article, gap.nearest_score = article, mean

    recognition = gap.reasons.get("intent_below_confidence_floor", 0)
    mostly_unrecognised = recognition > gap.size / 2
    gap.verdict = (
        RECOGNITION_GAP
        if mostly_unrecognised and gap.nearest_score >= floor
        else CONTENT_GAP
    )


def find_gaps(
    path: str | Path,
    settings: Settings | None = None,
    retriever: BM25Retriever | None = None,
    threshold: float = SIMILARITY,
    min_size: int = MIN_CLUSTER,
) -> tuple[list[Gap], list[Gap], int]:
    """Return the recurring gaps, the one-offs, and how many were examined."""
    settings = settings or Settings.from_env()
    retriever = retriever or BM25Retriever()

    unanswered = [d for d in read_decisions(path) if d.unanswered]
    gaps = cluster(unanswered, threshold)
    for gap in gaps:
        diagnose(gap, retriever, settings.min_retrieval_score)

    keep = [g for g in gaps if g.size >= min_size]
    keep.sort(key=lambda g: (-g.size, g.example))
    singles = [g for g in gaps if g.size < min_size]
    singles.sort(key=lambda g: g.example)
    return keep, singles, len(unanswered)


def render(
    gaps: list[Gap], singles: list[Gap], examined: int, floor: float
) -> str:
    if not examined:
        return "No unanswered contacts in the journal. Nothing to report.\n"

    covered = sum(g.size for g in gaps)
    lines = [
        "Knowledge gaps",
        "=" * 72,
        f"{examined} contacts went unanswered; {covered} of them fall into "
        f"{len(gaps)} recurring groups.",
        "",
    ]

    for rank, gap in enumerate(gaps, 1):
        channels = ", ".join(f"{c}x{n}" for c, n in gap.channels.most_common())
        lines.append(f"{rank}. {gap.size} contacts  [{channels}]")
        lines.append(f'   "{gap.example}"')
        for other in _other_phrasings(gap):
            lines.append(f'   also: "{other}"')
        if gap.verdict == RECOGNITION_GAP:
            lines.append(
                f"   -> RECOGNITION GAP. {gap.nearest_article} answers this "
                f"(score {gap.nearest_score}, above the {floor} floor), but the "
                "classifier did not recognise the question. Add the phrasing "
                "to the lexicon."
            )
        elif gap.nearest_article:
            lines.append(
                f"   -> CONTENT GAP. Nearest is {gap.nearest_article} at "
                f"{gap.nearest_score}, below the {floor} floor. Extend it, or "
                "write a new article."
            )
        else:
            lines.append("   -> CONTENT GAP. Nothing in the help centre is even close.")
        lines.append("")

    if singles:
        # Never end on "nothing to report" when thirteen people asked
        # something. Grouping is lexical and splits synonyms, so an ungrouped
        # tail is expected rather than a sign there was no signal.
        lines.append(f"Did not group ({len(singles)} one-off questions)")
        lines.append(
            "   Asked once each, or worded too differently to group "
            "lexically. Worth skimming for themes a machine cannot see."
        )
        for gap in singles[:15]:
            lines.append(f'   - "{gap.example}"')
        if len(singles) > 15:
            lines.append(f"   ... and {len(singles) - 15} more")
    return "\n".join(lines).rstrip() + "\n"


def _other_phrasings(gap: Gap, limit: int = 2) -> list[str]:
    seen = {gap.example}
    out = []
    for text in gap.texts:
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) == limit:
            break
    return out
