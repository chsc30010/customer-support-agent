"""Scoring, kept apart from running so the numbers are easy to check by hand."""

from __future__ import annotations

from dataclasses import dataclass

from harness import Outcome
from support_agent.models import Intent


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    support: int


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def intent_accuracy(outcomes: list[Outcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(o.intent_correct for o in outcomes) / len(outcomes)


def confusion(outcomes: list[Outcome]) -> dict[tuple[Intent, Intent], int]:
    matrix: dict[tuple[Intent, Intent], int] = {}
    for o in outcomes:
        key = (o.case.intent, o.predicted_intent)
        matrix[key] = matrix.get(key, 0) + 1
    return matrix


def per_intent(outcomes: list[Outcome]) -> dict[Intent, Score]:
    scores: dict[Intent, Score] = {}
    for intent in Intent:
        gold = [o for o in outcomes if o.case.intent is intent]
        predicted = [o for o in outcomes if o.predicted_intent is intent]
        hits = sum(1 for o in predicted if o.case.intent is intent)
        precision = hits / len(predicted) if predicted else 0.0
        recall = hits / len(gold) if gold else 0.0
        if gold or predicted:
            scores[intent] = Score(precision, recall, _f1(precision, recall), len(gold))
    return scores


def escalation(outcomes: list[Outcome]) -> tuple[Score, dict[str, int]]:
    """Scored with escalation as the positive class.

    A false negative is a customer who needed a person and did not get one --
    the failure this whole design exists to avoid -- so recall is the number to
    watch.
    """
    tp = sum(1 for o in outcomes if o.escalated and o.case.should_escalate)
    fp = sum(1 for o in outcomes if o.escalated and not o.case.should_escalate)
    fn = sum(1 for o in outcomes if not o.escalated and o.case.should_escalate)
    tn = sum(1 for o in outcomes if not o.escalated and not o.case.should_escalate)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return Score(precision, recall, _f1(precision, recall), tp + fn), counts


def retrieval(outcomes: list[Outcome]) -> tuple[float, float, int]:
    """hit@1 and hit@3 over the cases that have a labelled article."""
    scored = [o for o in outcomes if o.case.expected_article]
    if not scored:
        return 0.0, 0.0, 0
    at1 = sum(1 for o in scored if o.retrieved[:1] == [o.case.expected_article])
    at3 = sum(1 for o in scored if o.case.expected_article in o.retrieved[:3])
    return at1 / len(scored), at3 / len(scored), len(scored)


def resolution_rate(outcomes: list[Outcome]) -> tuple[float, int]:
    """End to end: answerable contacts actually answered from the right article."""
    answerable = [
        o for o in outcomes if not o.case.should_escalate and o.case.expected_article
    ]
    if not answerable:
        return 0.0, 0
    return sum(o.resolved_correctly for o in answerable) / len(answerable), len(answerable)


def reason_breakdown(outcomes: list[Outcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in outcomes:
        if o.escalated:
            counts[o.reason] = counts.get(o.reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
