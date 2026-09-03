"""Report card for the support agent.

    python evals/run_evals.py
    python evals/run_evals.py --show-failures
    python evals/run_evals.py --min-escalation-recall 1.0   # for CI

Exit code is non-zero when a threshold is not met, so this is usable as a gate
on a pull request rather than something someone remembers to look at.
"""

from __future__ import annotations

import argparse
import json
import sys

from harness import GOLDEN_SET, Outcome, load_cases, run
from metrics import (
    confusion,
    escalation,
    intent_accuracy,
    per_intent,
    reason_breakdown,
    resolution_rate,
    retrieval,
)
from support_agent.config import Settings
from support_agent.models import Intent

SHORT = {
    Intent.ORDER_STATUS: "order",
    Intent.RETURNS_REFUND: "return",
    Intent.BILLING: "billing",
    Intent.ACCOUNT_ACCESS: "account",
    Intent.TECHNICAL_ISSUE: "tech",
    Intent.CANCELLATION: "cancel",
    Intent.PRODUCT_INFO: "product",
    Intent.COMPLAINT: "complaint",
    Intent.UNKNOWN: "unknown",
}


def print_confusion(outcomes: list[Outcome]) -> None:
    matrix = confusion(outcomes)
    intents = [i for i in Intent if any(o.case.intent is i for o in outcomes)]
    predicted = [i for i in Intent]
    header = "  gold / pred  " + " ".join(f"{SHORT[i]:>9}" for i in predicted)
    print(header)
    for gold in intents:
        row = " ".join(f"{matrix.get((gold, p), 0):>9}" for p in predicted)
        print(f"  {SHORT[gold]:<12} {row}")


def print_report(outcomes: list[Outcome], settings: Settings) -> None:
    accuracy = intent_accuracy(outcomes)
    esc, counts = escalation(outcomes)
    hit1, hit3, retrieval_n = retrieval(outcomes)
    resolved, answerable_n = resolution_rate(outcomes)

    print("Support agent evaluation")
    print("=" * 72)
    print(f"cases            {len(outcomes)}")
    print(f"classifier       {settings.llm_provider if settings.llm_enabled else 'heuristic'}")
    print(f"intent floor     {settings.min_intent_confidence}")
    print(f"retrieval floor  {settings.min_retrieval_score}")
    print()
    print(f"Intent accuracy      {accuracy:6.1%}  ({sum(o.intent_correct for o in outcomes)}/{len(outcomes)})")
    print(f"Escalation recall    {esc.recall:6.1%}  ({counts['tp']}/{counts['tp'] + counts['fn']} contacts that needed a human got one)")
    print(f"Escalation precision {esc.precision:6.1%}  ({counts['fp']} handed over unnecessarily)")
    print(f"Escalation F1        {esc.f1:6.1%}")
    print(f"Retrieval hit@1      {hit1:6.1%}  hit@3 {hit3:6.1%}  (over {retrieval_n} labelled cases)")
    print(f"Resolved correctly   {resolved:6.1%}  ({answerable_n} answerable contacts)")
    print()

    print("Confusion matrix")
    print_confusion(outcomes)
    print()

    print("Per intent")
    print(f"  {'intent':<12} {'prec':>7} {'recall':>7} {'f1':>7} {'n':>4}")
    for intent, score in per_intent(outcomes).items():
        print(
            f"  {SHORT[intent]:<12} {score.precision:7.2f} {score.recall:7.2f} "
            f"{score.f1:7.2f} {score.support:4}"
        )
    print()

    print("Why it handed over")
    for reason, count in reason_breakdown(outcomes).items():
        print(f"  {count:>3}  {reason}")


def print_failures(outcomes: list[Outcome]) -> None:
    wrong_intent = [o for o in outcomes if not o.intent_correct]
    wrong_escalation = [o for o in outcomes if not o.escalation_correct]
    if wrong_intent:
        print("\nIntent misses")
        for o in wrong_intent:
            print(
                f"  {o.case.id} [{o.case.channel.value}] {SHORT[o.case.intent]} "
                f"-> {SHORT[o.predicted_intent]} ({o.confidence:.2f})"
            )
            print(f"        {o.case.text}")
    if wrong_escalation:
        print("\nEscalation misses")
        for o in wrong_escalation:
            kind = "handed over but should not have" if o.escalated else "MISSED a handoff"
            print(f"  {o.case.id} [{o.case.channel.value}] {kind} ({o.reason or 'answered'})")
            print(f"        {o.case.text}")


def sweep(cases, settings: Settings) -> None:
    """What the retrieval floor is actually buying.

    It is the one number in this system with no obviously right value, so show
    the tradeoff instead of defending the default.
    """
    print("\nRetrieval floor sweep")
    print(f"  {'floor':>6} {'esc recall':>11} {'esc prec':>9} {'resolved':>9}")
    for floor in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        trial = Settings(**{**settings.__dict__, "min_retrieval_score": floor})
        outcomes = run(cases, trial)
        esc, _ = escalation(outcomes)
        resolved, _ = resolution_rate(outcomes)
        marker = "  <- current" if floor == settings.min_retrieval_score else ""
        print(
            f"  {floor:6.1f} {esc.recall:11.1%} {esc.precision:9.1%} "
            f"{resolved:9.1%}{marker}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", default=str(GOLDEN_SET))
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="score across retrieval floors")
    parser.add_argument("--json", action="store_true", help="machine readable summary")
    parser.add_argument("--min-intent-accuracy", type=float, default=0.0)
    parser.add_argument("--min-escalation-recall", type=float, default=0.0)
    parser.add_argument("--min-hit1", type=float, default=0.0)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    from pathlib import Path

    cases = load_cases(Path(args.golden_set))
    outcomes = run(cases, settings)

    accuracy = intent_accuracy(outcomes)
    esc, _ = escalation(outcomes)
    hit1, hit3, _ = retrieval(outcomes)
    resolved, _ = resolution_rate(outcomes)

    if args.json:
        print(
            json.dumps(
                {
                    "cases": len(outcomes),
                    "intent_accuracy": round(accuracy, 4),
                    "escalation_recall": round(esc.recall, 4),
                    "escalation_precision": round(esc.precision, 4),
                    "retrieval_hit_at_1": round(hit1, 4),
                    "retrieval_hit_at_3": round(hit3, 4),
                    "resolution_rate": round(resolved, 4),
                },
                indent=2,
            )
        )
    else:
        print_report(outcomes, settings)
        if args.show_failures:
            print_failures(outcomes)
        if args.sweep:
            sweep(cases, settings)

    failures = []
    if accuracy < args.min_intent_accuracy:
        failures.append(f"intent accuracy {accuracy:.1%} < {args.min_intent_accuracy:.1%}")
    if esc.recall < args.min_escalation_recall:
        failures.append(f"escalation recall {esc.recall:.1%} < {args.min_escalation_recall:.1%}")
    if hit1 < args.min_hit1:
        failures.append(f"retrieval hit@1 {hit1:.1%} < {args.min_hit1:.1%}")
    if failures:
        print("\nFAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
