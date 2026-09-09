import json

from support_agent.config import Settings
from support_agent.gaps import (
    CONTENT_GAP,
    RECOGNITION_GAP,
    Gap,
    cluster,
    diagnose,
    find_gaps,
    inverse_frequency,
    render,
    similarity,
)
from support_agent.journal import Decision
from support_agent.kb import BM25Retriever

RETRIEVER = BM25Retriever()
SETTINGS = Settings()


def decision(text, reason="intent_below_confidence_floor", channel="chat"):
    return Decision(
        at="2026-09-09T12:00:00+00:00", conversation_id="c", channel=channel,
        turn=1, text=text, intent="unknown", confidence=0.0, sentiment="neutral",
        classifier="heuristic", escalated=True, reason=reason, queue="general_support",
        priority="normal", grounded=False, answer_source="", citation="",
        retrieval_score=0.0, latency_ms=1,
    )


def write(tmp_path, decisions):
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        "\n".join(json.dumps(d.__dict__) for d in decisions) + "\n", encoding="utf-8"
    )
    return path


def test_similarity_is_symmetric_and_bounded():
    a, b = {"swap", "colour"}, {"swap", "model"}
    assert similarity(a, b) == similarity(b, a)
    assert similarity(a, a) == 1.0
    assert similarity(a, set()) == 0.0


def test_rare_shared_words_count_for_more_than_common_ones():
    docs = [{"swap", "order"}, {"order", "refund"}, {"order", "track"}]
    weights = inverse_frequency(docs)
    # "order" is in every document here, so it distinguishes nothing.
    assert weights["swap"] > weights["order"]

    common_only = similarity({"order", "aaa"}, {"order", "bbb"}, weights)
    rare_only = similarity({"swap", "aaa"}, {"swap", "bbb"}, weights)
    assert rare_only > common_only


def test_near_duplicates_group_together():
    gaps = cluster(
        [
            decision("I need to swap this for a different colour"),
            decision("I would like to swap my camera for a different one"),
            decision("where is my order"),
        ]
    )
    sizes = sorted(g.size for g in gaps)
    assert sizes == [1, 2]


def test_grouping_does_not_depend_on_arrival_order():
    texts = [
        "are you open on bank holidays",
        "where is my order",
        "what are your opening hours on christmas day",
    ]
    forward = sorted(g.size for g in cluster([decision(t) for t in texts]))
    backward = sorted(g.size for g in cluster([decision(t) for t in reversed(texts)]))
    assert forward == backward


def test_a_junk_nearest_article_is_not_reported():
    gap = Gap(texts=["do you sponsor local football teams"])
    diagnose(gap, RETRIEVER, SETTINGS.min_retrieval_score)
    # Something always scores highest; below the floor it is a coincidence,
    # and naming it would send a writer to the wrong article.
    assert gap.nearest_article == ""
    assert gap.verdict == CONTENT_GAP


def test_an_answerable_question_the_classifier_missed_is_a_recognition_gap():
    gap = Gap(
        texts=["how do I get a return label", "how do I get a return label"],
        reasons={"intent_below_confidence_floor": 2},
    )
    diagnose(gap, RETRIEVER, SETTINGS.min_retrieval_score)
    assert gap.nearest_article == "returns-and-refunds"
    assert gap.verdict == RECOGNITION_GAP


def test_only_unanswered_contacts_are_examined(tmp_path):
    path = write(
        tmp_path,
        [
            decision("do you have a shop in leeds I can visit"),
            decision("is there a store near me I can visit"),
            # Correct escalations: the system working, not a knowledge gap.
            decision("get me a person", reason="customer_asked_for_human"),
            decision("this is unacceptable", reason="customer_is_angry"),
        ],
    )
    gaps, singles, examined = find_gaps(path, SETTINGS, RETRIEVER)
    assert examined == 2
    assert sum(g.size for g in gaps) + sum(g.size for g in singles) == 2


def test_one_off_questions_are_surfaced_rather_than_dropped(tmp_path):
    path = write(tmp_path, [decision("do you offer trade pricing for installers")])
    gaps, singles, examined = find_gaps(path, SETTINGS, RETRIEVER)
    assert gaps == []
    assert len(singles) == 1
    report = render(gaps, singles, examined, SETTINGS.min_retrieval_score)
    assert "trade pricing" in report  # never silently swallowed


def test_the_report_says_so_when_there_is_nothing_to_report(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    gaps, singles, examined = find_gaps(path, SETTINGS, RETRIEVER)
    assert "Nothing to report" in render(gaps, singles, examined, 2.5)


def test_channels_are_counted_per_group(tmp_path):
    path = write(
        tmp_path,
        [
            decision("are you open on bank holidays", channel="sms"),
            decision("what are your opening hours on christmas day", channel="voice"),
        ],
    )
    gaps, _, _ = find_gaps(path, SETTINGS, RETRIEVER)
    assert gaps and set(gaps[0].channels) == {"sms", "voice"}
