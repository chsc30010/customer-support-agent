from support_agent.kb import BM25Retriever, load_passages, parse_article, stem, tokenize
from support_agent.models import Intent

ARTICLE = """---
id: demo
title: Demo article
intents: billing
url: https://example.test/demo
handoff_sections: Second section
---

## First section
Some body text about invoices.

## Second section
More body text about receipts.
"""


def test_parse_article_splits_on_headings():
    passages = parse_article(ARTICLE)
    assert [p.section for p in passages] == ["First section", "Second section"]
    assert passages[0].article_id == "demo"
    assert passages[0].intents == (Intent.BILLING,)
    assert passages[0].url == "https://example.test/demo"


def test_handoff_sections_mark_only_the_named_section():
    passages = parse_article(ARTICLE)
    assert passages[0].requires_human is False
    assert passages[1].requires_human is True


def test_stemmer_folds_inflections_without_merging_distinct_words():
    assert stem("refunds") == stem("refunded") == stem("refund")
    assert stem("recordings") == stem("recording")
    assert stem("cancelling") == stem("cancel")
    assert stem("business") == "business"
    assert stem("billing") != stem("battery")


def test_tokenize_drops_stopwords():
    assert tokenize("where is my order") == ["order"]


def test_every_article_loads():
    passages = load_passages()
    assert len(passages) > 40
    assert all(p.article_id and p.section and p.text for p in passages)


def test_search_finds_the_right_article():
    retriever = BM25Retriever()
    pairs = [
        ("how do I get a return label", "returns-and-refunds"),
        ("my camera keeps going offline", "camera-offline"),
        ("I was charged twice this month", "billing-charges"),
        ("reset my password", "sign-in-problems"),
        ("does it work with homekit", "compatibility"),
    ]
    for query, expected in pairs:
        assert retriever.search(query, top_k=1)[0].article_id == expected, query


def test_intent_expansion_only_applies_to_short_queries():
    retriever = BM25Retriever()
    bare = retriever.search("order", top_k=1)
    guided = retriever.search("order", top_k=1, intent=Intent.ORDER_STATUS)
    assert guided[0].score > bare[0].score

    long_query = "I cannot log in to the app"
    plain = retriever.search(long_query, top_k=1)
    with_intent = retriever.search(long_query, top_k=1, intent=Intent.ACCOUNT_ACCESS)
    # Only the multiplicative intent boost applies, not query expansion, so the
    # ranking is unchanged.
    assert plain[0].article_id == with_intent[0].article_id


def test_search_returns_nothing_for_an_empty_query():
    assert BM25Retriever().search("   ") == []
