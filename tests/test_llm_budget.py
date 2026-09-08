"""The turn budget, which is what keeps a slow model call from dropping a call."""

import json
from types import SimpleNamespace

from support_agent.answer.extractive import ExtractiveAnswerEngine
from support_agent.answer.llm import LLMAnswerEngine
from support_agent.classify.heuristic import HeuristicClassifier
from support_agent.classify.llm import LLMClassifier
from support_agent.config import Settings
from support_agent.llm import (
    DEFAULT_TURN_BUDGET,
    TURN_BUDGET_SECONDS,
    ClaudeClient,
    Deadline,
)
from support_agent.models import (
    Channel,
    Classification,
    InboundMessage,
    Intent,
    Passage,
    Sentiment,
)


class _Error(Exception):
    pass


FAKE_ANTHROPIC = SimpleNamespace(
    RateLimitError=type("RateLimitError", (_Error,), {}),
    APIStatusError=type("APIStatusError", (_Error,), {}),
    APITimeoutError=type("APITimeoutError", (_Error,), {}),
    APIConnectionError=type("APIConnectionError", (_Error,), {}),
)


class FakeClient:
    """Stands in for anthropic.Anthropic, recording how it was called."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.options = []
        self.messages = SimpleNamespace(create=self._create)

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
        )


def claude(payload):
    client = ClaudeClient(Settings())
    client._client = FakeClient(payload)
    client._anthropic = FAKE_ANTHROPIC
    return client


CLASSIFICATION = {
    "intent": "billing",
    "confidence": 0.9,
    "sentiment": "neutral",
    "wants_human": False,
    "evidence": ["charged twice"],
}
SCHEMA = {"type": "object"}
PASSAGE = Passage(
    article_id="billing-charges",
    article_title="Charges",
    section="Charged twice",
    text="A duplicate charge is almost always a retry. It drops off in three days.",
    score=9.0,
)


def message(text="I was charged twice", channel=Channel.VOICE):
    return InboundMessage(conversation_id="t", channel=channel, text=text)


def test_voice_gets_the_tightest_budget():
    # Twilio drops a voice call at about 15 seconds; email has nobody waiting.
    assert TURN_BUDGET_SECONDS[Channel.VOICE] < TURN_BUDGET_SECONDS[Channel.CHAT]
    assert TURN_BUDGET_SECONDS[Channel.CHAT] < TURN_BUDGET_SECONDS[Channel.EMAIL]
    assert TURN_BUDGET_SECONDS[Channel.VOICE] < 15.0


def test_every_channel_has_a_budget():
    for channel in Channel:
        assert Deadline.for_channel(channel).budget == TURN_BUDGET_SECONDS.get(
            channel, DEFAULT_TURN_BUDGET
        )


def test_a_fresh_deadline_allows_a_call_and_an_exhausted_one_does_not():
    assert Deadline(budget=10.0).allows_call()
    assert not Deadline(budget=0.0).allows_call()
    # Too little left to be worth starting: the call cannot finish in time.
    assert not Deadline(budget=0.2).allows_call()


def test_remaining_never_goes_negative():
    assert Deadline(budget=0.0).remaining == 0.0


def test_an_exhausted_deadline_skips_the_call_entirely():
    client = claude(CLASSIFICATION)
    result = client.json_call(
        system="s", prompt="p", schema=SCHEMA, deadline=Deadline(budget=0.0)
    )
    assert result is None
    assert client._client.calls == []  # never reached the network


def test_a_deadline_sets_a_timeout_and_disables_retries():
    client = claude(CLASSIFICATION)
    client.json_call(
        system="s", prompt="p", schema=SCHEMA, deadline=Deadline(budget=5.0)
    )
    assert len(client._client.options) == 1
    options = client._client.options[0]
    # Retries must be off alongside a timeout: the SDK retries timeouts, so
    # otherwise a 5 second limit is really a 15 second call.
    assert options["max_retries"] == 0
    assert 0 < options["timeout"] <= 5.0


def test_without_a_deadline_the_client_is_left_alone():
    client = claude(CLASSIFICATION)
    client.json_call(system="s", prompt="p", schema=SCHEMA)
    assert client._client.options == []
    assert len(client._client.calls) == 1


def test_the_classifier_falls_back_when_the_budget_is_gone():
    client = claude(CLASSIFICATION)
    classifier = LLMClassifier(
        Settings(), client=client, fallback=HeuristicClassifier()
    )
    result = classifier.classify(message(), None, Deadline(budget=0.0))
    assert result.source == "heuristic"
    assert result.intent is Intent.BILLING  # still answered, just deterministically
    assert client._client.calls == []


def test_the_classifier_uses_the_model_when_there_is_time():
    client = claude(CLASSIFICATION)
    classifier = LLMClassifier(
        Settings(), client=client, fallback=HeuristicClassifier()
    )
    result = classifier.classify(message(), None, Deadline(budget=10.0))
    assert result.source == "llm"
    assert len(client._client.calls) == 1


def test_a_human_request_survives_a_model_that_missed_it():
    client = claude({**CLASSIFICATION, "wants_human": False})
    classifier = LLMClassifier(
        Settings(), client=client, fallback=HeuristicClassifier()
    )
    result = classifier.classify(
        message("put me through to a person please"), None, Deadline(budget=10.0)
    )
    assert result.wants_human is True


def test_the_answerer_falls_back_when_classification_ate_the_budget():
    client = claude({"answerable": True, "answer": "drafted", "sources": ["1"]})
    engine = LLMAnswerEngine(
        Settings(), client=client, fallback=ExtractiveAnswerEngine(Settings())
    )
    answer = engine.answer(
        "why was I charged twice",
        [PASSAGE],
        Classification(intent=Intent.BILLING, confidence=0.9, sentiment=Sentiment.NEUTRAL),
        None,
        Deadline(budget=0.0),
    )
    assert answer.source == "extractive"
    assert answer.grounded  # still a real, grounded answer
    assert client._client.calls == []


def test_the_answerer_uses_the_model_when_there_is_time():
    client = claude({"answerable": True, "answer": "drafted", "sources": ["1"]})
    engine = LLMAnswerEngine(
        Settings(), client=client, fallback=ExtractiveAnswerEngine(Settings())
    )
    answer = engine.answer(
        "why was I charged twice",
        [PASSAGE],
        Classification(intent=Intent.BILLING, confidence=0.9, sentiment=Sentiment.NEUTRAL),
        None,
        Deadline(budget=10.0),
    )
    assert answer.source == "llm"
    assert answer.text == "drafted"


def test_the_two_calls_in_a_turn_share_one_clock():
    """The bug this whole mechanism exists for: two calls that each look fine
    on their own, but together overrun the turn."""
    deadline = Deadline(budget=10.0)
    classifier_client = claude(CLASSIFICATION)
    LLMClassifier(
        Settings(), client=classifier_client, fallback=HeuristicClassifier()
    ).classify(message(), None, deadline)

    answer_client = claude({"answerable": True, "answer": "drafted", "sources": ["1"]})
    LLMAnswerEngine(
        Settings(), client=answer_client, fallback=ExtractiveAnswerEngine(Settings())
    ).answer(
        "why",
        [PASSAGE],
        Classification(intent=Intent.BILLING, confidence=0.9),
        None,
        deadline,
    )

    first = classifier_client._client.options[0]["timeout"]
    second = answer_client._client.options[0]["timeout"]
    assert second <= first  # the clock ran down; it did not reset
