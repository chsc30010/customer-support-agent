"""Whether the model path is on should depend on whether it can actually run.

A stand-in replaces the anthropic package, and real credentials are cleared,
so nothing on the machine running the tests decides the outcome.
"""

import sys
from types import SimpleNamespace

import pytest

from support_agent.answer import LLMAnswerEngine, build_answer_engine
from support_agent.classify import LLMClassifier, build_classifier
from support_agent.config import Settings


def fake_anthropic(constructs=True):
    class Anthropic:
        def __init__(self):
            if not constructs:
                raise RuntimeError("no credentials found anywhere")

    return SimpleNamespace(Anthropic=Anthropic)


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_no_provider_is_never_enabled(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic())
    assert Settings(llm_provider="none").llm_enabled is False


def test_credentials_without_an_api_key_enable_the_model(monkeypatch):
    # A client that builds with no ANTHROPIC_API_KEY is what an `ant auth login`
    # profile, or ANTHROPIC_AUTH_TOKEN, looks like. The model used to stay
    # off here, because only the API key was checked.
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic())
    assert Settings(llm_provider="anthropic", anthropic_api_key="").llm_enabled is True


def test_credentials_that_fail_leave_the_model_off(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic(constructs=False))
    assert Settings(llm_provider="anthropic").llm_enabled is False


def test_an_api_key_does_not_enable_a_model_that_cannot_run(monkeypatch):
    # The key is set but the anthropic package is not installed. This used to
    # count as enabled, so the model path was built and then fell back to the
    # deterministic one on every single call.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    settings = Settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")
    assert settings.llm_enabled is False


def test_the_factories_pick_the_model_path_when_it_can_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic())
    settings = Settings(llm_provider="anthropic", anthropic_api_key="")
    assert isinstance(build_classifier(settings), LLMClassifier)
    assert isinstance(build_answer_engine(settings), LLMAnswerEngine)
