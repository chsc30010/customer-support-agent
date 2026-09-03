"""Drafting a reply from retrieved knowledge base passages."""

from ..config import Settings
from .base import AnswerEngine
from .extractive import ExtractiveAnswerEngine
from .llm import LLMAnswerEngine

__all__ = [
    "AnswerEngine",
    "ExtractiveAnswerEngine",
    "LLMAnswerEngine",
    "build_answer_engine",
]


def build_answer_engine(settings: Settings | None = None) -> AnswerEngine:
    settings = settings or Settings.from_env()
    if settings.llm_enabled:
        return LLMAnswerEngine(settings)
    return ExtractiveAnswerEngine(settings)
