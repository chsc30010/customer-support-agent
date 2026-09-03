"""Intent and sentiment classification."""

from ..config import Settings
from .base import Classifier
from .heuristic import HeuristicClassifier
from .llm import LLMClassifier

__all__ = ["Classifier", "HeuristicClassifier", "LLMClassifier", "build_classifier"]


def build_classifier(settings: Settings | None = None) -> Classifier:
    """Pick the best classifier the current configuration can actually run."""
    settings = settings or Settings.from_env()
    if settings.llm_enabled:
        return LLMClassifier(settings)
    return HeuristicClassifier()
