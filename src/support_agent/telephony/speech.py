"""The speech seam.

The agent reasons over text. Voice reaches it as a transcript and leaves it as
speech markup, and this module is the only place that knows how that
conversion happens. Today Twilio does both -- its ``<Gather input="speech">``
returns a transcript with a confidence score, and its ``<Say>`` renders SSML --
so there is no audio to move around. Swapping in a different recogniser or a
different voice means writing one more :class:`SpeechAdapter`; nothing above
this line changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from ..render.base import RenderedReply
from . import twiml


@dataclass
class Heard:
    """One recognised customer turn on a voice call."""

    text: str
    #: Recogniser confidence 0.0-1.0, or None when the turn was keypresses.
    confidence: float | None = None
    digits: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.digits


#: Keypresses become sentences so the rest of the pipeline never has to care
#: that this turn arrived as DTMF. Zero is the escape hatch every caller
#: already knows, and it has to work even when the line is too noisy to hear.
DTMF_MEANING = {
    "0": "I would like to speak to an agent",
    "1": "yes",
    "2": "no",
}


class SpeechAdapter(ABC):
    @abstractmethod
    def hear(self, payload: Mapping[str, str]) -> Heard: ...

    @abstractmethod
    def speak(self, rendered: RenderedReply) -> str:
        """Return the markup that makes the caller hear ``rendered``."""


class TwilioSpeechAdapter(SpeechAdapter):
    def hear(self, payload: Mapping[str, str]) -> Heard:
        digits = (payload.get("Digits") or "").strip()
        text = (payload.get("SpeechResult") or "").strip()
        confidence: float | None = None
        raw = payload.get("Confidence")
        if raw:
            try:
                confidence = float(raw)
            except ValueError:
                confidence = None

        if not text and digits:
            # A keypress carries no confidence: it is not a guess.
            return Heard(text=DTMF_MEANING.get(digits, f"I pressed {digits}"), digits=digits)
        return Heard(text=text, confidence=confidence, digits=digits)

    def speak(self, rendered: RenderedReply) -> str:
        return twiml.say(ssml=rendered.ssml, plain=rendered.text)
