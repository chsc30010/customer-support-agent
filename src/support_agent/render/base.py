"""Turning one channel-neutral reply into what a channel can actually carry.

The agent writes plain prose. It does not know about SSML, SMS segment limits
or markdown, and it should not: the same decision has to be expressible as a
spoken sentence, a 153 character text and an email.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..models import AgentReply

_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_MARKS = re.compile(r"[*_`#]+")
URL = re.compile(r"\b(?:https?://\S+|www\.\S+|[a-z0-9-]+\.(?:com|net|org|io)\S*)")


def strip_markdown(text: str) -> str:
    """Markdown is a formatting language for exactly one of our four channels."""
    text = _MD_LINK.sub(r"\1 (\2)", text)
    text = _MD_MARKS.sub("", text)
    return " ".join(text.split())


@dataclass
class RenderedReply:
    """What actually goes out on the wire."""

    #: Plain text form. Always populated, on every channel.
    text: str
    #: Speech markup, voice only.
    ssml: str = ""
    #: One entry per outbound message, SMS only.
    segments: list[str] = field(default_factory=list)
    #: Email only.
    subject: str = ""

    def as_dict(self) -> dict:
        out: dict = {"text": self.text}
        if self.ssml:
            out["ssml"] = self.ssml
        if self.segments:
            out["segments"] = self.segments
        if self.subject:
            out["subject"] = self.subject
        return out


class Renderer(ABC):
    @abstractmethod
    def render(self, reply: AgentReply) -> RenderedReply: ...
