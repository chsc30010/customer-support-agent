"""Rendering for the three read rather than heard channels."""

from __future__ import annotations

from ..models import AgentReply, Intent
from .base import Renderer, RenderedReply, strip_markdown

#: A concatenated SMS carries 153 GSM-7 characters per part, not 160 -- the
#: rest of the first segment is spent on the header that reassembles them.
SEGMENT_CHARS = 153
SINGLE_CHARS = 160
#: Beyond this the customer should be reading a page, not a text thread.
MAX_SEGMENTS = 3


def segment(text: str) -> list[str]:
    """Split into SMS parts on word boundaries, numbering them only if split."""
    text = " ".join(text.split())
    if len(text) <= SINGLE_CHARS:
        return [text]

    budget = SEGMENT_CHARS - len(" (9/9)")
    words = text.split(" ")
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= budget:
            current = candidate
        else:
            parts.append(current)
            current = word
    if current:
        parts.append(current)

    if len(parts) > MAX_SEGMENTS:
        parts = parts[:MAX_SEGMENTS]
        parts[-1] = parts[-1].rstrip(" .,") + " ..."
    total = len(parts)
    return ["{} ({}/{})".format(p, i, total) for i, p in enumerate(parts, 1)]


class SMSRenderer(Renderer):
    def render(self, reply: AgentReply) -> RenderedReply:
        text = strip_markdown(reply.text)
        return RenderedReply(text=text, segments=segment(text))


class ChatRenderer(Renderer):
    """The one channel where markdown and a clickable citation are useful."""

    def render(self, reply: AgentReply) -> RenderedReply:
        parts = [reply.text.strip()]
        for passage in _citations(reply):
            if passage.url:
                parts.append("[{}]({})".format(passage.citation, passage.url))
            else:
                parts.append(passage.citation)
        return RenderedReply(text="\n\n".join(p for p in parts if p))


SUBJECTS = {
    Intent.ORDER_STATUS: "Your order",
    Intent.RETURNS_REFUND: "Your return",
    Intent.BILLING: "Your billing question",
    Intent.ACCOUNT_ACCESS: "Getting back into your account",
    Intent.TECHNICAL_ISSUE: "Your Kestrel camera",
    Intent.CANCELLATION: "Your Kestrel Cloud plan",
    Intent.PRODUCT_INFO: "Your question about Kestrel",
    Intent.COMPLAINT: "Your complaint",
    Intent.UNKNOWN: "Your message to Kestrel support",
}


class EmailRenderer(Renderer):
    def render(self, reply: AgentReply) -> RenderedReply:
        body = [reply.text.strip()]
        links = [p for p in _citations(reply) if p.url]
        if links:
            body.append(
                "More detail: "
                + ", ".join("{} ({})".format(p.citation, p.url) for p in links)
            )
        body.append("Kestrel Home Support")
        return RenderedReply(
            text="\n\n".join(part for part in body if part),
            subject="Re: " + SUBJECTS.get(reply.classification.intent, SUBJECTS[Intent.UNKNOWN]),
        )


def _citations(reply: AgentReply):
    if reply.answer is None or not reply.answer.grounded:
        return []
    return reply.answer.citations[:2]
