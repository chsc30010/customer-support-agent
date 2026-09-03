"""Rendering a reply for a phone call.

Speech is not text read out. The differences that matter, in the order they
bite: a URL is useless out loud, an order number has to be spelled rather than
pronounced, long sentences lose the listener, and a caller needs a beat between
ideas to interrupt. Each of those is handled here so the agent itself never has
to think about the medium.
"""

from __future__ import annotations

import re
from html import escape

from ..models import AgentReply
from .base import URL, Renderer, RenderedReply, strip_markdown

#: Things the caller has to write down: an order reference like KH-482913, or a
#: bare run of digits long enough to be an identifier rather than a quantity.
#: One pattern rather than two, because running two passes nested the second
#: say-as inside the first on any reference that contained both.
_SPELL_OUT = re.compile(r"\b[A-Z]{2,4}-?\d{4,}\b|\b\d{6,}\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: Written forms that a speech engine reliably mangles.
PRONUNCIATION = {
    "2.4GHz": "two point four gigahertz",
    "5GHz": "five gigahertz",
    "1080p": "ten eighty p",
    "2K": "two K",
    "IP65": "I P sixty five",
    "RTSP": "R T S P",
    "wifi": "wi-fi",
    "VAT": "V A T",
    "QR": "Q R",
    "8am": "8 a m",
    "8pm": "8 p m",
    "2pm": "2 p m",
}

#: A sentence longer than this gets extra pauses at its commas. A listener
#: loses the thread well before a reader does, and a pause is somewhere to
#: interrupt -- callers barge in at pauses, not mid-clause.
MAX_SPOKEN_WORDS = 18

LINK_SUBSTITUTE = "I can text you the link if that helps"


def _despell(text: str) -> str:
    for written, spoken in PRONUNCIATION.items():
        text = re.sub(re.escape(written), spoken, text, flags=re.IGNORECASE)
    return text


def _drop_links(text: str) -> tuple[str, bool]:
    """A spoken URL is noise. Say we can send it instead."""
    if not URL.search(text):
        return text, False
    cleaned = URL.sub("our help centre", text)
    return " ".join(cleaned.split()), True


def spoken_sentences(text: str) -> list[str]:
    """Real sentences only.

    An earlier version chopped long sentences into shorter fake ones at their
    commas. It read badly in the transcript and changed what the customer was
    told; the sentence is fine, it just needs breathing room, which is what
    SSML breaks are for.
    """
    flat = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]


def _mark_up(sentence: str) -> str:
    """Escape for XML, then spell out anything the caller has to write down."""
    marked = escape(sentence, quote=False)

    def spell_out(match: re.Match[str]) -> str:
        token = match.group(0)
        # "digits" reads 4-8-2-9-1-3; "characters" also spells the letters and
        # the hyphen, which is what an order reference needs.
        how = "digits" if token.isdigit() else "characters"
        return '<say-as interpret-as="{}">{}</say-as>'.format(how, token)

    marked = _SPELL_OUT.sub(spell_out, marked)
    if len(sentence.split()) > MAX_SPOKEN_WORDS:
        marked = marked.replace(", ", ',<break time="200ms"/> ')
    return marked


class VoiceRenderer(Renderer):
    """Produces SSML for the telephony leg, plus a plain transcript for logs."""

    def render(self, reply: AgentReply) -> RenderedReply:
        text = strip_markdown(reply.text)
        text, had_link = _drop_links(text)
        text = _despell(text)
        if had_link:
            text = "{} {}.".format(text.rstrip("."), LINK_SUBSTITUTE)

        sentences = spoken_sentences(text)
        body = '<break time="350ms"/>'.join(_mark_up(s) for s in sentences)
        return RenderedReply(
            text=" ".join(sentences),
            ssml="<speak>{}</speak>".format(body),
        )
