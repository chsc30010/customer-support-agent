"""Building the TwiML that drives a call or a text.

Twilio speaks XML. This module is the only place that knows that, and it is
deliberately a set of small string builders rather than a dependency: TwiML is
a handful of elements, and the official helper library would be a large
addition for what amounts to four tags.
"""

from __future__ import annotations

import re
from html import escape

#: Twilio wraps the contents of <Say> in its own <speak>, so the portable
#: wrapper our renderer produces has to come off on the way out.
_SPEAK_WRAPPER = re.compile(r"^\s*<speak>(.*)</speak>\s*$", re.DOTALL)

#: Twilio neural voices. Amy reads British English clearly at speed, which
#: matters more on a support line than sounding impressive.
VOICE = "Polly.Amy-Neural"
LANGUAGE = "en-GB"

#: How long to wait for a caller who has gone quiet before re-prompting.
SPEECH_TIMEOUT = "auto"
GATHER_TIMEOUT = 6


def speech_body(ssml: str, plain: str) -> str:
    """The inner markup for a <Say>, falling back to escaped plain text."""
    match = _SPEAK_WRAPPER.match(ssml or "")
    if match:
        return match.group(1)
    return escape(plain, quote=False)


def say(ssml: str = "", plain: str = "") -> str:
    return '<Say voice="{}" language="{}">{}</Say>'.format(
        VOICE, LANGUAGE, speech_body(ssml, plain)
    )


def gather(action: str, prompt: str = "") -> str:
    """Listen for speech or a keypress.

    ``input="speech dtmf"`` is what makes "press zero" work alongside "say
    agent" -- and pressing zero is the escape hatch every caller already knows.
    """
    return (
        '<Gather input="speech dtmf" action="{action}" method="POST" '
        'speechTimeout="{speech}" timeout="{timeout}" numDigits="1" '
        'language="{language}" actionOnEmptyResult="true">{prompt}</Gather>'
    ).format(
        action=escape(action, quote=True),
        speech=SPEECH_TIMEOUT,
        timeout=GATHER_TIMEOUT,
        language=LANGUAGE,
        prompt=prompt,
    )


def dial(number: str, caller_id: str = "") -> str:
    attrs = ' callerId="{}"'.format(escape(caller_id, quote=True)) if caller_id else ""
    return "<Dial{}>{}</Dial>".format(attrs, escape(number, quote=False))


def hangup() -> str:
    return "<Hangup/>"


def enqueue(queue_name: str) -> str:
    """Park the caller in a Twilio queue when no transfer number is configured."""
    return '<Enqueue>{}</Enqueue>'.format(escape(queue_name, quote=False))


def message(body: str) -> str:
    return "<Message>{}</Message>".format(escape(body, quote=False))


def voice_response(*parts: str) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response>{}</Response>'.format(
        "".join(parts)
    )


def messaging_response(*parts: str) -> str:
    return voice_response(*parts)
