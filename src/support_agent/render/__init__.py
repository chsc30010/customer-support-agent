"""Channel-specific rendering."""

from ..models import AgentReply, Channel
from .base import Renderer, RenderedReply, strip_markdown
from .text import ChatRenderer, EmailRenderer, SMSRenderer, segment
from .voice import VoiceRenderer

RENDERERS: dict[Channel, Renderer] = {
    Channel.VOICE: VoiceRenderer(),
    Channel.SMS: SMSRenderer(),
    Channel.CHAT: ChatRenderer(),
    Channel.EMAIL: EmailRenderer(),
}

__all__ = [
    "ChatRenderer",
    "EmailRenderer",
    "RENDERERS",
    "Renderer",
    "RenderedReply",
    "SMSRenderer",
    "VoiceRenderer",
    "render",
    "segment",
    "strip_markdown",
]


def render(reply: AgentReply) -> RenderedReply:
    return RENDERERS[reply.channel].render(reply)
