"""Command line entry point.

``serve`` runs the webhooks. The other three exist so the agent can be worked
on without a phone: ``ask`` is one turn, ``simulate`` replays a scripted
conversation, and ``kb`` shows what retrieval actually returned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import SupportAgent
from .config import Settings
from .models import Channel, InboundMessage
from .render import render
from .telephony.speech import DTMF_MEANING


def _channel(name: str) -> Channel:
    return Channel(name.lower())


def _show(reply, rendered) -> None:
    c = reply.classification
    print(
        "  [{} {:.2f} | {} | {}]".format(
            c.intent.value, c.confidence, c.sentiment.value, c.source
        )
    )
    print("  agent: {}".format(rendered.text))
    if rendered.segments and len(rendered.segments) > 1:
        print("  sms segments: {}".format(len(rendered.segments)))
    if rendered.ssml:
        print("  ssml : {}".format(rendered.ssml))
    if rendered.subject:
        print("  subject: {}".format(rendered.subject))
    if reply.answer and reply.answer.citations and reply.answer.grounded:
        for passage in reply.answer.citations:
            print(
                "  source: {} ({:.2f})".format(passage.citation, passage.score)
            )
    if reply.escalated:
        e = reply.escalation
        print("  --> HANDOFF to {} [{}] because {}".format(e.queue.value, e.priority, e.reason.value))
        for line in e.summary.splitlines():
            print("      {}".format(line))


def cmd_ask(args: argparse.Namespace) -> int:
    agent = SupportAgent()
    message = InboundMessage(
        conversation_id=args.conversation_id,
        channel=_channel(args.channel),
        text=args.text,
    )
    reply = agent.handle(message)
    print("customer: {}".format(args.text))
    _show(reply, render(reply))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Replay a scripted conversation, one JSON object per line.

    Each line needs a ``text`` and may set ``digits`` or ``speech_confidence``
    to reproduce a keypress or a transcript the recogniser was unsure of.
    """
    path = Path(args.transcript)
    agent = SupportAgent()
    channel = _channel(args.channel)
    conversation_id = args.conversation_id or path.stem

    number = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        number += 1
        turn = json.loads(line)
        digits = turn.get("digits", "")
        text = turn.get("text", "")
        if not text and digits:
            # Same mapping the phone leg applies, so a simulated call and a
            # real one see identical input.
            text = DTMF_MEANING.get(digits, f"I pressed {digits}")
        message = InboundMessage(
            conversation_id=conversation_id,
            channel=channel,
            text=text,
            digits=digits,
            speech_confidence=turn.get("speech_confidence"),
        )
        reply = agent.handle(message)
        print("\n[{}] customer: {}".format(number, message.text or message.digits))
        _show(reply, render(reply))
        if reply.escalated:
            print("\n(conversation handed to a human)")
            break
    return 0


def cmd_kb(args: argparse.Namespace) -> int:
    from .kb import BM25Retriever

    for passage in BM25Retriever().search(args.query, top_k=args.top_k):
        print("{:6.3f}  {}".format(passage.score, passage.citation))
        print("        {}".format(" ".join(passage.text.split())[:140]))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = Settings.from_env()
    if not settings.can_verify_webhooks and not settings.allow_unsigned_webhooks:
        print(
            "warning: TWILIO_AUTH_TOKEN and PUBLIC_BASE_URL are unset, so Twilio "
            "webhooks will be refused with a 503. See .env.example.",
            file=sys.stderr,
        )
    uvicorn.run(
        "support_agent.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="support-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="handle a single message")
    ask.add_argument("text")
    ask.add_argument("--channel", default="chat", choices=[c.value for c in Channel])
    ask.add_argument("--conversation-id", default="cli")
    ask.set_defaults(func=cmd_ask)

    simulate = sub.add_parser("simulate", help="replay a scripted conversation")
    simulate.add_argument("transcript")
    simulate.add_argument("--channel", default="voice", choices=[c.value for c in Channel])
    simulate.add_argument("--conversation-id", default="")
    simulate.set_defaults(func=cmd_simulate)

    kb = sub.add_parser("kb", help="show what retrieval returns for a query")
    kb.add_argument("query")
    kb.add_argument("--top-k", type=int, default=3)
    kb.set_defaults(func=cmd_kb)

    serve = sub.add_parser("serve", help="run the webhook server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
