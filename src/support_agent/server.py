"""The FastAPI app: Twilio webhooks for voice and SMS, JSON for chat and email.

Every route does the same three things -- turn a transport payload into an
:class:`InboundMessage`, hand it to the agent, render the reply for the channel
-- and differ only in what the transport wants back.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .agent import SupportAgent
from .config import Settings
from .models import AgentReply, Channel, InboundMessage
from .phrases import CLOSING, GREETING, NO_INPUT
from .render import render
from .render.voice import VoiceRenderer
from .telephony import TwilioSpeechAdapter, signature, twiml

log = logging.getLogger(__name__)

TWIML_MEDIA_TYPE = "application/xml"
#: Two unheard turns in a row is a bad line or an empty room, not a
#: conversation. A third re-prompt annoys everyone.
MAX_EMPTY_TURNS = 2


def _twiml(body: str) -> Response:
    return Response(content=body, media_type=TWIML_MEDIA_TYPE)


def create_app(
    agent: SupportAgent | None = None, settings: Settings | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    agent = agent or SupportAgent(settings)
    speech = TwilioSpeechAdapter()
    voice_renderer = VoiceRenderer()
    app = FastAPI(title="Kestrel Home support agent", version="0.1.0")

    async def twilio_form(request: Request, path: str) -> dict[str, str]:
        """Read a Twilio POST body, refusing anything it did not sign."""
        form = {k: str(v) for k, v in (await request.form()).items()}
        if settings.allow_unsigned_webhooks:
            log.warning("accepting unsigned webhook on %s", path)
            return form
        if not settings.can_verify_webhooks:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Webhook verification is not configured. Set TWILIO_AUTH_TOKEN "
                    "and PUBLIC_BASE_URL, or ALLOW_UNSIGNED_WEBHOOKS=1 for local "
                    "testing only."
                ),
            )
        url = signature.canonical_url(
            settings.public_base_url, path, request.url.query
        )
        provided = request.headers.get("X-Twilio-Signature", "")
        if not signature.is_valid(settings.twilio_auth_token, url, form, provided):
            log.warning("rejected webhook with bad signature on %s", path)
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        return form

    def transfer(reply: AgentReply) -> str:
        """Where an escalated call goes.

        A configured number is dialled directly. Without one the caller is
        parked in a named Twilio queue -- which is still the right queue, so
        the routing decision is not lost just because the number is missing.
        """
        if settings.handoff_phone_number:
            return twiml.dial(settings.handoff_phone_number)
        return twiml.enqueue(reply.escalation.queue.value)

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "classifier": agent.classifier.name,
                "answer_engine": agent.answer_engine.name,
                "kb_passages": len(agent.retriever.passages),
                "webhook_verification": settings.can_verify_webhooks
                and not settings.allow_unsigned_webhooks,
                "active_conversations": len(agent.store),
            }
        )

    @app.post("/twilio/voice")
    async def voice(request: Request) -> Response:
        await twilio_form(request, "/twilio/voice")
        greeting = voice_renderer.render(
            AgentReply(
                conversation_id="",
                channel=Channel.VOICE,
                text=GREETING[Channel.VOICE],
                classification=_blank_classification(),
            )
        )
        # The greeting sits inside the Gather so a caller who already knows
        # what they want can talk over it instead of waiting it out.
        return _twiml(
            twiml.voice_response(
                twiml.gather("/twilio/voice/turn", speech.speak(greeting))
            )
        )

    @app.post("/twilio/voice/turn")
    async def voice_turn(request: Request) -> Response:
        form = await twilio_form(request, "/twilio/voice/turn")
        heard = speech.hear(form)
        call_sid = form.get("CallSid", "unknown-call")

        if heard.is_empty:
            empty = int(request.query_params.get("empty", "0")) + 1
            if empty > MAX_EMPTY_TURNS:
                return _twiml(
                    twiml.voice_response(
                        twiml.say(plain=CLOSING), twiml.hangup()
                    )
                )
            prompt = voice_renderer.render(
                AgentReply(
                    conversation_id=call_sid,
                    channel=Channel.VOICE,
                    text=NO_INPUT,
                    classification=_blank_classification(),
                )
            )
            return _twiml(
                twiml.voice_response(
                    twiml.gather(
                        f"/twilio/voice/turn?empty={empty}", speech.speak(prompt)
                    )
                )
            )

        reply = agent.handle(
            InboundMessage(
                conversation_id=call_sid,
                channel=Channel.VOICE,
                text=heard.text,
                sender=form.get("From", ""),
                digits=heard.digits,
                speech_confidence=heard.confidence,
                metadata={"call_sid": call_sid},
            )
        )
        spoken = speech.speak(render(reply))
        if reply.escalated:
            return _twiml(twiml.voice_response(spoken, transfer(reply)))
        return _twiml(
            twiml.voice_response(twiml.gather("/twilio/voice/turn", spoken))
        )

    _register_text_routes(app, agent, twilio_form)
    return app


def _blank_classification():
    from .models import Classification, Intent

    return Classification(intent=Intent.UNKNOWN, confidence=0.0)


def _reply_payload(reply: AgentReply) -> dict:
    rendered = render(reply).as_dict()
    rendered["conversation_id"] = reply.conversation_id
    rendered["intent"] = reply.classification.intent.value
    rendered["confidence"] = reply.classification.confidence
    rendered["sentiment"] = reply.classification.sentiment.value
    rendered["escalation"] = reply.escalation.as_dict()
    rendered["citations"] = [
        {"title": p.article_title, "section": p.section, "url": p.url, "score": p.score}
        for p in (reply.answer.citations if reply.answer else [])
    ]
    return rendered


def _register_text_routes(app: FastAPI, agent: SupportAgent, twilio_form) -> None:
    @app.post("/twilio/sms")
    async def sms(request: Request) -> Response:
        form = await twilio_form(request, "/twilio/sms")
        sender = form.get("From", "unknown")
        reply = agent.handle(
            InboundMessage(
                # One thread per phone number, not per message: a customer who
                # texts three times in a row is having one conversation.
                conversation_id=f"sms:{sender}",
                channel=Channel.SMS,
                text=form.get("Body", ""),
                sender=sender,
                metadata={"message_sid": form.get("MessageSid", "")},
            )
        )
        rendered = render(reply)
        parts = rendered.segments or [rendered.text]
        return _twiml(twiml.messaging_response(*(twiml.message(p) for p in parts)))

    @app.post("/chat")
    async def chat(request: Request) -> JSONResponse:
        """The web widget. Put your own session auth in front of this."""
        body = await request.json()
        reply = agent.handle(
            InboundMessage(
                conversation_id=str(body.get("conversation_id") or "chat:anonymous"),
                channel=Channel.CHAT,
                text=str(body.get("text", "")),
                sender=str(body.get("customer_ref", "")),
            )
        )
        return JSONResponse(_reply_payload(reply))

    @app.post("/email")
    async def email(request: Request) -> JSONResponse:
        body = await request.json()
        sender = str(body.get("from", "unknown"))
        reply = agent.handle(
            InboundMessage(
                conversation_id=str(body.get("thread_id") or f"email:{sender}"),
                channel=Channel.EMAIL,
                text="\n".join(
                    part
                    for part in (str(body.get("subject", "")), str(body.get("body", "")))
                    if part
                ),
                sender=sender,
            )
        )
        return JSONResponse(_reply_payload(reply))
