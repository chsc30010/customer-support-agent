"""The orchestrator: classify, retrieve, answer, decide.

One turn in, one decision out. Everything channel-shaped happens before this
(``telephony/``) or after it (``render/``), which is why the same object serves
a phone call and an email without a branch.
"""

from __future__ import annotations

import logging

from . import phrases, policy
from .answer import AnswerEngine, build_answer_engine
from .classify import Classifier, build_classifier
from .config import Settings
from .conversations import ConversationStore
from .kb import BM25Retriever
from .models import (
    AgentReply,
    Answer,
    Conversation,
    EscalationReason,
    InboundMessage,
    Intent,
)

log = logging.getLogger(__name__)

TOP_K = 3
#: A short follow-up ("the second one", "still not working") retrieves badly on
#: its own, so it is searched together with what came before it.
FOLLOW_UP_WORDS = 8


class SupportAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        classifier: Classifier | None = None,
        retriever: BM25Retriever | None = None,
        answer_engine: AnswerEngine | None = None,
        store: ConversationStore | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.classifier = classifier or build_classifier(self.settings)
        self.retriever = retriever or BM25Retriever()
        self.answer_engine = answer_engine or build_answer_engine(self.settings)
        self.store = store or ConversationStore()

    def greeting(self, channel) -> str:
        return phrases.GREETING[channel]

    def handle(self, message: InboundMessage) -> AgentReply:
        conversation = self.store.get_or_create(
            message.conversation_id, message.channel, message.sender
        )
        conversation.add("customer", message.text)

        classification = self.classifier.classify(message, conversation)
        if classification.intent is not Intent.UNKNOWN:
            conversation.last_intent = classification.intent

        answer: Answer | None = None
        early = policy.pre_answer_reason(classification, self.settings)
        if early is EscalationReason.NONE:
            passages = self._retrieve(conversation, message, classification)
            answer = self.answer_engine.answer(
                message.text, passages, classification, conversation
            )
            if answer.grounded:
                conversation.served.update(p.key for p in answer.citations)

        escalation = policy.decide(
            classification, answer, conversation, self.settings
        )
        text = self._compose(conversation, answer, escalation)

        conversation.add("agent", text)
        conversation.escalated = escalation.escalate
        if escalation.escalate:
            # Re-summarize now that the transcript includes this turn, so the
            # receiving agent sees what the customer was last told.
            escalation.transcript = conversation.transcript()
            log.info(
                "escalating %s to %s (%s, %s)",
                conversation.id,
                escalation.queue.value,
                escalation.reason.value,
                escalation.priority,
            )

        return AgentReply(
            conversation_id=conversation.id,
            channel=conversation.channel,
            text=text,
            classification=classification,
            answer=answer,
            escalation=escalation,
            expects_reply=not escalation.escalate,
        )

    def _retrieve(self, conversation, message, classification):
        """Retrieve, minus anything this conversation has already been told.

        A customer who says "I already tried that" needs the next step, not the
        same paragraph again. Filtering what has been served turns a repeated
        answer into either progress through the article or, when the article is
        exhausted, an honest handoff.
        """
        served = conversation.served
        passages = self.retriever.search(
            self._query(conversation, message),
            top_k=TOP_K + len(served),
            intent=classification.intent,
        )
        return [p for p in passages if p.key not in served][:TOP_K]

    def _query(self, conversation: Conversation, message: InboundMessage) -> str:
        turns = conversation.customer_turns
        if len(turns) > 1 and len(message.text.split()) <= FOLLOW_UP_WORDS:
            return turns[-2].text + " " + message.text
        return message.text

    def _compose(
        self, conversation: Conversation, answer: Answer | None, escalation
    ) -> str:
        if not escalation.escalate:
            parts = [answer.text if answer else ""]
            follow_up = phrases.FOLLOW_UP.get(conversation.channel)
            if follow_up:
                parts.append(follow_up)
            return " ".join(p for p in parts if p)

        parts = []
        # A grounded answer is still worth saying before handing over -- the
        # customer may not need the human once they have heard it.
        if answer is not None and answer.grounded and answer.text:
            parts.append(answer.text)
        parts.append(phrases.HANDOFF[escalation.reason])
        parts.append(phrases.TRANSFER[conversation.channel.is_realtime])
        return " ".join(p for p in parts if p)
