"""The agent's own words -- everything it says that is not from an article.

Kept in one file so the voice of the product is reviewable in a single place,
and so a support lead can change the wording without touching the logic.
"""

from __future__ import annotations

from .models import Channel, EscalationReason

GREETING = {
    Channel.VOICE: "Thanks for calling Kestrel Home. Tell me what you need and I will help, or say agent at any time for a person.",
    Channel.SMS: "Kestrel Home support here. What can I help with?",
    Channel.CHAT: "Hi, Kestrel Home support. What can I help with?",
    Channel.EMAIL: "Thanks for writing to Kestrel Home support.",
}

#: Said before handing over. Each one names the real reason, because a customer
#: who is told why they are being transferred complains less than one who is
#: simply transferred.
HANDOFF = {
    EscalationReason.CUSTOMER_ASKED: "Of course.",
    EscalationReason.ANGRY_CUSTOMER: "I am sorry, that is genuinely frustrating.",
    EscalationReason.SENSITIVE_INTENT: "I want this logged properly rather than handled by me.",
    EscalationReason.LOW_CONFIDENCE: "I do not want to guess at this one.",
    EscalationReason.NO_GROUNDING: "I could not find a reliable answer to that.",
    EscalationReason.LOOPING: "We are going round in circles, and that is on me.",
    EscalationReason.HUMAN_ONLY: "This last step has to be done by a person.",
}

#: How the handover itself is described depends on whether anyone is waiting.
TRANSFER = {
    True: "Let me put you through to someone now.",
    False: "I have passed this to our team and someone will come back to you shortly.",
}

FOLLOW_UP = {
    Channel.VOICE: "Does that sort it, or is there something else?",
    Channel.CHAT: "Does that help?",
}

NO_INPUT = "Sorry, I did not catch that. Could you say it again?"

CLOSING = "Thanks for calling Kestrel Home. Goodbye."
