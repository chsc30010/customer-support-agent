"""The Claude client, used for classification and for grounded answers.

Every call returns ``None`` rather than raising. A support line that stops
answering because a model call timed out is worse than one that falls back to
its deterministic path, so failure here is always a downgrade, never an outage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .models import Channel

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

#: How long one turn may spend waiting on models, per channel.
#:
#: These are dictated by the transport, not by taste. Twilio abandons a voice
#: webhook at about 15 seconds and drops the call, so on a phone line a slow
#: model call is not a slow answer -- it is a hung-up customer. Six seconds
#: leaves room for the reply to be synthesised and still reach Twilio in time.
#: An email has nobody waiting, so it can afford to be patient.
TURN_BUDGET_SECONDS: dict[Channel, float] = {
    Channel.VOICE: 6.0,
    Channel.SMS: 12.0,
    Channel.CHAT: 12.0,
    Channel.EMAIL: 30.0,
}
DEFAULT_TURN_BUDGET = 12.0

#: Below this there is no point starting a call: it cannot finish, and the
#: time spent failing is time the deterministic path could have used.
MIN_CALL_SECONDS = 1.5


@dataclass
class Deadline:
    """The model-call budget for one turn, shared by every call in it.

    A turn on the LLM path makes two calls -- classify, then draft -- and they
    are sequential. Timing them individually is not enough: two calls that each
    come in under their own limit can still blow the turn. So the budget is
    owned by the turn and both calls draw down the same clock.
    """

    budget: float
    started: float = field(default_factory=time.monotonic)

    @classmethod
    def for_channel(cls, channel: Channel) -> "Deadline":
        return cls(TURN_BUDGET_SECONDS.get(channel, DEFAULT_TURN_BUDGET))

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - (time.monotonic() - self.started))

    def allows_call(self, minimum: float = MIN_CALL_SECONDS) -> bool:
        return self.remaining >= minimum


class ClaudeClient:
    """Thin wrapper over the Anthropic Messages API.

    Only one shape of call is needed: prompt in, schema-validated JSON out.
    ``output_config.format`` guarantees the response parses, which removes the
    "the model wrapped its JSON in prose" failure mode entirely.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = self.settings.llm_model or DEFAULT_MODEL
        self._client: Any = None
        self._anthropic: Any = None
        self._unavailable_reason = ""

        if self.settings.llm_provider != "anthropic":
            self._unavailable_reason = "LLM_PROVIDER is not 'anthropic'"
            return
        try:
            import anthropic
        except ImportError:
            self._unavailable_reason = "the anthropic package is not installed"
            return
        self._anthropic = anthropic
        # A bare Anthropic() resolves ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN,
        # then an `ant auth login` profile -- no key needs threading through here.
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:  # no credentials anywhere
            self._unavailable_reason = f"no Anthropic credentials: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def json_call(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        effort: str = "low",
        max_tokens: int = 2048,
        deadline: Deadline | None = None,
    ) -> dict[str, Any] | None:
        """Ask for one JSON object matching ``schema``. ``None`` means fall back.

        ``effort`` is the cost/quality dial. Classification runs at "low" -- it
        is high volume, latency-sensitive, and the deterministic classifier is
        already close. Answer drafting runs higher, because a wrong answer to a
        customer costs more than the tokens.

        ``deadline`` is the turn's remaining time. Without one the SDK's own
        default applies, which is ten minutes -- fine for a script, fatal on a
        phone call.
        """
        if not self.available:
            return None
        if deadline is not None and not deadline.allows_call():
            log.warning("skipping model call: %.2fs left in the turn", deadline.remaining)
            return None

        client = self._client
        if deadline is not None:
            # max_retries must go to zero alongside a timeout. The SDK retries
            # timeouts by default, so a 4 second limit with two retries is a
            # twelve second call -- which is the bug this is here to prevent.
            client = client.with_options(timeout=deadline.remaining, max_retries=0)

        anthropic = self._anthropic
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.RateLimitError:
            log.warning("Claude rate limited; falling back")
            return None
        except anthropic.APIStatusError as exc:
            log.warning("Claude returned %s; falling back", exc.status_code)
            return None
        except anthropic.APITimeoutError:
            log.warning("Claude exceeded the turn budget; falling back")
            return None
        except anthropic.APIConnectionError as exc:
            log.warning("Claude unreachable (%s); falling back", exc)
            return None

        if response.stop_reason == "refusal":
            log.warning("Claude declined the request; falling back")
            return None

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            log.warning("Claude returned unparseable JSON; falling back")
            return None
        return parsed if isinstance(parsed, dict) else None
