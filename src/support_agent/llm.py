"""The Claude client, used for classification and for grounded answers.

Every call returns ``None`` rather than raising. A support line that stops
answering because a model call timed out is worse than one that falls back to
its deterministic path, so failure here is always a downgrade, never an outage.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import Settings

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"


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
    ) -> dict[str, Any] | None:
        """Ask for one JSON object matching ``schema``. ``None`` means fall back.

        ``effort`` is the cost/quality dial. Classification runs at "low" -- it
        is high volume, latency-sensitive, and the deterministic classifier is
        already close. Answer drafting runs higher, because a wrong answer to a
        customer costs more than the tokens.
        """
        if not self.available:
            return None

        anthropic = self._anthropic
        try:
            response = self._client.messages.create(
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
