"""Settings, read once from the environment.

Everything has a default that works offline. Missing credentials degrade a
capability -- they never crash the agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional convenience, not a hard dependency
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is in requirements.txt
    pass


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Twilio
    twilio_auth_token: str = ""
    twilio_account_sid: str = ""
    public_base_url: str = ""
    handoff_phone_number: str = ""
    allow_unsigned_webhooks: bool = False

    # Language model
    llm_provider: str = "none"
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5"

    # Decision thresholds
    min_intent_confidence: float = 0.35
    min_retrieval_score: float = 2.5
    max_turns_before_handoff: int = 4

    company_name: str = "Kestrel Home"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
            twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/"),
            handoff_phone_number=os.getenv("HANDOFF_PHONE_NUMBER", "").strip(),
            allow_unsigned_webhooks=_bool("ALLOW_UNSIGNED_WEBHOOKS", False),
            llm_provider=os.getenv("LLM_PROVIDER", "none").strip().lower() or "none",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip() or "claude-opus-5",
            min_intent_confidence=_float("MIN_INTENT_CONFIDENCE", 0.35),
            min_retrieval_score=_float("MIN_RETRIEVAL_SCORE", 2.5),
            max_turns_before_handoff=_int("MAX_TURNS_BEFORE_HANDOFF", 4),
            company_name=os.getenv("COMPANY_NAME", "Kestrel Home").strip(),
        )

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return False

    @property
    def can_verify_webhooks(self) -> bool:
        return bool(self.twilio_auth_token and self.public_base_url)
