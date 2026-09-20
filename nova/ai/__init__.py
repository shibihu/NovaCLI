"""AI provider layer.

Nova Core never talks to a vendor SDK directly — it talks to the
:class:`AIProvider` protocol. Provider implementations:
- Groq (:mod:`nova.ai.groq`)
- Ollama (:mod:`nova.ai.ollama`)
- Gemini (:mod:`nova.ai.gemini`)
- OpenRouter (:mod:`nova.ai.openrouter`)
- Cerebras (:mod:`nova.ai.cerebras`)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nova.core.models import AIResponse

__all__ = [
    "AIProvider",
    "AIProviderError",
    "MissingAPIKeyError",
    "get_provider",
    "normalize_provider_name",
    "provider_names",
    "RateLimitInfo",
    "parse_rate_limit_info",
]


import re
from dataclasses import dataclass

@dataclass
class RateLimitInfo:
    is_rate_limit: bool
    retry_after: int = 0
    limit_type: str = "RPM"
    provider: str = ""
    reason: str = ""
    is_permanent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_rate_limit": self.is_rate_limit,
            "retry_after": self.retry_after,
            "limit_type": self.limit_type,
            "provider": self.provider,
            "reason": self.reason,
            "is_permanent": self.is_permanent,
        }


_PERMANENT_PATTERNS = (
    "daily quota",
    "daily limit",
    "monthly quota",
    "monthly limit",
    "quota exceeded",
    "quota exhausted",
    "insufficient_quota",
    "billing",
    "account disabled",
    "invalid api key",
    "401",
    "unauthorized",
    "forbidden",
)

def parse_rate_limit_info(exc: Exception, provider_name: str = "") -> RateLimitInfo:
    """Extract structured provider rate limit details from exception or HTTP metadata."""
    p_name = provider_name
    p_model = ""
    retry_after = 0
    is_perm = False

    if isinstance(exc, AIProviderError):
        if exc.provider:
            p_name = exc.provider
        if exc.model:
            p_model = exc.model
        if exc.retry_after is not None and exc.retry_after > 0:
            retry_after = exc.retry_after
        if exc.is_permanent:
            is_perm = True

    msg = str(exc).lower()

    if is_perm or any(p in msg for p in _PERMANENT_PATTERNS):
        if any(p in msg for p in ("quota", "billing", "daily", "monthly")):
            return RateLimitInfo(
                is_rate_limit=True,
                retry_after=0,
                limit_type="DAILY_QUOTA" if "daily" in msg else "BILLING",
                provider=p_name,
                reason="Quota or billing limit exhausted",
                is_permanent=True,
            )
        return RateLimitInfo(
            is_rate_limit=False,
            retry_after=0,
            limit_type="PERMANENT",
            provider=p_name,
            reason="Authentication or authorization failed",
            is_permanent=True,
        )

    is_rl = ("rate limit" in msg or "429" in msg or "too many requests" in msg
             or "tpm" in msg or "rpm" in msg or "resource_exhausted" in msg)
    if not is_rl and not retry_after and not (isinstance(exc, AIProviderError) and exc.status_code == 429):
        return RateLimitInfo(is_rate_limit=False, provider=p_name)

    limit_type = "TPM" if "tpm" in msg or "token" in msg else "RPM"

    if not retry_after:
        match = re.search(r"(?:retry\s+after|try\s+again\s+in|resets?\s+in|wait)\s+(\d+)\s*s?", msg)
        if match:
            try:
                retry_after = int(match.group(1))
            except ValueError:
                retry_after = 0

    if not retry_after:
        match_sec = re.search(r"(\d+)\s*seconds?", msg)
        if match_sec:
            try:
                retry_after = int(match_sec.group(1))
            except ValueError:
                retry_after = 0

    return RateLimitInfo(
        is_rate_limit=True,
        retry_after=retry_after,
        limit_type=limit_type,
        provider=p_name,
        reason=f"Per-minute {limit_type} rate limit reached",
        is_permanent=False,
    )


class AIProviderError(Exception):
    """Any provider failure, already sanitised of secrets."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: int | None = None,
        provider: str = "",
        model: str = "",
        error_code: str | None = None,
        is_permanent: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.provider = provider
        self.model = model
        self.error_code = error_code
        self.is_permanent = is_permanent


class MissingAPIKeyError(AIProviderError):
    """No API key configured; the message explains how to fix it."""


@runtime_checkable
class AIProvider(Protocol):
    """Minimal contract every model backend must satisfy."""

    @property
    def model_name(self) -> str:
        """Identifier of the model currently in use."""
        ...

    @property
    def configured(self) -> bool:
        """True when the provider has everything it needs to run."""
        ...

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
    ) -> AIResponse:
        """Return the assistant's response (text and/or native tool calls)."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


def provider_names() -> tuple[str, ...]:
    """Registered provider identifiers."""
    return ("groq", "gemini", "ollama", "openrouter", "cerebras")


def normalize_provider_name(name: str | None) -> str:
    """Normalize user input or aliases into canonical provider identifier."""
    if not name:
        return "groq"
    norm = str(name).strip().lower().replace("_", "").replace("-", "")
    if norm in ("groq",):
        return "groq"
    if norm in ("gemini", "google", "googlegemini"):
        return "gemini"
    if norm in ("ollama", "local"):
        return "ollama"
    if norm in ("openrouter", "openrouterai"):
        return "openrouter"
    if norm in ("cerebras", "cerebrasai"):
        return "cerebras"
    return str(name).strip().lower()


def get_provider(settings: object, name: str | None = None) -> AIProvider:
    """Build the configured provider implementation.

    The provider implementation is selected based on `name` (defaulting to `settings.provider`),
    and the model is resolved via `settings.model` or provider-specific configuration.
    """
    raw_name = name or getattr(settings, "provider", "groq")
    prov_name = normalize_provider_name(raw_name)

    if prov_name == "groq":
        from .groq import GroqProvider

        return GroqProvider(
            api_key=getattr(settings, "groq_api_key", None),
            model=getattr(settings, "groq_model", None) or getattr(settings, "model", "openai/gpt-oss-20b"),
            timeout=float(getattr(settings, "llm_timeout", 1800)),
            reasoning_effort=getattr(settings, "reasoning_effort", None),
        )

    if prov_name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(
            base_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
            model=getattr(settings, "ollama_model", None) or getattr(settings, "model", "qwen3:4b"),
            timeout=float(getattr(settings, "llm_timeout", 1800)),
        )

    if prov_name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(
            api_key=getattr(settings, "gemini_api_key", None),
            model=getattr(settings, "gemini_model", None) or getattr(settings, "model", "gemini-2.5-flash"),
            timeout=float(getattr(settings, "llm_timeout", 1800)),
        )

    if prov_name == "openrouter":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider(
            api_key=getattr(settings, "openrouter_api_key", None),
            model=getattr(settings, "openrouter_model", None) or getattr(settings, "model", "openai/gpt-oss-20b"),
            timeout=float(getattr(settings, "llm_timeout", 1800)),
        )

    if prov_name == "cerebras":
        from .cerebras import CerebrasProvider

        return CerebrasProvider(
            api_key=getattr(settings, "cerebras_api_key", None),
            model=getattr(settings, "cerebras_model", None) or getattr(settings, "model", "llama3.1-8b"),
            timeout=float(getattr(settings, "llm_timeout", 1800)),
        )

    raise AIProviderError(
        f"Unknown provider {raw_name!r}. Available: {', '.join(provider_names())}"
    )
