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
    "parse_retry_after",
    "retry_after_from_headers",
]


import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

#: Header names that carry a retry hint, in decreasing priority.
RETRY_AFTER_HEADERS = (
    "retry-after",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset",
)

#: Compact duration form used by Groq's reset headers, e.g. ``2m59.56s``.
_DURATION_RE = re.compile(
    r"^(?:(?P<minutes>\d+(?:\.\d+)?)m)?(?:(?P<seconds>\d+(?:\.\d+)?)s)?$"
)


def parse_retry_after(value: Any) -> int | None:
    """Normalise a ``Retry-After`` value into whole seconds.

    Supports the forms providers actually emit:

    * integer/float seconds — ``Retry-After: 60``
    * an RFC 7231 HTTP-date — ``Retry-After: Wed, 21 Oct 2026 07:28:00 GMT``
    * a compact duration — ``2m59.56s`` (Groq reset headers)

    Returns ``None`` when the value cannot be interpreted.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        seconds = int(value)
        return seconds if seconds >= 0 else None

    text = str(value).strip()
    if not text:
        return None

    # Plain seconds: "60", "12.5", "17"
    try:
        seconds = int(float(text))
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        return seconds if seconds >= 0 else None

    # Compact duration: "59s", "2m59.56s", "1m"
    duration = _DURATION_RE.match(text)
    if duration and (duration.group("minutes") or duration.group("seconds")):
        total = 0.0
        if duration.group("minutes"):
            total += float(duration.group("minutes")) * 60
        if duration.group("seconds"):
            total += float(duration.group("seconds"))
        return max(0, int(math.ceil(total)))

    # RFC 7231 HTTP-date.
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = (parsed - datetime.now(timezone.utc)).total_seconds()
    return max(0, int(math.ceil(delta)))


def retry_after_from_headers(headers: Any) -> int | None:
    """First parseable retry hint from an HTTP response's headers."""
    if not headers:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    for name in RETRY_AFTER_HEADERS:
        try:
            raw = getter(name)
        except Exception:  # noqa: BLE001 - malformed header mapping
            continue
        parsed = parse_retry_after(raw)
        if parsed is not None:
            return parsed
    return None


@dataclass
class RateLimitInfo:
    is_rate_limit: bool
    retry_after: int = 0
    limit_type: str = "RPM"
    provider: str = ""
    reason: str = ""
    is_permanent: bool = False
    status_code: int | None = None
    model: str = ""
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_rate_limit": self.is_rate_limit,
            "retry_after": self.retry_after,
            "limit_type": self.limit_type,
            "provider": self.provider,
            "model": self.model,
            "status_code": self.status_code,
            "error_code": self.error_code,
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
    status_code: int | None = None
    error_code: str | None = None

    if isinstance(exc, AIProviderError):
        if exc.provider:
            p_name = exc.provider
        if exc.model:
            p_model = exc.model
        if exc.status_code is not None:
            status_code = exc.status_code
        if exc.error_code:
            error_code = exc.error_code
        if exc.retry_after is not None and exc.retry_after > 0:
            retry_after = exc.retry_after
        if exc.is_permanent:
            is_perm = True

    # 401/403 and authentication/quota error codes are never worth retrying.
    if status_code in (401, 403):
        is_perm = True
    if error_code and error_code.upper() in ("INVALID_API_KEY", "UNAUTHENTICATED", "PERMISSION_DENIED"):
        is_perm = True

    msg = str(exc).lower()

    if is_perm or any(p in msg for p in _PERMANENT_PATTERNS):
        if any(p in msg for p in ("quota", "billing", "daily", "monthly")):
            return RateLimitInfo(
                is_rate_limit=True,
                retry_after=0,
                limit_type="DAILY_QUOTA" if "daily" in msg else "BILLING",
                provider=p_name,
                model=p_model,
                status_code=status_code,
                error_code=error_code,
                reason="Quota or billing limit exhausted",
                is_permanent=True,
            )
        return RateLimitInfo(
            is_rate_limit=False,
            retry_after=0,
            limit_type="PERMANENT",
            provider=p_name,
            model=p_model,
            status_code=status_code,
            error_code=error_code,
            reason="Authentication or authorization failed",
            is_permanent=True,
        )

    is_rl = ("rate limit" in msg or "429" in msg or "too many requests" in msg
             or "tpm" in msg or "rpm" in msg or "resource_exhausted" in msg)
    if not is_rl and not retry_after and status_code != 429:
        return RateLimitInfo(is_rate_limit=False, provider=p_name, model=p_model)

    if not retry_after:
        parsed_header = parse_retry_after(
            _header_hint_from_message(str(exc))
        )
        if parsed_header is not None:
            retry_after = parsed_header

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
        model=p_model,
        status_code=status_code,
        error_code=error_code,
        reason=f"Per-minute {limit_type} rate limit reached",
        is_permanent=False,
    )


def _header_hint_from_message(message: str) -> str | None:
    """Pull a ``Retry-After: <value>`` hint out of a provider error string."""
    match = re.search(
        r"retry[-_\s]?after\s*[:=]\s*([^\n;,]+)", message, re.IGNORECASE
    )
    if not match:
        return None
    return match.group(1).strip()


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
