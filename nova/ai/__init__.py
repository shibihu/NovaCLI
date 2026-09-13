"""AI provider layer.

Nova Core never talks to a vendor SDK directly — it talks to the
:class:`AIProvider` protocol. Groq is the v1.0 implementation
(:mod:`nova.ai.groq`); Gemini, OpenAI or a local model can be added later by
implementing the same members and registering them in
:func:`get_provider`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nova.core.models import AIResponse

__all__ = [
    "AIProvider",
    "AIProviderError",
    "MissingAPIKeyError",
    "get_provider",
    "provider_names",
]


class AIProviderError(Exception):
    """Any provider failure, already sanitised of secrets."""


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
    return ("groq",)


def get_provider(settings: object, name: str = "groq") -> AIProvider:
    """Build the configured provider."""
    if name == "groq":
        from .groq import GroqProvider

        return GroqProvider(
            api_key=getattr(settings, "groq_api_key", None),
            model=getattr(settings, "groq_model", None) or "openai/gpt-oss-20b",
            timeout=float(getattr(settings, "command_timeout", 30)),
            reasoning_effort=getattr(settings, "reasoning_effort", None),
        )
    raise AIProviderError(
        f"Unknown provider {name!r}. Available: {', '.join(provider_names())}"
    )
