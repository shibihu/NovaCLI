"""AI provider layer.

Nova Core never talks to a vendor SDK directly — it talks to the
:class:`AIProvider` protocol. Groq is the v1.0 implementation
(:mod:`nova.ai.groq`); Gemini, OpenAI or a local model can be added later by
implementing the same four members and registering them in
:func:`get_provider`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
    """Minimal contract every model backend must satisfy.

    Deliberately narrow: one async method returning text. Everything the agent
    needs (tool use, planning, final answers) is expressed in the prompt and
    parsed out of that text, so adding a backend requires no agent changes.
    """

    @property
    def model_name(self) -> str:
        """Identifier of the model currently in use."""
        ...

    @property
    def configured(self) -> bool:
        """True when the provider has everything it needs to run."""
        ...

    async def complete(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Return the assistant's reply for ``messages``."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


def provider_names() -> tuple[str, ...]:
    """Registered provider identifiers."""
    return ("groq",)


def get_provider(settings: object, name: str = "groq") -> AIProvider:
    """Build the configured provider.

    Imported lazily so that ``import nova`` never pulls in a vendor SDK.
    """
    if name == "groq":
        from .groq import GroqProvider

        return GroqProvider(
            api_key=getattr(settings, "groq_api_key", None),
            model=getattr(settings, "groq_model", None) or "llama-3.3-70b-versatile",
            timeout=float(getattr(settings, "command_timeout", 30)),
        )
    raise AIProviderError(
        f"Unknown provider {name!r}. Available: {', '.join(provider_names())}"
    )
