"""Groq provider — the v1.0 model backend.

Uses the official ``groq`` Python SDK's ``AsyncGroq`` client. All failure modes
are converted into :class:`AIProviderError` subclasses with messages that are
safe to show a user: no API key, no headers, no raw request/response dumps.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nova.config import API_KEY_HINT

from . import AIProviderError, MissingAPIKeyError

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2

#: Substrings that indicate an authentication problem, across SDK versions.
_AUTH_MARKERS = ("401", "unauthorized", "invalid api key", "authentication")


class GroqProvider:
    """Async chat-completion client for Groq.

    Args:
        api_key: Groq key. ``None``/empty disables the provider but does not
            raise — the error surfaces with setup instructions on first use.
        model: Model id; defaults to ``llama-3.3-70b-versatile``.
        timeout: Per-request timeout in seconds.
        client: Injected client, used by tests. When omitted, ``AsyncGroq`` is
            imported and constructed lazily on first call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        client: Any | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._model = model or DEFAULT_MODEL
        self._timeout = max(1.0, float(timeout))
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client
        self._owns_client = client is None

    # -- Introspection ---------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    @property
    def timeout(self) -> float:
        return self._timeout

    def __repr__(self) -> str:
        """Never reveals the key."""
        return (
            f"GroqProvider(model={self._model!r}, "
            f"configured={self.configured}, timeout={self._timeout!r})"
        )

    # -- Client ----------------------------------------------------------

    def _get_client(self) -> Any:
        """Return the SDK client, creating it on first use."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise MissingAPIKeyError(API_KEY_HINT)
        try:
            from groq import AsyncGroq  # imported lazily: optional at import time
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise AIProviderError(
                "The 'groq' package is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from exc
        try:
            self._client = AsyncGroq(api_key=self._api_key, timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 - SDK raises many types
            raise AIProviderError(self._sanitize(f"Could not create Groq client: {exc}")) from exc
        return self._client

    # -- API -------------------------------------------------------------

    async def complete(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Return the assistant reply for ``messages``.

        Raises:
            MissingAPIKeyError: no key configured.
            AIProviderError: transport failure, timeout, API error, or a
                response that contains no usable text.
        """
        if not self._api_key and self._client is None:
            raise MissingAPIKeyError(API_KEY_HINT)
        if not messages:
            raise AIProviderError("No messages were supplied to the model.")

        clean = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        if not any(m["role"] == "system" for m in clean):
            clean.insert(0, {"role": "system", "content": "You are Nova, a coding agent."})

        client = self._get_client()
        target_model = model or self._model

        try:
            response = await client.chat.completions.create(
                model=target_model,
                messages=clean,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except asyncio.TimeoutError as exc:
            raise AIProviderError(
                f"Groq request timed out after {self._timeout:.0f}s. "
                "Check your connection and try again."
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise every SDK error
            raise AIProviderError(self._describe_error(exc)) from exc

        return self._extract_text(response)

    def _describe_error(self, exc: Exception) -> str:
        """Turn an arbitrary SDK exception into a safe, actionable message."""
        name = type(exc).__name__
        detail = self._sanitize(str(exc) or name)

        lowered = detail.lower()
        if any(marker in lowered for marker in _AUTH_MARKERS):
            return f"Groq rejected the API key ({name}). " + API_KEY_HINT
        if "rate limit" in lowered or "429" in lowered:
            return f"Groq rate limit reached ({name}). Wait a moment and retry. Details: {detail}"
        if "timeout" in lowered or "timed out" in lowered:
            return f"Groq request timed out after {self._timeout:.0f}s."
        if "connection" in lowered or "network" in lowered:
            return f"Could not reach Groq ({name}): {detail}"
        # A bad model id is the most common configuration mistake, and it is
        # reported inconsistently across SDK versions, so match on the text.
        if "model" in lowered and any(
            phrase in lowered for phrase in ("not found", "does not exist", "invalid", "decommissioned", "unsupported")
        ):
            return (
                f"Groq rejected model {self._model!r} ({name}): {detail}. "
                "Set GROQ_MODEL to a supported model id."
            )
        if name in {"APIStatusError", "BadRequestError", "NotFoundError"}:
            return f"Groq returned an error ({name}): {detail}"
        return f"Groq request failed ({name}): {detail}"

    def _sanitize(self, text: str) -> str:
        """Null out anything that looks like a credential."""
        from nova.core.safety import redact_secrets

        return redact_secrets(text, self._api_key)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull text out of the SDK response, or explain why it is unusable."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise AIProviderError("Groq returned an empty response (no choices).")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise AIProviderError("Groq response contained no message.")

        content = getattr(message, "content", None)
        if isinstance(content, list):
            # Some SDK versions return content parts.
            content = "".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(getattr(part, "text", ""))
                for part in content
            )
        if content is None:
            content = ""

        text = str(content).strip()
        if not text:
            refusal = getattr(message, "refusal", None)
            if refusal:
                raise AIProviderError(f"The model declined to answer: {refusal}")
            raise AIProviderError("Groq returned an empty completion.")
        return text

    # -- Lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we created it."""
        client, self._client = self._client, None
        if client is None or not self._owns_client:
            return
        closer = getattr(client, "close", None) or getattr(client, "aclose", None)
        if closer is None:
            return
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001 - closing must never raise
            pass

    async def __aenter__(self) -> "GroqProvider":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
