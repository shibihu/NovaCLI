"""Groq provider — native tool calling support.

Uses the official ``groq`` Python SDK's ``AsyncGroq`` client. All failure modes
are converted into :class:`AIProviderError` subclasses with messages that are
safe to show a user.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from nova.config import API_KEY_HINT
from nova.core.models import AIResponse, ToolCall

from . import AIProviderError, MissingAPIKeyError

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2

_AUTH_MARKERS = ("401", "unauthorized", "invalid api key", "authentication")


class GroqProvider:
    """Async chat-completion client for Groq supporting native tool calls."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._model = model or DEFAULT_MODEL
        self._timeout = max(1.0, float(timeout))
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._client = client
        self._owns_client = client is None

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
        return (
            f"GroqProvider(model={self._model!r}, "
            f"configured={self.configured}, timeout={self._timeout!r})"
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise MissingAPIKeyError(API_KEY_HINT)
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise AIProviderError(
                "The 'groq' package is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from exc
        try:
            self._client = AsyncGroq(api_key=self._api_key, timeout=self._timeout)
        except Exception as exc:
            raise AIProviderError(self._sanitize(f"Could not create Groq client: {exc}")) from exc
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
    ) -> AIResponse:
        """Return the assistant response (text and/or tool calls)."""
        if not self._api_key and self._client is None:
            raise MissingAPIKeyError(API_KEY_HINT)
        if not messages:
            raise AIProviderError("No messages were supplied to the model.")

        clean_messages: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": str(m.get("role", "user"))}
            if "content" in m and m["content"] is not None:
                msg["content"] = str(m["content"])
            if "tool_calls" in m and m["tool_calls"] is not None:
                msg["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m and m["tool_call_id"] is not None:
                msg["tool_call_id"] = str(m["tool_call_id"])
            if "name" in m and m["name"] is not None:
                msg["name"] = str(m["name"])
            clean_messages.append(msg)

        if not any(m["role"] == "system" for m in clean_messages):
            clean_messages.insert(0, {"role": "system", "content": "You are Nova, a coding agent with access to real workspace tools. Always use appropriate tools to inspect files or run commands before answering, and never fabricate tool output."})

        client = self._get_client()
        target_model = model or self._model

        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": clean_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        if self._reasoning_effort and "gpt-oss" in target_model.lower():
            kwargs["reasoning_effort"] = self._reasoning_effort

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        try:
            response = await client.chat.completions.create(**kwargs)
        except asyncio.TimeoutError as exc:
            raise AIProviderError(
                f"Groq request timed out after {self._timeout:.0f}s. "
                "Check your connection and try again."
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise AIProviderError(self._describe_error(exc)) from exc

        return self._extract_response(response)

    def _describe_error(self, exc: Exception) -> str:
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
        from nova.core.safety import redact_secrets

        return redact_secrets(text, self._api_key)

    @staticmethod
    def _extract_response(response: Any) -> AIResponse:
        choices = getattr(response, "choices", None)
        if not choices:
            raise AIProviderError("Groq returned an empty response (no choices).")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise AIProviderError("Groq response contained no message.")

        raw_tool_calls = getattr(message, "tool_calls", None)
        parsed_tool_calls: list[ToolCall] = []

        if raw_tool_calls:
            for tc in raw_tool_calls:
                call_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "") or "call_unknown"
                func = getattr(tc, "function", None)
                if not func and isinstance(tc, dict):
                    func = tc.get("function")

                name = ""
                raw_args_str = ""
                parsed_args: dict[str, Any] | None = None

                if func:
                    name = getattr(func, "name", None) or (func.get("name") if isinstance(func, dict) else "") or ""
                    raw_args = getattr(func, "arguments", None) if not isinstance(func, dict) else func.get("arguments")

                    if isinstance(raw_args, str):
                        raw_args_str = raw_args
                        try:
                            decoded = json.loads(raw_args)
                            if isinstance(decoded, dict):
                                parsed_args = decoded
                            else:
                                parsed_args = None
                        except ValueError:
                            parsed_args = None
                    elif isinstance(raw_args, dict):
                        raw_args_str = json.dumps(raw_args)
                        parsed_args = raw_args
                    elif raw_args is None:
                        raw_args_str = "{}"
                        parsed_args = {}

                if name:
                    parsed_tool_calls.append(
                        ToolCall(
                            id=str(call_id),
                            name=str(name),
                            arguments=parsed_args,
                            raw_arguments=raw_args_str,
                        )
                    )

        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(getattr(part, "text", ""))
                for part in content
            )
        text = str(content).strip() if content is not None else None

        if not text and not parsed_tool_calls:
            refusal = getattr(message, "refusal", None)
            if refusal:
                raise AIProviderError(f"The model declined to answer: {refusal}")
            raise AIProviderError("Groq returned an empty completion.")

        return AIResponse(text=text, tool_calls=parsed_tool_calls)

    async def aclose(self) -> None:
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
        except Exception:
            pass

    async def __aenter__(self) -> "GroqProvider":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
