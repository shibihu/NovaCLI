"""Gemini provider — Google Gemini API integration.

Uses lightweight HTTP client via Gemini API / OpenAI compatibility endpoint.
All failure modes are converted into :class:`AIProviderError` subclasses with
messages safe to show to the user.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from nova.core.models import AIResponse, ToolCall
from . import AIProviderError, MissingAPIKeyError

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT = 60.0

MISSING_KEY_HINT = """No Gemini API key configured.

Set GEMINI_API_KEY in environment or credentials file (~/.nova/credentials.json).
"""


class GeminiProvider:
    """Async client for Google Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._model = model or DEFAULT_MODEL
        self._timeout = max(1.0, float(timeout))
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
            f"GeminiProvider(model={self._model!r}, "
            f"configured={self.configured}, timeout={self._timeout!r})"
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
    ) -> AIResponse:
        """Return the assistant response from Gemini."""
        if not self._api_key and self._client is None:
            raise MissingAPIKeyError(MISSING_KEY_HINT)
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

        target_model = model or self._model
        client = self._get_client()

        headers = {
            "Authorization": f"Bearer {self._api_key or ''}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": clean_messages,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        # Use Gemini OpenAI-compatible completions endpoint
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 401 or resp.status_code == 403:
                raise MissingAPIKeyError(f"Gemini rejected the API key (HTTP {resp.status_code}). {MISSING_KEY_HINT}")
            if resp.status_code != 200:
                error_body = self._sanitize(resp.text[:300])
                raise AIProviderError(f"Gemini returned HTTP {resp.status_code}: {error_body}")

            data = resp.json()
            return self._extract_response(data)
        except MissingAPIKeyError:
            raise
        except httpx.ConnectError as exc:
            raise AIProviderError(f"Could not reach Gemini ({type(exc).__name__}): {self._sanitize(str(exc))}") from exc
        except httpx.TimeoutException as exc:
            raise AIProviderError(f"Gemini request timed out after {self._timeout:.0f}s.") from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise AIProviderError(f"Gemini request failed: {self._sanitize(str(exc))}") from exc

    def _sanitize(self, text: str) -> str:
        from nova.core.safety import redact_secrets
        return redact_secrets(text, self._api_key)

    def _extract_response(self, data: dict[str, Any]) -> AIResponse:
        choices = data.get("choices") or []
        if not choices:
            raise AIProviderError("Gemini returned an empty response (no choices).")

        message = choices[0].get("message") or {}
        raw_tool_calls = message.get("tool_calls")
        parsed_tool_calls: list[ToolCall] = []

        if raw_tool_calls:
            for idx, tc in enumerate(raw_tool_calls):
                call_id = tc.get("id") or f"call_gemini_{idx}"
                func = tc.get("function") or {}
                name = func.get("name") or ""
                raw_args = func.get("arguments") or {}

                if isinstance(raw_args, dict):
                    raw_args_str = json.dumps(raw_args)
                    parsed_args = raw_args
                elif isinstance(raw_args, str):
                    raw_args_str = raw_args
                    try:
                        parsed_args = json.loads(raw_args)
                        if not isinstance(parsed_args, dict):
                            parsed_args = None
                    except Exception:
                        parsed_args = None
                else:
                    raw_args_str = "{}"
                    parsed_args = {}

                # Extract provider-specific metadata (e.g. Gemini thought_signature)
                provider_data: dict[str, Any] = {}
                for key in ("thought_signature", "extra_content", "thinking"):
                    if key in tc and tc[key] is not None:
                        provider_data[key] = tc[key]
                    elif isinstance(func, dict) and key in func and func[key] is not None:
                        provider_data[key] = func[key]

                if name:
                    parsed_tool_calls.append(
                        ToolCall(
                            id=str(call_id),
                            name=str(name),
                            arguments=parsed_args,
                            raw_arguments=raw_args_str,
                            provider_data=provider_data,
                        )
                    )

        content = message.get("content")
        text = str(content).strip() if content is not None else None

        if not text and not parsed_tool_calls:
            raise AIProviderError("Gemini returned an empty completion.")

        return AIResponse(text=text, tool_calls=parsed_tool_calls)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> GeminiProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
