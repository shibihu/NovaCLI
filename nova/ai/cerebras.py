"""Cerebras provider — Cerebras API integration.

Uses Cerebras OpenAI-compatible API endpoint. All failure modes are converted
into :class:`AIProviderError` subclasses with messages safe to show to the user.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from nova.core.models import AIResponse, ToolCall
from . import AIProviderError, MissingAPIKeyError, retry_after_from_headers

DEFAULT_MODEL = "llama3.1-8b"
DEFAULT_TIMEOUT = 60.0

MISSING_KEY_HINT = """No Cerebras API key configured.

Set CEREBRAS_API_KEY in environment or credentials file (~/.nova/credentials.json).
Get a key at https://cloud.cerebras.ai
"""


class CerebrasProvider:
    """Async client for Cerebras API."""

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
    def provider_id(self) -> str:
        return "cerebras"

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
            f"CerebrasProvider(model={self._model!r}, "
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
        """Return the assistant response from Cerebras."""
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

        url = "https://api.cerebras.ai/v1/chat/completions"
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code in (401, 403):
                raise MissingAPIKeyError(
                    f"Cerebras rejected the API key (HTTP {resp.status_code}). {MISSING_KEY_HINT}",
                    status_code=resp.status_code,
                    provider="cerebras",
                    model=target_model,
                    is_permanent=True,
                )
            if resp.status_code != 200:
                error_body = self._sanitize(resp.text[:300])
                # ``Retry-After: 60`` or an HTTP-date.
                raise AIProviderError(
                    f"Cerebras returned HTTP {resp.status_code}: {error_body}",
                    status_code=resp.status_code,
                    retry_after=retry_after_from_headers(resp.headers),
                    provider="cerebras",
                    model=target_model,
                    error_code=self._error_code(resp),
                    is_permanent=resp.status_code in (400, 404),
                )

            data = resp.json()
            return self._extract_response(data)
        except MissingAPIKeyError:
            raise
        except httpx.ConnectError as exc:
            raise AIProviderError(
                f"Could not reach Cerebras ({type(exc).__name__}): {self._sanitize(str(exc))}",
                provider="cerebras",
                model=target_model,
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIProviderError(
                f"Cerebras request timed out after {self._timeout:.0f}s.",
                provider="cerebras",
                model=target_model,
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise AIProviderError(
                f"Cerebras request failed: {self._sanitize(str(exc))}",
                provider="cerebras",
                model=target_model,
            ) from exc

    def _error_code(self, resp: httpx.Response) -> str | None:
        """Cerebras' ``error.code`` / ``error.type`` when present."""
        try:
            payload = resp.json()
        except (ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            if code is not None:
                return str(code)
        return None

    def _sanitize(self, text: str) -> str:
        from nova.core.safety import redact_secrets
        return redact_secrets(text, self._api_key)

    def _extract_response(self, data: dict[str, Any]) -> AIResponse:
        choices = data.get("choices") or []
        if not choices:
            raise AIProviderError("Cerebras returned an empty response (no choices).")

        message = choices[0].get("message") or {}
        raw_tool_calls = message.get("tool_calls")
        parsed_tool_calls: list[ToolCall] = []

        if raw_tool_calls:
            for idx, tc in enumerate(raw_tool_calls):
                call_id = tc.get("id") or f"call_cerebras_{idx}"
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

                if name:
                    parsed_tool_calls.append(
                        ToolCall(
                            id=str(call_id),
                            name=str(name),
                            arguments=parsed_args,
                            raw_arguments=raw_args_str,
                        )
                    )

        content = message.get("content")
        text = str(content).strip() if content is not None else None

        if not text and not parsed_tool_calls:
            raise AIProviderError("Cerebras returned an empty completion.")

        return AIResponse(text=text, tool_calls=parsed_tool_calls)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> CerebrasProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
