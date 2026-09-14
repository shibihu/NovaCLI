"""Ollama provider — local AI inference.

Communicates with Ollama server via HTTP. All failure modes are converted
into :class:`AIProviderError` subclasses with clear, secret-safe messages.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from nova.core.models import AIResponse, ToolCall
from . import AIProviderError

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 600


class OllamaProvider:
    """Async client for Ollama local inference."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        raw_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"http://{raw_url}"
        self._base_url = raw_url
        self._model = model or DEFAULT_MODEL
        self._timeout = max(1.0, float(timeout))
        self._client = client
        self._owns_client = client is None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        # Ollama does not require an API key
        return True

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    def __repr__(self) -> str:
        return f"OllamaProvider(model={self._model!r}, base_url={self._base_url!r})"

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
        """Return the assistant response from Ollama."""
        if not messages:
            raise AIProviderError("No messages were supplied to the model.")

        clean_messages: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": str(m.get("role", "user"))}
            if "content" in m and m["content"] is not None:
                msg["content"] = str(m["content"])
            elif m.get("role") == "assistant" and "tool_calls" in m:
                msg["content"] = ""
            if "tool_calls" in m and m["tool_calls"] is not None:
                norm_tool_calls = []
                for tc in m["tool_calls"]:
                    tc_copy = dict(tc)
                    if "function" in tc_copy and isinstance(tc_copy["function"], dict):
                        func_copy = dict(tc_copy["function"])
                        raw_args = func_copy.get("arguments")
                        if isinstance(raw_args, str):
                            try:
                                func_copy["arguments"] = json.loads(raw_args)
                            except Exception:
                                pass
                        tc_copy["function"] = func_copy
                    norm_tool_calls.append(tc_copy)
                msg["tool_calls"] = norm_tool_calls
            if "tool_call_id" in m and m["tool_call_id"] is not None:
                msg["tool_call_id"] = str(m["tool_call_id"])
            if "name" in m and m["name"] is not None:
                msg["name"] = str(m["name"])
                msg["tool_name"] = str(m["name"])
            clean_messages.append(msg)

        if not any(m["role"] == "system" for m in clean_messages):
            clean_messages.insert(0, {
                "role": "system",
                "content": "You are Nova, a coding agent with access to real workspace tools. Always use appropriate tools to inspect files or run commands before answering, and never fabricate tool output."
            })

        target_model = model or self._model
        client = self._get_client()

        # Try Ollama OpenAI-compatible chat endpoint first, fallback to native /api/chat
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": clean_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        url = f"{self._base_url}/v1/chat/completions"
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 404:
                # Fallback to native /api/chat
                url = f"{self._base_url}/api/chat"
                native_payload: dict[str, Any] = {
                    "model": target_model,
                    "messages": clean_messages,
                    "stream": False,
                }
                if tools:
                    native_payload["tools"] = tools
                resp = await client.post(url, json=native_payload)

            if resp.status_code != 200:
                error_body = resp.text[:300]
                raise AIProviderError(
                    f"Ollama returned HTTP status {resp.status_code}: {error_body}"
                )

            data = resp.json()
            return self._extract_response(data)
        except httpx.ConnectError as exc:
            raise AIProviderError(
                f"Cannot connect to Ollama at {self._base_url}. Check if Ollama service is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise AIProviderError(
                f"Ollama request timed out after {self._timeout:.0f}s."
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise AIProviderError(f"Ollama request failed: {exc}") from exc

    def _extract_response(self, data: dict[str, Any]) -> AIResponse:
        # OpenAI style response structure
        if "choices" in data:
            choices = data.get("choices") or []
            if not choices:
                raise AIProviderError("Ollama returned an empty response (no choices).")
            msg = choices[0].get("message") or {}
            raw_tool_calls = msg.get("tool_calls")
            content = msg.get("content")
        # Native Ollama /api/chat response structure
        elif "message" in data:
            msg = data.get("message") or {}
            raw_tool_calls = msg.get("tool_calls")
            content = msg.get("content")
        else:
            raise AIProviderError("Unrecognized Ollama response structure.")

        parsed_tool_calls: list[ToolCall] = []
        if raw_tool_calls:
            for idx, tc in enumerate(raw_tool_calls):
                call_id = tc.get("id") or f"call_ollama_{idx}"
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

        text = str(content).strip() if content is not None else None
        if not text and not parsed_tool_calls:
            raise AIProviderError("Ollama returned an empty completion.")

        return AIResponse(text=text, tool_calls=parsed_tool_calls)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OllamaProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
