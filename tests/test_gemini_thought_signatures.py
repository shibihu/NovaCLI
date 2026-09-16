"""Tests for Gemini thought signature preservation in tool calling loop."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import httpx

from nova.ai import AIProviderError
from nova.ai.gemini import GeminiProvider
from nova.ai.groq import GroqProvider
from nova.core.agent import NovaAgent
from nova.core.models import AIResponse, ToolCall
from nova.workspace.files import Workspace


@pytest.mark.asyncio
async def test_gemini_single_tool_call_preserves_thought_signature():
    sent_payloads: list[dict] = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        sent_payloads.append(payload)

        # First request: model returns a tool call with thought_signature
        if len(sent_payloads) == 1:
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_gemini_1",
                                    "type": "function",
                                    "thought_signature": "sig_gemini_secret_12345",
                                    "function": {
                                        "name": "project_summary",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        else:
            # Second request (after tool execution): model returns final text
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Project summary complete.",
                        }
                    }
                ]
            }
        return httpx.Response(200, json=resp_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = GeminiProvider(api_key="gemini_test_key", client=client)

    ws = Workspace(Path.cwd())
    agent = NovaAgent(provider, workspace=ws)

    result = await agent.run("Summarize project")
    assert result.ok is True
    assert result.answer == "Project summary complete."

    # Verify second request payload sent to Gemini
    assert len(sent_payloads) == 2
    second_request_messages = sent_payloads[1]["messages"]

    # Find the assistant message in second request
    assistant_msg = next(m for m in second_request_messages if m["role"] == "assistant")
    assert "tool_calls" in assistant_msg
    tc = assistant_msg["tool_calls"][0]

    # Verify thought_signature is present in the serialized request payload
    assert tc.get("thought_signature") == "sig_gemini_secret_12345" or tc.get("function", {}).get("thought_signature") == "sig_gemini_secret_12345"


@pytest.mark.asyncio
async def test_gemini_parallel_tool_calls_preserve_signatures():
    sent_payloads: list[dict] = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        sent_payloads.append(payload)

        if len(sent_payloads) == 1:
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_gemini_p1",
                                    "type": "function",
                                    "thought_signature": "sig_p1",
                                    "function": {"name": "project_summary", "arguments": "{}"},
                                },
                                {
                                    "id": "call_gemini_p2",
                                    "type": "function",
                                    "thought_signature": "sig_p2",
                                    "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"},
                                },
                            ],
                        }
                    }
                ]
            }
        else:
            resp_data = {
                "choices": [
                    {"message": {"role": "assistant", "content": "Parallel tools executed."}}
                ]
            }
        return httpx.Response(200, json=resp_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = GeminiProvider(api_key="gemini_test_key", client=client)

    ws = Workspace(Path.cwd())
    agent = NovaAgent(provider, workspace=ws)

    result = await agent.run("Run parallel summary and listing")
    assert result.ok is True

    # Verify second request payload
    assert len(sent_payloads) == 2
    assistant_msg = next(m for m in sent_payloads[1]["messages"] if m["role"] == "assistant")
    tcs = assistant_msg["tool_calls"]
    assert len(tcs) == 2
    assert tcs[0].get("thought_signature") == "sig_p1" or tcs[0].get("function", {}).get("thought_signature") == "sig_p1"
    assert tcs[1].get("thought_signature") == "sig_p2" or tcs[1].get("function", {}).get("thought_signature") == "sig_p2"


@pytest.mark.asyncio
async def test_gemini_normal_text_response_works():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello, I am Nova."}}
            ]
        }
        return httpx.Response(200, json=resp_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = GeminiProvider(api_key="gemini_test_key", client=client)

    res = await provider.complete([{"role": "user", "content": "hi"}])
    assert res.text == "Hello, I am Nova."
    assert res.has_tool_calls is False


@pytest.mark.asyncio
async def test_non_gemini_providers_unaffected():
    # Groq provider with standard tool calls without thought_signature
    def mock_transport(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_groq_1",
                                "type": "function",
                                "function": {"name": "project_summary", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        return httpx.Response(200, json=resp_data)

    # Simple model object
    tc = ToolCall(id="call_groq_1", name="project_summary", arguments={}, raw_arguments="{}")
    assert tc.provider_data == {}
