"""Tests for Gemini thought signature preservation in tool calling loop."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import httpx

from nova.ai.gemini import GeminiProvider
from nova.core.agent import NovaAgent
from nova.core.models import ToolCall
from nova.workspace.files import Workspace


@pytest.mark.asyncio
async def test_gemini_single_tool_call_preserves_thought_signature():
    sent_payloads: list[dict] = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        sent_payloads.append(payload)

        # First request: model returns a tool call with extra_content.google.thought_signature
        if len(sent_payloads) == 1:
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "function-call-1",
                                    "type": "function",
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": "sig_gemini_3_secret_12345"
                                        }
                                    },
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

    # Verify thought_signature is in the exact documented Google location
    assert "extra_content" in tc
    assert "google" in tc["extra_content"]
    assert tc["extra_content"]["google"]["thought_signature"] == "sig_gemini_3_secret_12345"


@pytest.mark.asyncio
async def test_gemini_sequential_tool_calls_preserve_signatures():
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
                                    "id": "call_turn_1",
                                    "type": "function",
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": "sig_turn_1_abc"
                                        }
                                    },
                                    "function": {"name": "project_summary", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }
        elif len(sent_payloads) == 2:
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_turn_2",
                                    "type": "function",
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": "sig_turn_2_xyz"
                                        }
                                    },
                                    "function": {"name": "list_files", "arguments": "{\"path\": \".\"}"},
                                }
                            ],
                        }
                    }
                ]
            }
        else:
            resp_data = {
                "choices": [
                    {"message": {"role": "assistant", "content": "Sequential tools complete."}}
                ]
            }
        return httpx.Response(200, json=resp_data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = GeminiProvider(api_key="gemini_test_key", client=client)

    ws = Workspace(Path.cwd())
    agent = NovaAgent(provider, workspace=ws)

    result = await agent.run("Run sequential tools")
    assert result.ok is True
    assert result.answer == "Sequential tools complete."

    assert len(sent_payloads) == 3

    # Turn 2 request messages contain Turn 1 assistant message with sig_turn_1_abc
    assistant_1 = [m for m in sent_payloads[1]["messages"] if m["role"] == "assistant"][0]
    tc_1 = assistant_1["tool_calls"][0]
    assert tc_1["extra_content"]["google"]["thought_signature"] == "sig_turn_1_abc"

    # Turn 3 request messages contain Turn 2 assistant message with sig_turn_2_xyz
    assistant_2 = [m for m in sent_payloads[2]["messages"] if m["role"] == "assistant"][1]
    tc_2 = assistant_2["tool_calls"][0]
    assert tc_2["extra_content"]["google"]["thought_signature"] == "sig_turn_2_xyz"


@pytest.mark.asyncio
async def test_gemini_parallel_tool_calls_preserve_signatures_correctly():
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
                                    "id": "call_p1",
                                    "type": "function",
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": "sig_parallel_1"
                                        }
                                    },
                                    "function": {"name": "project_summary", "arguments": "{}"},
                                },
                                {
                                    "id": "call_p2",
                                    "type": "function",
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

    # Part 1 has the signature
    assert tcs[0]["extra_content"]["google"]["thought_signature"] == "sig_parallel_1"

    # Part 2 does NOT have a fake signature added
    assert "extra_content" not in tcs[1]


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


def test_non_gemini_tool_calls_unaffected():
    tc = ToolCall(id="call_groq_1", name="project_summary", arguments={}, raw_arguments="{}")
    assert tc.provider_data == {}
