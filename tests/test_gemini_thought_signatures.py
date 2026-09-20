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


# ---------------------------------------------------------------------------
# Persistence round trip: response -> persist -> restart -> next request
# ---------------------------------------------------------------------------


def test_gemini_thought_signature_survives_session_persistence(
    monkeypatch, settings
) -> None:
    """``extra_content.google.thought_signature`` must survive a restart.

    A restored Gemini conversation that lost its signature would be rejected on
    the next tool-calling turn, so this is asserted end to end: the exact
    structure received from Gemini is written to disk, reloaded by a brand new
    application instance, and replayed in the next request's payload.
    """
    pytest.importorskip("fastapi", reason="fastapi is required for the web tests")
    from fastapi.testclient import TestClient

    from nova.web.app import create_app

    sent_payloads: list[dict] = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read().decode("utf-8"))
        sent_payloads.append(payload)
        if len(sent_payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_gemini_persist_1",
                                        "type": "function",
                                        "extra_content": {
                                            "google": {
                                                "thought_signature": "sig_persisted_abc"
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
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Done."}}]},
        )

    def make_provider() -> GeminiProvider:
        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        return GeminiProvider(api_key="gemini_test_key", client=client)

    monkeypatch.setattr("nova.web.routes.get_provider", lambda _settings: make_provider())

    def run(client: TestClient, task: str, session_id: str | None = None) -> str:
        body: dict[str, object] = {"task": task}
        if session_id:
            body["session_id"] = session_id
        created = client.post("/api/agent", json=body)
        assert created.status_code == 201, created.text
        sid = created.json()["session_id"]
        with client.stream("GET", f"/api/agent/stream?session_id={sid}") as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:") :].strip())
                if event["type"] in ("final", "error", "cancelled"):
                    assert event["type"] == "final", event
        return sid

    with TestClient(create_app(settings)) as client:
        sid = run(client, "summarize the project")
        assert len(sent_payloads) == 2
        stored = client.get(f"/api/agent/sessions/{sid}").json()["session"]["messages"]

    signature = (
        stored[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
    )
    assert signature == "sig_persisted_abc"
    assert stored[2]["tool_call_id"] == "call_gemini_persist_1"

    # The raw manifest on disk keeps the metadata verbatim.
    manifest = (
        settings.project_root / ".nova" / "sessions" / sid / "manifest.json"
    ).read_text(encoding="utf-8")
    assert "sig_persisted_abc" in manifest
    assert "thought_signature" in manifest

    # A fresh application instance (restart) continues the conversation.
    with TestClient(create_app(settings)) as restarted:
        run(restarted, "anything else?", session_id=sid)

        assert len(sent_payloads) == 3
        continuation_messages = sent_payloads[2]["messages"]
        assistant = next(m for m in continuation_messages if m["role"] == "assistant")
        tool_call = assistant["tool_calls"][0]
        assert tool_call["id"] == "call_gemini_persist_1"
        assert tool_call["extra_content"]["google"]["thought_signature"] == (
            "sig_persisted_abc"
        )
        tool_message = next(m for m in continuation_messages if m["role"] == "tool")
        assert tool_message["tool_call_id"] == "call_gemini_persist_1"

        # The prior turn is replayed once, not duplicated.
        assert sum(1 for m in continuation_messages if m["role"] == "user") == 2
