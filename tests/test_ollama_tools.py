"""Regression and unit tests for Ollama tool calling and runtime execution."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import httpx

from nova.ai.ollama import OllamaProvider
from nova.ai.groq import GroqProvider
from nova.ai.gemini import GeminiProvider
from nova.ai.openrouter import OpenRouterProvider
from nova.config import load_settings
from nova.core.agent import AgentController, NovaAgent
from nova.workspace.files import Workspace
from nova.core.runner import CommandRunner
from nova.core.safety import SafetyPolicy


@pytest.mark.asyncio
async def test_ollama_tool_call_is_detected():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "list_files",
                                    "arguments": {"path": "."},
                                },
                            }
                        ],
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    res = await provider.complete([{"role": "user", "content": "list files"}], tools=[])

    assert res.has_tool_calls is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "list_files"
    assert res.tool_calls[0].arguments == {"path": "."}


@pytest.mark.asyncio
async def test_ollama_tool_call_string_arguments():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_456",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path": "test.txt", "content": "hello"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    res = await provider.complete([{"role": "user", "content": "write test.txt"}], tools=[])

    assert res.has_tool_calls is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "write_file"
    assert res.tool_calls[0].arguments == {"path": "test.txt", "content": "hello"}


@pytest.mark.asyncio
async def test_ollama_tool_call_xml_tag_fallback():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = {
            "message": {
                "role": "assistant",
                "content": '<think>Thinking about tool</think>\n<tool_call>\n{"name": "write_file", "arguments": {"path": "script.py", "content": "print(1)"}}\n</tool_call>',
            }
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    res = await provider.complete([{"role": "user", "content": "create script.py"}], tools=[])

    assert res.has_tool_calls is True
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "write_file"
    assert res.tool_calls[0].arguments == {"path": "script.py", "content": "print(1)"}


@pytest.mark.asyncio
async def test_ollama_sends_tools_and_tool_choice():
    sent_payload = None

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal sent_payload
        sent_payload = json.loads(request.content)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Hello",
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    tools_schema = [{"type": "function", "function": {"name": "test_func", "parameters": {}}}]
    await provider.complete(
        [{"role": "user", "content": "hi"}], tools=tools_schema, tool_choice="auto"
    )

    assert sent_payload is not None
    assert "tools" in sent_payload
    assert sent_payload["tools"] == tools_schema
    assert sent_payload.get("tool_choice") == "auto"


@pytest.mark.asyncio
async def test_ollama_single_tool_creates_file(tmp_path: Path):
    call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_write",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": {
                                            "path": "app.py",
                                            "content": "# empty python script\n",
                                        },
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "I have created app.py for you.",
                        }
                    }
                ]
            }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)
    controller = AgentController(auto_approve=True)

    result = await agent.run("Create a Python empty script in this directory for me.", controller=controller)
    assert result.ok is True
    created_file = tmp_path / "app.py"
    assert created_file.exists()
    assert created_file.read_text(encoding="utf-8") == "# empty python script\n"


@pytest.mark.asyncio
async def test_ollama_tool_result_is_sent_back(tmp_path: Path):
    received_requests = []

    def mock_transport(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        received_requests.append(data)
        if len(received_requests) == 1:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_ls",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": {"path": "."},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            payload = {"choices": [{"message": {"content": "File list complete."}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    await agent.run("List the files in this directory.")
    assert len(received_requests) == 2
    second_req_msgs = received_requests[1]["messages"]
    tool_resp = next(m for m in second_req_msgs if m["role"] == "tool")
    assert tool_resp["tool_call_id"] == "call_ls"
    assert tool_resp["name"] == "list_files"


@pytest.mark.asyncio
async def test_multiple_sequential_tool_calls(tmp_path: Path):
    turn = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        if turn == 1:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": {"path": "test/sub.txt", "content": "subcontent"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        elif turn == 2:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c2",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": {"path": "test/sub.txt"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            payload = {"choices": [{"message": {"content": "Created test/sub.txt and verified its content."}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)
    controller = AgentController(auto_approve=True)

    result = await agent.run("Create a folder test and write sub.txt then read it.", controller=controller)
    assert result.ok is True
    assert len(result.steps) == 3
    assert result.steps[0].action == "write_file"
    assert result.steps[1].action == "read_file"
    assert (tmp_path / "test" / "sub.txt").read_text(encoding="utf-8") == "subcontent"


@pytest.mark.asyncio
async def test_normal_conversational_request(tmp_path: Path):
    def mock_transport(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Python is a high-level interpreted programming language.",
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("Explain what Python is.")
    assert result.ok is True
    assert "high-level" in result.answer
    assert len(result.steps) == 1
    assert result.steps[0].action is None


@pytest.mark.asyncio
async def test_operational_request_invokes_real_tool(tmp_path: Path):
    (tmp_path / "sample.py").write_text("print('sample')", encoding="utf-8")
    turn = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        if turn == 1:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c_list",
                                    "function": {
                                        "name": "list_files",
                                        "arguments": {"path": "."},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            data = json.loads(request.content)
            tool_msg = data["messages"][-1]
            assert "sample.py" in tool_msg["content"]
            payload = {"choices": [{"message": {"content": "Found sample.py in the project."}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("List the files in this project.")
    assert result.ok is True
    assert result.steps[0].action == "list_files"


@pytest.mark.asyncio
async def test_tool_failure_is_returned_to_model(tmp_path: Path):
    turn = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        if turn == 1:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": {"path": "non_existent.txt"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            data = json.loads(request.content)
            tool_msg = data["messages"][-1]
            assert "FileNotFoundError" in tool_msg["content"] or "failed" in tool_msg["content"] or "Error" in tool_msg["content"]
            payload = {"choices": [{"message": {"content": "File non_existent.txt does not exist."}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("Read non_existent.txt")
    assert result.ok is True
    assert "does not exist" in result.answer


@pytest.mark.asyncio
async def test_safety_block_is_preserved(tmp_path: Path):
    turn = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        if turn == 1:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "run_command",
                                        "arguments": {"command": "rm -rf /"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            data = json.loads(request.content)
            tool_msg = data["messages"][-1]
            assert "REFUSED" in tool_msg["content"]
            payload = {"choices": [{"message": {"content": "Command refused for safety reasons."}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("Delete all system files")
    assert result.ok is True
    assert result.steps[0].result.blocked is True


@pytest.mark.asyncio
async def test_provider_non_regression():
    groq_provider = GroqProvider("gsk_fake_key_12345678901234567890")
    gemini_provider = GeminiProvider("fake_gemini_key")
    openrouter_provider = OpenRouterProvider("fake_openrouter_key")

    assert groq_provider.model_name == "openai/gpt-oss-20b"
    assert gemini_provider.model_name == "gemini-2.5-flash"
    assert openrouter_provider.model_name == "openai/gpt-oss-20b"
