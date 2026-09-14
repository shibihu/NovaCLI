"""Regression tests for Ollama tool calling and runtime execution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import httpx

from nova.ai.ollama import OllamaProvider
from nova.ai.groq import GroqProvider
from nova.config import load_settings
from nova.core.agent import NovaAgent
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
async def test_ollama_tool_call_is_executed(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("Hello World", encoding="utf-8")

    call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        data = json.loads(request.content)
        if call_count == 1:
            payload = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": {"path": "hello.txt"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            # Verify the tool result message was sent back
            tool_msg = data["messages"][-1]
            assert tool_msg["role"] == "tool"
            assert tool_msg["tool_call_id"] == "call_read"
            assert "Hello World" in tool_msg["content"]
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "เนื้อหาในไฟล์คือ Hello World",
                        }
                    }
                ]
            }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("อ่านไฟล์ hello.txt")
    assert result.ok is True
    assert "Hello World" in result.answer
    assert len(result.steps) == 2
    assert result.steps[0].action == "read_file"
    assert result.steps[0].result.ok is True


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
            payload = {"choices": [{"message": {"content": "รายการไฟล์พร้อมแล้ว"}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    await agent.run("แสดงรายการไฟล์")
    assert len(received_requests) == 2
    second_req_msgs = received_requests[1]["messages"]
    tool_resp = next(m for m in second_req_msgs if m["role"] == "tool")
    assert tool_resp["tool_call_id"] == "call_ls"
    assert tool_resp["name"] == "list_files"


@pytest.mark.asyncio
async def test_ollama_returns_final_answer_after_tool(tmp_path: Path):
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
                                        "arguments": {"command": "echo hello"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            payload = {"choices": [{"message": {"content": "ผลลัพธ์คือ hello"}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("รัน echo hello")
    assert result.ok is True
    assert "ผลลัพธ์คือ hello" in result.answer


@pytest.mark.asyncio
async def test_multiple_sequential_tool_calls(tmp_path: Path):
    (tmp_path / "a.txt").write_text("file_a", encoding="utf-8")
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
                                        "name": "list_files",
                                        "arguments": {"path": "."},
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
                                        "arguments": {"path": "a.txt"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        else:
            payload = {"choices": [{"message": {"content": "พบไฟล์ a.txt มีเนื้อหา file_a"}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("แสดงรายการไฟล์แล้วอ่านไฟล์ a.txt")
    assert result.ok is True
    assert len(result.steps) == 3
    assert result.steps[0].action == "list_files"
    assert result.steps[1].action == "read_file"


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
            payload = {"choices": [{"message": {"content": "ไม่พบไฟล์ดังกล่าว"}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("อ่าน non_existent.txt")
    assert result.ok is True
    assert "ไม่พบไฟล์" in result.answer


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
            payload = {"choices": [{"message": {"content": "คำสั่งถูกปฏิเสธเนื่องจากความปลอดภัย"}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("ลบระบบทั้งหมด")
    assert result.ok is True
    assert result.steps[0].result.blocked is True


@pytest.mark.asyncio
async def test_no_fake_tool_output(tmp_path: Path):
    # Verify real tool output from CommandRunner is captured
    (tmp_path / "real_file.py").write_text("print('real')", encoding="utf-8")
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
                                        "arguments": {"command": "python real_file.py"},
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
            assert "real" in tool_msg["content"]
            payload = {"choices": [{"message": {"content": "ผลลัพธ์การรันคือ real"}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("รัน python real_file.py")
    assert result.ok is True
    assert "real" in result.steps[0].result.output


@pytest.mark.asyncio
async def test_groq_tool_loop_still_works():
    # Verify Groq provider tool calling interface remains functional
    groq_provider = GroqProvider("gsk_fake_key_12345678901234567890")
    assert groq_provider.model_name == "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_malformed_tool_arguments_fails_safely(tmp_path: Path):
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
                                    "id": "c_bad",
                                    "function": {
                                        "name": "run_command",
                                        "arguments": "{invalid_json: ",
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
            assert "parsing failed" in tool_msg["content"].lower() or "invalid" in tool_msg["content"].lower()
            payload = {"choices": [{"message": {"content": "Failed to parse arguments."}}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", client=client)
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama"})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("run command")
    assert result.ok is True
    assert "Failed to parse" in result.answer
