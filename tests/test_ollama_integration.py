"""Integration tests for Ollama with real local Ollama server if available.

Skipped cleanly if Ollama server is unreachable.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import httpx

from nova.ai.ollama import OllamaProvider
from nova.config import load_settings
from nova.core.agent import NovaAgent
from nova.core.models import EventType


def is_ollama_reachable(base_url: str) -> bool:
    try:
        url = f"{base_url.rstrip('/')}/api/version"
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def ollama_env():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
    if not is_ollama_reachable(base_url):
        pytest.skip(f"Ollama server is not reachable at {base_url}")
    return {"base_url": base_url, "model": model}


@pytest.mark.asyncio
async def test_ollama_real_native_tool_lifecycle(ollama_env, tmp_path: Path):
    (tmp_path / "sample.txt").write_text("sample content", encoding="utf-8")
    provider = OllamaProvider(base_url=ollama_env["base_url"], model=ollama_env["model"])
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_PROVIDER": "ollama", "OLLAMA_MODEL": ollama_env["model"]}
    )
    agent = NovaAgent(provider=provider, settings=settings)

    events = []
    async for event in agent.stream("use list_files to inspect the directory"):
        events.append(event)

    tool_call_events = [e for e in events if e.type == EventType.TOOL_CALL]
    tool_result_events = [e for e in events if e.type == EventType.TOOL_RESULT]

    assert len(tool_call_events) >= 1, "Agent did not emit any TOOL_CALL event"
    assert len(tool_result_events) >= 1, "Agent did not emit any TOOL_RESULT event"

    tool_name = tool_call_events[0].data.get("tool")
    assert tool_name in {"list_files", "run_command", "search"}, f"Unexpected tool name: {tool_name}"

    assert tool_result_events[0].data.get("ok") is True
    assert "sample.txt" in str(tool_result_events[0].data.get("output"))


@pytest.mark.asyncio
async def test_ollama_real_read_and_write_file(ollama_env, tmp_path: Path):
    provider = OllamaProvider(base_url=ollama_env["base_url"], model=ollama_env["model"])
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_PROVIDER": "ollama", "OLLAMA_MODEL": ollama_env["model"]}
    )
    agent = NovaAgent(provider=provider, settings=settings)

    events = []
    async for event in agent.stream("Create a file named hello.txt containing 'Hello World'"):
        events.append(event)

    final_events = [e for e in events if e.type == EventType.FINAL]
    assert len(final_events) == 1
    assert (tmp_path / "hello.txt").exists()
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8").strip() == "Hello World"
