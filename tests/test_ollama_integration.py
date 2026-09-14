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


def is_ollama_reachable(base_url: str) -> bool:
    try:
        url = f"{base_url.rstrip('/')}/api/version"
        resp = httpx.get(url, timeout=2.0)
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
async def test_ollama_real_list_directory(ollama_env, tmp_path: Path):
    (tmp_path / "sample.txt").write_text("sample content", encoding="utf-8")
    provider = OllamaProvider(base_url=ollama_env["base_url"], model=ollama_env["model"])
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama", "OLLAMA_MODEL": ollama_env["model"]})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("list files in current directory")
    assert result.ok is True
    assert len(result.steps) >= 1


@pytest.mark.asyncio
async def test_ollama_real_read_and_write_file(ollama_env, tmp_path: Path):
    provider = OllamaProvider(base_url=ollama_env["base_url"], model=ollama_env["model"])
    settings = load_settings(project_root=tmp_path, env={"NOVA_PROVIDER": "ollama", "OLLAMA_MODEL": ollama_env["model"]})
    agent = NovaAgent(provider=provider, settings=settings)

    result = await agent.run("Create a file named hello.txt containing 'Hello World'")
    assert result.ok is True
    assert (tmp_path / "hello.txt").exists()
