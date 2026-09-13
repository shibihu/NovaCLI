"""Optional real Groq integration test for openai/gpt-oss-20b native tool calling.

Runs ONLY when GROQ_API_KEY is present in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from nova.ai.groq import GroqProvider
from nova.core.agent import AgentController, NovaAgent
from nova.config import load_settings


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set in environment — skipping real Groq integration test.",
)
async def test_real_groq_gpt_oss_native_tool_calling(tmp_path: Path) -> None:
    api_key = os.environ["GROQ_API_KEY"]
    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": api_key, "GROQ_MODEL": "openai/gpt-oss-20b"},
    )
    provider = GroqProvider(api_key=api_key, model="openai/gpt-oss-20b")
    agent = NovaAgent(provider, settings)

    controller = AgentController(auto_approve=True)
    result = await agent.run("List the files in the current project directory.", controller=controller)

    assert result.ok is True
    assert len(result.steps) >= 1
    # Verify at least one step was a list_files action
    actions = [s.action for s in result.steps if s.action]
    assert "list_files" in actions
    assert len(result.answer) > 0
