"""Tests for Agent task integration with Checkpoints."""

from __future__ import annotations

from pathlib import Path
import pytest

from nova.core.agent import NovaAgent
from nova.core.models import AIResponse
from nova.workspace.files import Workspace
from conftest import FakeProvider


async def test_agent_run_creates_checkpoint(tmp_path: Path):
    (tmp_path / "hello.py").write_text("v1", encoding="utf-8")
    provider = FakeProvider(replies=[AIResponse(text="Done!")])

    agent = NovaAgent(provider, workspace=Workspace(tmp_path))
    res = await agent.run("Check project")

    assert res.ok is True
    assert res.checkpoint_id is not None
    assert res.checkpoint_id.startswith("cp_")
