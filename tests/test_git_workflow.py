"""Unit tests for GitService in nova.core.git."""

from __future__ import annotations

import subprocess
from pathlib import Path
import pytest

from nova.core.git import GitService
from nova.workspace.files import Workspace


@pytest.fixture
def git_repo(tmp_path: Path) -> Workspace:
    try:
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)
        (tmp_path / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)
    except Exception:
        pytest.skip("git CLI not available")
    return Workspace(tmp_path)


async def test_git_service_status_and_diff(git_repo: Workspace):
    git_svc = GitService(git_repo)

    assert git_svc.is_repo() is True
    branch = await git_svc.current_branch()
    assert branch in ("main", "master")

    # Make a change
    (git_repo.root / "README.md").write_text("# Test Repo\nUpdated line\n", encoding="utf-8")

    status = await git_svc.status()
    assert status["clean"] is False
    assert "README.md" in status["modified"]

    diff_res = await git_svc.diff("README.md")
    assert "+Updated line" in diff_res["diff"]
    assert diff_res["additions"] == 1


async def test_git_service_commit_preview(git_repo: Workspace):
    git_svc = GitService(git_repo)

    (git_repo.root / "new_file.txt").write_text("new file content", encoding="utf-8")
    preview = await git_svc.commit_preview()

    assert preview["has_git"] is True
    assert preview["total_unstaged"] >= 1
    assert "new_file.txt" in preview["unstaged_files"]
