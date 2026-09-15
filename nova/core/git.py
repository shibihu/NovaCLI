"""Git Workflow Service for NovaCLI.

Encapsulates Git operations using structured argument arrays (CommandRunner.run_args)
without shell command string interpolation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nova.core.runner import CommandRunner
from nova.core.safety import SafetyError, redact_secrets
from nova.workspace.files import Workspace


class GitService:
    """Provides structured, safe Git operations for NovaCLI."""

    def __init__(self, workspace: Workspace, runner: CommandRunner | None = None) -> None:
        self.workspace = workspace
        self.root = workspace.root
        self.safety = workspace.safety
        self.runner = runner or CommandRunner(
            self.root, timeout=15, safety=self.safety
        )

    def is_repo(self) -> bool:
        """True if the workspace is inside a Git repository."""
        return (self.root / ".git").exists()

    async def current_branch(self) -> str:
        """Return active branch name or empty string if detached/unavailable."""
        if not self.is_repo():
            return ""
        res = await self.runner.run_args(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], check_safety=True
        )
        return res.stdout.strip() if res.ok else ""

    async def status(self) -> dict[str, Any]:
        """Return structured Git status summary."""
        if not self.is_repo():
            return {
                "has_git": False,
                "branch": "",
                "status": [],
                "clean": True,
                "modified": [],
                "staged": [],
                "untracked": [],
                "deleted": [],
            }

        branch = await self.current_branch()
        res = await self.runner.run_args(
            ["git", "status", "--porcelain"], check_safety=True
        )
        if not res.ok:
            return {
                "has_git": True,
                "branch": branch,
                "status": [],
                "clean": True,
                "modified": [],
                "staged": [],
                "untracked": [],
                "deleted": [],
            }

        status_entries: list[dict[str, str]] = []
        modified, staged, untracked, deleted = [], [], [], []

        for line in res.stdout.strip().splitlines():
            if len(line) >= 3:
                xy = line[:2]
                p = line[2:].strip()
                status_entries.append({"state": xy, "path": p})

                index_status = xy[0]
                work_status = xy[1]

                if "M" in (index_status, work_status):
                    modified.append(p)
                if index_status in ("M", "A", "R"):
                    staged.append(p)
                if index_status == "D" or work_status == "D":
                    deleted.append(p)
                if index_status == "?" and work_status == "?":
                    untracked.append(p)

        return {
            "has_git": True,
            "branch": branch,
            "status": status_entries,
            "clean": len(status_entries) == 0,
            "modified": modified,
            "staged": staged,
            "untracked": untracked,
            "deleted": deleted,
        }

    async def diff(self, path: str | None = None) -> dict[str, Any]:
        """Return safe Git diff for a path or entire repository."""
        if not self.is_repo():
            return {"has_git": False, "path": path, "diff": "", "additions": 0, "deletions": 0}

        args = ["git", "diff"]
        if path:
            target = self.workspace.resolve(path)
            rel_path = self.workspace.relative(target)
            args = ["git", "diff", "--", rel_path]
        else:
            args = ["git", "diff", "--"]

        res = await self.runner.run_args(args, check_safety=True)
        raw_diff = res.stdout if res.ok else ""
        redacted = self.safety.redact(raw_diff)

        # Count additions and deletions
        additions = sum(
            1 for line in redacted.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1 for line in redacted.splitlines() if line.startswith("-") and not line.startswith("---")
        )

        return {
            "has_git": True,
            "path": path,
            "diff": redacted,
            "additions": additions,
            "deletions": deletions,
        }

    async def commit_preview(self) -> dict[str, Any]:
        """Return a preview of what changes would be included in a commit."""
        status_info = await self.status()
        diff_info = await self.diff()

        return {
            "has_git": status_info["has_git"],
            "branch": status_info["branch"],
            "staged_files": status_info["staged"],
            "unstaged_files": status_info["modified"] + status_info["untracked"],
            "total_staged": len(status_info["staged"]),
            "total_unstaged": len(status_info["modified"]) + len(status_info["untracked"]),
            "diff_summary": {
                "additions": diff_info["additions"],
                "deletions": diff_info["deletions"],
            },
        }


__all__ = ["GitService"]
