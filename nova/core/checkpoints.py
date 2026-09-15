"""Agent Checkpoint and Rollback Subsystem for NovaCLI.

Checkpoints capture recovery state before/during Agent task execution.
Key principles:
1. Never destroy pre-existing user uncommitted work.
2. Checkpoints are stored in .nova/checkpoints/ and ignored by Git.
3. Rollback only affects Agent-owned changes.
4. Credential files (.env, keys) are never snapshotted.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.core.models import to_jsonable, utc_now_iso
from nova.core.safety import (
    SENSITIVE_FILENAMES,
    SENSITIVE_PATH_PARTS,
    SENSITIVE_READ_ALLOWLIST,
    SENSITIVE_SUFFIXES,
    SafetyError,
)
from nova.workspace.files import Workspace

_VALID_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _file_hash(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def is_sensitive_path(path_str: str) -> bool:
    """True if path matches credential material or sensitive files."""
    name = Path(path_str).name
    lowered = path_str.replace("\\", "/").lower()

    if name in SENSITIVE_READ_ALLOWLIST:
        return False
    if name in SENSITIVE_FILENAMES or name.startswith(".env."):
        return True
    if Path(name).suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    for part in SENSITIVE_PATH_PARTS:
        if part in lowered:
            return True
    if ".nova/checkpoints" in lowered or ".git/" in lowered:
        return True
    return False


@dataclass
class CheckpointFile:
    path: str
    state: str = "clean"  # clean, created, modified, deleted
    before_hash: str | None = None
    after_hash: str | None = None
    existed_before: bool = True
    existed_after: bool = True
    is_user_owned: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class Checkpoint:
    id: str
    task_id: str
    created_at: str = field(default_factory=utc_now_iso)
    root: str = ""
    files: list[CheckpointFile] = field(default_factory=list)
    user_dirty_files: list[str] = field(default_factory=list)
    git_head: str | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class RollbackResult:
    ok: bool
    checkpoint_id: str
    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


class CheckpointManager:
    """Manages creation, inspection, and safe rollback of agent checkpoints."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.root
        self.checkpoints_dir = self.root / ".nova" / "checkpoints"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure .nova/checkpoints exists and is ignored by Git."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.root / ".nova" / ".gitignore"
        if not gitignore.exists():
            try:
                gitignore.parent.mkdir(parents=True, exist_ok=True)
                gitignore.write_text("*\n", encoding="utf-8")
            except OSError:
                pass

    def _validate_id(self, checkpoint_id: str) -> str:
        clean_id = (checkpoint_id or "").strip()
        if not clean_id or not _VALID_ID_RE.match(clean_id):
            raise ValueError(f"Invalid checkpoint ID: {checkpoint_id!r}")
        return clean_id

    def _get_user_dirty_files(self) -> list[str]:
        """Collect paths dirty before the Agent starts to protect user changes."""
        dirty: set[str] = set()
        git_dir = self.root / ".git"
        if git_dir.exists():
            try:
                import subprocess

                out = subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=self.root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                for line in out.splitlines():
                    if len(line) >= 3:
                        p = line[3:].strip()
                        if p and not is_sensitive_path(p):
                            dirty.add(p)
            except Exception:
                pass
        return sorted(dirty)

    def _get_git_head(self) -> str | None:
        git_dir = self.root / ".git"
        if git_dir.exists():
            try:
                import subprocess

                return subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                return None
        return None

    def create(self, task_id: str, session_id: str | None = None) -> Checkpoint:
        """Create a new checkpoint before/during Agent task execution."""
        clean_task = re.sub(r"[^a-zA-Z0-9_\-]", "", task_id)[:8] or "task"
        cp_id = f"cp_{int(time.time() * 1000)}_{clean_task}"
        cp_dir = self.checkpoints_dir / cp_id
        snapshots_dir = cp_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        user_dirty = self._get_user_dirty_files()
        git_head = self._get_git_head()

        files: list[CheckpointFile] = []
        workspace_files = self.workspace.iter_files()

        for file_path in workspace_files:
            rel_path = self.workspace.relative(file_path)
            if is_sensitive_path(rel_path):
                continue

            try:
                content = file_path.read_bytes()
                chash = _file_hash(content)
                is_dirty = rel_path in user_dirty

                # Snapshot file text/bytes if size <= 1MB
                if file_path.stat().st_size <= 1_000_000:
                    snap_target = snapshots_dir / rel_path
                    snap_target.parent.mkdir(parents=True, exist_ok=True)
                    snap_target.write_bytes(content)

                files.append(
                    CheckpointFile(
                        path=rel_path,
                        state="clean",
                        before_hash=chash,
                        after_hash=chash,
                        existed_before=True,
                        existed_after=True,
                        is_user_owned=is_dirty,
                    )
                )
            except (OSError, ValueError):
                continue

        checkpoint = Checkpoint(
            id=cp_id,
            task_id=task_id,
            created_at=utc_now_iso(),
            root=str(self.root),
            files=files,
            user_dirty_files=user_dirty,
            git_head=git_head,
            status="active",
        )

        manifest_data = {
            "metadata": checkpoint.to_dict(),
            "session_id": session_id,
        }
        (cp_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return checkpoint

    def get(self, checkpoint_id: str) -> Checkpoint | None:
        """Retrieve a checkpoint by ID."""
        try:
            cp_id = self._validate_id(checkpoint_id)
        except ValueError:
            return None

        cp_dir = self.checkpoints_dir / cp_id
        manifest_path = cp_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta = data.get("metadata", {})
            files = [CheckpointFile(**f) for f in meta.get("files", [])]
            return Checkpoint(
                id=meta.get("id", cp_id),
                task_id=meta.get("task_id", ""),
                created_at=meta.get("created_at", ""),
                root=meta.get("root", str(self.root)),
                files=files,
                user_dirty_files=meta.get("user_dirty_files", []),
                git_head=meta.get("git_head"),
                status=meta.get("status", "active"),
            )
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def list(self) -> list[Checkpoint]:
        """List all available checkpoints for this workspace."""
        cps: list[Checkpoint] = []
        if not self.checkpoints_dir.exists():
            return cps

        for item in self.checkpoints_dir.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                cp = self.get(item.name)
                if cp:
                    cps.append(cp)

        cps.sort(key=lambda c: c.created_at, reverse=True)
        return cps

    def inspect(self, checkpoint_id: str) -> dict[str, Any]:
        """Compare current workspace against checkpoint to list changed files."""
        cp = self.get(checkpoint_id)
        if cp is None:
            raise KeyError(f"Checkpoint {checkpoint_id!r} not found")

        cp_files = {f.path: f for f in cp.files}
        user_dirty = set(cp.user_dirty_files)

        changes: list[dict[str, Any]] = []
        current_files = {self.workspace.relative(p): p for p in self.workspace.iter_files()}

        all_paths = sorted(set(cp_files.keys()) | set(current_files.keys()))

        for path in all_paths:
            if is_sensitive_path(path):
                continue

            cp_file = cp_files.get(path)
            curr_path = current_files.get(path)

            if cp_file and not curr_path:
                changes.append(
                    {
                        "path": path,
                        "status": "deleted",
                        "is_user_owned": cp_file.is_user_owned,
                        "before_hash": cp_file.before_hash,
                        "after_hash": None,
                    }
                )
            elif not cp_file and curr_path:
                try:
                    chash = _file_hash(curr_path.read_bytes())
                except OSError:
                    chash = ""
                changes.append(
                    {
                        "path": path,
                        "status": "created",
                        "is_user_owned": path in user_dirty,
                        "before_hash": None,
                        "after_hash": chash,
                    }
                )
            elif cp_file and curr_path:
                try:
                    curr_hash = _file_hash(curr_path.read_bytes())
                except OSError:
                    curr_hash = ""
                if curr_hash != cp_file.before_hash:
                    changes.append(
                        {
                            "path": path,
                            "status": "modified",
                            "is_user_owned": cp_file.is_user_owned,
                            "before_hash": cp_file.before_hash,
                            "after_hash": curr_hash,
                        }
                    )

        return {
            "checkpoint_id": cp.id,
            "task_id": cp.task_id,
            "created_at": cp.created_at,
            "changed_files": changes,
            "total_changes": len(changes),
        }

    def rollback(self, checkpoint_id: str, *, force: bool = False) -> RollbackResult:
        """Safely restore workspace state to checkpoint baseline."""
        cp_id = self._validate_id(checkpoint_id)
        cp = self.get(cp_id)
        if cp is None:
            raise KeyError(f"Checkpoint {cp_id!r} not found")

        cp_dir = self.checkpoints_dir / cp_id
        snapshots_dir = cp_dir / "snapshots"

        inspection = self.inspect(cp_id)
        changed_files = inspection["changed_files"]

        restored: list[str] = []
        removed: list[str] = []
        preserved: list[str] = []
        errors: list[str] = []

        for change in changed_files:
            rel_path = change["path"]
            status = change["status"]
            is_user_owned = change["is_user_owned"]

            if is_sensitive_path(rel_path):
                preserved.append(rel_path)
                continue

            try:
                target = self.workspace.resolve(rel_path, for_write=True)
            except SafetyError as exc:
                errors.append(f"SafetyError for {rel_path}: {exc}")
                preserved.append(rel_path)
                continue

            if is_user_owned and not force:
                preserved.append(f"{rel_path} (preserved_due_to_ownership_conflict)")
                continue

            if status == "created":
                if target.exists() and target.is_file():
                    try:
                        target.unlink()
                        removed.append(rel_path)
                    except OSError as exc:
                        errors.append(f"Failed to remove {rel_path}: {exc}")
                        preserved.append(rel_path)

            elif status in ("modified", "deleted"):
                snap_file = snapshots_dir / rel_path
                if snap_file.exists():
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(snap_file, target)
                        restored.append(rel_path)
                    except OSError as exc:
                        errors.append(f"Failed to restore {rel_path}: {exc}")
                        preserved.append(rel_path)
                else:
                    errors.append(f"Snapshot missing for {rel_path}")
                    preserved.append(rel_path)

        return RollbackResult(
            ok=len(errors) == 0,
            checkpoint_id=cp_id,
            restored=restored,
            removed=removed,
            preserved=preserved,
            errors=errors,
        )

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint record and its snapshots."""
        try:
            cp_id = self._validate_id(checkpoint_id)
        except ValueError:
            return False

        cp_dir = self.checkpoints_dir / cp_id
        if cp_dir.exists() and cp_dir.is_dir():
            shutil.rmtree(cp_dir, ignore_errors=True)
            return True
        return False

    def cleanup(self, max_keep: int = 20) -> int:
        """Prune old checkpoints keeping at most max_keep."""
        cps = self.list()
        removed = 0
        if len(cps) > max_keep:
            for old_cp in cps[max_keep:]:
                if self.delete(old_cp.id):
                    removed += 1
        return removed


__all__ = [
    "Checkpoint",
    "CheckpointFile",
    "CheckpointManager",
    "RollbackResult",
    "is_sensitive_path",
]
