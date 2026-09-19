"""Agent Checkpoint and Rollback Subsystem for NovaCLI.

Checkpoints capture recovery state before/during Agent task execution.
Key principles:
1. Never destroy pre-existing user uncommitted work.
2. Checkpoints are stored in .nova/checkpoints/ and ignored by Git.
3. Rollback only affects Agent-owned changes.
4. Credential files (.env, keys) are never snapshotted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.core.git import GitService
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


def _is_safe_rel_path(path_str: str) -> bool:
    if not path_str or not isinstance(path_str, str):
        return False
    clean = path_str.replace("\\", "/").strip()
    if not clean or clean.startswith("/") or clean.startswith("\\"):
        return False
    if len(clean) >= 2 and clean[1] == ":":
        return False
    if clean.startswith("//"):
        return False

    p = Path(clean)
    if ".." in p.parts:
        return False
    return True
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
    session_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    root: str = ""
    files: list[CheckpointFile] = field(default_factory=list)
    user_dirty_files: list[str] = field(default_factory=list)
    git_head: str | None = None
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))



@dataclass
class RedoResult:
    ok: bool
    checkpoint_id: str
    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

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
        self.root = workspace.root.resolve()
        self.checkpoints_dir = self.root / ".nova" / "checkpoints"
        self.git_svc = GitService(self.workspace)
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure .nova/checkpoints exists and is ignored by Git without symlink redirection."""
        nova_dir = self.root / ".nova"

        # Reject .nova if it is a symlink, junction, or reparse point
        if nova_dir.exists() and (nova_dir.is_symlink() or os.path.islink(nova_dir)):
            raise SafetyError("Storage root .nova is a symlink or reparse point", None)

        if nova_dir.exists():
            try:
                if self.root not in nova_dir.resolve().parents and nova_dir.resolve() != self.root:
                    raise SafetyError("Storage root .nova resolves outside workspace root", None)
            except OSError as exc:
                raise SafetyError(f"Cannot resolve .nova: {exc}", None) from exc

        cp_dir = nova_dir / "checkpoints"
        if cp_dir.exists() and (cp_dir.is_symlink() or os.path.islink(cp_dir)):
            raise SafetyError("Storage root .nova/checkpoints is a symlink or reparse point", None)

        if cp_dir.exists():
            try:
                if self.root not in cp_dir.resolve().parents:
                    raise SafetyError("Storage root .nova/checkpoints resolves outside workspace root", None)
            except OSError as exc:
                raise SafetyError(f"Cannot resolve .nova/checkpoints: {exc}", None) from exc

        nova_dir.mkdir(parents=True, exist_ok=True)
        cp_dir.mkdir(parents=True, exist_ok=True)

        gitignore = nova_dir / ".gitignore"
        if not gitignore.exists():
            try:
                gitignore.write_text("*\n", encoding="utf-8")
            except OSError:
                pass

    def _validate_id(self, checkpoint_id: str) -> str:
        clean_id = (checkpoint_id or "").strip()
        if not clean_id or not _VALID_ID_RE.match(clean_id) or ".." in clean_id:
            raise ValueError(f"Invalid checkpoint ID: {checkpoint_id!r}")
        return clean_id

    def _get_user_dirty_files(self) -> list[str]:
        """Collect paths dirty before the Agent starts using GitService."""
        dirty: set[str] = set()
        if self.git_svc.is_repo():
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        st = pool.submit(lambda: asyncio.run(self.git_svc.status())).result()
                else:
                    st = asyncio.run(self.git_svc.status())

                for entry in st.get("status", []):
                    p = entry.get("path", "").strip()
                    if p and not is_sensitive_path(p):
                        dirty.add(p)
            except Exception:
                pass
        return sorted(dirty)

    def _get_git_head(self) -> str | None:
        if self.git_svc.is_repo():
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(lambda: asyncio.run(self.git_svc.current_head())).result()
                else:
                    return asyncio.run(self.git_svc.current_head())
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

        snapshots_dir_resolved = snapshots_dir.resolve()

        for file_path in workspace_files:
            rel_path = self.workspace.relative(file_path)
            if is_sensitive_path(rel_path):
                continue

            # Reject/skip file symlinks
            if file_path.is_symlink():
                continue

            # Resolve canonical path and verify it is a regular file inside workspace root
            try:
                resolved_file = file_path.resolve()
            except (OSError, ValueError):
                continue

            if not resolved_file.is_file():
                continue

            if self.root not in resolved_file.parents and resolved_file != self.root:
                continue

            # Verify destination snapshot path stays strictly inside snapshots_dir
            try:
                snap_target = (snapshots_dir / rel_path).resolve()
            except (OSError, ValueError):
                continue

            if snapshots_dir_resolved not in snap_target.parents:
                continue

            try:
                content = resolved_file.read_bytes()
                chash = _file_hash(content)
                is_dirty = rel_path in user_dirty

                if resolved_file.stat().st_size <= 1_000_000:
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
            session_id=session_id,
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

    def get(self, checkpoint_id: str, session_id: str | None = None) -> Checkpoint | None:
        """Retrieve a checkpoint by ID enforcing strict manifest schema and session/workspace ownership."""
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
            if not isinstance(data, dict):
                return None

            meta = data.get("metadata")
            if not isinstance(meta, dict):
                return None

            # Strict top-level fields validation
            cp_id_meta = meta.get("id")
            if not isinstance(cp_id_meta, str) or not _VALID_ID_RE.match(cp_id_meta):
                return None

            task_id_meta = meta.get("task_id")
            if not isinstance(task_id_meta, str):
                return None

            created_at_meta = meta.get("created_at")
            if not isinstance(created_at_meta, str):
                return None

            cp_root = meta.get("root")
            if not isinstance(cp_root, str) or (cp_root and Path(cp_root).resolve() != self.root):
                return None

            status_meta = meta.get("status", "active")
            if not isinstance(status_meta, str):
                return None

            stored_session = data.get("session_id") or meta.get("session_id")
            if stored_session is not None and not isinstance(stored_session, str):
                return None

            # Mandatory Session ownership verification
            if stored_session:
                if not session_id or session_id != stored_session:
                    raise PermissionError(
                        f"Session ownership mismatch: checkpoint {checkpoint_id!r} requires session {stored_session!r}"
                    )

            # Strict user_dirty_files validation
            raw_user_dirty = meta.get("user_dirty_files")
            if not isinstance(raw_user_dirty, list):
                return None
            user_dirty_files: list[str] = []
            for item in raw_user_dirty:
                if not isinstance(item, str) or not _is_safe_rel_path(item):
                    return None
                user_dirty_files.append(item)

            # Strict files list validation
            raw_files = meta.get("files")
            if not isinstance(raw_files, list):
                return None

            files: list[CheckpointFile] = []
            valid_states = {"clean", "created", "modified", "deleted"}

            for f in raw_files:
                if not isinstance(f, dict):
                    return None

                p = f.get("path")
                if not isinstance(p, str) or not _is_safe_rel_path(p):
                    return None

                state = f.get("state")
                if not isinstance(state, str) or state not in valid_states:
                    return None

                b_hash = f.get("before_hash")
                if b_hash is not None and not isinstance(b_hash, str):
                    return None

                a_hash = f.get("after_hash")
                if a_hash is not None and not isinstance(a_hash, str):
                    return None

                ex_before = f.get("existed_before", True)
                if type(ex_before) is not bool:
                    return None

                ex_after = f.get("existed_after", True)
                if type(ex_after) is not bool:
                    return None

                is_user = f.get("is_user_owned", False)
                if type(is_user) is not bool:
                    return None

                files.append(
                    CheckpointFile(
                        path=p,
                        state=state,
                        before_hash=b_hash,
                        after_hash=a_hash,
                        existed_before=ex_before,
                        existed_after=ex_after,
                        is_user_owned=is_user,
                    )
                )

            return Checkpoint(
                id=cp_id_meta,
                task_id=task_id_meta,
                session_id=stored_session,
                created_at=created_at_meta,
                root=cp_root,
                files=files,
                user_dirty_files=user_dirty_files,
                git_head=str(meta["git_head"]) if isinstance(meta.get("git_head"), str) else None,
                status=status_meta,
            )
        except PermissionError:
            raise
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def list(self, session_id: str | None = None) -> list[Checkpoint]:
        """List all available checkpoints for this workspace/session."""
        cps: list[Checkpoint] = []
        if not self.checkpoints_dir.exists():
            return cps

        for item in self.checkpoints_dir.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                try:
                    cp = self.get(item.name, session_id=session_id)
                    if cp:
                        cps.append(cp)
                except PermissionError:
                    continue

        cps.sort(key=lambda c: c.created_at, reverse=True)
        return cps

    def inspect(self, checkpoint_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Compare current workspace against checkpoint to list changed files."""
        cp = self.get(checkpoint_id, session_id=session_id)
        if cp is None:
            raise KeyError(f"Checkpoint {checkpoint_id!r} not found or inaccessible")

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
            "session_id": cp.session_id,
            "created_at": cp.created_at,
            "changed_files": changes,
            "total_changes": len(changes),
        }

    def rollback(self, checkpoint_id: str, *, session_id: str | None = None) -> RollbackResult:
        """Safely restore workspace state to checkpoint baseline.

        NEVER overwrites or deletes pre-existing user-owned files.
        Captures pre-undo state so Redo can restore agent changes.
        """
        cp_id = self._validate_id(checkpoint_id)
        cp = self.get(cp_id, session_id=session_id)
        if cp is None:
            raise KeyError(f"Checkpoint {cp_id!r} not found or inaccessible")

        cp_dir = (self.checkpoints_dir / cp_id).resolve()
        snapshots_dir = (cp_dir / "snapshots").resolve()

        # Storage directory jail check
        if self.checkpoints_dir not in cp_dir.parents and cp_dir != self.checkpoints_dir:
            raise SafetyError("Checkpoint directory is outside storage root", None)

        inspection = self.inspect(cp_id, session_id=session_id)
        changed_files = inspection["changed_files"]

        # Prepare Redo snapshot directory
        redo_dir = cp_dir / "redo"
        redo_snapshots = redo_dir / "snapshots"
        if redo_dir.exists():
            shutil.rmtree(redo_dir, ignore_errors=True)
        redo_snapshots.mkdir(parents=True, exist_ok=True)
        redo_snapshots_resolved = redo_snapshots.resolve()

        redo_file_manifests: list[dict[str, Any]] = []

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

            # INVIOLABLE SECURITY RULE: Pre-existing user-owned files are NEVER touched
            if is_user_owned:
                preserved.append(f"{rel_path} (preserved_due_to_ownership_conflict)")
                continue

            # Capture state BEFORE rollback for Redo
            existed_before_rollback = target.exists() and not target.is_symlink() and target.is_file()
            hash_before_rollback = None
            if existed_before_rollback:
                try:
                    content_before = target.read_bytes()
                    hash_before_rollback = _file_hash(content_before)
                    redo_snap_target = (redo_snapshots / rel_path).resolve()
                    if redo_snapshots_resolved in redo_snap_target.parents and len(content_before) <= 1_000_000:
                        redo_snap_target.parent.mkdir(parents=True, exist_ok=True)
                        redo_snap_target.write_bytes(content_before)
                except OSError:
                    pass

            # Calculate expected state AFTER rollback
            expected_after_undo_hash = None
            existed_after_undo = False
            if status in ("modified", "deleted"):
                try:
                    snap_file = (snapshots_dir / rel_path).resolve()
                    if snapshots_dir in snap_file.parents and snap_file.exists() and snap_file.is_file():
                        existed_after_undo = True
                        expected_after_undo_hash = _file_hash(snap_file.read_bytes())
                except Exception:
                    pass
            elif status == "created":
                existed_after_undo = False
                expected_after_undo_hash = None

            # Execute Rollback
            if status == "created":
                if target.exists() or target.is_symlink():
                    try:
                        if target.is_dir() and not target.is_symlink():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                        removed.append(rel_path)
                    except OSError as exc:
                        errors.append(f"Failed to remove {rel_path}: {exc}")
                        preserved.append(rel_path)
                        continue

            elif status in ("modified", "deleted"):
                try:
                    snap_file = (snapshots_dir / rel_path).resolve()
                except Exception:
                    errors.append(f"Invalid snapshot path for {rel_path}")
                    preserved.append(rel_path)
                    continue

                # Ensure snapshot path stays inside snapshots_dir
                if snapshots_dir not in snap_file.parents and snap_file != snapshots_dir:
                    errors.append(f"Snapshot path traversal attempt blocked for {rel_path}")
                    preserved.append(rel_path)
                    continue

                if snap_file.exists() and snap_file.is_file():
                    try:
                        # Symlink safety: if target is a symlink, unlink it first so copy2 writes real file
                        if target.is_symlink():
                            target.unlink()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(snap_file, target)
                        restored.append(rel_path)
                    except OSError as exc:
                        errors.append(f"Failed to restore {rel_path}: {exc}")
                        preserved.append(rel_path)
                        continue
                else:
                    errors.append(f"Snapshot missing for {rel_path}")
                    preserved.append(rel_path)
                    continue

            redo_file_manifests.append({
                "path": rel_path,
                "status": status,
                "existed_pre_undo": existed_before_rollback,
                "pre_undo_hash": hash_before_rollback,
                "existed_post_undo": existed_after_undo,
                "expected_post_undo_hash": expected_after_undo_hash,
            })

        # Save Redo manifest and update checkpoint status
        redo_manifest_data = {
            "checkpoint_id": cp.id,
            "session_id": cp.session_id,
            "created_at": utc_now_iso(),
            "files": redo_file_manifests,
        }
        (redo_dir / "manifest.json").write_text(json.dumps(redo_manifest_data, indent=2), encoding="utf-8")

        # Update checkpoint status in manifest
        manifest_path = cp_dir / "manifest.json"
        try:
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            mdata["metadata"]["status"] = "undone"
            manifest_path.write_text(json.dumps(mdata, indent=2), encoding="utf-8")
        except Exception:
            pass

        return RollbackResult(
            ok=len(errors) == 0,
            checkpoint_id=cp_id,
            restored=restored,
            removed=removed,
            preserved=preserved,
            errors=errors,
        )

    def redo(self, checkpoint_id: str, *, session_id: str | None = None) -> RedoResult:
        """Redo a previously undone checkpoint safely.

        Restores agent modifications that existed immediately before Undo.
        If a file was modified by the user after Undo, it is preserved as a conflict.
        """
        cp_id = self._validate_id(checkpoint_id)
        cp = self.get(cp_id, session_id=session_id)
        if cp is None:
            raise KeyError(f"Checkpoint {cp_id!r} not found or inaccessible")

        cp_dir = (self.checkpoints_dir / cp_id).resolve()
        redo_dir = (cp_dir / "redo").resolve()
        redo_snapshots = (redo_dir / "snapshots").resolve()

        if self.checkpoints_dir not in cp_dir.parents and cp_dir != self.checkpoints_dir:
            raise SafetyError("Checkpoint directory is outside storage root", None)

        redo_manifest_path = redo_dir / "manifest.json"
        if not redo_manifest_path.exists():
            raise KeyError(f"No redo state available for checkpoint {cp_id!r}")

        try:
            redo_data = json.loads(redo_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(redo_data, dict):
                raise ValueError("Invalid redo manifest")

            stored_session = redo_data.get("session_id")
            if stored_session and session_id and session_id != stored_session:
                raise PermissionError(f"Session ownership mismatch for redo: requires {stored_session!r}")

            raw_redo_files = redo_data.get("files")
            if not isinstance(raw_redo_files, list):
                raise ValueError("Invalid redo files list")
        except PermissionError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to parse redo manifest: {exc}") from exc

        restored: list[str] = []
        removed: list[str] = []
        preserved: list[str] = []
        errors: list[str] = []

        for item in raw_redo_files:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("path")
            if not isinstance(rel_path, str) or not _is_safe_rel_path(rel_path) or is_sensitive_path(rel_path):
                continue

            existed_pre_undo = bool(item.get("existed_pre_undo"))
            pre_undo_hash = item.get("pre_undo_hash")
            existed_post_undo = bool(item.get("existed_post_undo"))
            expected_post_undo_hash = item.get("expected_post_undo_hash")

            try:
                target = self.workspace.resolve(rel_path, for_write=True)
            except SafetyError as exc:
                errors.append(f"SafetyError for {rel_path}: {exc}")
                preserved.append(rel_path)
                continue

            current_exists = target.exists() and not target.is_symlink() and target.is_file()
            current_hash = None
            if current_exists:
                try:
                    current_hash = _file_hash(target.read_bytes())
                except OSError:
                    pass

            # CONFLICT DETECTION: check if user modified the file after Undo
            conflict = False
            if existed_post_undo != current_exists:
                conflict = True
            elif current_exists and expected_post_undo_hash is not None and current_hash != expected_post_undo_hash:
                conflict = True

            if conflict:
                preserved.append(f"{rel_path} (preserved_due_to_conflict_after_undo)")
                continue

            # Perform Redo
            if existed_pre_undo:
                try:
                    redo_snap_file = (redo_snapshots / rel_path).resolve()
                except Exception:
                    errors.append(f"Invalid redo snapshot path for {rel_path}")
                    preserved.append(rel_path)
                    continue

                if redo_snapshots not in redo_snap_file.parents and redo_snap_file != redo_snapshots:
                    errors.append(f"Redo snapshot path traversal attempt blocked for {rel_path}")
                    preserved.append(rel_path)
                    continue

                if redo_snap_file.exists() and redo_snap_file.is_file():
                    try:
                        if target.is_symlink():
                            target.unlink()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(redo_snap_file, target)
                        restored.append(rel_path)
                    except OSError as exc:
                        errors.append(f"Failed to redo restore {rel_path}: {exc}")
                        preserved.append(rel_path)
                else:
                    errors.append(f"Redo snapshot missing for {rel_path}")
                    preserved.append(rel_path)
            else:
                # Pre-undo it did not exist, so Redo removes it
                if target.exists() or target.is_symlink():
                    try:
                        if target.is_dir() and not target.is_symlink():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                        removed.append(rel_path)
                    except OSError as exc:
                        errors.append(f"Failed to redo remove {rel_path}: {exc}")
                        preserved.append(rel_path)

        # Remove consumed redo state directory
        shutil.rmtree(redo_dir, ignore_errors=True)

        # Update checkpoint status back to "active"
        manifest_path = cp_dir / "manifest.json"
        try:
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            mdata["metadata"]["status"] = "active"
            manifest_path.write_text(json.dumps(mdata, indent=2), encoding="utf-8")
        except Exception:
            pass

        return RedoResult(
            ok=len(errors) == 0,
            checkpoint_id=cp_id,
            restored=restored,
            removed=removed,
            preserved=preserved,
            errors=errors,
        )

    def delete(self, checkpoint_id: str, session_id: str | None = None) -> bool:
        """Delete a checkpoint record and its snapshots."""
        try:
            cp_id = self._validate_id(checkpoint_id)
            cp = self.get(cp_id, session_id=session_id)
            if cp is None:
                return False
        except (ValueError, PermissionError):
            return False

        cp_dir = (self.checkpoints_dir / cp_id).resolve()
        if self.checkpoints_dir in cp_dir.parents and cp_dir.exists() and cp_dir.is_dir():
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
    "RedoResult",
    "RollbackResult",
    "is_sensitive_path",
]
