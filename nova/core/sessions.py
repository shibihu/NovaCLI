"""Persistent Chat Sessions Subsystem for NovaCLI.

Saves conversation transcripts and metadata in `.nova/sessions/<session_id>/`
separated from agent checkpoints.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nova.core.models import to_jsonable, utc_now_iso
from nova.core.safety import SafetyError, redact_secrets
from nova.workspace.files import Workspace

_VALID_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
MAX_STORED_OUTPUT_CHARS = 4_000


def _is_safe_rel_path(path_str: str) -> bool:
    if not path_str or not isinstance(path_str, str):
        return False
    clean = path_str.replace("\\", "/").strip()
    if not clean or clean.startswith("/") or clean.startswith("\\"):
        return False
    if len(clean) >= 2 and clean[1] == ":":
        return False
    p = Path(clean)
    return ".." not in p.parts


def generate_chat_title(task: str) -> str:
    """Generate a readable, short title from the user's first task (up to 70 chars)."""
    if not task or not isinstance(task, str):
        return "New Chat"

    # Remove code fences
    clean = re.sub(r"```[a-zA-Z0-9]*", "", task).strip()
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    clean = lines[0] if lines else ""

    clean = re.sub(r"^[#\-\*\s]+", "", clean)
    clean = re.sub(r"[`\*_]", "", clean)
    clean = clean.strip(".,;:?!'\"-")

    if not clean:
        return "New Chat"

    if len(clean) > 70:
        words = clean[:70].rsplit(" ", 1)[0]
        clean = words if words else clean[:70]

    clean = clean[0].upper() + clean[1:] if len(clean) > 1 else clean.upper()
    return clean or "New Chat"


@dataclass
class ChatSession:
    id: str
    project_root: str
    title: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    model: str = ""
    status: str = "done"
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    def to_messages(self) -> list[Any]:
        """Reconstruct canonical list of Message objects from session history."""
        from nova.core.models import Message
        messages: list[Message] = []
        raw_items = self.messages
        if not raw_items:
            return messages

        current_tool_calls = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            if role in ("user", "system"):
                if item.get("content"):
                    messages.append(Message(role=role, content=str(item["content"])))
                continue
            elif role == "assistant":
                messages.append(
                    Message(
                        role="assistant",
                        content=item.get("content"),
                        tool_calls=item.get("tool_calls"),
                    )
                )
                continue
            elif role == "tool":
                messages.append(
                    Message(
                        role="tool",
                        content=str(item.get("content", "")),
                        tool_call_id=item.get("tool_call_id"),
                        name=item.get("name"),
                    )
                )
                continue

            evt_type = item.get("type")
            data = item.get("data") or {}

            if evt_type == "agent_start":
                task_text = data.get("task")
                if task_text and not messages:
                    messages.append(Message.user(str(task_text)))

            elif evt_type == "thought":
                text = data.get("text")
                if text:
                    messages.append(Message.assistant(content=str(text)))

            elif evt_type == "tool_call":
                tc_id = data.get("id") or f"call_{len(messages)}"
                t_name = data.get("tool") or data.get("name") or "tool"
                t_input = data.get("input") or {}

                raw_args = json.dumps(t_input) if isinstance(t_input, dict) else str(t_input)
                tc_dict = {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": t_name,
                        "arguments": raw_args,
                    },
                }
                if data.get("thought_signature"):
                    tc_dict["thought_signature"] = data["thought_signature"]
                    tc_dict["function"]["thought_signature"] = data["thought_signature"]

                current_tool_calls.append(tc_dict)

            elif evt_type in ("tool_result", "blocked"):
                if current_tool_calls:
                    messages.append(Message.assistant(content=None, tool_calls=list(current_tool_calls)))
                    current_tool_calls.clear()

                tc_id = data.get("tool_call_id") or data.get("id") or f"call_{len(messages)}"
                t_name = data.get("name") or data.get("tool") or "tool"
                t_out = data.get("output") or data.get("reason") or data.get("error") or ""
                messages.append(Message.tool_result(tc_id, t_name, str(t_out)))

            elif evt_type == "final":
                if current_tool_calls:
                    messages.append(Message.assistant(content=None, tool_calls=list(current_tool_calls)))
                    current_tool_calls.clear()
                answer = data.get("answer")
                if answer:
                    messages.append(Message.assistant(content=str(answer)))

        if current_tool_calls:
            messages.append(Message.assistant(content=None, tool_calls=list(current_tool_calls)))
            current_tool_calls.clear()

        return messages



class SessionStorageManager:
    """Manages persistent chat session storage in .nova/sessions/."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.root.resolve()
        self.sessions_dir = self.root / ".nova" / "sessions"
        self._secrets = workspace.safety.secret_values if hasattr(workspace, "safety") else ()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure .nova/sessions exists and is safe from symlink redirection."""
        nova_dir = self.root / ".nova"

        if nova_dir.exists() and (nova_dir.is_symlink() or os.path.islink(nova_dir)):
            raise SafetyError("Storage root .nova is a symlink or reparse point", None)

        if nova_dir.exists():
            try:
                if self.root not in nova_dir.resolve().parents and nova_dir.resolve() != self.root:
                    raise SafetyError("Storage root .nova resolves outside workspace root", None)
            except OSError as exc:
                raise SafetyError(f"Cannot resolve .nova: {exc}", None) from exc

        sess_dir = nova_dir / "sessions"
        if sess_dir.exists() and (sess_dir.is_symlink() or os.path.islink(sess_dir)):
            raise SafetyError("Storage root .nova/sessions is a symlink or reparse point", None)

        if sess_dir.exists():
            try:
                if self.root not in sess_dir.resolve().parents:
                    raise SafetyError("Storage root .nova/sessions resolves outside workspace root", None)
            except OSError as exc:
                raise SafetyError(f"Cannot resolve .nova/sessions: {exc}", None) from exc

        nova_dir.mkdir(parents=True, exist_ok=True)
        sess_dir.mkdir(parents=True, exist_ok=True)

    def _validate_id(self, session_id: str) -> str:
        clean_id = (session_id or "").strip()
        if not clean_id or not _VALID_SESSION_ID_RE.match(clean_id) or ".." in clean_id:
            raise ValueError(f"Invalid session ID: {session_id!r}")
        return clean_id

    def _redact_data(self, data: Any) -> Any:
        if isinstance(data, str):
            res = redact_secrets(data, *self._secrets)
            if len(res) > MAX_STORED_OUTPUT_CHARS:
                return res[:MAX_STORED_OUTPUT_CHARS] + f"\n… [truncated {len(res) - MAX_STORED_OUTPUT_CHARS} chars]"
            return res
        if isinstance(data, dict):
            return {k: self._redact_data(v) for k, v in data.items() if k not in ("authorization", "api_key", "secret")}
        if isinstance(data, list):
            return [self._redact_data(v) for v in data]
        return data

    def create(
        self,
        task: str,
        session_id: str | None = None,
        title: str | None = None,
        model: str = "",
    ) -> ChatSession:
        """Create a new persistent chat session record."""
        sid = self._validate_id(session_id) if session_id else f"sess_{int(time.time() * 1000)}_{os.urandom(3).hex()}"
        s_title = title or generate_chat_title(task)
        now = utc_now_iso()

        session = ChatSession(
            id=sid,
            project_root=str(self.root),
            title=s_title,
            created_at=now,
            updated_at=now,
            model=model,
            status="pending",
            messages=[],
        )

        self.save(session)
        return session

    def save(self, session: ChatSession) -> None:
        """Save chat session manifest safely with secret redaction and output truncation."""
        sid = self._validate_id(session.id)
        s_dir = (self.sessions_dir / sid).resolve()

        if self.sessions_dir not in s_dir.parents and s_dir != self.sessions_dir:
            raise SafetyError("Session directory is outside storage root", None)

        s_dir.mkdir(parents=True, exist_ok=True)

        session.updated_at = utc_now_iso()
        data = self._redact_data(session.to_dict())

        manifest_file = s_dir / "manifest.json"
        manifest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, session_id: str) -> ChatSession | None:
        """Load session record enforcing workspace root containment and schema checks."""
        try:
            sid = self._validate_id(session_id)
        except ValueError:
            return None

        s_dir = (self.sessions_dir / sid).resolve()
        manifest_file = s_dir / "manifest.json"
        if not manifest_file.exists():
            return None

        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None

            s_root = data.get("project_root")
            if not isinstance(s_root, str) or Path(s_root).resolve() != self.root:
                # Workspace ownership mismatch
                return None

            sid_loaded = data.get("id")
            if sid_loaded != sid:
                return None

            title = data.get("title") or "Untitled Chat"
            created_at = data.get("created_at") or utc_now_iso()
            updated_at = data.get("updated_at") or created_at
            model = data.get("model") or ""
            status = data.get("status") or "done"
            messages = data.get("messages") or []

            return ChatSession(
                id=sid,
                project_root=str(self.root),
                title=title,
                created_at=created_at,
                updated_at=updated_at,
                model=model,
                status=status,
                messages=messages if isinstance(messages, list) else [],
            )
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def list(self) -> list[ChatSession]:
        """List all persistent chat sessions for this workspace sorted by updated_at desc."""
        sessions: list[ChatSession] = []
        if not self.sessions_dir.exists():
            return sessions

        for item in self.sessions_dir.iterdir():
            if item.is_dir() and (item / "manifest.json").exists():
                sess = self.get(item.name)
                if sess:
                    sessions.append(sess)

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def rename(self, session_id: str, new_title: str) -> ChatSession:
        """Rename a chat session."""
        sess = self.get(session_id)
        if not sess:
            raise KeyError(f"Session {session_id!r} not found")

        clean_title = (new_title or "").strip()
        if not clean_title:
            raise ValueError("Title cannot be empty")
        if len(clean_title) > 100:
            clean_title = clean_title[:100]

        clean_title = re.sub(r"[\r\n\t\x00-\x1f]", " ", clean_title).strip()
        if not clean_title:
            raise ValueError("Invalid title format")

        sess.title = clean_title
        self.save(sess)
        return sess

    def delete(self, session_id: str) -> bool:
        """Delete a chat session directory."""
        try:
            sid = self._validate_id(session_id)
            sess = self.get(sid)
            if not sess:
                return False
        except (ValueError, PermissionError):
            return False

        s_dir = (self.sessions_dir / sid).resolve()
        if self.sessions_dir in s_dir.parents and s_dir.exists() and s_dir.is_dir():
            shutil.rmtree(s_dir, ignore_errors=True)
            return True
        return False


__all__ = [
    "ChatSession",
    "SessionStorageManager",
    "generate_chat_title",
]
