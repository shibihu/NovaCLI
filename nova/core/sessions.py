"""Persistent Chat Sessions Subsystem for NovaCLI.

Saves the canonical conversation (``Message[]``) and metadata in
`.nova/sessions/<session_id>/`, separated from agent checkpoints.

The stored ``messages`` list is the *canonical* conversation the LLM was given —
user turns, assistant turns, tool calls and tool results — not a transcript of
``AgentEvent`` progress notifications. Session files written by older releases
stored an event list instead; those are detected and converted on read
(:meth:`ChatSession.is_legacy_history`), never silently corrupted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from nova.core.conversation import (
    CONVERSATION_SCHEMA_VERSION,
    is_canonical_history,
    is_legacy_history,
    messages_from_dicts,
    messages_to_dicts,
)
from nova.core.models import Message, to_jsonable, utc_now_iso
from nova.core.safety import SafetyError, redact_secrets
from nova.workspace.files import Workspace

_VALID_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
MAX_STORED_OUTPUT_CHARS = 4_000

#: Any title that means "not named yet" — replaced by the first task's title.
PLACEHOLDER_TITLES = frozenset({"", "new chat", "untitled chat"})


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
    """One persistent chat: metadata plus its canonical conversation."""

    id: str
    project_root: str
    title: str
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    model: str = ""
    status: str = "done"
    messages: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = CONVERSATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @property
    def has_placeholder_title(self) -> bool:
        """True while the chat still carries its generated "New Chat" title."""
        return (self.title or "").strip().lower() in PLACEHOLDER_TITLES

    @property
    def is_empty(self) -> bool:
        return not self.messages

    def is_legacy_history(self) -> bool:
        """True for session files written before canonical persistence.

        Those files stored a raw ``AgentEvent`` list (entries with ``type`` and
        no ``role``). Anything already in canonical shape is treated as such
        even if the version marker was lost.
        """
        if is_legacy_history(self.messages):
            return True
        if self.schema_version >= CONVERSATION_SCHEMA_VERSION:
            return False
        return not is_canonical_history(self.messages)

    def to_messages(self) -> list[Message]:
        """Return the canonical conversation as :class:`Message` objects."""
        if self.is_legacy_history():
            return self._legacy_messages_from_events()
        return messages_from_dicts(self.messages)

    def _legacy_messages_from_events(self) -> list[Message]:
        """Best-effort conversion of a pre-canonical event list.

        Internal ``thought``/``progress``/``step_start`` notifications are
        deliberately **not** turned into assistant messages: they were UI
        status output, and replaying them to a provider would fabricate a
        conversation the model never had.
        """
        messages: list[Message] = []
        raw_items = self.messages
        if not raw_items:
            return messages

        current_tool_calls: list[dict[str, Any]] = []

        def flush_tool_calls() -> None:
            nonlocal current_tool_calls
            if current_tool_calls:
                messages.append(Message.assistant(content=None, tool_calls=list(current_tool_calls)))
                current_tool_calls = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            if role in ("user", "system"):
                if item.get("content"):
                    messages.append(Message(role=role, content=str(item["content"])))
                continue
            if role == "assistant":
                messages.append(
                    Message(
                        role="assistant",
                        content=item.get("content"),
                        tool_calls=item.get("tool_calls"),
                    )
                )
                continue
            if role == "tool":
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
                if task_text and not any(m.role == "user" for m in messages):
                    messages.append(Message.user(str(task_text)))

            elif evt_type == "tool_call":
                tc_id = data.get("id") or f"call_{len(messages)}"
                t_name = data.get("tool") or data.get("name") or "tool"
                t_input = data.get("input") or {}

                raw_args = json.dumps(t_input) if isinstance(t_input, dict) else str(t_input)
                tc_dict: dict[str, Any] = {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": t_name,
                        "arguments": raw_args,
                    },
                }
                signature = data.get("thought_signature")
                if signature:
                    tc_dict["thought_signature"] = signature
                    tc_dict["function"]["thought_signature"] = signature

                current_tool_calls.append(tc_dict)

            elif evt_type in ("tool_result", "blocked"):
                flush_tool_calls()

                tc_id = data.get("tool_call_id") or data.get("id") or f"call_{len(messages)}"
                t_name = data.get("name") or data.get("tool") or "tool"
                t_out = data.get("output") or data.get("reason") or data.get("error") or ""
                messages.append(Message.tool_result(tc_id, t_name, str(t_out)))

            elif evt_type == "final":
                flush_tool_calls()
                answer = data.get("answer")
                if answer:
                    messages.append(Message.assistant(content=str(answer)))

        flush_tool_calls()
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

    def _redact_data(self, data: Any, *, truncate: bool = True) -> Any:
        """Redact secrets; optionally cap string length.

        Canonical conversations are redacted but **not** truncated: a tool
        result is part of the conversation the model will be shown again, so
        silently shortening it would corrupt the session.
        """
        if isinstance(data, str):
            res = redact_secrets(data, *self._secrets)
            if truncate and len(res) > MAX_STORED_OUTPUT_CHARS:
                return res[:MAX_STORED_OUTPUT_CHARS] + f"\n… [truncated {len(res) - MAX_STORED_OUTPUT_CHARS} chars]"
            return res
        if isinstance(data, dict):
            return {
                k: self._redact_data(v, truncate=truncate)
                for k, v in data.items()
                if k not in ("authorization", "api_key", "secret")
            }
        if isinstance(data, list):
            return [self._redact_data(v, truncate=truncate) for v in data]
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
            schema_version=CONVERSATION_SCHEMA_VERSION,
        )

        self.save(session)
        return session

    def replace_history(self, session_id: str, messages: Iterable[Message]) -> ChatSession | None:
        """Overwrite a session's canonical conversation with ``messages``.

        The caller is responsible for :meth:`save`. Returns ``None`` when the
        session does not exist (or is not owned by this workspace).
        """
        sess = self.get(session_id)
        if sess is None:
            return None
        sess.schema_version = CONVERSATION_SCHEMA_VERSION
        sess.messages = messages_to_dicts(messages)
        return sess

    def save(self, session: ChatSession, *, touch: bool = True) -> None:
        """Save chat session manifest with secret redaction.

        ``touch=False`` leaves ``updated_at`` alone so metadata-only writes do
        not reshuffle Recent Chats; a real conversation update is one logical
        write (see :meth:`replace_history`).
        """
        sid = self._validate_id(session.id)
        s_dir = (self.sessions_dir / sid).resolve()

        if self.sessions_dir not in s_dir.parents and s_dir != self.sessions_dir:
            raise SafetyError("Session directory is outside storage root", None)

        s_dir.mkdir(parents=True, exist_ok=True)

        if touch:
            session.updated_at = utc_now_iso()
        legacy = session.is_legacy_history()
        data = self._redact_data(session.to_dict(), truncate=legacy)
        data["schema_version"] = session.schema_version

        manifest_file = s_dir / "manifest.json"
        manifest_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

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
            # 0 means "written before canonical persistence"; the message shape
            # then decides between canonical and legacy handling.
            raw_version = data.get("schema_version")
            schema_version = raw_version if isinstance(raw_version, int) else 0

            return ChatSession(
                id=sid,
                project_root=str(self.root),
                title=title,
                created_at=created_at,
                updated_at=updated_at,
                model=model,
                status=status,
                messages=messages if isinstance(messages, list) else [],
                schema_version=schema_version,
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
    "PLACEHOLDER_TITLES",
    "SessionStorageManager",
    "generate_chat_title",
]
