"""Plain data types shared across NovaCLI.

Everything here is a ``dataclass`` with an explicit ``to_dict()`` so the CLI,
the agent loop and the web layer all speak the same JSON dialect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclass-friendly values into JSON-safe ones."""
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def as_dict(obj: Any) -> dict[str, Any]:
    """``dataclasses.asdict`` plus enum/path normalisation."""
    return to_jsonable(asdict(obj))


# --- Enumerations -----------------------------------------------------------


class RiskLevel(StrEnum):
    """How dangerous an action is judged to be."""

    SAFE = "safe"
    MODERATE = "moderate"
    DANGEROUS = "dangerous"
    FORBIDDEN = "forbidden"


class SafetyMode(StrEnum):
    """Policy profile controlling what runs without asking."""

    STRICT = "strict"
    SMART = "smart"
    PERMISSIVE = "permissive"


class AgentStatus(StrEnum):
    """Lifecycle state of a single agent run."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

    @classmethod
    def terminal(cls) -> frozenset["AgentStatus"]:
        return frozenset({cls.DONE, cls.ERROR, cls.CANCELLED})


class EventType(StrEnum):
    """Event names emitted while an agent works.

    The CLI renders these as terminal output; the web IDE streams them over
    Server-Sent Events. Both consume the same vocabulary.
    """

    AGENT_START = "agent_start"
    STEP_START = "step_start"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESOLVED = "approval_resolved"
    TOOL_RESULT = "tool_result"
    BLOCKED = "blocked"
    PROGRESS = "progress"
    RATE_LIMIT_WAIT = "rate_limit_wait"
    FINAL = "final"
    ERROR = "error"
    CANCELLED = "cancelled"


TERMINAL_EVENTS: frozenset[str] = frozenset(
    {EventType.FINAL.value, EventType.ERROR.value, EventType.CANCELLED.value}
)


# --- Tool plumbing ----------------------------------------------------------


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str = ""
    provider_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)


@dataclass
class AIResponse:
    """Structured response returned by an AI provider."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# --- Messages ---------------------------------------------------------------


@dataclass
class Message:
    """A single chat turn in the provider-agnostic OpenAI-style shape.

    ``tool_calls`` keeps the provider's exact structure (including vendor
    extensions such as ``extra_content.google.thought_signature``) and
    ``provider_data`` carries any message-level provider metadata, so a
    persisted conversation can be replayed to the same provider unchanged.
    """

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    provider_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls is not None:
            data["tool_calls"] = to_jsonable(self.tool_calls)
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        if self.provider_data:
            data["provider_data"] = to_jsonable(self.provider_data)
        return data

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[dict[str, Any]] | None = None
    ) -> "Message":
        return cls("assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, content: str) -> "Message":
        return cls("tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class ToolResult:
    """Outcome of executing a :class:`ToolCall`."""

    name: str
    ok: bool
    output: str = ""
    error: str | None = None
    blocked: bool = False
    duration_ms: int = 0
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)

    def render(self, *, limit: int = 4_000) -> str:
        """Human/model-readable observation string."""
        body = self.output if self.ok else (self.error or self.output or "failed")
        if self.blocked:
            body = f"BLOCKED BY SAFETY POLICY: {body}"
        if len(body) > limit:
            body = body[:limit] + f"\n... [truncated, {len(body)} chars total]"
        return body


@dataclass
class ApprovalRequest:
    """A safety gate waiting on a human decision."""

    id: str
    tool: str
    summary: str
    detail: str = ""
    level: RiskLevel = RiskLevel.DANGEROUS
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)


class ApprovalDecision(StrEnum):
    """Answer to an :class:`ApprovalRequest`."""

    APPROVE = "approve"
    DENY = "deny"
    ALWAYS = "always"


# --- Steps & events ---------------------------------------------------------


@dataclass
class AgentStep:
    """One think/act cycle of the agent loop."""

    index: int
    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    final_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)


@dataclass
class AgentEvent:
    """A progress notification streamed while the agent works."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": to_jsonable(self.data),
            "step": self.step,
            "timestamp": self.timestamp,
        }

    @property
    def is_terminal(self) -> bool:
        return self.type in TERMINAL_EVENTS


@dataclass
class AgentResult:
    """The complete outcome of one agent run."""

    task: str
    status: AgentStatus = AgentStatus.PENDING
    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None
    model: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    checkpoint_id: str | None = None
    changed_files: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == AgentStatus.DONE

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)


# --- Execution --------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a shell command executed through :class:`CommandRunner`."""

    command: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    blocked: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.blocked and not self.timed_out and self.exit_code == 0

    def render(self, *, limit: int = 4_000) -> str:
        """Compact transcript suitable for an LLM observation."""
        if self.blocked:
            return f"Command blocked by safety policy: {self.reason}"
        parts = [f"$ {self.command}", f"exit_code: {self.exit_code}"]
        if self.timed_out:
            parts.append(f"TIMED OUT after {self.duration_ms}ms")
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.rstrip()}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self.stderr.rstrip()}")
        if not self.stdout.strip() and not self.stderr.strip():
            parts.append("(no output)")
        body = "\n".join(parts)
        if len(body) > limit:
            body = body[:limit] + f"\n... [truncated, {len(body)} chars total]"
        return body

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)


# --- Workspace --------------------------------------------------------------


@dataclass
class FileEntry:
    """A file or directory inside the workspace."""

    path: str
    is_dir: bool = False
    size: int = 0
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)


@dataclass
class SearchHit:
    """A single matching line from a workspace search."""

    path: str
    line: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.text.strip()}"


@dataclass
class ProjectSummary:
    """NovaCLI's understanding of the user's project."""

    name: str
    root: str
    languages: dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    total_bytes: int = 0
    key_files: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    commands: dict[str, str] = field(default_factory=dict)
    has_git: bool = False
    readme: str = ""

    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)
