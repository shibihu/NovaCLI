"""The Nova agent — reasoning, tool use, progress reporting and controls.

This module is the single implementation of agent behaviour. The CLI and the
web IDE both drive :class:`NovaAgent`; they differ only in how they render the
:class:`~nova.core.models.AgentEvent` stream and how they answer approval
requests.

Supports both native Groq/OpenAI tool calling and text JSON protocol fallback.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from nova.ai import AIProvider, AIProviderError
from nova.config import Settings, load_settings
from nova.workspace.files import Workspace
from nova.workspace.projects import ProjectAnalyzer
from nova.core.checkpoints import CheckpointManager

from .context import ContextBuilder
from .models import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    AgentStep,
    ApprovalDecision,
    ApprovalRequest,
    EventType,
    Message,
    RiskLevel,
    ToolResult,
    utc_now_iso,
)
from .runner import CommandRunner
from .safety import SafetyError, SafetyPolicy, SafetyVerdict

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Nova, an autonomous coding agent working inside a developer's project with access to real workspace tools.

When the user asks you to inspect files, execute commands, read files, write files, search files, or perform another operation that an available tool can perform, use the appropriate tool directly to perform the task. Do not merely explain how to perform the operation or output code instructions when a tool can do it.

Never invent, guess, or fabricate file contents, command output, or tool results.
Never claim that a file was created or modified unless a real tool call was executed and returned success.
Only report filesystem or command output returned by a real tool.

Keep code, commands, filenames, API names, and technical identifiers unchanged.

## Rules

1. Inspect before you modify. Read files before editing them. Never invent file contents.
2. Paths are always relative to the project root.
3. Prefer searching or listing directory contents over guessing paths.
4. After making changes, verify them when possible (run tests or review the code).
5. Credential files (.env, keys, tokens, SSH keys) are unavailable. Do not attempt to read or create them.
6. Some commands require user approval and will be refused if denied — adapt accordingly.
7. Be concise in your final responses. Use short markdown formatting.
8. If a tool fails repeatedly, stop and explain the blocker instead of looping."""

MAX_OBSERVATION_CHARS = 6_000
MAX_HISTORY_MESSAGES = 20


# ---------------------------------------------------------------------------
# Response parsing (for text JSON fallback)
# ---------------------------------------------------------------------------


@dataclass
class AgentDecision:
    """A parsed model turn."""

    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final: str | None = None
    raw: str = ""
    parse_error: str | None = None

    @property
    def is_action(self) -> bool:
        return self.action is not None and self.final is None

    @property
    def is_final(self) -> bool:
        return self.final is not None


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_ACTION_KEYS = ("action", "tool", "tool_name", "name", "command")
_INPUT_KEYS = ("action_input", "input", "args", "arguments", "parameters", "params")
_FINAL_KEYS = ("final_answer", "final", "answer", "response", "done", "result", "message")
_THOUGHT_KEYS = ("thought", "reasoning", "thinking", "plan")


def extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` object in ``text``, or ``None``."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _first_present(payload: dict[str, Any], keys: Sequence[str]) -> tuple[str, Any]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return key, payload[key]
    return "", None


def parse_agent_response(text: str) -> AgentDecision:
    """Parse a text model reply into an :class:`AgentDecision`."""
    raw = (text or "").strip()
    if not raw:
        return AgentDecision(raw=text, parse_error="empty response")

    candidate = _FENCE_RE.sub("", raw).strip()
    blob = extract_json_object(candidate)

    payload: dict[str, Any] | None = None
    if blob:
        try:
            loaded = json.loads(blob)
            if isinstance(loaded, dict):
                payload = loaded
        except (json.JSONDecodeError, ValueError):
            payload = None

    if payload is None:
        return AgentDecision(final=raw, raw=text, thought="")

    _, final_value = _first_present(payload, _FINAL_KEYS)
    _, action_value = _first_present(payload, _ACTION_KEYS)
    _, input_value = _first_present(payload, _INPUT_KEYS)
    _, thought_value = _first_present(payload, _THOUGHT_KEYS)

    thought = str(thought_value).strip() if isinstance(thought_value, (str, int, float)) else ""
    action_input = input_value if isinstance(input_value, dict) else {}

    action: str | None = None
    if isinstance(action_value, str) and action_value.strip():
        action = action_value.strip()
        if " " in action and not action_input:
            action, _, remainder = action.partition(" ")
            action_input = {"command": remainder.strip()}
        elif action == "run_command" and not action_input and isinstance(payload.get("command"), str):
            action_input = {"command": payload["command"]}

    if final_value is not None and not action:
        if isinstance(final_value, (dict, list)):
            final_text = json.dumps(final_value, indent=2)
        else:
            final_text = str(final_value).strip()
        if final_text:
            return AgentDecision(thought=thought, final=final_text, raw=text)

    if action:
        return AgentDecision(
            thought=thought, action=action, action_input=action_input, raw=text
        )

    return AgentDecision(
        thought=thought,
        raw=text,
        parse_error="JSON object contained neither an action nor a final_answer",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of an agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolOutcome:
    """Result of running a tool."""

    ok: bool
    output: str = ""
    blocked: bool = False
    error: str | None = None

    def to_tool_result(self, name: str, duration_ms: int = 0, tool_call_id: str | None = None) -> ToolResult:
        return ToolResult(
            name=name,
            ok=self.ok,
            output=self.output,
            error=self.error,
            blocked=self.blocked,
            duration_ms=duration_ms,
            tool_call_id=tool_call_id,
        )


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "read_file",
        "Read a UTF-8 text file from the project.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative file path to read"}},
            "required": ["path"],
        },
    ),
    ToolSpec(
        "write_file",
        "Create or overwrite a project file.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path to write"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        "list_files",
        "List a directory in the project.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path (default '.')"},
                "depth": {"type": "integer", "description": "Recursion depth (default 1)"},
            },
            "required": [],
        },
    ),
    ToolSpec(
        "search",
        "Search file contents.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text or regex query to search"},
                "glob": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        "run_command",
        "Run a shell command in the project root.",
        {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
            "required": ["command"],
        },
    ),
    ToolSpec(
        "project_summary",
        "Describe the project: languages, entry points, commands.",
        {"type": "object", "properties": {}, "required": []},
    ),
)

TOOL_NAMES: frozenset[str] = frozenset(spec.name for spec in TOOL_SPECS)


def render_tool_catalog() -> str:
    """Tool list rendered for legacy text fallback."""
    return "\n".join(f"- {s.name}: {s.description}" for s in TOOL_SPECS)


def get_native_tools_schema() -> list[dict[str, Any]]:
    """Convert TOOL_SPECS into native OpenAI/Groq tool schema format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in TOOL_SPECS
    ]


class ToolBox:
    """Executes agent tools against the workspace and runner."""

    def __init__(
        self,
        workspace: Workspace,
        runner: CommandRunner,
        *,
        analyzer: ProjectAnalyzer | None = None,
        safety: SafetyPolicy | None = None,
    ) -> None:
        self.workspace = workspace
        self.runner = runner
        self.analyzer = analyzer or ProjectAnalyzer(workspace)
        self.safety = safety or workspace.safety

    def spec(self, name: str) -> ToolSpec | None:
        for spec in TOOL_SPECS:
            if spec.name == name:
                return spec
        return None

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolOutcome:
        """Run a tool, converting every failure into a ToolOutcome."""
        args = arguments or {}
        if self.spec(name) is None:
            return ToolOutcome(
                ok=False,
                error=f"Tool execution failed: unknown tool {name!r}. Available tools: {', '.join(sorted(TOOL_NAMES))}",
            )
        try:
            handler = getattr(self, f"_tool_{name}")
        except AttributeError:
            return ToolOutcome(ok=False, error=f"Tool execution failed: {name!r} is not implemented.")

        try:
            result = handler(args)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except SafetyError as exc:
            return ToolOutcome(ok=False, blocked=True, error=str(exc))
        except (OSError, ValueError) as exc:
            return ToolOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")

    # -- Handlers --------------------------------------------------------

    def _tool_read_file(self, args: dict[str, Any]) -> ToolOutcome:
        path = str(args.get("path") or "").strip()
        if not path:
            return ToolOutcome(ok=False, error="read_file requires a 'path'.")
        content = self.workspace.read_text(path, max_bytes=200_000)
        return ToolOutcome(ok=True, output=self.safety.redact(content))

    def _tool_write_file(self, args: dict[str, Any]) -> ToolOutcome:
        path = str(args.get("path") or "").strip()
        if not path:
            return ToolOutcome(ok=False, error="write_file requires a 'path'.")
        if "content" not in args:
            return ToolOutcome(ok=False, error="write_file requires 'content'.")
        content = args.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2)

        existed = self.workspace.exists(path)
        entry = self.workspace.write_text(path, content)
        verb = "Updated" if existed else "Created"
        return ToolOutcome(
            ok=True,
            output=f"{verb} {entry.path} ({entry.size} bytes, {content.count(chr(10)) + 1} lines).",
        )

    def _tool_list_files(self, args: dict[str, Any]) -> ToolOutcome:
        path = str(args.get("path") or ".")
        depth = int(args.get("depth") or 1)
        if depth <= 1:
            entries = self.workspace.list_dir(path)
            if not entries:
                return ToolOutcome(ok=True, output=f"{path}: (empty)")
            lines = [
                f"{'[dir] ' if e.is_dir else '      '}{e.path}" for e in entries[:200]
            ]
            return ToolOutcome(ok=True, output="\n".join(lines))
        tree = self.workspace.tree(path, max_depth=min(depth, 6), max_entries=300)
        return ToolOutcome(ok=True, output=tree)

    def _tool_search(self, args: dict[str, Any]) -> ToolOutcome:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolOutcome(ok=False, error="search requires a 'query'.")
        glob = args.get("glob")
        hits = self.workspace.search(
            query,
            glob=str(glob) if glob else None,
            max_results=int(args.get("max_results") or 40),
            regex=bool(args.get("regex")),
        )
        if not hits:
            return ToolOutcome(ok=True, output=f"No matches for {query!r}.")
        body = "\n".join(hit.render() for hit in hits)
        return ToolOutcome(ok=True, output=self.safety.redact(body))

    async def _tool_run_command(self, args: dict[str, Any]) -> ToolOutcome:
        command = str(args.get("command") or "").strip()
        if not command:
            return ToolOutcome(ok=False, error="run_command requires a 'command'.")
        result = await self.runner.run(command, approved=True)
        if result.blocked:
            return ToolOutcome(ok=False, blocked=True, error=result.reason)
        return ToolOutcome(ok=result.ok, output=result.render(), error=None if result.ok else result.render())

    def _tool_project_summary(self, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(ok=True, output=self.analyzer.render_summary())


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

ApprovalHandler = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]


class AgentController:
    """Pause, resume, cancel and approve an in-flight agent run."""

    def __init__(
        self,
        approval_handler: ApprovalHandler | None = None,
        *,
        auto_approve: bool = False,
        approval_timeout: float = 300.0,
    ) -> None:
        self._approval_handler = approval_handler
        self.auto_approve = auto_approve
        self.approval_timeout = approval_timeout
        self._cancelled = False
        self._always_allow: set[str] = set()
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        for future in list(self._pending.values()):
            if not future.done():
                future.set_result(ApprovalDecision.DENY)
        self._pending.clear()

    @property
    def pending_request_id(self) -> str | None:
        for request_id, future in self._pending.items():
            if not future.done():
                return request_id
        return None

    def resolve(self, request_id: str, decision: ApprovalDecision | str) -> bool:
        try:
            resolved = ApprovalDecision(decision)
        except ValueError:
            resolved = ApprovalDecision.DENY
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(resolved)
        return True

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        if self.auto_approve or request.tool in self._always_allow:
            return ApprovalDecision.APPROVE
        if self._cancelled:
            return ApprovalDecision.DENY

        decision: ApprovalDecision
        if self._approval_handler is not None:
            try:
                decision = await self._approval_handler(request)
            except (EOFError, KeyboardInterrupt):
                decision = ApprovalDecision.DENY
            except asyncio.CancelledError:
                raise
            except Exception:
                decision = ApprovalDecision.DENY
        else:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[ApprovalDecision] = loop.create_future()
            self._pending[request.id] = future
            try:
                decision = await asyncio.wait_for(future, timeout=self.approval_timeout)
            except asyncio.TimeoutError:
                decision = ApprovalDecision.DENY
            except asyncio.CancelledError:
                self._pending.pop(request.id, None)
                raise
            finally:
                self._pending.pop(request.id, None)

        try:
            decision = ApprovalDecision(decision)
        except ValueError:
            decision = ApprovalDecision.DENY
        if decision == ApprovalDecision.ALWAYS:
            self._always_allow.add(request.tool)
            return ApprovalDecision.APPROVE
        return decision


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class NovaAgent:
    """The reasoning loop shared by the CLI and the web IDE."""

    def __init__(
        self,
        provider: AIProvider,
        settings: Settings | None = None,
        *,
        workspace: Workspace | None = None,
        safety: SafetyPolicy | None = None,
        runner: CommandRunner | None = None,
        analyzer: ProjectAnalyzer | None = None,
        context_builder: ContextBuilder | None = None,
        toolbox: ToolBox | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        root = self.settings.project_root
        self.safety = safety or SafetyPolicy(
            root,
            self.settings.safety_mode,
            secret_values=self.settings.active_secrets,
        )
        self.workspace = workspace or Workspace(
            root, safety=self.safety, extra_ignore=self.settings.extra_ignore
        )
        self.runner = runner or CommandRunner(
            root, self.settings.command_timeout, safety=self.safety
        )
        self.analyzer = analyzer or ProjectAnalyzer(self.workspace)
        self.context_builder = context_builder or ContextBuilder(
            self.workspace,
            analyzer=self.analyzer,
            safety=self.safety,
            max_chars=self.settings.max_context_chars,
        )
        self.toolbox = toolbox or ToolBox(
            self.workspace, self.runner, analyzer=self.analyzer, safety=self.safety
        )
        self.checkpoint_manager = CheckpointManager(self.workspace)
        self.provider = provider
        self.max_steps = max_steps or self.settings.max_steps
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def system_message(self) -> Message:
        return Message.system(self.system_prompt)

    def build_messages(
        self, task: str, history: Sequence[Message] | None = None
    ) -> tuple[list[Message], list[str]]:
        context = self.context_builder.build(task)
        messages: list[Message] = [self.system_message()]
        if history:
            messages.extend(list(history)[-MAX_HISTORY_MESSAGES:])
        messages.append(
            Message.user(f"{context.text}\n\n# Task\n{task}")
        )
        return messages, context.files

    async def stream(
        self,
        task: str,
        *,
        history: Sequence[Message] | None = None,
        controller: AgentController | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        controller = controller or AgentController()
        limit = max_steps or self.max_steps

        checkpoint = self.checkpoint_manager.create(task_id=task[:20])
        checkpoint_id = checkpoint.id

        yield AgentEvent(
            EventType.AGENT_START,
            {
                "task": task,
                "model": self.provider.model_name,
                "project": str(self.workspace.root),
                "max_steps": limit,
                "safety_mode": str(self.safety.mode),
                "checkpoint_id": checkpoint_id,
            },
        )

        if not getattr(self.provider, "configured", True):
            from nova.config import get_api_key_hint

            yield AgentEvent(EventType.ERROR, {"message": get_api_key_hint(self.settings.provider)})
            return

        try:
            messages, files = self.build_messages(task, history)
        except (OSError, ValueError) as exc:
            yield AgentEvent(EventType.ERROR, {"message": f"Could not read the project: {exc}"})
            return

        if files:
            yield AgentEvent(EventType.PROGRESS, {"files": files, "percent": 5})

        steps: list[AgentStep] = []
        tools_schema = get_native_tools_schema()

        for index in range(1, limit + 1):
            if controller.cancelled:
                yield AgentEvent(EventType.CANCELLED, {"steps": len(steps)}, step=index)
                return

            yield AgentEvent(EventType.STEP_START, {"index": index, "of": limit}, step=index)
            yield AgentEvent(
                EventType.PROGRESS,
                {"percent": int(5 + 90 * (index - 1) / max(1, limit)), "step": index},
                step=index,
            )

            try:
                ai_response = await self.provider.complete(
                    [m.to_dict() for m in messages],
                    tools=tools_schema,
                    tool_choice="auto",
                )
            except AIProviderError as exc:
                yield AgentEvent(EventType.ERROR, {"message": str(exc)}, step=index)
                return
            except asyncio.CancelledError:
                yield AgentEvent(EventType.CANCELLED, {"steps": len(steps)}, step=index)
                return

            # Native tool calling path
            if ai_response.has_tool_calls:
                assistant_tool_calls = []
                for tc in ai_response.tool_calls:
                    tc_dict = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments if tc.raw_arguments else json.dumps(tc.arguments or {}),
                        },
                    }
                    if tc.provider_data:
                        for k, v in tc.provider_data.items():
                            if v is not None:
                                tc_dict[k] = v
                                if k == "thought_signature" and isinstance(tc_dict.get("function"), dict):
                                    tc_dict["function"]["thought_signature"] = v
                    assistant_tool_calls.append(tc_dict)

                messages.append(Message.assistant(content=ai_response.text or None, tool_calls=assistant_tool_calls))

                for tc in ai_response.tool_calls:
                    action = tc.name
                    arguments = tc.arguments
                    tool_call_id = tc.id

                    yield AgentEvent(
                        EventType.TOOL_CALL,
                        {"tool": action, "input": arguments if arguments is not None else {"raw": tc.raw_arguments}},
                        step=index,
                    )

                    # Handle malformed arguments
                    if arguments is None:
                        reason = f"Tool argument parsing failed: invalid JSON arguments: {tc.raw_arguments!r}"
                        yield AgentEvent(
                            EventType.BLOCKED,
                            {"tool": action, "reason": reason, "input": {"raw": tc.raw_arguments}},
                            step=index,
                        )
                        result = ToolResult(
                            name=action, ok=False, blocked=True, error=reason, tool_call_id=tool_call_id
                        )
                        steps.append(
                            AgentStep(
                                index=index, thought="", action=action,
                                action_input={"raw": tc.raw_arguments}, result=result,
                            )
                        )
                        messages.append(Message.tool_result(tool_call_id, action, f"Tool argument parsing failed: invalid JSON arguments."))
                        continue

                    # Safety check
                    verdict: SafetyVerdict | None = None
                    if action in TOOL_NAMES:
                        verdict = self.safety.check_tool_call(action, arguments)
                        if verdict.level == RiskLevel.FORBIDDEN or (
                            not verdict.allowed and not verdict.requires_approval
                        ):
                            reason = verdict.reason or "blocked by safety policy"
                            yield AgentEvent(
                                EventType.BLOCKED,
                                {"tool": action, "reason": reason, "input": arguments},
                                step=index,
                            )
                            result = ToolResult(
                                name=action, ok=False, blocked=True, error=reason, tool_call_id=tool_call_id
                            )
                            steps.append(
                                AgentStep(
                                    index=index, thought="", action=action,
                                    action_input=arguments, result=result,
                                )
                            )
                            obs = f"REFUSED by safety layer: {reason}"
                            messages.append(Message.tool_result(tool_call_id, action, obs))
                            continue

                        if verdict.requires_approval and not controller.auto_approve:
                            request = ApprovalRequest(
                                id=uuid.uuid4().hex[:12],
                                tool=action,
                                summary=self._approval_summary(action, arguments),
                                detail=self._approval_detail(action, arguments),
                                level=verdict.level,
                                reason=verdict.reason,
                            )
                            yield AgentEvent(EventType.APPROVAL_REQUEST, request.to_dict(), step=index)
                            decision_ = await controller.request_approval(request)
                            if controller.cancelled:
                                yield AgentEvent(EventType.CANCELLED, {"steps": len(steps)}, step=index)
                                return
                            yield AgentEvent(
                                EventType.APPROVAL_RESOLVED,
                                {"id": request.id, "decision": str(decision_), "tool": action},
                                step=index,
                            )
                            if decision_ == ApprovalDecision.DENY:
                                reason = f"denied by user ({verdict.reason})"
                                yield AgentEvent(EventType.BLOCKED, {"tool": action, "reason": "denied by user"}, step=index)
                                result = ToolResult(
                                    name=action, ok=False, blocked=True, error=reason, tool_call_id=tool_call_id
                                )
                                steps.append(
                                    AgentStep(
                                        index=index, thought="", action=action,
                                        action_input=arguments, result=result,
                                    )
                                )
                                messages.append(Message.tool_result(tool_call_id, action, f"DENIED by user: {reason}"))
                                continue

                    # Execute tool
                    started = asyncio.get_running_loop().time()
                    outcome = await self.toolbox.execute(action, arguments)
                    duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
                    result = outcome.to_tool_result(action, duration_ms, tool_call_id=tool_call_id)
                    result.output = self.safety.redact(result.output)
                    if result.error:
                        result.error = self.safety.redact(result.error)

                    steps.append(
                        AgentStep(
                            index=index, thought="", action=action,
                            action_input=arguments, result=result,
                        )
                    )
                    yield AgentEvent(EventType.TOOL_RESULT, result.to_dict(), step=index)

                    obs = result.render(limit=MAX_OBSERVATION_CHARS)
                    messages.append(Message.tool_result(tool_call_id, action, obs))

                continue

            # Fallback or final answer text
            reply_text = ai_response.text or ""
            decision = parse_agent_response(reply_text)

            if decision.thought:
                yield AgentEvent(EventType.THOUGHT, {"text": decision.thought}, step=index)

            if decision.is_final:
                steps.append(AgentStep(index=index, thought=decision.thought, final_answer=decision.final))
                yield AgentEvent(EventType.PROGRESS, {"percent": 100}, step=index)
                inspection = self.checkpoint_manager.inspect(checkpoint_id)
                changed_files = inspection.get("changed_files", [])
                yield AgentEvent(
                    EventType.FINAL,
                    {
                        "answer": decision.final,
                        "steps": len(steps),
                        "files": files,
                        "checkpoint_id": checkpoint_id,
                        "changed_files": changed_files,
                    },
                    step=index,
                )
                return

            if decision.is_action:
                # Model returned text JSON instead of native tool calls
                action = decision.action or ""
                arguments = dict(decision.action_input)
                yield AgentEvent(EventType.TOOL_CALL, {"tool": action, "input": arguments}, step=index)

                verdict: SafetyVerdict | None = None
                if action in TOOL_NAMES:
                    verdict = self.safety.check_tool_call(action, arguments)
                    if verdict.level == RiskLevel.FORBIDDEN or (
                        not verdict.allowed and not verdict.requires_approval
                    ):
                        reason = verdict.reason or "blocked by safety policy"
                        yield AgentEvent(EventType.BLOCKED, {"tool": action, "reason": reason, "input": arguments}, step=index)
                        result = ToolResult(name=action, ok=False, blocked=True, error=reason)
                        steps.append(AgentStep(index=index, thought=decision.thought, action=action, action_input=arguments, result=result))
                        messages.append(Message.assistant(reply_text))
                        messages.append(Message.user(f"Observation:\nREFUSED: {reason}"))
                        continue

                started = asyncio.get_running_loop().time()
                outcome = await self.toolbox.execute(action, arguments)
                duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
                result = outcome.to_tool_result(action, duration_ms)
                result.output = self.safety.redact(result.output)
                if result.error:
                    result.error = self.safety.redact(result.error)

                steps.append(AgentStep(index=index, thought=decision.thought, action=action, action_input=arguments, result=result))
                yield AgentEvent(EventType.TOOL_RESULT, result.to_dict(), step=index)
                obs = result.render(limit=MAX_OBSERVATION_CHARS)
                messages.append(Message.assistant(reply_text))
                messages.append(Message.user(f"Observation ({action}):\n{obs}"))
                continue

            # Unusable response
            steps.append(AgentStep(index=index, thought=decision.thought))
            messages.append(Message.assistant(reply_text))
            messages.append(
                Message.user(
                    "Your last response was unusable. Please perform a tool call or give a final answer."
                )
            )

        best = next((s.final_answer for s in reversed(steps) if s.final_answer), None)
        answer = best or (
            f"Stopped after the maximum of {limit} steps without a final answer.\n\n"
            + self._progress_digest(steps)
        )
        inspection = self.checkpoint_manager.inspect(checkpoint_id)
        changed_files = inspection.get("changed_files", [])
        yield AgentEvent(
            EventType.FINAL,
            {
                "answer": answer,
                "steps": len(steps),
                "limit_reached": True,
                "files": files,
                "checkpoint_id": checkpoint_id,
                "changed_files": changed_files,
            },
            step=limit,
        )

    async def run(
        self,
        task: str,
        *,
        history: Sequence[Message] | None = None,
        controller: AgentController | None = None,
        max_steps: int | None = None,
    ) -> AgentResult:
        result = AgentResult(
            task=task, model=self.provider.model_name, status=AgentStatus.RUNNING
        )
        steps: dict[int, AgentStep] = {}

        async for event in self.stream(
            task, history=history, controller=controller, max_steps=max_steps
        ):
            if event.type == EventType.STEP_START:
                steps.setdefault(event.step, AgentStep(index=event.step))
            elif event.type == EventType.THOUGHT:
                steps.setdefault(event.step, AgentStep(index=event.step)).thought = str(
                    event.data.get("text", "")
                )
            elif event.type == EventType.TOOL_CALL:
                step = steps.setdefault(event.step, AgentStep(index=event.step))
                step.action = str(event.data.get("tool", ""))
                step.action_input = dict(event.data.get("input") or {})
            elif event.type == EventType.TOOL_RESULT:
                step = steps.setdefault(event.step, AgentStep(index=event.step))
                step.result = ToolResult(
                    name=str(event.data.get("name", "")),
                    ok=bool(event.data.get("ok")),
                    output=str(event.data.get("output", "")),
                    error=event.data.get("error"),
                    blocked=bool(event.data.get("blocked")),
                    duration_ms=int(event.data.get("duration_ms") or 0),
                )
            elif event.type == EventType.BLOCKED:
                step = steps.setdefault(event.step, AgentStep(index=event.step))
                step.action = str(event.data.get("tool", ""))
                step.action_input = dict(event.data.get("input") or {})
                step.result = ToolResult(
                    name=step.action,
                    ok=False,
                    blocked=True,
                    error=str(event.data.get("reason", "")),
                )
            elif event.type == EventType.AGENT_START:
                result.checkpoint_id = event.data.get("checkpoint_id")
            elif event.type == EventType.FINAL:
                result.answer = str(event.data.get("answer", ""))
                result.status = AgentStatus.DONE
                result.checkpoint_id = str(event.data.get("checkpoint_id", "") or result.checkpoint_id)
                result.changed_files = list(event.data.get("changed_files") or [])
                steps.setdefault(event.step, AgentStep(index=event.step)).final_answer = result.answer
            elif event.type == EventType.ERROR:
                result.error = str(event.data.get("message", "unknown error"))
                result.status = AgentStatus.ERROR
            elif event.type == EventType.CANCELLED:
                result.status = AgentStatus.CANCELLED
                result.answer = result.answer or "Cancelled before completion."

        result.steps = [steps[key] for key in sorted(steps)]
        result.finished_at = utc_now_iso()
        return result

    def run_sync(self, task: str, **kwargs: Any) -> AgentResult:
        return asyncio.run(self.run(task, **kwargs))

    def _approval_summary(self, action: str, arguments: dict[str, Any]) -> str:
        if action == "run_command":
            return f"Run command: {arguments.get('command', '')}"
        if action == "write_file":
            path = arguments.get("path", "?")
            content = arguments.get("content")
            size = len(content) if isinstance(content, str) else 0
            return f"Write {size} bytes to {path}"
        return f"{action} {arguments}"

    def _approval_detail(self, action: str, arguments: dict[str, Any]) -> str:
        if action == "write_file":
            content = arguments.get("content")
            if isinstance(content, str):
                preview = self.safety.redact(content[:1_500])
                return preview + ("\n..." if len(content) > 1_500 else "")
        if action == "run_command":
            return str(arguments.get("command", ""))
        return json.dumps(arguments, indent=2)[:1_500]

    @staticmethod
    def _progress_digest(steps: Sequence[AgentStep]) -> str:
        lines: list[str] = ["Here is what was accomplished:"]
        for step in steps:
            if step.result:
                status = "ok" if step.result.ok else ("blocked" if step.result.blocked else "failed")
                lines.append(f"- step {step.index}: {step.action} [{status}]")
            elif step.action:
                lines.append(f"- step {step.index}: {step.action} [no result]")
        return "\n".join(lines)


def build_agent(
    settings: Settings | None = None,
    *,
    provider: AIProvider | None = None,
    **kwargs: Any,
) -> NovaAgent:
    settings = settings or load_settings()
    if provider is None:
        from nova.ai import get_provider

        provider = get_provider(settings)
    return NovaAgent(provider, settings, **kwargs)


__all__ = [
    "AgentController",
    "AgentDecision",
    "NovaAgent",
    "SYSTEM_PROMPT",
    "TOOL_NAMES",
    "TOOL_SPECS",
    "ToolBox",
    "ToolOutcome",
    "ToolSpec",
    "build_agent",
    "extract_json_object",
    "get_native_tools_schema",
    "parse_agent_response",
    "render_tool_catalog",
]
