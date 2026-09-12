"""The Nova agent — reasoning, tool use, progress reporting and controls.

This module is the single implementation of agent behaviour. The CLI and the
web IDE both drive :class:`NovaAgent`; they differ only in how they render the
:class:`~nova.core.models.AgentEvent` stream and how they answer approval
requests.

The model speaks a strict JSON protocol (below) rather than vendor tool-calling
APIs, which keeps nova.ai.groq interchangeable with any future provider that
can return text.

Protocol, one object per turn::

    {"thought": "...", "action": "read_file", "action_input": {"path": "a.py"}}
    {"thought": "...", "final_answer": "..."}
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

SYSTEM_PROMPT = """You are Nova, an autonomous coding agent working inside a developer's project.

You investigate the codebase with tools, then answer or change things. You are
precise, terse and honest: if you do not know something, you say so.

## Tools

- `read_file`   {"path": "relative/path.py"}          Read a text file.
- `write_file`  {"path": "relative/path.py", "content": "..."}  Create or overwrite a file.
- `list_files`  {"path": ".", "depth": 2}             List a directory.
- `search`      {"query": "text", "glob": "**/*.py"}  Search file contents.
- `run_command` {"command": "pytest -q"}              Run a shell command in the project root.
- `project_summary` {}                                Project languages, entry points, commands.

## Response format

Reply with EXACTLY ONE JSON object and nothing else — no markdown fence, no prose.

To use a tool:
{"thought": "why you are doing this", "action": "read_file", "action_input": {"path": "nova/config.py"}}

To finish:
{"thought": "why you are done", "final_answer": "your answer to the user"}

## Rules

1. One action per turn. Wait for the observation before the next step.
2. Paths are always relative to the project root.
3. Read a file before you modify it. Never invent file contents.
4. Prefer `search` and `list_files` over guessing paths.
5. After a change, verify it: run the project's tests or the code you touched.
6. Credential files (.env, keys, tokens) are unavailable. Do not try to read them.
7. Some commands require user approval and will be refused silently if denied — adapt.
8. Be concise in `final_answer`. Use short markdown; the user may be on a phone.
9. If a tool fails repeatedly, stop and explain the blocker instead of looping.
"""

MAX_OBSERVATION_CHARS = 6_000
MAX_HISTORY_MESSAGES = 20


# ---------------------------------------------------------------------------
# Response parsing
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
    """Return the first balanced ``{...}`` object in ``text``, or ``None``.

    Brace counting is string- and escape-aware, so JSON containing ``{`` inside
    a string literal does not terminate the scan early.
    """
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
    """Parse a model reply into an :class:`AgentDecision`.

    Lenient by design: small models on a phone often wrap JSON in prose or a
    markdown fence. A reply with no usable JSON is treated as a final answer
    rather than an error, so a run never dies on a formatting slip.
    """
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
        # No JSON at all — treat the whole reply as the answer.
        return AgentDecision(final=raw, raw=text, thought="")

    _, final_value = _first_present(payload, _FINAL_KEYS)
    _, action_value = _first_present(payload, _ACTION_KEYS)
    _, input_value = _first_present(payload, _INPUT_KEYS)
    _, thought_value = _first_present(payload, _THOUGHT_KEYS)

    thought = str(thought_value).strip() if isinstance(thought_value, (str, int, float)) else ""
    action_input = input_value if isinstance(input_value, dict) else {}

    # A bare action string may carry its argument, e.g. {"action": "ls -la"}.
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

    # Valid JSON, but it named neither an action nor an answer: the model
    # needs another turn rather than being taken at its word.
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
    parameters: str = ""


@dataclass
class ToolOutcome:
    """Result of running a tool."""

    ok: bool
    output: str = ""
    blocked: bool = False
    error: str | None = None

    def to_tool_result(self, name: str, duration_ms: int = 0) -> ToolResult:
        return ToolResult(
            name=name,
            ok=self.ok,
            output=self.output,
            error=self.error,
            blocked=self.blocked,
            duration_ms=duration_ms,
        )


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("read_file", "Read a UTF-8 text file from the project.", '{"path": "..."}'),
    ToolSpec("write_file", "Create or overwrite a project file.", '{"path": "...", "content": "..."}'),
    ToolSpec("list_files", "List a directory in the project.", '{"path": ".", "depth": 2}'),
    ToolSpec("search", "Search file contents.", '{"query": "...", "glob": "**/*.py"}'),
    ToolSpec("run_command", "Run a shell command in the project root.", '{"command": "..."}'),
    ToolSpec("project_summary", "Describe the project: languages, entry points, commands.", "{}"),
)

TOOL_NAMES: frozenset[str] = frozenset(spec.name for spec in TOOL_SPECS)


def render_tool_catalog() -> str:
    """Tool list rendered for the system prompt."""
    return "\n".join(f"- {s.name}{(' ' + s.parameters) if s.parameters else ''}: {s.description}" for s in TOOL_SPECS)


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
                error=f"Unknown tool {name!r}. Available tools: {', '.join(sorted(TOOL_NAMES))}",
            )
        try:
            handler = getattr(self, f"_tool_{name}")
        except AttributeError:  # pragma: no cover - spec/handler drift guard
            return ToolOutcome(ok=False, error=f"Tool {name!r} is not implemented.")

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
        except Exception as exc:  # noqa: BLE001 - a tool must never kill a run
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
    """Pause, resume, cancel and approve an in-flight agent run.

    The CLI supplies an :class:`ApprovalHandler` that prompts on stdin. The web
    IDE leaves the handler unset and resolves approvals out-of-band through
    :meth:`resolve`, which is what makes the approval modal work over HTTP.
    """

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

    # -- Cancellation ----------------------------------------------------

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation; the loop stops at the next checkpoint."""
        self._cancelled = True
        for future in list(self._pending.values()):
            if not future.done():
                future.set_result(ApprovalDecision.DENY)
        self._pending.clear()

    # -- Approvals -------------------------------------------------------

    @property
    def pending_request_id(self) -> str | None:
        """Id of the approval currently being awaited, if any."""
        for request_id, future in self._pending.items():
            if not future.done():
                return request_id
        return None

    def resolve(self, request_id: str, decision: ApprovalDecision | str) -> bool:
        """Answer a pending approval from outside the loop.

        Returns ``True`` when a waiting request was satisfied.
        """
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
        """Ask for a decision, honouring ``always`` and timeouts."""
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
            except Exception:  # noqa: BLE001 - a broken handler must not hang a run
                decision = ApprovalDecision.DENY
        else:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[ApprovalDecision] = loop.create_future()
            self._pending[request.id] = future
            try:
                decision = await asyncio.wait_for(future, timeout=self.approval_timeout)
            except asyncio.TimeoutError:
                # Nobody answered in time: fail closed.
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
            secret_values=[self.settings.groq_api_key],
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
        self.provider = provider
        self.max_steps = max_steps or self.settings.max_steps
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    # -- Prompt assembly -------------------------------------------------

    def system_message(self) -> Message:
        return Message.system(
            f"{self.system_prompt}\n\n## Available tools\n\n{render_tool_catalog()}"
        )

    def build_messages(
        self, task: str, history: Sequence[Message] | None = None
    ) -> tuple[list[Message], list[str]]:
        """Assemble the initial message list and return the chosen files."""
        context = self.context_builder.build(task)
        messages: list[Message] = [self.system_message()]
        if history:
            messages.extend(list(history)[-MAX_HISTORY_MESSAGES:])
        messages.append(
            Message.user(f"{context.text}\n\n# Task\n{task}\n\nRespond with ONE JSON object.")
        )
        return messages, context.files

    # -- Run (streaming) -------------------------------------------------

    async def stream(
        self,
        task: str,
        *,
        history: Sequence[Message] | None = None,
        controller: AgentController | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent, yielding progress events as they happen.

        The final event is always one of ``final``, ``error`` or ``cancelled``,
        so a consumer never has to guess whether the run ended.
        """
        controller = controller or AgentController()
        limit = max_steps or self.max_steps

        yield AgentEvent(
            EventType.AGENT_START,
            {
                "task": task,
                "model": self.provider.model_name,
                "project": str(self.workspace.root),
                "max_steps": limit,
                "safety_mode": str(self.safety.mode),
            },
        )

        if not getattr(self.provider, "configured", True):
            from nova.config import API_KEY_HINT

            yield AgentEvent(EventType.ERROR, {"message": API_KEY_HINT})
            return

        try:
            messages, files = self.build_messages(task, history)
        except (OSError, ValueError) as exc:
            yield AgentEvent(EventType.ERROR, {"message": f"Could not read the project: {exc}"})
            return

        if files:
            yield AgentEvent(EventType.PROGRESS, {"files": files, "percent": 5})

        steps: list[AgentStep] = []

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
                reply = await self.provider.complete(
                    [message.to_dict() for message in messages]
                )
            except AIProviderError as exc:
                yield AgentEvent(EventType.ERROR, {"message": str(exc)}, step=index)
                return
            except asyncio.CancelledError:
                yield AgentEvent(EventType.CANCELLED, {"steps": len(steps)}, step=index)
                return

            decision = parse_agent_response(reply)

            if decision.thought:
                yield AgentEvent(EventType.THOUGHT, {"text": decision.thought}, step=index)

            # -- Final answer -------------------------------------------
            if decision.is_final:
                steps.append(AgentStep(index=index, thought=decision.thought, final_answer=decision.final))
                yield AgentEvent(
                    EventType.PROGRESS, {"percent": 100}, step=index
                )
                yield AgentEvent(
                    EventType.FINAL,
                    {
                        "answer": decision.final,
                        "steps": len(steps),
                        "files": files,
                    },
                    step=index,
                )
                return

            # -- Malformed turn -----------------------------------------
            if not decision.is_action:
                steps.append(AgentStep(index=index, thought=decision.thought))
                yield AgentEvent(
                    EventType.THOUGHT,
                    {"text": "(retrying: reply was not a usable action or answer)"},
                    step=index,
                )
                messages.append(Message.assistant(reply))
                messages.append(
                    Message.user(
                        f"Your last reply was unusable ({decision.parse_error}). "
                        "Reply with exactly ONE JSON object containing either "
                        '"action" + "action_input" or "final_answer".'
                    )
                )
                continue

            action = decision.action or ""
            arguments = dict(decision.action_input)
            yield AgentEvent(
                EventType.TOOL_CALL,
                {"tool": action, "input": arguments},
                step=index,
            )

            # -- Safety gate -------------------------------------------
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
                    observation = (
                        f"REFUSED by the safety layer: {reason}. "
                        "Do not retry this action; try a different approach."
                    )
                    result = ToolResult(name=action, ok=False, blocked=True, error=reason)
                    steps.append(
                        AgentStep(
                            index=index, thought=decision.thought, action=action,
                            action_input=arguments, result=result,
                        )
                    )
                    messages.append(Message.assistant(reply))
                    messages.append(Message.user(f"Observation:\n{observation}"))
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
                    yield AgentEvent(
                        EventType.APPROVAL_REQUEST, request.to_dict(), step=index
                    )
                    decision_ = await controller.request_approval(request)
                    if controller.cancelled:
                        yield AgentEvent(
                            EventType.CANCELLED, {"steps": len(steps)}, step=index
                        )
                        return
                    yield AgentEvent(
                        EventType.APPROVAL_RESOLVED,
                        {"id": request.id, "decision": str(decision_), "tool": action},
                        step=index,
                    )
                    if decision_ == ApprovalDecision.DENY:
                        observation = (
                            f"The user DENIED permission to run {action}. "
                            "Do not retry it; choose another approach or explain the blocker."
                        )
                        result = ToolResult(
                            name=action, ok=False, blocked=True,
                            error=f"denied by user ({verdict.reason})",
                        )
                        yield AgentEvent(
                            EventType.BLOCKED,
                            {"tool": action, "reason": "denied by user"},
                            step=index,
                        )
                        steps.append(
                            AgentStep(
                                index=index, thought=decision.thought, action=action,
                                action_input=arguments, result=result,
                            )
                        )
                        messages.append(Message.assistant(reply))
                        messages.append(Message.user(f"Observation:\n{observation}"))
                        continue

            # -- Execute ------------------------------------------------
            started = asyncio.get_running_loop().time()
            outcome = await self.toolbox.execute(action, arguments)
            duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            result = outcome.to_tool_result(action, duration_ms)
            result.output = self.safety.redact(result.output)
            if result.error:
                result.error = self.safety.redact(result.error)

            steps.append(
                AgentStep(
                    index=index, thought=decision.thought, action=action,
                    action_input=arguments, result=result,
                )
            )
            yield AgentEvent(EventType.TOOL_RESULT, result.to_dict(), step=index)

            observation = result.render(limit=MAX_OBSERVATION_CHARS)
            messages.append(Message.assistant(reply))
            messages.append(Message.user(f"Observation ({action}):\n{observation}"))

        # -- Step budget exhausted --------------------------------------
        best = next(
            (s.final_answer for s in reversed(steps) if s.final_answer), None
        )
        answer = best or (
            f"Stopped after the maximum of {limit} steps without a final answer.\n\n"
            + self._progress_digest(steps)
        )
        yield AgentEvent(
            EventType.FINAL,
            {"answer": answer, "steps": len(steps), "limit_reached": True, "files": files},
            step=limit,
        )

    # -- Run (collecting) ------------------------------------------------

    async def run(
        self,
        task: str,
        *,
        history: Sequence[Message] | None = None,
        controller: AgentController | None = None,
        max_steps: int | None = None,
    ) -> AgentResult:
        """Run to completion and return an :class:`AgentResult`.

        A thin collector over :meth:`stream` — it records steps and terminal
        state, but rendering stays the caller's job.
        """
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
            elif event.type == EventType.FINAL:
                result.answer = str(event.data.get("answer", ""))
                result.status = AgentStatus.DONE
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
        """Blocking convenience wrapper (used by the CLI)."""
        return asyncio.run(self.run(task, **kwargs))

    # -- Helpers ---------------------------------------------------------

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
    """Construct a fully wired agent — the shared factory for CLI and web."""
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
    "parse_agent_response",
    "render_tool_catalog",
]
