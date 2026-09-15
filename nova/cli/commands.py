"""NovaCLI command line.

Rendering only: every subcommand drives Nova Core
(:class:`~nova.core.agent.NovaAgent`) and formats the resulting event stream.
Keeping the logic in core is what guarantees the CLI and the web IDE behave
identically.

Designed for a phone first: short output, no giant tables, and colours that
switch off automatically when the terminal (like Termux's default) cannot
handle them.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from nova import __version__
from nova.ai import normalize_provider_name, provider_names
from nova.config import (
    DEFAULT_USER_CONFIG_PATH,
    DEFAULT_USER_CREDENTIALS_PATH,
    ConfigError,
    NovaConfigStore,
    Settings,
    get_api_key_hint,
    load_settings,
    prompt_and_save_api_key,
)
from nova.core.agent import AgentController, build_agent
from nova.core.models import (
    ApprovalDecision,
    ApprovalRequest,
    EventType,
    Message,
    RiskLevel,
)
from nova.core.runner import CommandRunner
from nova.core.safety import SafetyError, SafetyPolicy
from nova.workspace.files import Workspace
from nova.workspace.projects import ProjectAnalyzer
from nova.intelligence.cache import IntelligenceCache

# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------

_COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


class Console:
    """Minimal, dependency-free styled output."""

    def __init__(self, color: bool | None = None, *, stream: Any = None) -> None:
        if color is None:
            color = (
                stream is None or stream is sys.stdout
            ) and sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ
        self.color = bool(color)
        self.stream = stream or sys.stdout

    def paint(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        prefix = "".join(_COLORS.get(style, "") for style in styles)
        return f"{prefix}{text}{_COLORS['reset']}"

    def write(self, text: str = "", *, end: str = "\n") -> None:
        print(text, end=end, file=self.stream, flush=True)

    def rule(self, label: str = "") -> None:
        width = min(shutil.get_terminal_size((60, 24)).columns, 60)
        if label:
            self.write(self.paint(f"── {label} ".ljust(width, "─"), "dim"))
        else:
            self.write(self.paint("─" * width, "dim"))

    def info(self, text: str) -> None:
        self.write(f"{self.paint('•', 'cyan')} {text}")

    def ok(self, text: str) -> None:
        self.write(f"{self.paint('✓', 'green')} {text}")

    def warn(self, text: str) -> None:
        self.write(f"{self.paint('!', 'yellow')} {text}")

    def error(self, text: str) -> None:
        self.write(f"{self.paint('✗', 'red')} {text}", end="\n")

    def title(self, text: str) -> None:
        self.write(self.paint(text, "bold", "magenta"))


RISK_STYLES = {
    RiskLevel.SAFE: "green",
    RiskLevel.MODERATE: "yellow",
    RiskLevel.DANGEROUS: "yellow",
    RiskLevel.FORBIDDEN: "red",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_workspace(settings: Settings) -> Workspace:
    safety = SafetyPolicy(
        settings.project_root,
        settings.safety_mode,
        secret_values=settings.active_secrets,
    )
    return Workspace(
        settings.project_root, safety=safety, extra_ignore=settings.extra_ignore
    )


def cli_approval_handler(console: Console, *, auto_yes: bool):
    """Build an approval handler that prompts on the terminal.

    Non-interactive runs (pipes, CI) deny rather than hang — failing closed.
    """

    async def handler(request: ApprovalRequest) -> ApprovalDecision:
        if request.level == RiskLevel.FORBIDDEN:
            return ApprovalDecision.DENY

        console.write()
        console.write(console.paint(f"  ⚠ approval needed — {request.summary}", "yellow", "bold"))
        if request.reason:
            console.write(console.paint(f"    reason: {request.reason}", "dim"))
        if request.detail:
            preview = "\n".join(f"    {line}" for line in request.detail.splitlines()[:12])
            console.write(console.paint(preview, "dim"))

        if auto_yes:
            console.write(console.paint("    auto-approved (--yes)", "dim"))
            return ApprovalDecision.APPROVE
        if not sys.stdin.isatty():
            console.write(console.paint("    no terminal available → denied", "dim"))
            return ApprovalDecision.DENY

        try:
            answer = await asyncio.to_thread(
                input, "    approve? [y]es / [n]o / [a]lways: "
            )
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENY

        choice = (answer or "").strip().lower()
        if choice in {"y", "yes"}:
            return ApprovalDecision.APPROVE
        if choice in {"a", "always"}:
            return ApprovalDecision.ALWAYS
        return ApprovalDecision.DENY

    return handler


def _print_event(console: Console, event: Any, *, verbose: bool) -> None:
    """Render one agent event to the terminal."""
    data = event.data
    kind = event.type

    if kind == EventType.AGENT_START:
        console.write()
        console.write(
            console.paint(f"◆ Nova · {data.get('model')} · {data.get('safety_mode')} mode", "magenta")
        )
        console.write(console.paint(f"  workspace {data.get('project')}", "dim"))
    elif kind == EventType.THOUGHT:
        text = str(data.get("text", "")).strip()
        if text:
            console.write(f"  {console.paint('▸', 'cyan')} {console.paint(text, 'dim')}")
    elif kind == EventType.TOOL_CALL:
        tool = data.get("tool")
        console.write(f"  {console.paint('🔧', 'blue')} {console.paint(str(tool), 'bold')} {console.paint(_preview(data.get('input')), 'dim')}")
    elif kind == EventType.TOOL_RESULT:
        ok = bool(data.get("ok"))
        mark = console.paint("✓", "green") if ok else console.paint("✗", "red")
        summary = _result_summary(data)
        console.write(f"    {mark} {console.paint(summary, 'dim')}")
        if verbose and data.get("output"):
            for line in str(data["output"]).splitlines()[:20]:
                console.write(console.paint(f"      {line}", "dim"))
    elif kind == EventType.BLOCKED:
        console.write(
            f"    {console.paint('⛔', 'red')} blocked: {console.paint(str(data.get('reason')), 'red')}"
        )
    elif kind == EventType.APPROVAL_RESOLVED:
        decision = str(data.get("decision"))
        style = "green" if decision == "approve" else "yellow"
        console.write(f"    {console.paint('→', style)} {console.paint(decision, style)}")
    elif kind == EventType.FINAL:
        console.write()
        console.rule("answer")
        console.write(str(data.get("answer", "")).strip())
        console.write()
        console.write(console.paint(f"  ({data.get('steps', 0)} steps)", "dim"))
    elif kind == EventType.ERROR:
        console.write()
        console.error(str(data.get("message", "unknown error")))
    elif kind == EventType.CANCELLED:
        console.write()
        console.warn("Cancelled.")


def _preview(value: Any, limit: int = 90) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _result_summary(data: dict[str, Any]) -> str:
    if data.get("blocked"):
        return f"blocked: {data.get('error') or 'policy'}"
    if not data.get("ok"):
        return f"failed: {_preview(data.get('error') or '', 100)}"
    output = str(data.get("output", ""))
    first = output.strip().splitlines()[0] if output.strip() else "done"
    return _preview(first, 100)


async def _run_streaming(
    agent: Any,
    task: str,
    console: Console,
    *,
    controller: AgentController,
    history: Sequence[Message] | None = None,
    verbose: bool = False,
) -> tuple[str, list[Message], bool]:
    answer = ""
    failed = False
    async for event in agent.stream(task, history=history, controller=controller):
        _print_event(console, event, verbose=verbose)
        if event.type == EventType.FINAL:
            answer = str(event.data.get("answer", ""))
        elif event.type == EventType.ERROR:
            failed = True

    updated = list(history or [])
    updated.append(Message.user(task))
    if answer:
        updated.append(Message.assistant(answer))
    return answer, updated, failed


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace, console: Console) -> int:
    """One-shot task for the agent."""
    settings = _settings_from_args(args)
    task = " ".join(args.task).strip()
    if not task:
        console.error("Nothing to do — provide a task, e.g. `nova ask \"explain this project\"`.")
        return 2

    agent = None
    try:
        settings.require_api_key()

        controller = AgentController(
            approval_handler=cli_approval_handler(console, auto_yes=args.yes),
            auto_approve=args.yes,
            approval_timeout=settings.approval_timeout,
        )
        agent = build_agent(settings)
        narration = Console(color=False, stream=io.StringIO()) if args.json else console
        answer, _, failed = asyncio.run(
            _run_streaming(
                agent, task, narration, controller=controller, verbose=args.verbose
            )
        )
    except ConfigError as exc:
        console.error(str(exc))
        return 3
    except KeyboardInterrupt:
        console.write()
        console.warn("Interrupted.")
        return 130
    finally:
        if agent is not None:
            _close_provider(agent)

    if args.json:
        console.write(
            json.dumps({"task": task, "answer": answer, "ok": not failed}, indent=2)
        )
    return 1 if failed else 0


def cmd_chat(args: argparse.Namespace, console: Console) -> int:
    """Interactive REPL."""
    settings = _settings_from_args(args)
    console.title(f"NovaCLI v{__version__} · chat")
    console.write(
        console.paint(
            "  Type a task and press enter. /exit to quit, /new to clear context, "
            "/files to list the workspace.",
            "dim",
        )
    )
    if not settings.has_api_key:
        console.warn("No API key configured — the agent cannot answer yet.")
        console.write(console.paint(get_api_key_hint(settings.provider), "dim"))

    agent = build_agent(settings)
    history: list[Message] = []

    try:
        while True:
            try:
                line = input(console.paint("\nyou › ", "cyan", "bold"))
            except (EOFError, KeyboardInterrupt):
                console.write()
                break

            task = line.strip()
            if not task:
                continue
            if task in {"/exit", "/quit", ":q"}:
                break
            if task == "/new":
                history.clear()
                console.ok("Context cleared.")
                continue
            if task in {"/files", "/ls"}:
                for entry in _make_workspace(settings).list_dir("."):
                    console.write(f"  {'[dir] ' if entry.is_dir else '      '}{entry.path}")
                continue
            if task == "/help":
                console.write("  /exit  /new  /files  /help")
                continue

            controller = AgentController(
                approval_handler=cli_approval_handler(console, auto_yes=args.yes),
                auto_approve=args.yes,
                approval_timeout=settings.approval_timeout,
            )
            try:
                _, history, _ = asyncio.run(
                    _run_streaming(
                        agent, task, console, controller=controller,
                        history=history, verbose=args.verbose,
                    )
                )
            except KeyboardInterrupt:
                console.write()
                console.warn("Cancelled current task.")
            except ConfigError as exc:
                console.error(str(exc))
                break
    finally:
        _close_provider(agent)

    console.write()
    console.info("Session ended.")
    return 0


def cmd_run(args: argparse.Namespace, console: Console) -> int:
    """Run a shell command through the safety layer and runner."""
    settings = _settings_from_args(args)
    parts = list(getattr(args, "shell_command", None) or [])
    if parts and parts[0] == "--":
        parts = parts[1:]
    command = " ".join(parts).strip()
    if not command:
        console.error("No command given. Try: nova run pytest -q")
        return 2

    safety = SafetyPolicy(
        settings.project_root, settings.safety_mode, secret_values=settings.active_secrets
    )
    verdict = safety.check_command(command)

    style = RISK_STYLES.get(verdict.level, "yellow")
    console.write(
        f"{console.paint('risk:', 'dim')} {console.paint(str(verdict.level), style)} "
        f"{console.paint(f'({verdict.reason})', 'dim')}"
    )

    if not verdict.allowed:
        console.error(f"Refused: {verdict.reason}")
        return 4
    if verdict.requires_approval and not args.yes:
        console.warn("This command needs approval. Re-run with --yes to allow it.")
        return 4

    runner = CommandRunner(
        settings.project_root,
        settings.command_timeout,
        safety=safety,
    )
    result = asyncio.run(runner.run(command, approved=args.yes, check_safety=False))

    if result.stdout:
        console.write(result.stdout.rstrip())
    if result.stderr:
        console.write(console.paint(result.stderr.rstrip(), "red"))
    console.write(
        console.paint(
            f"[exit {result.exit_code} in {result.duration_ms}ms"
            f"{' TIMEOUT' if result.timed_out else ''}]",
            "dim",
        )
    )
    return 0 if result.ok else 1


def cmd_tree(args: argparse.Namespace, console: Console) -> int:
    settings = _settings_from_args(args)
    workspace = _make_workspace(settings)
    try:
        console.write(workspace.tree(args.path, max_depth=args.depth, max_entries=args.limit))
    except (OSError, ValueError) as exc:
        console.error(str(exc))
        return 1
    return 0


def cmd_ls(args: argparse.Namespace, console: Console) -> int:
    settings = _settings_from_args(args)
    workspace = _make_workspace(settings)
    try:
        entries = workspace.list_dir(args.path)
    except (OSError, ValueError) as exc:
        console.error(str(exc))
        return 1
    if not entries:
        console.info("(empty)")
        return 0
    for entry in entries:
        size = f"{entry.size:>8}" if not entry.is_dir else "     dir"
        console.write(f"  {console.paint(size, 'dim')}  {entry.path}")
    return 0


def cmd_read(args: argparse.Namespace, console: Console) -> int:
    settings = _settings_from_args(args)
    workspace = _make_workspace(settings)
    try:
        content = workspace.read_text(args.path)
    except (OSError, ValueError) as exc:
        console.error(str(exc))
        return 1
    if args.json:
        console.write(json.dumps({"path": args.path, "content": content}, indent=2))
    else:
        console.write(content)
    return 0


def cmd_search(args: argparse.Namespace, console: Console) -> int:
    settings = _settings_from_args(args)
    workspace = _make_workspace(settings)
    try:
        hits = workspace.search(
            " ".join(args.query), glob=args.glob, max_results=args.limit, regex=args.regex
        )
    except (OSError, ValueError) as exc:
        console.error(str(exc))
        return 1
    if not hits:
        console.info("No matches.")
        return 0
    for hit in hits:
        console.write(f"  {console.paint(f'{hit.path}:{hit.line}', 'cyan')} {hit.text.strip()}")
    console.write(console.paint(f"  {len(hits)} match(es)", "dim"))
    return 0


def cmd_summary(args: argparse.Namespace, console: Console) -> int:
    settings = _settings_from_args(args)
    workspace = _make_workspace(settings)
    analyzer = ProjectAnalyzer(workspace)
    if args.json:
        console.write(json.dumps(analyzer.summarize().to_dict(), indent=2))
    else:
        console.write(analyzer.render_summary())
    return 0


def cmd_project(args: argparse.Namespace, console: Console) -> int:
    """Project Intelligence 2.0 CLI command."""
    settings = _settings_from_args(args)
    workspace = _make_workspace(settings)
    cache = IntelligenceCache(workspace)

    subcommand = getattr(args, "project_subcommand", None)
    force_refresh = (subcommand == "scan")

    info = cache.get_or_scan(force_refresh=force_refresh)

    if args.json:
        console.write(json.dumps(info.to_dict(), indent=2))
        return 0

    console.title("◆ Nova Project Intelligence 2.0")
    console.write()
    console.write(f"  Name:             {info.name}")
    console.write(f"  Root:             {info.root}")
    console.write(f"  Ecosystems:       {', '.join(info.ecosystems) if info.ecosystems else 'None'}")
    console.write(f"  Frameworks:       {', '.join(info.frameworks) if info.frameworks else 'None'}")
    console.write(f"  Package Managers: {', '.join(info.package_managers) if info.package_managers else 'None'}")
    console.write(f"  Files:            {info.total_files} ({info.total_bytes / 1024:.1f} KiB)")
    if info.entry_points:
        console.write(f"  Entry points:     {', '.join(ep.path for ep in info.entry_points[:5])}")
    if info.tests.frameworks or info.tests.total_tests:
        console.write(f"  Tests:            {info.tests.total_tests} files ({', '.join(info.tests.frameworks)})")
    if info.git.has_git:
        console.write(f"  Git Branch:       {info.git.branch or 'unknown'}")
        console.write(f"  Git Modified:     {len(info.git.modified_files)} file(s)")
        console.write(f"  Git Untracked:    {len(info.git.untracked_files)} file(s)")
    return 0


def cmd_config(args: argparse.Namespace, console: Console) -> int:
    store = NovaConfigStore()
    subcommand = getattr(args, "config_subcommand", None)

    if subcommand == "provider":
        prov_target = getattr(args, "provider_name", "").strip()
        if not prov_target:
            console.error("Provider name required.")
            return 2
        norm = normalize_provider_name(prov_target)
        if norm not in provider_names():
            console.error(f"Unknown provider '{prov_target}'. Available: {', '.join(provider_names())}")
            return 2
        old_prov = load_settings().provider
        store.save_config({"provider": norm})
        console.ok(f"Provider changed: {old_prov} -> {norm}")
        return 0

    if subcommand == "model":
        model_target = getattr(args, "model_name", "").strip()
        if not model_target:
            console.error("Model name required.")
            return 2
        current_settings = load_settings()
        prov = current_settings.provider
        model_field = f"{prov}_model" if prov != "groq" else "groq_model"
        store.save_config({model_field: model_target})
        console.ok(f"Model for provider {prov} changed to: {model_target}")
        return 0

    if subcommand == "set-key":
        key = getattr(args, "key", "").strip()
        if not key:
            console.error("API key cannot be empty.")
            return 2
        current_prov = load_settings().provider
        if current_prov == "ollama":
            console.error("Ollama does not require an API key.")
            return 2
        key_field = f"{current_prov}_api_key"
        store.save_credentials({key_field: key})
        console.ok(f"Saved global {current_prov.capitalize()} API key to {store.credentials_path}")
        return 0

    settings = _settings_from_args(args)
    data = settings.to_public_dict()

    config_exists = store.config_path.exists()
    creds_exists = store.credentials_path.exists()

    if args.json:
        output_data = dict(data)
        output_data.update({
            "global_config_path": str(store.config_path),
            "global_config_exists": config_exists,
            "global_credentials_path": str(store.credentials_path),
            "global_credentials_exists": creds_exists,
        })
        console.write(json.dumps(output_data, indent=2))
        return 0

    console.title("◆ Nova Configuration")
    console.write()
    console.write("  Global config:")
    console.write(f"    {store.config_path}  {'✓' if config_exists else '✗'}")
    console.write()
    console.write("  Credentials:")
    console.write(f"    {store.credentials_path}  {'✓' if creds_exists else '✗'}")
    console.write()
    console.write(f"  Active Provider: {settings.provider}")
    console.write(f"  Active Model:    {settings.model}")
    console.write()
    console.write("  Available Providers:")
    for p in provider_names():
        check = "[x]" if p == settings.provider else "[ ]"
        console.write(f"    {check} {p.capitalize()}")
    console.write()

    if settings.provider == "ollama":
        console.write("  API key:")
        console.write(f"    {console.paint('not required', 'cyan')}")
        console.write(f"  Base URL: {settings.ollama_base_url}")
    elif settings.has_api_key:
        source = f"(from {settings.api_key_source})"
        console.write("  API key:")
        console.write(
            f"    {console.paint('configured', 'green')} "
            f"{console.paint(settings.masked_api_key, 'dim')} "
            f"{console.paint(source, 'dim')}"
        )
    else:
        console.write("  API key:")
        console.write(f"    {console.paint('missing', 'red')}")
        console.write()
        console.write(console.paint(get_api_key_hint(settings.provider), "dim"))
    console.write()
    console.write(f"  Project root:  {data['project_root']}")
    console.write(f"  Timeout:       {data['command_timeout']}s")
    console.write(f"  Safety mode:   {data['safety_mode']}")
    return 0


def cmd_doctor(args: argparse.Namespace, console: Console) -> int:
    """Diagnose the environment — the first thing to run on a new device."""
    settings = _settings_from_args(args)
    console.title("NovaCLI doctor")
    console.write()
    failures = 0

    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        console.ok(f"Python {major}.{minor} (supported)")
    else:
        console.error(f"Python {major}.{minor} is too old — 3.11+ required")
        failures += 1

    console.write(f"  Selected Provider: {settings.provider}")
    console.write(f"  Selected Model:    {settings.model}")

    if settings.provider == "ollama":
        console.ok("API key: not required for Ollama")
        # Check Ollama server reachability
        import urllib.request
        try:
            req = urllib.request.Request(f"{settings.ollama_base_url}/api/version")
            with urllib.request.urlopen(req, timeout=3):
                console.ok(f"Ollama server reachable at {settings.ollama_base_url}")
        except Exception:
            console.warn(f"Cannot reach Ollama server at {settings.ollama_base_url}")
    else:
        if settings.has_api_key:
            console.ok(f"API key configured (from {settings.api_key_source})")
        else:
            console.warn(f"API key missing for {settings.provider} — set API key or use `nova config set-key`")
            failures += 1

    if settings.project_root.is_dir():
        console.ok(f"workspace {settings.project_root}")

    workspace = _make_workspace(settings)
    if os.access(workspace.root, os.W_OK):
        console.ok("workspace is writable")
    else:
        console.warn("workspace is read-only — writes will fail")
        failures += 1

    shell = shutil.which("sh") or shutil.which("bash")
    if shell:
        console.ok(f"shell available ({shell})")
    else:
        console.warn("no POSIX shell found — command execution disabled")

    # Terminal & environment diagnostics
    try:
        from nova.core.pty import get_terminal_diagnostics
        diag = get_terminal_diagnostics(cwd=str(settings.project_root), env=dict(settings.environment))
        backend = diag.get("backend", "unknown")
        shell_path = diag.get("shell") or "not detected"
        if shell_path and shell_path != "not detected":
            console.ok(f"terminal PTY: {backend}")
            console.write(f"    shell: {shell_path}")
        else:
            console.warn(f"terminal PTY unavailable: no shell found")
            failures += 1

        tools = diag.get("tool_resolutions", {})
        console.write("  command resolution:")
        for tool_name, resolved_path in tools.items():
            if resolved_path:
                console.write(f"    {tool_name:8s} -> {resolved_path}")
            else:
                console.write(f"    {tool_name:8s} -> (not found in PATH)")
    except Exception as e:
        console.warn(f"could not diagnose PTY backend: {e}")

    console.write()
    if failures:
        console.warn(f"{failures} issue(s) need attention.")
        return 1
    console.ok("Everything looks good.")
    return 0


def cmd_init(args: argparse.Namespace, console: Console) -> int:
    """Create a .env file in the project so setup is one command."""
    target = Path(args.path or Path.cwd()).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    env_path = target / ".env"

    if env_path.exists() and not args.force:
        console.warn(f"{env_path} already exists — use --force to overwrite.")
        return 1

    template = "\n".join(
        [
            "# NovaCLI configuration — created by `nova init`",
            "NOVA_PROVIDER=groq",
            "GROQ_API_KEY=",
            "GROQ_MODEL=openai/gpt-oss-20b",
            "OLLAMA_MODEL=qwen3:4b",
            "OLLAMA_BASE_URL=http://localhost:11434",
            f"NOVA_PROJECT_ROOT={target}",
            "NOVA_COMMAND_TIMEOUT=30",
            "NOVA_MAX_STEPS=8",
            "NOVA_SAFETY_MODE=smart",
            "NOVA_HOST=127.0.0.1",
            "NOVA_PORT=8000",
            "",
        ]
    )
    env_path.write_text(template, encoding="utf-8")
    console.ok(f"Wrote {env_path}")
    console.write(console.paint("  Next: set your API key in .env or ~/.nova/credentials.json", "dim"))
    console.write(console.paint("  Then: nova doctor", "dim"))
    return 0


def cmd_serve(args: argparse.Namespace, console: Console) -> int:
    """Start the FastAPI web IDE."""
    settings = _settings_from_args(args)
    try:
        import uvicorn
    except ImportError:
        console.error("uvicorn is not installed. Run: python -m pip install -r requirements.txt")
        return 1

    from nova.web.app import create_app

    host = args.host or settings.host
    port = args.port or settings.port

    console.title(f"NovaCLI web IDE · http://{host}:{port}")
    console.write(console.paint(f"  workspace {settings.project_root}", "dim"))
    if not settings.has_api_key:
        console.warn("No API key configured — the IDE will open but the agent cannot run.")
    console.write(console.paint("  Ctrl+C to stop", "dim"))
    console.write()

    app = create_app(settings)
    try:
        uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    except KeyboardInterrupt:
        console.write()
        console.info("Server stopped.")
    return 0


def cmd_version(args: argparse.Namespace, console: Console) -> int:
    if args.json:
        console.write(json.dumps({"name": "NovaCLI", "version": __version__}))
    else:
        console.write(f"NovaCLI v{__version__}")
        console.write(f"Python {sys.version.split()[0]} on {sys.platform}")
    return 0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def _settings_from_args(args: argparse.Namespace, **extra: Any) -> Settings:
    """Build Settings from global flags, then per-command overrides."""
    overrides: dict[str, Any] = {}
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "model", None):
        overrides["groq_model"] = args.model
        overrides["ollama_model"] = args.model
        overrides["gemini_model"] = args.model
        overrides["openrouter_model"] = args.model
        overrides["cerebras_model"] = args.model
    if getattr(args, "timeout", None):
        overrides["command_timeout"] = args.timeout
    if getattr(args, "safety", None):
        overrides["safety_mode"] = args.safety
    overrides.update(extra)

    return load_settings(
        project_root=getattr(args, "project_root", None),
        require_api_key=False,
        **overrides,
    )


def _close_provider(agent: Any) -> None:
    """Best-effort cleanup of the HTTP client."""
    try:
        asyncio.run(agent.provider.aclose())
    except Exception:  # noqa: BLE001 - cleanup must never mask the real result
        pass


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argparse tree."""
    parser = argparse.ArgumentParser(
        prog="nova",
        description="NovaCLI — an AI-powered developer environment (agent + CLI + mobile Web IDE).",
        epilog=(
            "examples:\n"
            "  nova ask \"what does this project do?\"\n"
            "  nova ask \"add tests for config.py and run them\" --yes\n"
            "  nova chat\n"
            "  nova serve --host 0.0.0.0\n"
            "  nova summary\n"
            "  nova doctor\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"NovaCLI {__version__}")

    # Global options, accepted both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-C", "--project-root", help="Workspace root (default: current directory)")
    common.add_argument("--provider", help="Override provider (groq, gemini, ollama, openrouter, cerebras)")
    common.add_argument("--model", help="Override active model id")
    common.add_argument("--timeout", type=int, help="Per-command timeout in seconds")
    common.add_argument(
        "--safety",
        choices=("smart", "strict", "permissive"),
        help="Safety profile (default: smart)",
    )
    common.add_argument("--json", action="store_true", help="Machine-readable output")
    common.add_argument("-v", "--verbose", action="store_true", help="Show full tool output")
    common.add_argument("--no-color", action="store_true", help="Disable ANSI colours")
    common.add_argument(
        "-y", "--yes", action="store_true",
        help="Auto-approve actions that would normally ask (use with care)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, help_text: str, **kwargs: Any) -> argparse.ArgumentParser:
        return subparsers.add_parser(
            name, parents=[common], help=help_text, description=help_text, **kwargs
        )

    p_ask = add("ask", "Ask the agent to perform a task, once.")
    p_ask.add_argument("task", nargs="+", help="The task, in plain language")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = add("chat", "Start an interactive agent session.")
    p_chat.set_defaults(func=cmd_chat)

    p_serve = add("serve", "Run the mobile-first web IDE.")
    p_serve.add_argument("--host", help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, help="Port (default: 8000)")
    p_serve.add_argument(
        "--log-level", default="warning", choices=("critical", "error", "warning", "info", "debug")
    )
    p_serve.set_defaults(func=cmd_serve)

    p_run = add("run", "Run a shell command through the safety layer.")
    p_run.add_argument(
        "shell_command",
        nargs=argparse.REMAINDER,
        metavar="COMMAND",
        help="Command to execute, e.g. `nova run -C ~/app pytest -q`",
    )
    p_run.set_defaults(func=cmd_run)

    p_summary = add("summary", "Show what NovaCLI understands about the project.")
    p_summary.set_defaults(func=cmd_summary)

    p_project = add("project", "Project Intelligence 2.0 details.")
    p_project_sub = p_project.add_subparsers(dest="project_subcommand")
    p_project_info = p_project_sub.add_parser("info", parents=[common], help="Show project intelligence info.")
    p_project_info.set_defaults(func=cmd_project)
    p_project_scan = p_project_sub.add_parser("scan", parents=[common], help="Force refresh project scan.")
    p_project_scan.set_defaults(func=cmd_project)
    p_project.set_defaults(func=cmd_project)

    p_tree = add("tree", "Print a project tree.")
    p_tree.add_argument("path", nargs="?", default=".")
    p_tree.add_argument("-d", "--depth", type=int, default=3)
    p_tree.add_argument("-l", "--limit", type=int, default=200)
    p_tree.set_defaults(func=cmd_tree)

    p_ls = add("ls", "List a directory.")
    p_ls.add_argument("path", nargs="?", default=".")
    p_ls.set_defaults(func=cmd_ls)

    p_read = add("read", "Print a file from the workspace.")
    p_read.add_argument("path")
    p_read.set_defaults(func=cmd_read)

    p_search = add("search", "Search file contents.")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("-g", "--glob", help="Glob filter, e.g. '**/*.py'")
    p_search.add_argument("-l", "--limit", type=int, default=50)
    p_search.add_argument("--regex", action="store_true", help="Treat the query as a regex")
    p_search.set_defaults(func=cmd_search)

    p_config = add("config", "Show or manage the configuration and global credentials.")
    config_sub = p_config.add_subparsers(dest="config_subcommand")
    p_config_status = config_sub.add_parser("status", parents=[common], help="Show configuration status.")
    p_config_status.set_defaults(func=cmd_config)

    p_config_prov = config_sub.add_parser("provider", parents=[common], help="Set active provider.")
    p_config_prov.add_argument("provider_name", help="Provider name (groq, gemini, ollama, openrouter, cerebras)")
    p_config_prov.set_defaults(func=cmd_config)

    p_config_mod = config_sub.add_parser("model", parents=[common], help="Set model for active provider.")
    p_config_mod.add_argument("model_name", help="Model name / identifier")
    p_config_mod.set_defaults(func=cmd_config)

    p_config_set = config_sub.add_parser("set-key", parents=[common], help="Set global API key for active provider.")
    p_config_set.add_argument("key", help="The API key")
    p_config_set.set_defaults(func=cmd_config)
    p_config.set_defaults(func=cmd_config)

    p_doctor = add("doctor", "Diagnose Python, dependencies, API key and workspace.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_init = add("init", "Create a .env file for this project.")
    p_init.add_argument("path", nargs="?", default=None)
    p_init.add_argument("--force", action="store_true", help="Overwrite an existing .env")
    p_init.set_defaults(func=cmd_init)

    p_version = add("version", "Print the NovaCLI version.")
    p_version.set_defaults(func=cmd_version)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    console = Console(color=False if getattr(args, "no_color", False) else None)

    try:
        return int(args.func(args, console) or 0)
    except ConfigError as exc:
        console.error(str(exc))
        return 3
    except SafetyError as exc:
        console.error(str(exc))
        return 1
    except KeyboardInterrupt:
        console.write()
        console.warn("Interrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
