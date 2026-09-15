"""HTTP and Server-Sent-Events endpoints for the web IDE.

Every endpoint either reads workspace state or delegates to Nova Core. No agent
logic is duplicated here — starting a task just constructs the same
:class:`~nova.core.agent.NovaAgent` the CLI builds, then streams its events.
"""

from __future__ import annotations

import shlex
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nova.ai import AIProviderError, get_provider
from nova.config import get_api_key_hint, NovaConfigStore, Settings
from nova.core.agent import AgentController, build_agent
from nova.core.checkpoints import CheckpointManager
from nova.core.git import GitService
from nova.core.models import ApprovalDecision, RiskLevel
from nova.core.runner import CommandRunner
from nova.core.safety import SafetyError, SafetyPolicy
from nova.workspace.files import Workspace, detect_language
from nova.workspace.projects import ProjectAnalyzer
from nova.intelligence.cache import IntelligenceCache
from nova.web.app import TEMPLATES_DIR
from nova.web.events import SessionRegistry, stream_session

router = APIRouter()

# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class AgentRequest(BaseModel):
    """Body of ``POST /api/agent``."""

    task: str = Field(min_length=1, max_length=20_000)
    auto_approve: bool = False


class ApprovalBody(BaseModel):
    """Body of ``POST /api/agent/approve``."""

    session_id: str
    request_id: str
    decision: str = "approve"


class SessionBody(BaseModel):
    """Body of session-scoped control endpoints."""

    session_id: str


class RunBody(BaseModel):
    """Body of ``POST /api/run``."""

    command: str = Field(min_length=1, max_length=4_000)
    cwd: str | None = None
    approve: bool = False


class WriteBody(BaseModel):
    """Body of ``PUT /api/file``."""

    path: str = Field(min_length=1, max_length=1_000)
    content: str = Field(max_length=2_000_000)


class RenameBody(BaseModel):
    """Body of ``POST /api/file/rename``."""

    old_path: str = Field(min_length=1, max_length=1_000)
    new_path: str = Field(min_length=1, max_length=1_000)


class TestRunBody(BaseModel):
    """Body of ``POST /api/tests/run``."""

    approve: bool = False


class CheckpointCreateBody(BaseModel):
    task_id: str = Field(min_length=1, max_length=100)
    session_id: str | None = None


class RollbackBody(BaseModel):
    confirm: bool = False


class MkdirBody(BaseModel):
    """Body of ``POST /api/file/mkdir``."""

    path: str = Field(min_length=1, max_length=1_000)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _registry(request: Request) -> SessionRegistry:
    return request.app.state.registry


def _safety(settings: Settings) -> SafetyPolicy:
    return SafetyPolicy(
        settings.project_root,
        settings.safety_mode,
        secret_values=settings.active_secrets,
    )


def _workspace(settings: Settings) -> Workspace:
    return Workspace(
        settings.project_root,
        safety=_safety(settings),
        extra_ignore=settings.extra_ignore,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    """Serve the single-page IDE shell."""
    settings = _settings(request)
    try:
        from fastapi.templating import Jinja2Templates
    except ImportError:  # pragma: no cover - jinja2 is a declared dependency
        html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "version": request.app.state.version,
            "model": settings.model,
            "has_api_key": settings.has_api_key,
            "project_name": settings.project_root.name,
            "safety_mode": settings.safety_mode,
        },
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness plus the flag the UI needs to show setup instructions."""
    settings = _settings(request)
    return {
        "status": "ok",
        "version": request.app.state.version,
        "model": settings.model,
        "has_api_key": settings.has_api_key,
        "project_root": str(settings.project_root),
        "safety_mode": settings.safety_mode,
    }


@router.get("/api/config")
async def config(request: Request) -> dict[str, Any]:
    """Effective configuration. Never contains secret material."""
    return _settings(request).to_public_dict()


@router.get("/api/config/status")
async def config_status(request: Request) -> dict[str, Any]:
    """Status of global config & credentials without exposing secrets."""
    settings = _settings(request)
    store = NovaConfigStore()
    return {
        "has_api_key": settings.has_api_key,
        "api_key_source": settings.api_key_source,
        "api_key_preview": settings.masked_api_key,
        "global_config_path": str(store.config_path),
        "global_config_exists": store.config_path.exists(),
        "global_credentials_path": str(store.credentials_path),
        "global_credentials_exists": store.credentials_path.exists(),
        "model": settings.model,
    }


# ---------------------------------------------------------------------------
# Project / files / Intelligence
# ---------------------------------------------------------------------------


@router.get("/api/project")
async def project(request: Request) -> dict[str, Any]:
    """Project summary plus a shallow tree — powers the Project tab."""
    settings = _settings(request)
    workspace = _workspace(settings)
    analyzer = ProjectAnalyzer(workspace)
    summary = analyzer.summarize()
    return {
        "summary": summary.to_dict(),
        "tree": workspace.tree(".", max_depth=3, max_entries=200),
        "skills": analyzer.detect_commands(),
    }


@router.get("/api/project/intelligence")
async def project_intelligence(request: Request) -> dict[str, Any]:
    """Return Project Intelligence 2.0 info."""
    settings = _settings(request)
    workspace = _workspace(settings)
    cache = IntelligenceCache(workspace)
    info = cache.get_or_scan()
    return info.to_dict()


@router.post("/api/project/refresh")
async def project_intelligence_refresh(request: Request) -> dict[str, Any]:
    """Force re-scan Project Intelligence 2.0 info."""
    settings = _settings(request)
    workspace = _workspace(settings)
    cache = IntelligenceCache(workspace)
    info = cache.get_or_scan(force_refresh=True)
    return info.to_dict()


@router.get("/api/tree")
async def tree(
    request: Request,
    path: str = Query(".", max_length=1_000),
    depth: int = Query(2, ge=1, le=6),
) -> dict[str, Any]:
    workspace = _workspace(_settings(request))
    try:
        return {"path": path, "tree": workspace.tree(path, max_depth=depth, max_entries=300)}
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/api/files")
async def list_files(
    request: Request,
    path: str = Query(".", max_length=1_000),
) -> dict[str, Any]:
    workspace = _workspace(_settings(request))
    try:
        entries = workspace.list_dir(path)
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"path": path, "entries": [entry.to_dict() for entry in entries]}


@router.get("/api/file")
async def read_file(
    request: Request,
    path: str = Query(..., min_length=1, max_length=1_000),
) -> dict[str, Any]:
    """Read one file. Credential files are refused by the safety layer."""
    workspace = _workspace(_settings(request))
    try:
        content = workspace.read_text(path, max_bytes=500_000)
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {
        "path": workspace.relative(path),
        "content": content,
        "language": detect_language(path) or "text",
        "size": len(content.encode("utf-8")),
        "lines": content.count("\n") + 1,
    }


@router.put("/api/file")
async def write_file(request: Request, body: WriteBody) -> dict[str, Any]:
    """Write one file, honouring the same safety rules as the agent."""
    settings = _settings(request)
    workspace = _workspace(settings)
    try:
        entry = workspace.write_text(body.path, body.content)
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"ok": True, "file": entry.to_dict()}


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


@router.post("/api/run")
async def run_command(request: Request, body: RunBody) -> dict[str, Any]:
    """Run a shell command through the safety layer and runner."""
    settings = _settings(request)
    safety = _safety(settings)
    verdict = safety.check_command(body.command)

    if verdict.level == RiskLevel.FORBIDDEN:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Refused: {verdict.reason}",
        )
    if verdict.requires_approval and not body.approve:
        return {
            "requires_approval": True,
            "level": str(verdict.level),
            "reason": verdict.reason,
            "command": body.command,
        }

    runner = CommandRunner(
        settings.project_root, settings.command_timeout, safety=safety
    )
    result = await runner.run(
        body.command, cwd=body.cwd, approved=body.approve, check_safety=True
    )
    return {"requires_approval": False, "result": result.to_dict()}


# ---------------------------------------------------------------------------
# Agent sessions
# ---------------------------------------------------------------------------


def _agent_factory(settings: Settings):
    """Build the factory the session registry calls to create an agent."""

    def factory():
        try:
            provider = get_provider(settings)
        except AIProviderError:
            provider = None
        return build_agent(settings, provider=provider, max_steps=settings.max_steps)

    return factory


@router.post("/api/agent", status_code=status.HTTP_201_CREATED)
async def start_agent(request: Request, body: AgentRequest) -> JSONResponse:
    """Create a session and start the agent in the background."""
    settings = _settings(request)
    registry = _registry(request)

    if not settings.has_api_key:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, get_api_key_hint(settings.provider))

    controller = AgentController(
        auto_approve=body.auto_approve, approval_timeout=settings.approval_timeout
    )
    session = registry.create(body.task, controller=controller, model=settings.model)
    registry.start(session, _agent_factory(settings))

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"session_id": session.id, "status": str(session.status)},
    )


@router.get("/api/agent/stream")
async def agent_stream(
    request: Request,
    session_id: str = Query(..., min_length=1, max_length=64),
) -> StreamingResponse:
    """Stream a session's events as Server-Sent Events."""
    registry = _registry(request)
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id!r}")

    return StreamingResponse(
        stream_session(session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/agent/result")
async def agent_result(
    request: Request,
    session_id: str = Query(..., min_length=1, max_length=64),
    events: bool = Query(False),
) -> dict[str, Any]:
    """Poll a session's state, for clients that cannot hold an SSE stream."""
    registry = _registry(request)
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {session_id!r}")
    return session.to_dict(include_events=events)


@router.get("/api/agent/sessions")
async def agent_sessions(request: Request) -> dict[str, Any]:
    """List sessions, newest first."""
    registry = _registry(request)
    return {"sessions": [session.to_dict() for session in registry.list()]}


@router.post("/api/agent/approve")
async def agent_approve(request: Request, body: ApprovalBody) -> dict[str, Any]:
    """Answer a pending approval request from the UI."""
    registry = _registry(request)
    try:
        decision = ApprovalDecision(body.decision)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"decision must be one of {[d.value for d in ApprovalDecision]}",
        ) from exc

    if not registry.approve(body.session_id, body.request_id, decision):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No approval is waiting for that session (it may have timed out).",
        )
    return {"ok": True, "decision": str(decision)}


@router.post("/api/agent/cancel")
async def agent_cancel(request: Request, body: SessionBody) -> dict[str, Any]:
    """Cancel a running session."""
    registry = _registry(request)
    if not registry.cancel(body.session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown session {body.session_id!r}")
    return {"ok": True, "status": "cancelled"}


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Extended Workspace & Developer APIs
# ---------------------------------------------------------------------------


@router.get("/api/search")
async def search_files(
    request: Request,
    q: str = Query(..., min_length=1, max_length=1_000),
    glob: str | None = Query(None, max_length=200),
    case_sensitive: bool = Query(False),
    regex: bool = Query(False),
) -> dict[str, Any]:
    """Project-wide text search."""
    workspace = _workspace(_settings(request))
    try:
        hits = workspace.search(
            query=q,
            glob=glob,
            case_sensitive=case_sensitive,
            regex=regex,
        )
        return {"query": q, "hits": [hit.to_dict() for hit in hits]}
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.delete("/api/file")
async def delete_file(
    request: Request,
    path: str = Query(..., min_length=1, max_length=1_000),
) -> dict[str, Any]:
    """Delete a file within workspace safely."""
    workspace = _workspace(_settings(request))
    try:
        deleted = workspace.delete(path)
        if not deleted:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"File {path!r} not found")
        return {"ok": True, "path": path}
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError, IsADirectoryError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/api/file/rename")
async def rename_file(request: Request, body: RenameBody) -> dict[str, Any]:
    """Rename/move a file safely within workspace."""
    workspace = _workspace(_settings(request))
    try:
        old_target = workspace.resolve(body.old_path, for_write=True)
        new_target = workspace.resolve(body.new_path, for_write=True)

        if not old_target.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Path {body.old_path!r} not found")
        if new_target.exists():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Target {body.new_path!r} already exists")

        new_target.parent.mkdir(parents=True, exist_ok=True)
        old_target.rename(new_target)
        return {"ok": True, "old_path": workspace.relative(old_target), "new_path": workspace.relative(new_target)}
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/api/file/mkdir")
async def mkdir(request: Request, body: MkdirBody) -> dict[str, Any]:
    """Create a directory safely within workspace."""
    workspace = _workspace(_settings(request))
    try:
        target = workspace.resolve(body.path, for_write=True)
        target.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": workspace.relative(target)}
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/api/git/status")
async def git_status(request: Request) -> dict[str, Any]:
    """Return basic git branch and status summary."""
    settings = _settings(request)
    workspace = _workspace(settings)
    runner = CommandRunner(settings.project_root, timeout=10, safety=_safety(settings))
    git_svc = GitService(workspace, runner)
    return await git_svc.status()


@router.get("/api/git/diff")
async def git_diff(
    request: Request,
    path: str | None = Query(None, max_length=1_000),
) -> dict[str, Any]:
    """Return git diff for a file or entire repository."""
    settings = _settings(request)
    workspace = _workspace(settings)
    runner = CommandRunner(settings.project_root, timeout=15, safety=_safety(settings))
    git_svc = GitService(workspace, runner)
    try:
        return await git_svc.diff(path)
    except SafetyError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/api/git/commit_preview")
async def git_commit_preview(request: Request) -> dict[str, Any]:
    """Return a preview of staged and unstaged changes for commit."""
    settings = _settings(request)
    workspace = _workspace(settings)
    runner = CommandRunner(settings.project_root, timeout=15, safety=_safety(settings))
    git_svc = GitService(workspace, runner)
    return await git_svc.commit_preview()


@router.get("/api/agent/checkpoints")
async def list_checkpoints(request: Request) -> dict[str, Any]:
    """List all agent checkpoints in workspace."""
    workspace = _workspace(_settings(request))
    cpm = CheckpointManager(workspace)
    cps = cpm.list()
    return {"checkpoints": [cp.to_dict() for cp in cps]}


@router.get("/api/agent/checkpoint/{checkpoint_id}")
async def get_checkpoint(request: Request, checkpoint_id: str) -> dict[str, Any]:
    """Inspect details and changed files for a checkpoint."""
    workspace = _workspace(_settings(request))
    cpm = CheckpointManager(workspace)
    try:
        return cpm.inspect(checkpoint_id)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/api/agent/checkpoint")
async def create_checkpoint(request: Request, body: CheckpointCreateBody) -> dict[str, Any]:
    """Manually create an agent checkpoint."""
    workspace = _workspace(_settings(request))
    cpm = CheckpointManager(workspace)
    cp = cpm.create(body.task_id, session_id=body.session_id)
    return {"ok": True, "checkpoint": cp.to_dict()}


@router.post("/api/agent/checkpoint/{checkpoint_id}/rollback")
async def rollback_checkpoint(
    request: Request, checkpoint_id: str, body: RollbackBody | None = None
) -> dict[str, Any]:
    """Safely rollback workspace state to a checkpoint baseline."""
    workspace = _workspace(_settings(request))
    cpm = CheckpointManager(workspace)
    confirm = body.confirm if body else False
    try:
        res = cpm.rollback(checkpoint_id, force=confirm)
        return res.to_dict()
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.delete("/api/agent/checkpoint/{checkpoint_id}")
async def delete_checkpoint(request: Request, checkpoint_id: str) -> dict[str, Any]:
    """Delete a checkpoint record and its snapshots."""
    workspace = _workspace(_settings(request))
    cpm = CheckpointManager(workspace)
    deleted = cpm.delete(checkpoint_id)
    if not deleted:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Checkpoint {checkpoint_id!r} not found"
        )
    return {"ok": True, "checkpoint_id": checkpoint_id}


@router.post("/api/tests/run")
async def run_tests(
    request: Request,
    body: TestRunBody | None = None,
) -> dict[str, Any]:
    """Discover and safely execute detected project test command."""
    settings = _settings(request)
    workspace = _workspace(settings)
    analyzer = ProjectAnalyzer(workspace)
    commands = analyzer.detect_commands()
    test_cmd = commands.get("test")

    if not test_cmd:
        return {"ok": False, "error": "No test runner command detected for this project.", "output": ""}

    approve = body.approve if body else False
    safety = _safety(settings)
    verdict = safety.check_command(test_cmd)

    if verdict.level == RiskLevel.FORBIDDEN:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Refused: {verdict.reason}",
        )
    if verdict.requires_approval and not approve:
        return {
            "requires_approval": True,
            "level": str(verdict.level),
            "reason": verdict.reason,
            "command": test_cmd,
        }

    runner = CommandRunner(settings.project_root, timeout=60, safety=safety)
    res = await runner.run(test_cmd, approved=approve, check_safety=True)

    if res.blocked:
        return {
            "ok": False,
            "command": test_cmd,
            "error": res.reason or "Command blocked by safety policy",
            "output": res.stderr or res.reason or "",
            "exit_code": None,
        }

    out = res.stdout + ("\n" + res.stderr if res.stderr else "")
    return {
        "ok": res.exit_code == 0,
        "command": test_cmd,
        "exit_code": res.exit_code,
        "output": out,
        "duration_ms": res.duration_ms,
    }


# Error handling
# ---------------------------------------------------------------------------


def register_exception_handlers(app: Any) -> None:
    """Map NovaCLI's own exceptions onto sensible HTTP status codes."""
    from nova.config import ConfigError
    from nova.core.safety import SafetyError as CoreSafetyError, redact_secrets

    @app.exception_handler(ConfigError)
    async def _config_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        settings = getattr(request.app.state, "settings", None)
        secrets = getattr(settings, "active_secrets", ()) if settings else ()
        msg = redact_secrets(str(exc), *secrets)
        return JSONResponse(status_code=400, content={"detail": msg})

    @app.exception_handler(CoreSafetyError)
    async def _safety_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        settings = getattr(request.app.state, "settings", None)
        secrets = getattr(settings, "active_secrets", ()) if settings else ()
        msg = redact_secrets(str(exc), *secrets)
        return JSONResponse(status_code=403, content={"detail": msg})


__all__ = ["router", "register_exception_handlers"]
