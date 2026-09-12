"""HTTP and Server-Sent-Events endpoints for the web IDE.

Every endpoint either reads workspace state or delegates to Nova Core. No agent
logic is duplicated here — starting a task just constructs the same
:class:`~nova.core.agent.NovaAgent` the CLI builds, then streams its events.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nova.ai import AIProviderError, get_provider
from nova.config import API_KEY_HINT, Settings
from nova.core.agent import AgentController, build_agent
from nova.core.models import ApprovalDecision, RiskLevel
from nova.core.runner import CommandRunner
from nova.core.safety import SafetyError, SafetyPolicy
from nova.workspace.files import Workspace, detect_language
from nova.workspace.projects import ProjectAnalyzer
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
        secret_values=[settings.groq_api_key],
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
            "model": settings.groq_model,
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
        "model": settings.groq_model,
        "has_api_key": settings.has_api_key,
        "project_root": str(settings.project_root),
        "safety_mode": settings.safety_mode,
    }


@router.get("/api/config")
async def config(request: Request) -> dict[str, Any]:
    """Effective configuration. Never contains secret material."""
    return _settings(request).to_public_dict()


# ---------------------------------------------------------------------------
# Project / files
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


@router.get("/api/tree")
async def tree(
    request: Request,
    path: str = Query(".", max_length=1_000),
    depth: int = Query(2, ge=1, le=6),
) -> dict[str, Any]:
    workspace = _workspace(_settings(request))
    try:
        return {"path": path, "tree": workspace.tree(path, max_depth=depth, max_entries=300)}
    except (OSError, ValueError, SafetyError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.get("/api/files")
async def list_files(
    request: Request,
    path: str = Query(".", max_length=1_000),
) -> dict[str, Any]:
    workspace = _workspace(_settings(request))
    try:
        entries = workspace.list_dir(path)
    except (OSError, ValueError, SafetyError) as exc:
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
    """Run a shell command through the safety layer and runner.

    A command that needs approval but has not been approved returns
    ``requires_approval: true`` with HTTP 200, so the UI can show its confirm
    dialog and re-post with ``approve: true``. Refusals return HTTP 403.
    """
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
        body.command, cwd=body.cwd, approved=body.approve, check_safety=False
    )
    return {"requires_approval": False, "result": result.to_dict()}


# ---------------------------------------------------------------------------
# Agent sessions
# ---------------------------------------------------------------------------


def _agent_factory(settings: Settings):
    """Build the factory the session registry calls to create an agent.

    Construction is deferred so a missing dependency shows up as a streamed
    error event rather than an import failure at request time.
    """

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
        # Fail fast with setup instructions rather than a background error.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, API_KEY_HINT)

    controller = AgentController(
        auto_approve=body.auto_approve, approval_timeout=settings.approval_timeout
    )
    session = registry.create(body.task, controller=controller, model=settings.groq_model)
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
# Error handling
# ---------------------------------------------------------------------------


def register_exception_handlers(app: Any) -> None:
    """Map NovaCLI's own exceptions onto sensible HTTP status codes."""
    from nova.config import ConfigError
    from nova.core.safety import SafetyError as CoreSafetyError

    @app.exception_handler(ConfigError)
    async def _config_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(CoreSafetyError)
    async def _safety_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=403, content={"detail": str(exc)})


__all__ = ["router", "register_exception_handlers"]
