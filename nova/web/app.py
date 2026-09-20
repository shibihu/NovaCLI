"""FastAPI application factory for the NovaCLI web IDE.

The app is intentionally thin. It holds a :class:`~nova.config.Settings`
object and a :class:`~nova.web.events.SessionRegistry`, mounts static assets,
and delegates every agent action to Nova Core.
"""

from __future__ import annotations

from pathlib import Path

from nova.config import Settings, load_settings

WEB_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_ROOT / "templates"
STATIC_DIR = WEB_ROOT / "static"


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from nova.web.terminal import pty_manager

    async def _cleanup_loop():
        while True:
            try:
                await asyncio.sleep(30.0)
                pty_manager.cleanup_orphans()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        if not cleanup_task.done():
            cleanup_task.cancel()

def create_app(settings: Settings | None = None) -> "FastAPI":  # noqa: F821
    """Build the FastAPI application.

    Settings are loaded lazily and may legitimately have no API key — the
    server starts (so the UI can show setup instructions) and the error
    surfaces when a task is submitted.
    """
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    from nova import __version__
    from nova.web.events import SessionRegistry
    from nova.web.routes import register_exception_handlers, router

    settings = settings or load_settings()

    app = FastAPI(
        lifespan=lifespan,
        title="NovaCLI",
        version=__version__,
        description="AI-powered developer environment — agent, CLI and mobile web IDE.",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.state.settings = settings
    app.state.registry = SessionRegistry()
    app.state.version = __version__

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        path = request.url.path
        if (
            path in {"/", "/api/health", "/api/docs", "/api/openapi.json"}
            or path.startswith("/static/")
            or path == "/static"
            or path.startswith("/ws/")
        ):
            return await call_next(request)

        if path.startswith("/api/"):
            current_settings = getattr(app.state, "settings", None)
            web_token = current_settings.web_token if current_settings else None

            client_host = request.client.host if request.client else "127.0.0.1"
            is_localhost = client_host in {
                "127.0.0.1",
                "::1",
                "localhost",
                "testclient",
            }

            if web_token or not is_localhost:
                provided_token = None
                auth_header = request.headers.get("Authorization", "")
                if auth_header.lower().startswith("bearer "):
                    provided_token = auth_header[7:].strip()
                if not provided_token:
                    provided_token = request.headers.get("X-Nova-Web-Token")
                # Query-string token parameter is deprecated for REST APIs and allowed only for EventSource streams
                if not provided_token and path == "/api/agent/stream":
                    provided_token = request.query_params.get("token")

                if not web_token and not is_localhost:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "detail": (
                                "Authentication token required for non-localhost access. "
                                "Set NOVA_WEB_TOKEN environment variable."
                            )
                        },
                    )

                if not provided_token or provided_token != web_token:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing authentication token."},
                    )

        return await call_next(request)

    # Local single-user tool: permissive CORS keeps the phone browser happy
    # when the IDE is reached over a LAN address or a Termux port-forward.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    from nova.web.terminal import router as terminal_router

    app.include_router(terminal_router)
    app.include_router(router)
    register_exception_handlers(app)



    return app


def get_settings(app: "FastAPI") -> Settings:  # noqa: F821
    """Fetch the settings attached to a running app."""
    return app.state.settings


def get_registry(app: "FastAPI") -> object:  # noqa: F821
    """Fetch the session registry attached to a running app."""
    return app.state.registry


__all__ = ["create_app", "get_registry", "get_settings", "STATIC_DIR", "TEMPLATES_DIR"]
