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


def create_app(settings: Settings | None = None) -> "FastAPI":  # noqa: F821
    """Build the FastAPI application.

    Settings are loaded lazily and may legitimately have no API key — the
    server starts (so the UI can show setup instructions) and the error
    surfaces when a task is submitted.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    from nova import __version__
    from nova.web.events import SessionRegistry
    from nova.web.routes import register_exception_handlers, router

    settings = settings or load_settings()

    app = FastAPI(
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
