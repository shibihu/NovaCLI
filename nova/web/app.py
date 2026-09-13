"""FastAPI application factory for the NovaCLI web IDE.

The app holds a :class:`~nova.config.Settings` object and a
:class:`~nova.web.events.SessionRegistry`, mounts static assets, enforces
authentication for non-localhost/LAN or configured token access, and
delegates every agent action to Nova Core.
"""

from __future__ import annotations

from pathlib import Path

from nova.config import Settings, load_settings

WEB_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_ROOT / "templates"
STATIC_DIR = WEB_ROOT / "static"


def create_app(settings: Settings | None = None) -> "FastAPI":  # noqa: F821
    """Build the FastAPI application."""
    from fastapi import FastAPI, HTTPException, Request, status
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

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        """Enforce authentication on non-localhost or when a web token is required."""
        path = request.url.path

        # Unprotected static assets and index UI shell
        if path == "/" or path.startswith("/static") or path in ("/favicon.ico", "/api/openapi.json", "/api/docs"):
            return await call_next(request)

        req_settings: Settings = app.state.settings
        client_host = request.client.host if request.client else ""
        is_local = client_host in ("127.0.0.1", "::1", "localhost", "testclient")

        # Require authentication if configured token is set OR if request is coming from non-localhost (LAN/0.0.0.0)
        token_required = req_settings.has_web_token or not is_local

        if token_required:
            auth_header = request.headers.get("authorization", "")
            provided_token = ""

            if auth_header.lower().startswith("bearer "):
                provided_token = auth_header[7:].strip()

            if not provided_token:
                provided_token = request.query_params.get("token", "")

            expected_token = req_settings.web_token

            if not expected_token:
                # If bound to LAN without a token set, deny access to state-changing endpoints for safety
                if not provided_token or provided_token != expected_token:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Authentication required for non-localhost/LAN requests. Set NOVA_WEB_TOKEN."},
                    )
            elif provided_token != expected_token:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing Web API token."},
                )

        return await call_next(request)

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
    return app.state.settings


def get_registry(app: "FastAPI") -> object:  # noqa: F821
    return app.state.registry


__all__ = ["create_app", "get_registry", "get_settings", "STATIC_DIR", "TEMPLATES_DIR"]
