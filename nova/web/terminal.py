"""WebSocket terminal endpoint and PTY session transport for NovaCLI.

The endpoint is a thin transport between a browser xterm.js instance and a
platform PTY session. It only ever touches the common
:class:`~nova.core.pty.base.PTYSession` interface (``is_alive``, ``pid``,
``returncode``, ``resize``, ``write``, ``read``, ``close``) so it behaves
identically on Unix, WSL, Termux and Windows.

Wire protocol
-------------
Client -> server::

    {"type": "input",  "data": "ls\\r"}
    {"type": "resize", "cols": 120, "rows": 30}
    {"type": "ping"}

Server -> client::

    {"type": "output", "data": "..."}
    {"type": "exit",   "code": 0}
    {"type": "pong"}
    {"type": "error",  "message": "..."}

The server never echoes input itself — character echo and line editing are the
shell's job, provided by the PTY (POSIX line discipline or the Windows
pseudoconsole).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from nova.core.pty import PTYManager

logger = logging.getLogger(__name__)

router = APIRouter()
pty_manager = PTYManager()

#: A frame can hold at most this much console output (bytes read from the PTY).
_READ_CHUNK = 4096

#: How many times to re-drain the PTY after the shell has exited so the final
#: lines of output are not lost to the race between process exit and read.
_FINAL_DRAIN_ATTEMPTS = 6


def verify_ws_auth(websocket: WebSocket) -> bool:
    """Verify WebSocket client authentication against settings.

    Header credentials (``Authorization: Bearer`` and ``X-Nova-Web-Token``) are
    preferred. A ``?token=`` query parameter is still accepted because browser
    WebSocket clients cannot set request headers on the handshake.
    """
    app = websocket.app
    settings = getattr(app.state, "settings", None)
    web_token = settings.web_token if settings else None

    client_host = websocket.client.host if websocket.client else "127.0.0.1"
    is_localhost = client_host in {"127.0.0.1", "::1", "localhost", "testclient"}

    if not web_token and is_localhost:
        return True

    provided_token = None
    auth_header = websocket.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        provided_token = auth_header[7:].strip()
    if not provided_token:
        provided_token = websocket.headers.get("X-Nova-Web-Token")
    if not provided_token:
        provided_token = websocket.query_params.get("token")

    if not web_token and not is_localhost:
        return False

    return bool(provided_token and provided_token == web_token)


def _redact(message: str, websocket: WebSocket) -> str:
    """Remove any configured secret values from an error message.

    Terminal startup errors must stay diagnosable without ever leaking API keys,
    tokens or passwords into the browser or the logs.
    """
    settings = getattr(websocket.app.state, "settings", None)
    secrets = getattr(settings, "active_secrets", None) or ()
    text = str(message)
    for secret in secrets:
        if secret and len(str(secret)) >= 6:
            text = text.replace(str(secret), "***")
    return text


async def _safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    """Send JSON, tolerating a socket that is already closing."""
    try:
        await websocket.send_json(payload)
        return True
    except Exception:  # pragma: no cover - race with client disconnect
        return False


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send a structured, redacted error to the client."""
    await _safe_send_json(
        websocket, {"type": "error", "message": _redact(message, websocket)}
    )


async def _pump_pty_output(websocket: WebSocket, session) -> None:
    """Stream PTY output to the WebSocket until the shell exits.

    Runs as its own task so a slow or dead client can never stall the shell.
    """
    loop = asyncio.get_running_loop()
    try:
        while session.is_alive:
            data = await loop.run_in_executor(None, session.read, _READ_CHUNK)
            if not data:
                await asyncio.sleep(0.02)
                continue
            if not await _safe_send_json(
                websocket,
                {"type": "output", "data": data.decode("utf-8", errors="replace")},
            ):
                return
    except asyncio.CancelledError:
        raise
    except (WebSocketDisconnect, RuntimeError, OSError):
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("PTY output stream error (session %s): %s", session.id, exc)
        return

    # The shell exited: give the reader a moment to flush its last bytes.
    for _ in range(_FINAL_DRAIN_ATTEMPTS):
        try:
            data = await loop.run_in_executor(None, session.read, _READ_CHUNK)
        except Exception:
            break
        if data:
            if not await _safe_send_json(
                websocket,
                {"type": "output", "data": data.decode("utf-8", errors="replace")},
            ):
                return
        else:
            await asyncio.sleep(0.05)

    code = session.returncode
    if code is not None:
        await _safe_send_json(websocket, {"type": "exit", "code": code})


async def _terminal_input_loop(websocket: WebSocket, session) -> None:
    """Forward client messages into the PTY until the shell exits."""
    while session.is_alive:
        raw_msg = await websocket.receive_text()
        if not raw_msg:
            continue

        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            # Not JSON: treat the payload as raw keystroke data.
            session.write(raw_msg)
            continue

        if not isinstance(msg, dict):
            # A JSON scalar (string/number) is still usable as literal input.
            session.write(raw_msg)
            continue

        msg_type = msg.get("type", "input")

        if msg_type == "input":
            data = msg.get("data", "")
            if isinstance(data, str) and data:
                session.write(data)

        elif msg_type == "resize":
            try:
                new_cols = int(msg.get("cols", session.cols))
                new_rows = int(msg.get("rows", session.rows))
            except (TypeError, ValueError):
                await _send_error(websocket, "Invalid resize dimensions.")
                continue
            try:
                session.resize(new_cols, new_rows)
            except Exception as exc:
                logger.warning("PTY resize failed (session %s): %s", session.id, exc)
                await _send_error(websocket, f"Resize failed: {exc}")

        elif msg_type == "ping":
            await _safe_send_json(websocket, {"type": "pong"})


@router.websocket("/ws/terminal")
async def terminal_websocket(websocket: WebSocket) -> None:
    """Interactive WebSocket endpoint connected to an OS PTY shell."""
    await websocket.accept()

    if not verify_ws_auth(websocket):
        await _safe_send_json(
            websocket,
            {"type": "error", "message": "Invalid or missing authentication token."},
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    settings = getattr(websocket.app.state, "settings", None)
    project_root = settings.project_root if settings else "."

    # Get initial cols/rows if provided in query string
    try:
        cols = int(websocket.query_params.get("cols", 80))
        rows = int(websocket.query_params.get("rows", 24))
    except (ValueError, TypeError):
        cols, rows = 80, 24

    # Create the PTY session. Failures are surfaced to the client instead of
    # being swallowed, because a terminal that cannot start is unusable.
    try:
        server_env = dict(settings.environment) if settings and hasattr(settings, "environment") and settings.environment else None
        session = pty_manager.create(project_root, cols=cols, rows=rows, env=server_env)
    except Exception as exc:
        logger.error("Failed to start PTY terminal session: %s", exc)
        await _send_error(websocket, f"Failed to start terminal: {exc}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Pump output and input concurrently. Whichever finishes first ends the
    # session: the shell exiting tears down the socket, and the client
    # disconnecting stops the pump. This keeps the lifecycle deterministic
    # instead of waiting for the next client message to notice a dead shell.
    output_task = asyncio.create_task(_pump_pty_output(websocket, session))
    input_task = asyncio.create_task(_terminal_input_loop(websocket, session))
    tasks = (output_task, input_task)

    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
            elif not task.cancelled() and task.exception() is not None:
                exc = task.exception()
                if isinstance(exc, WebSocketDisconnect):
                    logger.debug("Terminal client disconnected (session %s)", session.id)
                else:
                    logger.warning(
                        "WebSocket terminal error (session %s): %s", session.id, exc
                    )
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        pty_manager.close(session.id)


__all__ = ["pty_manager", "router", "verify_ws_auth"]
