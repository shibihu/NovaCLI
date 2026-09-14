"""WebSocket terminal endpoint and PTY session transport for NovaCLI."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from nova.core.pty import PTYManager

logger = logging.getLogger(__name__)

router = APIRouter()
pty_manager = PTYManager()


def verify_ws_auth(websocket: WebSocket) -> bool:
    """Verify WebSocket client authentication against settings."""
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


@router.websocket("/ws/terminal")
async def terminal_websocket(websocket: WebSocket) -> None:
    """Interactive WebSocket endpoint connected to an OS PTY shell."""
    await websocket.accept()

    if not verify_ws_auth(websocket):
        await websocket.send_json(
            {"type": "error", "message": "Invalid or missing authentication token."}
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

    # Create new PTY session
    session = pty_manager.create(project_root, cols=cols, rows=rows)

    # Background task to stream PTY output to WebSocket
    async def stream_pty_output():
        loop = asyncio.get_running_loop()
        try:
            while session.is_alive:
                # Read output in thread to avoid blocking loop
                data = await loop.run_in_executor(None, session.read, 4096)
                if not data:
                    await asyncio.sleep(0.02)
                    continue
                decoded = data.decode("utf-8", errors="replace")
                await websocket.send_json({"type": "output", "data": decoded})
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        except Exception as exc:
            logger.debug("PTY output stream error: %s", exc)
        finally:
            if not session.is_alive and session.process.returncode is not None:
                try:
                    await websocket.send_json(
                        {"type": "exit", "code": session.process.returncode}
                    )
                except Exception:
                    pass

    output_task = asyncio.create_task(stream_pty_output())

    try:
        while session.is_alive:
            raw_msg = await websocket.receive_text()
            if not raw_msg:
                continue

            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                # Treat raw text input as keystroke data
                session.write(raw_msg)
                continue

            msg_type = msg.get("type", "input")
            if msg_type == "input":
                data = msg.get("data", "")
                if data:
                    session.write(data)
            elif msg_type == "resize":
                new_cols = int(msg.get("cols", session.cols))
                new_rows = int(msg.get("rows", session.rows))
                session.resize(new_cols, new_rows)
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket terminal error: %s", exc)
    finally:
        output_task.cancel()
        pty_manager.close(session.id)


__all__ = ["pty_manager", "router", "verify_ws_auth"]
