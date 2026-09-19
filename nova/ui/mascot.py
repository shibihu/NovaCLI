"""Nova Mascot Identity and State Machine.

Provides a unified visual identity and safe high-level status messages
for both CLI and Web IDE without exposing private internal chain-of-thought.
"""

from __future__ import annotations

from typing import Any

MASCOT_UNICODE = {
    "idle": "✦ Nova",
    "thinking": "✦ Nova …",
    "working": "✦ Nova ⚙",
    "success": "✦ Nova ✓",
    "error": "✦ Nova !",
    "waiting": "✦ Nova ⏳",
}

MASCOT_ASCII = {
    "idle": "[N] Nova",
    "thinking": "[N] Nova ...",
    "working": "[N] Nova *",
    "success": "[N] Nova OK",
    "error": "[N] Nova !",
    "waiting": "[N] Nova WAIT",
}

SAFE_STATUS_TEMPLATES = {
    "connecting": "Nova is connecting to workspace…",
    "inspecting": "Nova is inspecting the project…",
    "planning": "Nova is planning the changes…",
    "executing": "Nova is executing tasks…",
    "testing": "Nova is running tests…",
    "waiting_rate_limit": "Nova is waiting for the API rate limit…",
    "finishing": "Nova is checking the result…",
    "idle": "Nova is ready",
    "success": "Finished ✓",
    "error": "Encountered an error",
}


def render_mascot(state: str = "idle", use_unicode: bool = True) -> str:
    """Render the Nova mascot symbol for a given state."""
    clean_state = (state or "idle").lower().strip()
    table = MASCOT_UNICODE if use_unicode else MASCOT_ASCII
    return table.get(clean_state, table["idle"])


def get_safe_status(state: str, context: dict[str, Any] | None = None) -> str:
    """Return a safe high-level status string without exposing private CoT."""
    clean_state = (state or "idle").lower().strip()
    ctx = context or {}

    if clean_state in ("tool_call", "executing"):
        tool = ctx.get("tool") or ctx.get("name")
        if tool == "project_summary":
            return "Nova is inspecting project summary…"
        if tool in ("read_file", "search", "list_files"):
            return "Nova is inspecting project files…"
        if tool == "write_file":
            path = ctx.get("input", {}).get("path") or ctx.get("path")
            return f"Nova is editing {path}…" if path else "Nova is editing files…"
        if tool == "run_command":
            return "Nova is executing command…"
        return f"Nova is calling {tool}…"

    if clean_state in ("rate_limit_wait", "waiting"):
        delay = ctx.get("retry_after") or ctx.get("delay")
        if delay:
            return f"Nova is waiting for API rate limit ({delay}s)…"
        return "Nova is waiting for API rate limit…"

    return SAFE_STATUS_TEMPLATES.get(clean_state, "Nova is working…")


__all__ = [
    "MASCOT_ASCII",
    "MASCOT_UNICODE",
    "SAFE_STATUS_TEMPLATES",
    "get_safe_status",
    "render_mascot",
]
