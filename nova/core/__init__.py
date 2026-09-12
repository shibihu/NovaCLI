"""Nova Core — the shared brain used by every NovaCLI surface.

Modules:

* :mod:`nova.core.models`  — plain data types shared across the codebase
* :mod:`nova.core.safety`  — Smart Mode risk analysis and secret redaction
* :mod:`nova.core.runner`  — sandboxed shell command execution
* :mod:`nova.core.context` — project understanding / prompt assembly
* :mod:`nova.core.agent`   — the reasoning + tool-use loop
"""

from __future__ import annotations

__all__ = [
    "agent",
    "context",
    "models",
    "runner",
    "safety",
]
