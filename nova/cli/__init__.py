"""Command-line interface for NovaCLI.

:mod:`nova.cli.commands` holds the argparse app. It renders the shared
:class:`~nova.core.models.AgentEvent` stream as terminal output; it contains no
agent logic of its own.
"""

from __future__ import annotations

__all__ = ["commands"]
