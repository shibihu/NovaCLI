"""Mobile-first web IDE.

The web layer is a *pure view*: it starts the very same
:class:`~nova.core.agent.NovaAgent` the CLI uses and streams the resulting
event objects to the browser over Server-Sent Events.

* :mod:`nova.web.app`     — FastAPI application factory
* :mod:`nova.web.routes`  — HTTP + SSE endpoints
* :mod:`nova.web.events`  — agent sessions and the event bus
"""

from __future__ import annotations

__all__ = ["app", "events", "routes"]
