"""NovaCLI — an AI-powered developer environment.

Nova Core (:mod:`nova.core`, :mod:`nova.ai`, :mod:`nova.workspace`) is the
single implementation of agent behaviour. Both the CLI (:mod:`nova.cli`) and
the web IDE (:mod:`nova.web`) are thin shells over it, so the two surfaces can
never drift apart.

Typical programmatic use::

    import asyncio
    from nova.config import load_settings
    from nova.core.agent import NovaAgent
    from nova.ai import get_provider

    settings = load_settings()
    agent = NovaAgent(provider=get_provider(settings), settings=settings)
    result = asyncio.run(agent.run("Summarise this project"))
    print(result.answer)
"""

from __future__ import annotations

from typing import Any

__version__ = "2.0.0"
__all__ = ["__version__", "Settings", "ConfigError", "load_settings"]


def __getattr__(name: str) -> Any:
    """Lazily expose :mod:`nova.config` names without importing at startup.

    Keeping the package ``__init__`` import-free means ``import nova`` stays
    cheap and cannot raise a configuration error as a side effect.
    """
    if name in {"Settings", "ConfigError", "load_settings"}:
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
