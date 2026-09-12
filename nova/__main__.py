"""Support ``python -m nova``."""

from __future__ import annotations

from nova.cli.commands import main

if __name__ == "__main__":
    raise SystemExit(main())
