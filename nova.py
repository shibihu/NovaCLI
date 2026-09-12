#!/usr/bin/env python3
"""NovaCLI entry point.

Run directly:

    python nova.py ask "what does this project do?"

or, after ``pip install -e .``:

    nova ask "what does this project do?"

The real implementation lives in the ``nova`` package; this file exists so the
project can be run straight from a checkout on Termux.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `nova` importable when this script is invoked from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nova.cli.commands import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
