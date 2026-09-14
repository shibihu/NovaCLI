"""Static wiring checks for the web terminal frontend.

The project has no browser test infrastructure (no bundler, no headless runner),
so browser behaviour cannot be asserted here. What *can* be checked
deterministically is the wiring contract between the template, the vendored
xterm.js bundle and ``app.js``: exactly the things that silently break the
terminal — a missing addon, a missing ``focus()``/``onData()`` registration, a
wrong WebSocket URL, or an accidental fake input element.

This is deliberately a static check, not a claim that clicking the terminal has
been exercised in a real browser.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "nova" / "web"
TEMPLATE = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "static" / "js" / "app.js").read_text(encoding="utf-8")
VENDOR = WEB / "static" / "vendor" / "xterm"


def _terminal_section() -> str:
    """Return just the ``#view-term`` markup from the template."""
    match = re.search(
        r'<section[^>]*id="view-term".*?</section>', TEMPLATE, re.DOTALL
    )
    assert match, "terminal view section not found in index.html"
    return match.group(0)


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def test_xterm_assets_are_declared():
    assert "/static/vendor/xterm/xterm.js" in TEMPLATE
    assert "/static/vendor/xterm/xterm-addon-fit.js" in TEMPLATE
    assert "/static/vendor/xterm/xterm.css" in TEMPLATE


def test_xterm_assets_are_real_bundles():
    """Guard against a placeholder/empty vendor file silently breaking xterm."""
    xterm = (VENDOR / "xterm.js").read_bytes()
    fit = (VENDOR / "xterm-addon-fit.js").read_bytes()
    css = (VENDOR / "xterm.css").read_bytes()

    assert len(xterm) > 100_000, "xterm.js does not look like a real bundle"
    assert b"Terminal" in xterm
    assert b"FitAddon" in fit
    assert b"xterm" in css


# ---------------------------------------------------------------------------
# app.js wiring
# ---------------------------------------------------------------------------


def test_terminal_is_created_and_opened():
    assert "new Terminal(" in APP_JS
    assert re.search(r"\.open\(\s*container\s*\)", APP_JS), "terminal.open(container) missing"
    assert re.search(r"\bfunction initTerminal\b", APP_JS)


def test_terminal_is_focused_on_open_and_on_tab_show():
    """Without focus() keystrokes never reach the shell."""
    open_block = APP_JS[APP_JS.index("new Terminal("):]
    assert ".focus()" in open_block


def test_terminal_registers_input_and_resize():
    assert ".onData(" in APP_JS
    assert ".onResize(" in APP_JS


def test_fit_addon_is_loaded_and_fitted():
    assert "FitAddon" in APP_JS
    assert "loadAddon(" in APP_JS
    assert ".fit()" in APP_JS


def test_websocket_targets_terminal_endpoint():
    assert "new WebSocket(" in APP_JS
    assert "/ws/terminal" in APP_JS


def test_wire_protocol_shapes():
    assert re.search(r'type:\s*"input"', APP_JS), "input frames not sent"
    assert re.search(r'type:\s*"resize"', APP_JS), "resize frames not sent"
    assert re.search(r'msg\.type\s*===\s*"output"', APP_JS), "output frames not handled"
    assert re.search(r"\.write\(\s*msg\.data\s*\)", APP_JS), "output not written to xterm"
    assert re.search(r'msg\.type\s*===\s*"error"', APP_JS), "error frames not surfaced"


def test_resize_is_sent_as_cols_and_rows():
    assert re.search(r"cols:\s*size\.cols", APP_JS)
    assert re.search(r"rows:\s*size\.rows", APP_JS)


def test_failed_connection_is_reported_to_the_user():
    assert "onclose" in APP_JS or "onerror" in APP_JS


# ---------------------------------------------------------------------------
# No fake terminal
# ---------------------------------------------------------------------------


def test_terminal_view_has_no_input_or_textarea():
    """The xterm surface must be the only input; no fake command box."""
    section = _terminal_section()
    assert "<input" not in section
    assert "<textarea" not in section
    assert 'id="terminal-container"' in section


def test_terminal_container_is_the_only_input_surface():
    assert APP_JS.count("new Terminal(") == 1
    # Terminal input flows through WebSocket frames, never through a DOM field.
    assert "document.createElement(\"input\")" not in APP_JS
