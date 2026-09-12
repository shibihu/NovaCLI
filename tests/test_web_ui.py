"""Tests for the web IDE's UI assets.

These assert the properties that make the IDE usable on a phone: correct
viewport configuration, safe-area handling, large touch targets, the four
views, the approval dialog, and a client that consumes the shared event
stream.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB_ROOT = Path(__file__).resolve().parents[1] / "nova" / "web"
TEMPLATE = WEB_ROOT / "templates" / "index.html"
CSS = WEB_ROOT / "static" / "css" / "app.css"
JS = WEB_ROOT / "static" / "js" / "app.js"


@pytest.fixture(scope="module")
def html() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return JS.read_text(encoding="utf-8")


# --- Files exist ------------------------------------------------------------


def test_ui_files_exist() -> None:
    assert TEMPLATE.is_file()
    assert CSS.is_file()
    assert JS.is_file()


# --- Mobile readiness -------------------------------------------------------


def test_viewport_meta_is_mobile_ready(html: str) -> None:
    match = re.search(r'<meta name="viewport" content="([^"]+)"', html)
    assert match, "viewport meta tag is required"
    content = match.group(1)
    assert "width=device-width" in content
    assert "initial-scale=1" in content


def test_viewport_covers_notched_screens(html: str) -> None:
    assert "viewport-fit=cover" in html


def test_theme_colour_is_declared(html: str) -> None:
    assert 'name="theme-color"' in html


def test_installs_as_a_web_app(html: str) -> None:
    assert "mobile-web-app-capable" in html


def test_language_and_title(html: str) -> None:
    assert '<html lang="en">' in html
    assert "<title>" in html


def test_css_uses_safe_area_insets(css: str) -> None:
    assert "env(safe-area-inset-bottom" in css
    assert "env(safe-area-inset-top" in css


def test_css_defines_a_large_touch_target(css: str) -> None:
    match = re.search(r"--tap:\s*(\d+)px", css)
    assert match, "--tap must be defined"
    assert int(match.group(1)) >= 44


def test_css_is_dark_first_with_a_light_override(css: str) -> None:
    assert "--bg:" in css
    assert "prefers-color-scheme: light" in css


def test_css_prevents_horizontal_scroll(css: str) -> None:
    assert "overflow-x: hidden" in css


def test_css_has_a_desktop_breakpoint(css: str) -> None:
    assert "@media (min-width: 720px)" in css


def test_css_respects_reduced_motion(css: str) -> None:
    assert "prefers-reduced-motion" in css


def test_css_uses_dynamic_viewport_height(css: str) -> None:
    assert "dvh" in css


# --- Structure --------------------------------------------------------------


@pytest.mark.parametrize(
    "view",
    ["view-chat", "view-files", "view-term", "view-project"],
)
def test_all_four_views_are_present(html: str, view: str) -> None:
    assert f'id="{view}"' in html


@pytest.mark.parametrize("tab", ["chat", "files", "term", "project"])
def test_bottom_navigation_covers_every_view(html: str, tab: str) -> None:
    assert f'data-view="{tab}"' in html


def test_bottom_navigation_is_a_tablist(html: str) -> None:
    assert 'role="tablist"' in html
    assert html.count('role="tab"') >= 4


def test_chat_controls_exist(html: str) -> None:
    for element in ("composer", "task", "send", "stop", "messages", "progress-bar"):
        assert f'id="{element}"' in html


def test_approval_dialog_exists(html: str) -> None:
    assert 'id="approval-modal"' in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html


def test_approval_dialog_offers_all_decisions(html: str) -> None:
    for button in ("approve-once", "approve-always", "approve-deny"):
        assert f'id="{button}"' in html


def test_file_editor_controls_exist(html: str) -> None:
    for element in ("file-list", "editor", "editor-save", "editor-close", "files-path"):
        assert f'id="{element}"' in html


def test_terminal_controls_exist(html: str) -> None:
    assert 'id="term-form"' in html
    assert 'id="term-input"' in html
    assert 'id="terminal-out"' in html


def test_project_panel_exists(html: str) -> None:
    assert 'id="project-cards"' in html
    assert 'id="project-tree"' in html


def test_setup_banner_explains_missing_key(html: str) -> None:
    assert 'id="setup-banner"' in html
    assert "GROQ_API_KEY" in html
    assert "console.groq.com" in html


def test_assets_are_linked(html: str) -> None:
    assert "/static/css/app.css" in html
    assert "/static/js/app.js" in html


def test_labels_are_provided_for_inputs(html: str) -> None:
    assert 'class="sr-only"' in html


def test_empty_state_offers_suggestions(html: str) -> None:
    assert 'class="suggestion"' in html


# --- Client behaviour -------------------------------------------------------


def test_client_uses_event_source(js: str) -> None:
    assert "EventSource" in js


def test_client_calls_the_agent_endpoints(js: str) -> None:
    for endpoint in ("/api/agent", "/api/agent/stream", "/api/agent/approve", "/api/agent/cancel"):
        assert endpoint in js


def test_client_calls_the_workspace_endpoints(js: str) -> None:
    for endpoint in ("/api/files", "/api/file", "/api/run", "/api/project", "/api/health"):
        assert endpoint in js


def test_client_handles_every_event_type(js: str) -> None:
    for event_type in (
        "agent_start", "step_start", "thought", "tool_call", "tool_result",
        "approval_request", "approval_resolved", "blocked", "progress",
        "final", "error", "cancelled",
    ):
        assert f'"{event_type}"' in js, f"client does not handle {event_type}"


def test_client_is_wrapped_in_an_iife(js: str) -> None:
    """An IIFE keeps the client's internals out of the global scope."""
    assert "(function () {" in js
    assert js.rstrip().endswith("})();")
    assert '"use strict"' in js


def test_client_escapes_html(js: str) -> None:
    assert "&amp;" in js
    assert "innerHTML" in js


def test_client_has_no_framework_dependency(js: str) -> None:
    assert "import " not in js
    assert "require(" not in js


def test_client_manages_the_approval_modal(js: str) -> None:
    assert "approval-modal" in js
    assert "state.approvalId" in js
