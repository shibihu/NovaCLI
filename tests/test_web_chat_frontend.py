"""Static + API-contract checks for the Recent Chats frontend.

The project has no browser test infrastructure (no bundler, no headless runner),
so browser behaviour cannot be asserted here. What *can* be checked
deterministically is the two things that silently break the feature:

1. the wiring contract between ``index.html``, ``app.css`` and ``app.js`` —
   every element the JS binds, every endpoint it calls, the active-session id
   it propagates, and the confirmation gate before a destructive delete;
2. the REST contract those calls depend on, exercised against the real app.

This is deliberately not a claim that clicking a Recent Chat has been tested in
a real browser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi is required for the web tests")
pytest.importorskip("httpx", reason="httpx is required by fastapi's TestClient")

from fastapi.testclient import TestClient  # noqa: E402

from nova.web.app import create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "nova" / "web"
TEMPLATE = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "static" / "js" / "app.js").read_text(encoding="utf-8")
APP_CSS = (WEB / "static" / "css" / "app.css").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "element_id",
    [
        "chat-sidebar",
        "chat-session-list",
        "btn-new-chat",
        "btn-rename-chat",
        "btn-delete-chat",
        "sidebar-toggle",
        "active-chat-title",
        "messages",
    ],
)
def test_template_declares_the_recent_chats_elements(element_id: str) -> None:
    assert f'id="{element_id}"' in TEMPLATE


def test_sidebar_starts_collapsed_for_the_mobile_drawer() -> None:
    match = re.search(r'<aside[^>]*id="chat-sidebar"[^>]*>', TEMPLATE)
    assert match, "chat sidebar not found"
    assert "is-collapsed" in match.group(0)


# ---------------------------------------------------------------------------
# app.js wiring
# ---------------------------------------------------------------------------


def test_recent_chats_are_loaded_on_startup() -> None:
    assert 'api("/api/agent/sessions")' in APP_JS
    boot = APP_JS[APP_JS.index("async function boot()") :]
    assert "loadChatSessions()" in boot


def test_sessions_are_rendered_into_the_sidebar_list() -> None:
    assert re.search(r'\$\("chat-session-list"\)', APP_JS)
    assert re.search(r"\bfunction renderChatSessions\b", APP_JS)
    # Titles are escaped before being written as HTML.
    assert re.search(r"esc\(session\.title", APP_JS)
    assert re.search(r"\bfunction relativeTime\b", APP_JS)


def test_sessions_are_sorted_newest_first() -> None:
    assert re.search(r"\.sort\(", APP_JS)
    assert "updated_at" in APP_JS


def test_opening_a_chat_fetches_it_and_renders_its_conversation() -> None:
    assert re.search(r"\bfunction openChatSession\b", APP_JS)
    assert re.search(r'"/api/agent/sessions/"\s*\+\s*encodeURIComponent', APP_JS)
    assert re.search(r"\bfunction renderConversation\b", APP_JS)
    # Persisted turns are rendered from the canonical message shape.
    assert re.search(r'message\.role === "user"', APP_JS)
    assert re.search(r'message\.role === "tool"', APP_JS)
    assert re.search(r"message\.tool_calls", APP_JS)


def test_new_chat_creates_a_session_and_does_not_reuse_the_old_one() -> None:
    assert re.search(r'\$\("btn-new-chat"\)', APP_JS)
    assert re.search(r"\bfunction newChat\b", APP_JS)
    new_chat = APP_JS[APP_JS.index("async function newChat") :]
    assert re.search(r'api\("/api/agent/sessions",\s*\{\s*method:\s*"POST"', new_chat)
    assert "state.activeChatSessionId = session.id" in new_chat
    assert re.search(r"\bfunction resetToNewChat\b", APP_JS)


def test_rename_uses_patch_and_updates_the_visible_title() -> None:
    assert re.search(r"\bfunction renameActiveChat\b", APP_JS)
    rename = APP_JS[APP_JS.index("async function renameActiveChat") :]
    assert re.search(r'method:\s*"PATCH"', rename)
    assert "setActiveChatTitle(" in rename


def test_delete_asks_for_confirmation_and_resets_the_ui() -> None:
    assert re.search(r"\bfunction deleteActiveChat\b", APP_JS)
    delete = APP_JS[APP_JS.index("async function deleteActiveChat") :]
    assert re.search(r"window\.confirm\(", delete), "destructive delete must be confirmed"
    assert re.search(r'method:\s*"DELETE"', delete)
    # The UI must not keep pointing at the deleted session.
    assert "resetToNewChat()" in delete


def test_active_chat_session_id_is_maintained_and_sent_to_the_agent() -> None:
    assert "activeChatSessionId" in APP_JS
    assert re.search(
        r"session_id:\s*state\.activeChatSessionId", APP_JS
    ), "the agent request must carry the active chat session id"


def test_continuing_a_chat_reloads_the_recent_chats_list() -> None:
    final_branch = APP_JS[APP_JS.index('case "final":') :]
    assert "loadChatSessions()" in final_branch[:2000]


def test_sidebar_toggle_wires_the_mobile_drawer() -> None:
    assert re.search(r'\$\("sidebar-toggle"\)', APP_JS)
    assert re.search(r"\bfunction toggleSidebar\b", APP_JS)
    assert re.search(r"\bfunction collapseSidebar\b", APP_JS)


def test_no_duplicate_endpoint_is_invented() -> None:
    """The frontend uses the existing session API only."""
    assert "/api/sessions" not in APP_JS
    assert "/api/chats" not in APP_JS


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def test_sidebar_is_a_drawer_on_phones_and_a_column_on_desktop() -> None:
    assert ".chat-sidebar.is-collapsed" in APP_CSS
    assert "translateX(" in APP_CSS
    desktop = APP_CSS[APP_CSS.index("@media (min-width: 720px)") :]
    assert ".chat-sidebar" in desktop
    assert "position: static" in desktop


def test_session_list_and_header_are_styled() -> None:
    for selector in (
        ".chat-layout",
        ".chat-main",
        ".chat-header",
        ".active-chat-title",
        ".chat-session-list",
        ".session-item",
    ):
        assert selector in APP_CSS, f"{selector} is not styled"


# ---------------------------------------------------------------------------
# REST contract used by the wiring above
# ---------------------------------------------------------------------------


def test_sessions_endpoint_exposes_the_fields_the_ui_reads(settings) -> None:
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/agent/sessions", json={"task": "Recent chats contract"})
        assert created.status_code == 200
        session_id = created.json()["session"]["id"]

        listed = client.get("/api/agent/sessions").json()["sessions"]
        assert listed, "the sidebar would render nothing"
        first = listed[0]
        assert set(first) >= {"id", "title", "created_at", "updated_at", "status", "model"}
        assert "message_count" in first

        detail = client.get(f"/api/agent/sessions/{session_id}").json()["session"]
        assert isinstance(detail["messages"], list)
        assert detail["title"]

        assert client.delete(f"/api/agent/sessions/{session_id}").status_code == 200


def test_new_chat_button_payload_works_without_a_body(settings) -> None:
    """``newChat()`` posts an empty body; the endpoint must accept it."""
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/agent/sessions")
        assert created.status_code == 200
        assert created.json()["session"]["title"] == "New Chat"


def test_session_payloads_never_contain_raw_events(settings) -> None:
    """The UI renders canonical messages, so none of them may look like events."""
    with TestClient(create_app(settings)) as client:
        session_id = client.post("/api/agent/sessions", json={"task": "shape check"}).json()[
            "session"
        ]["id"]
        messages = client.get(f"/api/agent/sessions/{session_id}").json()["session"][
            "messages"
        ]
    assert messages == []
