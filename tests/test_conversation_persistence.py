"""Canonical conversation persistence — a session stores the real ``Message[]``.

The invariants under test:

* what is stored is the canonical conversation (user / assistant / tool), not a
  transcript of ``AgentEvent`` progress notifications;
* internal ``thought`` events never become assistant turns;
* native tool calls keep their id, name, raw arguments and provider metadata;
* a restart restores the same structure, and a continuation replays the prior
  turns exactly once (no duplication, no rebuild from display text);
* legacy event-shaped session files still load;
* ``updated_at`` moves once per assistant response, not per tool step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi is required for the web tests")
pytest.importorskip("httpx", reason="httpx is required by fastapi's TestClient")

from fastapi.testclient import TestClient  # noqa: E402

from nova.core.models import AIResponse, ToolCall  # noqa: E402
from nova.core.sessions import SessionStorageManager  # noqa: E402
from nova.workspace.files import Workspace  # noqa: E402
from nova.web.app import create_app  # noqa: E402

from conftest import FakeProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_sse(client: TestClient, url: str) -> list[dict]:
    """Collect JSON payloads from an SSE stream."""
    events: list[dict] = []
    with client.stream("GET", url) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            events.append(json.loads(line[len("data:") :].strip()))
    return events


def make_client(monkeypatch, settings, provider) -> TestClient:
    monkeypatch.setattr("nova.web.routes.get_provider", lambda _settings: provider)
    return TestClient(create_app(settings))


def run_task(client: TestClient, task: str, session_id: str | None = None) -> tuple[str, list[dict]]:
    body: dict[str, object] = {"task": task}
    if session_id:
        body["session_id"] = session_id
    created = client.post("/api/agent", json=body)
    assert created.status_code == 201, created.text
    sid = created.json()["session_id"]
    events = read_sse(client, f"/api/agent/stream?session_id={sid}")
    return sid, events


def session_messages(client: TestClient, session_id: str) -> list[dict]:
    response = client.get(f"/api/agent/sessions/{session_id}")
    assert response.status_code == 200, response.text
    return response.json()["session"]["messages"]


def roles(messages: list[dict]) -> list[str]:
    return [str(m.get("role")) for m in messages]


# ---------------------------------------------------------------------------
# A. Canonical shape + restart round trip
# ---------------------------------------------------------------------------


def test_session_stores_canonical_conversation_not_events(monkeypatch, settings) -> None:
    provider = FakeProvider(
        [
            FakeProvider.native_tool_call("read_file", call_id="call_1", path="main.py"),
            FakeProvider.final("It greets."),
        ]
    )
    with make_client(monkeypatch, settings, provider) as client:
        sid, events = run_task(client, "what is main.py?")

        assert [event["type"] for event in events][-1] == "final"
        # Progress/thought events still drive the UI...
        assert {"thought", "tool_call", "tool_result"} <= {event["type"] for event in events}

        # ...but the stored conversation is the canonical one.
        messages = session_messages(client, sid)

    assert roles(messages) == ["user", "assistant", "tool", "assistant"]

    # The stored user turn is the raw task, not the context-augmented prompt.
    assert messages[0]["content"] == "what is main.py?"

    assistant_call = messages[1]
    assert assistant_call["tool_calls"][0]["id"] == "call_1"
    assert assistant_call["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(assistant_call["tool_calls"][0]["function"]["arguments"]) == {
        "path": "main.py"
    }

    tool_result = messages[2]
    assert tool_result["tool_call_id"] == "call_1"
    assert tool_result["name"] == "read_file"
    assert "hello" in tool_result["content"]

    assert messages[3]["content"] == "It greets."

    # Nothing resembling a progress notification leaked into the conversation.
    for message in messages:
        assert "type" not in message
        content = str(message.get("content") or "")
        assert "Nova is" not in content
        assert not content.startswith("Analyzing")


def test_conversation_survives_an_application_restart(monkeypatch, settings) -> None:
    provider = FakeProvider(
        [
            FakeProvider.native_tool_call("project_summary", call_id="call_a"),
            FakeProvider.final("Summary complete."),
        ]
    )
    with make_client(monkeypatch, settings, provider) as client:
        sid, _ = run_task(client, "summarize the project")
        before = session_messages(client, sid)

    # A brand new application instance over the same workspace (a restart).
    restarted = FakeProvider([FakeProvider.final("unused")])
    with make_client(monkeypatch, settings, restarted) as client2:
        after = session_messages(client2, sid)

    assert after == before

    # ...and it deserialises back into an equivalent Message[].
    manager = SessionStorageManager(Workspace(settings.project_root))
    restored = manager.get(sid).to_messages()
    assert [m.role for m in restored] == ["user", "assistant", "tool", "assistant"]
    assert restored[1].tool_calls is not None
    assert restored[1].tool_calls[0]["id"] == "call_a"
    assert restored[2].tool_call_id == "call_a"
    assert restored[3].content == "Summary complete."


# ---------------------------------------------------------------------------
# B. Continuation
# ---------------------------------------------------------------------------


def test_continuation_replays_previous_turns_exactly_once(monkeypatch, settings) -> None:
    first = FakeProvider(
        [
            FakeProvider.native_tool_call("read_file", call_id="call_1", path="main.py"),
            FakeProvider.final("It greets."),
        ]
    )
    with make_client(monkeypatch, settings, first) as client:
        sid, _ = run_task(client, "what is main.py?")

    second = FakeProvider([FakeProvider.final("No tests yet.")])
    with make_client(monkeypatch, settings, second) as client2:
        run_task(client2, "are there tests?", session_id=sid)

        assert len(second.calls) == 1
        sent = second.calls[0]
        assert [m["role"] for m in sent] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
        ]
        contents = [str(m.get("content") or "") for m in sent]
        assert sum(1 for text in contents if text == "what is main.py?") == 1
        assert sum(1 for text in contents if text.endswith("are there tests?")) == 1
        assert sum(1 for m in sent if m["role"] == "tool") == 1

        persisted = session_messages(client2, sid)

    # The second turn appended; the first turn was not duplicated.
    assert roles(persisted) == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert [m["content"] for m in persisted if m["role"] == "user"] == [
        "what is main.py?",
        "are there tests?",
    ]


def test_repeated_continuations_do_not_duplicate_history(monkeypatch, settings) -> None:
    with make_client(monkeypatch, settings, FakeProvider([FakeProvider.final("A1")])) as client:
        sid, _ = run_task(client, "A")

    for index in range(2, 5):
        provider = FakeProvider([FakeProvider.final(f"A{index}")])
        with make_client(monkeypatch, settings, provider) as client:
            run_task(client, f"U{index}", session_id=sid)

    with make_client(monkeypatch, settings, FakeProvider([FakeProvider.final("x")])) as client:
        persisted = session_messages(client, sid)

    assert roles(persisted) == ["user", "assistant"] * 4
    assert [m["content"] for m in persisted if m["role"] == "user"] == ["A", "U2", "U3", "U4"]


# ---------------------------------------------------------------------------
# C. No thought pollution
# ---------------------------------------------------------------------------


def test_thought_events_never_become_assistant_messages(monkeypatch, settings) -> None:
    provider = FakeProvider(
        [
            FakeProvider.final("All done.", thought="I will inspect the code first."),
        ]
    )
    with make_client(monkeypatch, settings, provider) as client:
        sid, events = run_task(client, "fix it")
        messages = session_messages(client, sid)

    # The browser still receives the progress event...
    assert "thought" in {event["type"] for event in events}
    # ...but it is not part of the conversation.
    assert roles(messages) == ["user", "assistant"]
    assert messages[1]["content"] == "All done."
    assert not any(
        "I will inspect the code first" in str(m.get("content") or "") for m in messages
    )


def test_thought_events_are_not_persisted_even_with_tool_steps(monkeypatch, settings) -> None:
    provider = FakeProvider(
        [
            FakeProvider.native_tool_call("list_files", call_id="call_1", path="."),
            FakeProvider.final("Listed.", thought="Now I can answer."),
        ]
    )
    with make_client(monkeypatch, settings, provider) as client:
        sid, events = run_task(client, "what files are here?")
        messages = session_messages(client, sid)

    assert "thought" in {event["type"] for event in events}
    assert roles(messages) == ["user", "assistant", "tool", "assistant"]
    assert not any("Now I can answer" in str(m.get("content") or "") for m in messages)


# ---------------------------------------------------------------------------
# D. Provider metadata round trip
# ---------------------------------------------------------------------------


def test_tool_call_provider_metadata_round_trips(monkeypatch, settings) -> None:
    response = AIResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="call_gemini_1",
                name="project_summary",
                arguments={},
                raw_arguments="{}",
                provider_data={
                    "extra_content": {"google": {"thought_signature": "sig-round-trip-1"}}
                },
            )
        ],
    )
    provider = FakeProvider([response, FakeProvider.final("Done.")])
    with make_client(monkeypatch, settings, provider) as client:
        sid, _ = run_task(client, "summarize")
        messages = session_messages(client, sid)

    tool_call = messages[1]["tool_calls"][0]
    assert tool_call["id"] == "call_gemini_1"
    assert tool_call["function"]["arguments"] == "{}"
    assert tool_call["extra_content"]["google"]["thought_signature"] == "sig-round-trip-1"

    # Restoring the session preserves the exact structure for the next request.
    manager = SessionStorageManager(Workspace(settings.project_root))
    restored = manager.get(sid).to_messages()
    assert restored[1].tool_calls[0]["extra_content"]["google"]["thought_signature"] == (
        "sig-round-trip-1"
    )


def test_provider_metadata_is_replayed_on_a_continued_request(monkeypatch, settings) -> None:
    response = AIResponse(
        text=None,
        tool_calls=[
            ToolCall(
                id="call_gemini_2",
                name="list_files",
                arguments={"path": "."},
                raw_arguments='{"path": "."}',
                provider_data={"extra_content": {"google": {"thought_signature": "sig-2"}}},
            )
        ],
    )
    with make_client(monkeypatch, settings, FakeProvider([response, FakeProvider.final("Ok.")])) as client:
        sid, _ = run_task(client, "list files")

    second = FakeProvider([FakeProvider.final("Still ok.")])
    with make_client(monkeypatch, settings, second) as client2:
        run_task(client2, "and now?", session_id=sid)

    sent = second.calls[0]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig-2"
    # The tool result stayed attached to the same call id.
    tool_message = next(m for m in sent if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_gemini_2"


# ---------------------------------------------------------------------------
# E. Timestamps
# ---------------------------------------------------------------------------


def test_updated_at_moves_once_per_turn(monkeypatch, settings) -> None:
    saves: list[tuple[str, bool]] = []
    original = SessionStorageManager.save

    def recording_save(self, session, *, touch=True):
        saves.append((session.id, touch))
        return original(self, session, touch=touch)

    monkeypatch.setattr(SessionStorageManager, "save", recording_save)

    provider = FakeProvider(
        [
            FakeProvider.native_tool_call("list_files", call_id="call_1", path="."),
            FakeProvider.native_tool_call("list_files", call_id="call_2", path="tests"),
            FakeProvider.final("Listed everything."),
        ]
    )
    with make_client(monkeypatch, settings, provider) as client:
        created = client.post("/api/agent/sessions", json={"task": "list everything"})
        sid = created.json()["session"]["id"]
        saves.clear()  # only the writes caused by the run matter here
        run_task(client, "list everything", session_id=sid)

    turned = [entry for entry in saves if entry[0] == sid and entry[1]]
    assert len(turned) == 1, f"expected exactly one timestamp update, got {saves}"

    manager = SessionStorageManager(Workspace(settings.project_root))
    session = manager.get(sid)
    assert session.updated_at >= session.created_at


def test_recent_chats_are_ordered_by_updated_at(settings) -> None:
    manager = SessionStorageManager(Workspace(settings.project_root))
    older = manager.create("older chat")
    newer = manager.create("newer chat")

    older.updated_at = "2026-01-01T00:00:00+00:00"
    manager.save(older, touch=False)
    newer.updated_at = "2026-06-01T00:00:00+00:00"
    manager.save(newer, touch=False)

    listed = [s.id for s in manager.list()]
    assert listed[:2] == [newer.id, older.id]


def test_saving_without_touch_keeps_updated_at(settings) -> None:
    manager = SessionStorageManager(Workspace(settings.project_root))
    session = manager.create("stable")
    session.updated_at = "2026-03-03T03:03:03+00:00"
    manager.save(session, touch=False)
    assert manager.get(session.id).updated_at == "2026-03-03T03:03:03+00:00"


# ---------------------------------------------------------------------------
# F. Backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_event_session_file_still_loads(settings) -> None:
    """A file written by the previous implementation must not crash the app."""
    manager = SessionStorageManager(Workspace(settings.project_root))
    session = manager.create("legacy chat")
    session.schema_version = 0
    session.messages = [
        {"type": "agent_start", "data": {"task": "legacy chat"}},
        {"type": "thought", "data": {"text": "Analyzing..."}},
        {"type": "tool_call", "data": {"tool": "read_file", "input": {"path": "main.py"}}},
        {
            "type": "tool_result",
            "data": {"name": "read_file", "output": "def greet(): ..."},
        },
        {"type": "final", "data": {"answer": "It greets."}},
    ]
    manager.save(session, touch=False)

    raw = json.loads((manager.sessions_dir / session.id / "manifest.json").read_text("utf-8"))
    assert raw["schema_version"] == 0  # as an old release would have left it

    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/agent/sessions")
        assert listed.status_code == 200
        assert session.id in [s["id"] for s in listed.json()["sessions"]]

        detail = client.get(f"/api/agent/sessions/{session.id}")
        assert detail.status_code == 200

    reloaded = manager.get(session.id)
    assert reloaded.is_legacy_history() is True
    converted = reloaded.to_messages()
    assert [m.role for m in converted] == ["user", "assistant", "tool", "assistant"]
    assert converted[0].content == "legacy chat"
    assert converted[3].content == "It greets."


def test_unreadable_session_file_is_ignored_not_fatal(settings) -> None:
    manager = SessionStorageManager(Workspace(settings.project_root))
    broken = manager.sessions_dir / "sess_broken_1"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")

    assert manager.get("sess_broken_1") is None
    assert "sess_broken_1" not in [s.id for s in manager.list()]

    with TestClient(create_app(settings)) as client:
        assert client.get("/api/agent/sessions/sess_broken_1").status_code == 404
        assert client.get("/api/agent/sessions").status_code == 200


def test_foreign_session_file_is_not_readable(settings, tmp_path: Path) -> None:
    """Workspace containment is preserved: another project's session is refused."""
    manager = SessionStorageManager(Workspace(settings.project_root))
    session = manager.create("mine")
    manifest = manager.sessions_dir / session.id / "manifest.json"

    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["project_root"] = str(tmp_path / "somewhere-else")
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    assert manager.get(session.id) is None


# ---------------------------------------------------------------------------
# G. Persistent chat CRUD (API contract used by the Recent Chats UI)
# ---------------------------------------------------------------------------


def test_new_chat_gets_the_first_task_as_its_title(monkeypatch, settings) -> None:
    provider = FakeProvider([FakeProvider.final("Done.")])
    with make_client(monkeypatch, settings, provider) as client:
        created = client.post("/api/agent/sessions", json={})
        assert created.status_code == 200
        sid = created.json()["session"]["id"]
        assert created.json()["session"]["title"] == "New Chat"

        run_task(client, "Explain the calculator module", session_id=sid)

        session = client.get(f"/api/agent/sessions/{sid}").json()["session"]

    assert session["title"] == "Explain the calculator module"


def test_session_crud_contract(monkeypatch, settings) -> None:
    with make_client(monkeypatch, settings, FakeProvider([FakeProvider.final("x")])) as client:
        created = client.post("/api/agent/sessions", json={"task": "Add a feature"})
        assert created.status_code == 200
        sid = created.json()["session"]["id"]

        listed = client.get("/api/agent/sessions").json()["sessions"]
        assert [s["id"] for s in listed] == [sid]
        assert listed[0]["message_count"] == 0

        fetched = client.get(f"/api/agent/sessions/{sid}")
        assert fetched.status_code == 200
        assert fetched.json()["session"]["title"] == "Add a feature"

        renamed = client.patch(f"/api/agent/sessions/{sid}", json={"title": "Renamed"})
        assert renamed.status_code == 200
        assert renamed.json()["session"]["title"] == "Renamed"
        assert client.get(f"/api/agent/sessions/{sid}").json()["session"]["title"] == "Renamed"

        assert client.patch(f"/api/agent/sessions/{sid}", json={"title": ""}).status_code == 422
        assert client.patch("/api/agent/sessions/nope", json={"title": "x"}).status_code == 404

        assert client.delete(f"/api/agent/sessions/{sid}").status_code == 200
        assert client.get(f"/api/agent/sessions/{sid}").status_code == 404
        assert client.delete(f"/api/agent/sessions/{sid}").status_code == 404


def test_invalid_session_id_is_rejected_not_a_server_error(monkeypatch, settings) -> None:
    with make_client(monkeypatch, settings, FakeProvider([FakeProvider.final("x")])) as client:
        response = client.post(
            "/api/agent", json={"task": "hi", "session_id": "../escape"}
        )
        assert response.status_code == 400


def test_session_titles_do_not_leak_api_keys(settings) -> None:
    manager = SessionStorageManager(Workspace(settings.project_root))
    for session in manager.list():
        serialized = json.dumps(session.to_dict())
        assert "gsk_" not in serialized
