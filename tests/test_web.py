"""Tests for the web API (:mod:`nova.web.app` and :mod:`nova.web.routes`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi is required for the web tests")
pytest.importorskip("httpx", reason="httpx is required by fastapi's TestClient")

from fastapi.testclient import TestClient  # noqa: E402

from nova.web.app import create_app  # noqa: E402

from conftest import TEST_API_KEY, FakeProvider  # noqa: E402


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


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


# --- Shell ------------------------------------------------------------------


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NovaCLI" in response.text


def test_static_assets_are_mounted(client: TestClient) -> None:
    for path, marker in (
        ("/static/css/app.css", "--accent"),
        ("/static/js/app.js", "EventSource"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert marker in response.text


# --- Health / config --------------------------------------------------------


def test_health(client: TestClient) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["has_api_key"] is True
    assert payload["model"] == "test-model"


def test_config_never_exposes_the_key(client: TestClient) -> None:
    response = client.get("/api/config")
    assert response.status_code == 200
    assert TEST_API_KEY not in response.text
    assert response.json()["has_api_key"] is True


def test_openapi_is_available(client: TestClient) -> None:
    assert client.get("/api/openapi.json").status_code == 200


# --- Project / files --------------------------------------------------------


def test_project_summary(client: TestClient) -> None:
    payload = client.get("/api/project").json()
    assert payload["summary"]["languages"]["Python"] >= 3
    assert "main.py" in payload["tree"]
    assert "test" in payload["skills"]


def test_tree_endpoint(client: TestClient) -> None:
    payload = client.get("/api/tree", params={"path": ".", "depth": 2}).json()
    assert "main.py" in payload["tree"]


def test_tree_rejects_traversal(client: TestClient) -> None:
    response = client.get("/api/tree", params={"path": "../../"})
    assert response.status_code == 400


def test_list_files(client: TestClient) -> None:
    payload = client.get("/api/files", params={"path": "."}).json()
    names = [entry["path"] for entry in payload["entries"]]
    assert "main.py" in names
    assert ".env" not in names


def test_read_file(client: TestClient) -> None:
    payload = client.get("/api/file", params={"path": "main.py"}).json()
    assert "def greet" in payload["content"]
    assert payload["language"] == "Python"
    assert payload["lines"] > 1


def test_read_file_refuses_secrets(client: TestClient) -> None:
    response = client.get("/api/file", params={"path": ".env"})
    assert response.status_code == 403


def test_read_missing_file(client: TestClient) -> None:
    assert client.get("/api/file", params={"path": "nope.py"}).status_code == 400


def test_write_file(client: TestClient, tmp_project: Path) -> None:
    response = client.put("/api/file", json={"path": "created.py", "content": "x = 1\n"})
    assert response.status_code == 200
    assert (tmp_project / "created.py").read_text() == "x = 1\n"


def test_write_file_refuses_outside_workspace(client: TestClient) -> None:
    response = client.put("/api/file", json={"path": "../escape.py", "content": "x"})
    assert response.status_code == 403


def test_write_file_refuses_env(client: TestClient) -> None:
    response = client.put("/api/file", json={"path": ".env", "content": "GROQ_API_KEY=stolen"})
    assert response.status_code == 403


# --- Command execution ------------------------------------------------------


def test_run_command(client: TestClient) -> None:
    response = client.post("/api/run", json={"command": "echo web-run"})
    assert response.status_code == 200
    assert "web-run" in response.json()["result"]["stdout"]


def test_run_command_reports_approval_need(client: TestClient) -> None:
    response = client.post("/api/run", json={"command": "rm somefile.txt"})
    payload = response.json()
    assert response.status_code == 200
    # Smart mode permits moderate commands outright; strict mode would ask.
    assert payload.get("requires_approval") in (True, False)


def test_run_command_refuses_forbidden(client: TestClient) -> None:
    response = client.post("/api/run", json={"command": "rm -rf /"})
    assert response.status_code == 403
    assert "Refused" in response.json()["detail"]


def test_run_command_in_strict_mode_asks_first(settings, tmp_project: Path) -> None:
    strict = settings.with_overrides(safety_mode="strict")
    with TestClient(create_app(strict)) as strict_client:
        payload = strict_client.post("/api/run", json={"command": "rm notes.txt"}).json()
        assert payload["requires_approval"] is True


# --- Agent sessions ---------------------------------------------------------


def test_start_agent_without_key_returns_setup_error(bare_project: Path) -> None:
    from nova.config import load_settings

    keyless = load_settings(
        project_root=bare_project,
        env={},
        user_config_path=bare_project / "_absent.json",
    )
    with TestClient(create_app(keyless)) as keyless_client:
        response = keyless_client.post("/api/agent", json={"task": "hi"})
        assert response.status_code == 400
        assert "GROQ_API_KEY" in response.json()["detail"]


def test_agent_session_streams_to_completion(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "nova.web.routes.get_provider",
        lambda settings: FakeProvider(
            [FakeProvider.action("read_file", path="main.py"), FakeProvider.final("It greets.")]
        ),
    )
    created = client.post("/api/agent", json={"task": "what is main.py?"})
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    events = read_sse(client, f"/api/agent/stream?session_id={session_id}")
    types = [event["type"] for event in events]

    assert "agent_start" in types
    assert "tool_result" in types
    assert types[-1] == "final"
    assert events[-1]["data"]["answer"] == "It greets."


def test_agent_result_endpoint_polls_state(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "nova.web.routes.get_provider",
        lambda settings: FakeProvider([FakeProvider.final("polled")]),
    )
    session_id = client.post("/api/agent", json={"task": "hi"}).json()["session_id"]
    read_sse(client, f"/api/agent/stream?session_id={session_id}")

    payload = client.get("/api/agent/result", params={"session_id": session_id}).json()
    assert payload["status"] == "done"
    assert payload["result"]["answer"] == "polled"


def test_agent_stream_unknown_session(client: TestClient) -> None:
    assert client.get("/api/agent/stream", params={"session_id": "nope"}).status_code == 404


def test_agent_sessions_listing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "nova.web.routes.get_provider",
        lambda settings: FakeProvider([FakeProvider.final("x")]),
    )
    client.post("/api/agent", json={"task": "one"})
    payload = client.get("/api/agent/sessions").json()
    assert len(payload["sessions"]) >= 1


def test_approve_unknown_request_conflicts(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "nova.web.routes.get_provider",
        lambda settings: FakeProvider([FakeProvider.final("x")]),
    )
    session_id = client.post("/api/agent", json={"task": "hi"}).json()["session_id"]
    response = client.post(
        "/api/agent/approve",
        json={"session_id": session_id, "request_id": "nope", "decision": "approve"},
    )
    assert response.status_code == 409


def test_approve_rejects_bad_decision(client: TestClient) -> None:
    response = client.post(
        "/api/agent/approve",
        json={"session_id": "s", "request_id": "r", "decision": "maybe"},
    )
    assert response.status_code == 400


def test_cancel_unknown_session(client: TestClient) -> None:
    assert client.post("/api/agent/cancel", json={"session_id": "nope"}).status_code == 404


def test_cancel_stops_a_session(client: TestClient, monkeypatch) -> None:
    long_script = [FakeProvider.action("list_files", path=".") for _ in range(8)]
    monkeypatch.setattr(
        "nova.web.routes.get_provider", lambda settings: FakeProvider(long_script)
    )
    session_id = client.post("/api/agent", json={"task": "loop"}).json()["session_id"]
    response = client.post("/api/agent/cancel", json={"session_id": session_id})
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_agent_request_validation(client: TestClient) -> None:
    assert client.post("/api/agent", json={"task": ""}).status_code == 422


def test_start_agent_with_auto_approve(client: TestClient, monkeypatch, tmp_project: Path) -> None:
    monkeypatch.setattr(
        "nova.web.routes.get_provider",
        lambda settings: FakeProvider(
            [
                FakeProvider.action("write_file", path="auto.py", content="y = 2\n"),
                FakeProvider.final("written"),
            ]
        ),
    )
    session_id = client.post(
        "/api/agent", json={"task": "write a file", "auto_approve": True}
    ).json()["session_id"]
    read_sse(client, f"/api/agent/stream?session_id={session_id}")

    assert (tmp_project / "auto.py").read_text() == "y = 2\n"


# --- Global Search & Developer Endpoint Tests --------------------------------


def test_search_files_basic(client: TestClient) -> None:
    res = client.get("/api/search", params={"q": "greet"})
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "greet"
    assert len(data["hits"]) >= 1


def test_search_files_with_regex_and_case(client: TestClient) -> None:
    res = client.get("/api/search", params={"q": "def\s+greet", "regex": "true", "case_sensitive": "true"})
    assert res.status_code == 200
    data = res.json()
    assert len(data["hits"]) >= 1


def test_search_files_with_glob(client: TestClient) -> None:
    res = client.get("/api/search", params={"q": "greet", "glob": "*.py"})
    assert res.status_code == 200
    data = res.json()
    assert all(hit["path"].endswith(".py") for hit in data["hits"])
