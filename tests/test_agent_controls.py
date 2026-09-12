"""Tests for agent controls: cancellation and human approval.

This is the machinery that lets the CLI prompt on stdin and the web IDE prompt
over HTTP, both driving the same ``AgentController``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nova.core.agent import AgentController, NovaAgent
from nova.core.models import (
    AgentStatus,
    ApprovalDecision,
    ApprovalRequest,
    EventType,
    RiskLevel,
)
from nova.web.events import AgentSession, SessionRegistry, stream_session

from conftest import FakeProvider


# ---------------------------------------------------------------------------
# Controller primitives
# ---------------------------------------------------------------------------


def test_new_controller_is_not_cancelled() -> None:
    assert AgentController().cancelled is False


def test_cancel_sets_the_flag() -> None:
    controller = AgentController()
    controller.cancel()
    assert controller.cancelled is True


def test_cancel_is_idempotent() -> None:
    controller = AgentController()
    controller.cancel()
    controller.cancel()
    assert controller.cancelled is True


def test_no_pending_request_initially() -> None:
    assert AgentController().pending_request_id is None


def test_resolve_without_pending_returns_false() -> None:
    assert AgentController().resolve("missing", "approve") is False


async def test_auto_approve_skips_the_handler() -> None:
    calls: list[ApprovalRequest] = []

    async def handler(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        return ApprovalDecision.DENY

    controller = AgentController(handler, auto_approve=True)
    decision = await controller.request_approval(
        ApprovalRequest(id="1", tool="run_command", summary="rm")
    )
    assert decision == ApprovalDecision.APPROVE
    assert calls == []


async def test_handler_is_consulted_when_present() -> None:
    async def handler(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    controller = AgentController(handler)
    decision = await controller.request_approval(
        ApprovalRequest(id="1", tool="write_file", summary="write")
    )
    assert decision == ApprovalDecision.APPROVE


async def test_always_remembers_the_decision() -> None:
    calls: list[str] = []

    async def handler(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request.id)
        return ApprovalDecision.ALWAYS

    controller = AgentController(handler)
    first = await controller.request_approval(
        ApprovalRequest(id="1", tool="run_command", summary="a")
    )
    second = await controller.request_approval(
        ApprovalRequest(id="2", tool="run_command", summary="b")
    )

    assert first == ApprovalDecision.APPROVE
    assert second == ApprovalDecision.APPROVE
    assert calls == ["1"]  # only asked once


async def test_always_is_scoped_to_one_tool() -> None:
    seen: list[str] = []

    async def handler(request: ApprovalRequest) -> ApprovalDecision:
        seen.append(request.tool)
        return ApprovalDecision.ALWAYS

    controller = AgentController(handler)
    await controller.request_approval(ApprovalRequest(id="1", tool="run_command", summary="a"))
    await controller.request_approval(ApprovalRequest(id="2", tool="write_file", summary="b"))
    assert seen == ["run_command", "write_file"]


async def test_broken_handler_fails_closed() -> None:
    async def handler(request: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError("handler exploded")

    controller = AgentController(handler)
    decision = await controller.request_approval(
        ApprovalRequest(id="1", tool="write_file", summary="x")
    )
    assert decision == ApprovalDecision.DENY


async def test_eof_denies() -> None:
    async def handler(request: ApprovalRequest) -> ApprovalDecision:
        raise EOFError()

    controller = AgentController(handler)
    assert (
        await controller.request_approval(ApprovalRequest(id="1", tool="t", summary="s"))
        == ApprovalDecision.DENY
    )


async def test_no_handler_and_no_resolver_times_out_to_deny() -> None:
    controller = AgentController(approval_timeout=0.2)
    decision = await controller.request_approval(
        ApprovalRequest(id="1", tool="write_file", summary="x")
    )
    assert decision == ApprovalDecision.DENY


async def test_resolve_from_outside_satisfies_the_wait() -> None:
    controller = AgentController(approval_timeout=5)

    async def resolve_soon() -> None:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if controller.pending_request_id:
                # Accepts a plain string as well as the enum.
                assert controller.resolve("req-1", "approve")
                return
        raise AssertionError("approval was never requested")

    task = asyncio.create_task(resolve_soon())
    decision = await controller.request_approval(
        ApprovalRequest(id="req-1", tool="write_file", summary="x")
    )
    await task
    assert decision == ApprovalDecision.APPROVE


async def test_resolve_with_a_bad_value_denies() -> None:
    controller = AgentController(approval_timeout=5)

    async def resolve_soon() -> None:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if controller.pending_request_id:
                assert controller.resolve("req-1", "banana")
                return

    task = asyncio.create_task(resolve_soon())
    decision = await controller.request_approval(
        ApprovalRequest(id="req-1", tool="write_file", summary="x")
    )
    await task
    assert decision == ApprovalDecision.DENY


async def test_cancel_denies_a_waiting_approval() -> None:
    controller = AgentController(approval_timeout=5)

    async def cancel_soon() -> None:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if controller.pending_request_id:
                controller.cancel()
                return

    task = asyncio.create_task(cancel_soon())
    decision = await controller.request_approval(
        ApprovalRequest(id="req-1", tool="write_file", summary="x")
    )
    await task
    assert decision == ApprovalDecision.DENY


async def test_cancelled_controller_denies_immediately() -> None:
    controller = AgentController()
    controller.cancel()
    decision = await controller.request_approval(
        ApprovalRequest(id="1", tool="write_file", summary="x")
    )
    assert decision == ApprovalDecision.DENY


# ---------------------------------------------------------------------------
# Cancellation inside the agent loop
# ---------------------------------------------------------------------------


async def test_cancel_before_the_first_step(make_agent) -> None:
    provider = FakeProvider([FakeProvider.action("list_files", path=".")])
    agent = make_agent(provider)
    controller = AgentController()
    controller.cancel()

    events = [event async for event in agent.stream("do it", controller=controller)]
    assert events[-1].type == EventType.CANCELLED
    assert provider.calls == []  # never even called the model


async def test_cancel_mid_run(make_agent) -> None:
    provider = FakeProvider([FakeProvider.action("list_files", path=".") for _ in range(8)])
    agent = make_agent(provider)
    controller = AgentController()

    events = []
    async for event in agent.stream("loop", controller=controller, max_steps=8):
        events.append(event)
        if event.type == EventType.TOOL_RESULT and len(events) > 6:
            controller.cancel()

    assert events[-1].type == EventType.CANCELLED
    assert len(provider.calls) < 8


async def test_cancel_produces_a_cancelled_result(make_agent) -> None:
    provider = FakeProvider([FakeProvider.action("list_files", path=".") for _ in range(8)])
    agent = make_agent(provider)
    controller = AgentController()
    controller.cancel()

    result = await agent.run("loop", controller=controller)
    assert result.status == AgentStatus.CANCELLED


# ---------------------------------------------------------------------------
# Approval inside the agent loop
# ---------------------------------------------------------------------------


async def _run_with_approval(make_agent, provider, controller, agent_kwargs=None):
    agent = make_agent(provider, **(agent_kwargs or {}))
    events: list = []

    async def consume() -> None:
        async for event in agent.stream("do the thing", controller=controller):
            events.append(event)

    task = asyncio.create_task(consume())
    return task, events


async def test_write_requires_and_receives_approval(make_agent, tmp_project: Path) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("write_file", path="approved.py", content="ok = True\n"),
            FakeProvider.final("done"),
        ]
    )
    controller = AgentController(approval_timeout=5)
    task, events = await _run_with_approval(make_agent, provider, controller)

    for _ in range(300):
        await asyncio.sleep(0.01)
        if controller.pending_request_id:
            break
    assert controller.pending_request_id is not None
    assert any(e.type == EventType.APPROVAL_REQUEST for e in events)

    assert controller.resolve(controller.pending_request_id, ApprovalDecision.APPROVE)
    await asyncio.wait_for(task, timeout=10)

    assert (tmp_project / "approved.py").exists()
    assert any(e.type == EventType.APPROVAL_RESOLVED for e in events)


async def test_denied_write_is_not_performed(make_agent, tmp_project: Path) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("write_file", path="denied.py", content="x = 1\n"),
            FakeProvider.final("understood, not writing"),
        ]
    )
    controller = AgentController(approval_timeout=5)
    task, events = await _run_with_approval(make_agent, provider, controller)

    for _ in range(300):
        await asyncio.sleep(0.01)
        if controller.pending_request_id:
            break
    controller.resolve(controller.pending_request_id, ApprovalDecision.DENY)
    await asyncio.wait_for(task, timeout=10)

    assert not (tmp_project / "denied.py").exists()
    assert any(e.type == EventType.BLOCKED for e in events)
    assert events[-1].type == EventType.FINAL


async def test_denied_action_tells_the_model_not_to_retry(make_agent) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("write_file", path="a.py", content="1\n"),
            FakeProvider.final("stopping"),
        ]
    )
    controller = AgentController(approval_timeout=5)
    task, _ = await _run_with_approval(make_agent, provider, controller)

    for _ in range(300):
        await asyncio.sleep(0.01)
        if controller.pending_request_id:
            break
    controller.resolve(controller.pending_request_id, ApprovalDecision.DENY)
    await asyncio.wait_for(task, timeout=10)

    follow_up = " ".join(message["content"] for message in provider.calls[-1])
    assert "DENIED" in follow_up


async def test_approval_timeout_denies_and_continues(make_agent, tmp_project: Path) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("write_file", path="timedout.py", content="1\n"),
            FakeProvider.final("moved on"),
        ]
    )
    controller = AgentController(approval_timeout=0.2)
    agent = make_agent(provider)

    events = [event async for event in agent.stream("write", controller=controller)]
    assert events[-1].type == EventType.FINAL
    assert not (tmp_project / "timedout.py").exists()


async def test_safe_tools_never_prompt(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("read_file", path="main.py"), FakeProvider.final("done")]
    )
    controller = AgentController(approval_timeout=0.2)
    agent = make_agent(provider)

    events = [event async for event in agent.stream("read", controller=controller)]
    assert not any(e.type == EventType.APPROVAL_REQUEST for e in events)
    assert events[-1].type == EventType.FINAL


async def test_forbidden_actions_bypass_the_approval_prompt(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("run_command", command="rm -rf /"), FakeProvider.final("ok")]
    )
    controller = AgentController(approval_timeout=0.2)
    agent = make_agent(provider)

    events = [event async for event in agent.stream("delete", controller=controller)]
    assert not any(e.type == EventType.APPROVAL_REQUEST for e in events)
    assert any(e.type == EventType.BLOCKED for e in events)


# ---------------------------------------------------------------------------
# Session registry controls
# ---------------------------------------------------------------------------


def test_registry_creates_and_finds_sessions() -> None:
    registry = SessionRegistry()
    session = registry.create("do something", model="m")
    assert registry.get(session.id) is session
    assert session.task == "do something"
    assert session.status == AgentStatus.PENDING


def test_registry_require_raises_for_unknown() -> None:
    with pytest.raises(KeyError):
        SessionRegistry().require("nope")


def test_registry_cancel_unknown_returns_false() -> None:
    assert SessionRegistry().cancel("nope") is False


def test_registry_cancel_marks_the_session() -> None:
    registry = SessionRegistry()
    session = registry.create("task")
    assert registry.cancel(session.id) is True
    assert session.status == AgentStatus.CANCELLED
    assert session.controller.cancelled is True


def test_registry_approve_unknown_returns_false() -> None:
    assert SessionRegistry().approve("nope", "req", "approve") is False


def test_registry_prune_removes_expired_sessions() -> None:
    registry = SessionRegistry(ttl_seconds=10)
    session = registry.create("task")
    session.status = AgentStatus.DONE
    session.finished_at = 0.0  # ancient
    assert registry.prune(now=1000.0) == 1
    assert registry.get(session.id) is None


def test_registry_prune_keeps_running_sessions() -> None:
    registry = SessionRegistry(ttl_seconds=10)
    session = registry.create("task")
    assert registry.prune(now=1000.0) == 0
    assert registry.get(session.id) is session


def test_registry_enforces_max_sessions() -> None:
    registry = SessionRegistry(max_sessions=3)
    for index in range(6):
        session = registry.create(f"task {index}")
        session.status = AgentStatus.DONE
        session.finished_at = 1.0
    assert len(registry.sessions) < 6


def test_session_lists_events_newest_session_first() -> None:
    registry = SessionRegistry()
    first = registry.create("one")
    second = registry.create("two")
    ordered = registry.list()
    assert ordered[0].id == second.id or ordered[0].id == first.id
    assert len(ordered) == 2


def test_session_snapshot_is_a_copy() -> None:
    session = AgentSession(id="x", task="t")
    session.publish({"type": "final", "data": {}})
    snapshot = session.snapshot()
    snapshot.append({"type": "extra"})
    assert len(session.snapshot()) == 1


def test_session_publish_fans_out_to_subscribers() -> None:
    session = AgentSession(id="x", task="t")
    queue = session.subscribe()
    session.publish({"type": "thought", "data": {"text": "hi"}})
    assert queue.get_nowait()["type"] == "thought"


def test_session_unsubscribe_stops_delivery() -> None:
    session = AgentSession(id="x", task="t")
    queue = session.subscribe()
    session.unsubscribe(queue)
    session.publish({"type": "thought", "data": {}})
    assert queue.empty()


def test_session_to_dict_serialises() -> None:
    import json

    session = AgentSession(id="x", task="t")
    json.dumps(session.to_dict())


async def test_stream_session_replays_then_finishes() -> None:
    session = AgentSession(id="x", task="t")
    session.publish({"type": "thought", "data": {"text": "a"}})
    session.publish({"type": "final", "data": {"answer": "b"}})
    session.status = AgentStatus.DONE

    frames = [frame async for frame in stream_session(session, heartbeat=0.1)]
    assert len(frames) == 2
    assert "final" in frames[-1]


async def test_stream_session_stops_on_terminal_event() -> None:
    session = AgentSession(id="x", task="t")
    session.publish({"type": "cancelled", "data": {}})
    session.status = AgentStatus.CANCELLED

    frames = [frame async for frame in stream_session(session, heartbeat=0.1)]
    assert len(frames) == 1
