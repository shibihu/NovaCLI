"""Agent sessions and the server-sent event bus.

A browser cannot call an async generator directly, so the web layer keeps one
:class:`AgentSession` per task. The session owns an
:class:`~nova.core.agent.AgentController` (so the UI can cancel and approve)
and a fan-out queue of events that any number of SSE connections can read.

Because the controller is the same object the core agent uses, "cancel" and
"approve" from the browser are the exact same code paths the CLI drives from
stdin — one implementation, two front ends.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from nova.ai import AIProviderError
from nova.core.agent import AgentController, NovaAgent
from nova.core.models import (
    AgentEvent,
    AgentResult,
    AgentStatus,
    ApprovalDecision,
    EventType,
    TERMINAL_EVENTS,
)

#: Heartbeat interval so mobile browsers and proxies keep the SSE stream open.
HEARTBEAT_SECONDS = 15.0

#: Sessions kept in memory before the oldest finished ones are evicted.
MAX_SESSIONS = 40

#: Finished sessions older than this are pruned.
SESSION_TTL_SECONDS = 3_600.0

AgentFactory = Callable[[], NovaAgent]


def sse_format(event: dict[str, Any]) -> str:
    """Serialise an event dict as one Server-Sent Events frame."""
    import json

    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@dataclass
class AgentSession:
    """One agent run plus everything a client needs to watch it."""

    id: str
    task: str
    status: AgentStatus = AgentStatus.PENDING
    events: list[dict[str, Any]] = field(default_factory=list)
    controller: AgentController = field(default_factory=AgentController)
    result: AgentResult | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    model: str = ""
    checkpoint_id: str | None = None
    changed_files: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list, repr=False)
    run_task: asyncio.Task[Any] | None = field(default=None, repr=False)

    # -- Publishing ------------------------------------------------------

    def publish(self, event: AgentEvent | dict[str, Any]) -> dict[str, Any]:
        """Record an event and fan it out to every subscriber."""
        payload = event.to_dict() if isinstance(event, AgentEvent) else dict(event)
        self.events.append(payload)

        for queue in list(self.subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover - bounded by design
                pass
        return payload

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    # -- State -----------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.status in AgentStatus.terminal()

    @property
    def pending_approval_id(self) -> str | None:
        return self.controller.pending_request_id

    def snapshot(self) -> list[dict[str, Any]]:
        """All events so far, for a client that connects late."""
        return list(self.events)

    def to_dict(self, *, include_events: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "task": self.task,
            "status": str(self.status),
            "model": self.model,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "pending_approval_id": self.pending_approval_id,
            "checkpoint_id": self.checkpoint_id,
            "changed_files": self.changed_files,
            "result": self.result.to_dict() if self.result else None,
        }
        if include_events:
            data["events"] = self.snapshot()
        return data


class SessionRegistry:
    """In-memory registry of agent sessions."""

    def __init__(
        self,
        *,
        max_sessions: int = MAX_SESSIONS,
        ttl_seconds: float = SESSION_TTL_SECONDS,
    ) -> None:
        self.sessions: dict[str, AgentSession] = {}
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds

    # -- Lifecycle -------------------------------------------------------

    def create(
        self,
        task: str,
        *,
        session_id: str | None = None,
        controller: AgentController | None = None,
        model: str = "",
    ) -> AgentSession:
        self.prune()
        sid = session_id or uuid.uuid4().hex[:16]
        session = AgentSession(
            id=sid,
            task=task,
            controller=controller or AgentController(),
            model=model,
        )
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> AgentSession | None:
        return self.sessions.get(session_id)

    def require(self, session_id: str) -> AgentSession:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session {session_id!r}")
        return session

    def list(self) -> list[AgentSession]:
        return sorted(self.sessions.values(), key=lambda s: s.created_at, reverse=True)

    def prune(self, *, now: float | None = None) -> int:
        """Drop finished, expired sessions. Returns how many were removed."""
        now = time.time() if now is None else now
        removed = 0
        for session_id, session in list(self.sessions.items()):
            if session.is_terminal and session.finished_at is not None:
                if now - session.finished_at > self.ttl_seconds:
                    del self.sessions[session_id]
                    removed += 1

        if len(self.sessions) >= self.max_sessions:
            finished = [s for s in self.list() if s.is_terminal]
            while len(self.sessions) >= self.max_sessions and finished:
                oldest = finished.pop()
                self.sessions.pop(oldest.id, None)
                removed += 1
        return removed

    def clear(self) -> None:
        self.sessions.clear()

    # -- Running ---------------------------------------------------------

    async def run_session(self, session: AgentSession, factory: AgentFactory, storage_manager: Any = None) -> None:
        """Drive the agent for ``session``, publishing every event.

        Runs as a background task so the HTTP request that created the session
        can return immediately and the browser can attach to the stream.
        """
        session.status = AgentStatus.RUNNING
        agent: NovaAgent | None = None
        try:
            agent = factory()
            session.model = agent.provider.model_name
            # The agent itself emits AGENT_START, so we do not synthesise one.
            async for event in agent.stream(session.task, controller=session.controller):
                session.publish(event)
                if event.type == EventType.AGENT_START:
                    session.checkpoint_id = event.data.get("checkpoint_id")
                elif event.type == EventType.APPROVAL_REQUEST:
                    session.status = AgentStatus.WAITING_APPROVAL
                elif event.type == EventType.APPROVAL_RESOLVED:
                    session.status = AgentStatus.RUNNING
                elif event.type == EventType.FINAL:
                    session.status = AgentStatus.DONE
                    session.checkpoint_id = str(event.data.get("checkpoint_id", "") or session.checkpoint_id or "")
                    session.changed_files = list(event.data.get("changed_files") or [])
                    session.result = AgentResult(
                        task=session.task,
                        status=AgentStatus.DONE,
                        answer=str(event.data.get("answer", "")),
                        model=session.model,
                        checkpoint_id=session.checkpoint_id,
                        changed_files=session.changed_files,
                    )
                elif event.type == EventType.ERROR:
                    session.status = AgentStatus.ERROR
                    session.error = str(event.data.get("message", "unknown error"))
                elif event.type == EventType.CANCELLED:
                    session.status = AgentStatus.CANCELLED
        except asyncio.CancelledError:
            session.status = AgentStatus.CANCELLED
            session.publish(AgentEvent(EventType.CANCELLED, {"reason": "cancelled"}))
            raise
        except (AIProviderError, OSError, ValueError) as exc:
            session.status = AgentStatus.ERROR
            session.error = str(exc)
            session.publish(AgentEvent(EventType.ERROR, {"message": str(exc)}))
        except Exception as exc:  # noqa: BLE001 - never leave a session hanging
            session.status = AgentStatus.ERROR
            session.error = f"{type(exc).__name__}: {exc}"
            session.publish(AgentEvent(EventType.ERROR, {"message": session.error}))
        finally:
            if session.status not in AgentStatus.terminal():
                session.status = AgentStatus.DONE
            session.finished_at = time.time()
            if storage_manager is not None:
                try:
                    p_sess = storage_manager.get(session.id)
                    if p_sess:
                        p_sess.status = str(session.status)
                        p_sess.messages = list(session.events)
                        storage_manager.save(p_sess)
                except Exception:
                    pass
            if agent is not None:
                try:
                    await agent.provider.aclose()
                except Exception:  # noqa: BLE001 - cleanup is best effort
                    pass

    def start(self, session: AgentSession, factory: AgentFactory, storage_manager: Any = None) -> asyncio.Task[Any]:
        """Schedule :meth:`run_session` and remember the task."""
        task = asyncio.create_task(self.run_session(session, factory, storage_manager=storage_manager))
        session.run_task = task
        return task

    # -- Controls --------------------------------------------------------

    def cancel(self, session_id: str) -> bool:
        """Cancel a running session. Returns False for unknown sessions."""
        session = self.get(session_id)
        if session is None:
            return False
        session.controller.cancel()
        if session.status not in AgentStatus.terminal():
            session.status = AgentStatus.CANCELLED
        return True

    def approve(
        self, session_id: str, request_id: str, decision: ApprovalDecision | str
    ) -> bool:
        """Resolve a pending approval. Returns False if nothing was waiting."""
        session = self.get(session_id)
        if session is None:
            return False
        resolved = session.controller.resolve(request_id, decision)
        if resolved and session.status == AgentStatus.WAITING_APPROVAL:
            session.status = AgentStatus.RUNNING
        return resolved


async def stream_session(
    session: AgentSession, *, heartbeat: float = HEARTBEAT_SECONDS
) -> AsyncIterator[str]:
    """Yield SSE frames for ``session`` until it reaches a terminal state.

    Replays everything already recorded first, so a client that connects after
    the run started (or after it finished) still sees the whole transcript.
    """
    queue = session.subscribe()
    try:
        for event in session.snapshot():
            yield sse_format(event)
            if event.get("type") in TERMINAL_EVENTS:
                return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                if session.is_terminal and queue.empty():
                    return
                yield ": ping\n\n"
                continue

            yield sse_format(event)
            if event.get("type") in TERMINAL_EVENTS:
                return
    finally:
        session.unsubscribe(queue)


__all__ = [
    "AgentSession",
    "SessionRegistry",
    "sse_format",
    "stream_session",
]
