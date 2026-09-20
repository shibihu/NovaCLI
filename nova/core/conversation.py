"""Canonical conversation persistence helpers.

A chat session stores the *canonical* conversation — the exact ``Message[]``
list the LLM is given: user turns, assistant turns, tool calls and tool results.
It deliberately stores **nothing** derived from the :class:`AgentEvent` stream:
``thought``/``progress``/``step_start`` notifications are UI output and must
never be replayed to a provider as assistant content.

Everything here is pure: no I/O, no provider knowledge. ``sessions.py`` owns the
storage, ``events.py`` owns the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from nova.core.models import Message

#: Bumped whenever the persisted conversation shape changes. Session files
#: written before this existed have no ``schema_version`` and stored a raw
#: ``AgentEvent`` list instead — see :func:`is_legacy_history`.
CONVERSATION_SCHEMA_VERSION = 2

#: Message roles that may appear in a canonical conversation.
CANONICAL_ROLES = frozenset({"system", "user", "assistant", "tool"})

#: Keys preserved verbatim on a message so provider metadata round-trips.
_STRUCTURED_KEYS = ("tool_calls", "tool_call_id", "name", "provider_data")


def message_to_dict(message: Message) -> dict[str, Any]:
    """Serialize one canonical message.

    Structured fields (native tool calls, tool call ids, provider metadata such
    as Gemini's ``extra_content.google.thought_signature``) are kept as
    structures — never flattened into text.
    """
    return message.to_dict()


def message_from_dict(data: Any) -> Message | None:
    """Rebuild a :class:`Message` from a stored dict. ``None`` if unusable."""
    if not isinstance(data, dict):
        return None
    role = data.get("role")
    if not isinstance(role, str) or role not in CANONICAL_ROLES:
        return None

    content = data.get("content")
    content = None if content is None else str(content)

    tool_calls = data.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = None

    tool_call_id = data.get("tool_call_id")
    tool_call_id = None if tool_call_id is None else str(tool_call_id)

    name = data.get("name")
    name = None if name is None else str(name)

    provider_data = data.get("provider_data")
    if not isinstance(provider_data, dict):
        provider_data = {}

    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
        provider_data=dict(provider_data),
    )


def messages_to_dicts(messages: Iterable[Message]) -> list[dict[str, Any]]:
    """Serialize a canonical conversation."""
    return [message_to_dict(message) for message in messages]


def messages_from_dicts(items: Any) -> list[Message]:
    """Deserialize a stored canonical conversation, skipping unusable entries."""
    if not isinstance(items, list):
        return []
    restored: list[Message] = []
    for item in items:
        message = message_from_dict(item)
        if message is not None:
            restored.append(message)
    return restored


def is_canonical_history(items: Any) -> bool:
    """True when ``items`` already looks like a stored ``Message[]``."""
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if isinstance(item, dict) and item.get("role") in CANONICAL_ROLES:
            return True
    return False


def is_legacy_history(items: Any) -> bool:
    """True when ``items`` is a pre-canonical raw ``AgentEvent`` list."""
    if not isinstance(items, list):
        return False
    for item in items:
        if isinstance(item, dict) and "role" not in item and "type" in item:
            return True
    return False


def trim_history(messages: Sequence[Message], limit: int) -> list[Message]:
    """Keep the newest ``limit`` messages without orphaning tool results.

    Dropping the oldest entries can leave ``tool`` messages at the front whose
    originating assistant tool call was removed; providers reject that, so the
    boundary is advanced past them.
    """
    if limit <= 0:
        return []
    ordered = list(messages)
    if len(ordered) <= limit:
        return ordered
    trimmed = ordered[-limit:]
    while trimmed and trimmed[0].role == "tool":
        trimmed.pop(0)
    return trimmed


# ---------------------------------------------------------------------------
# Live run trace
# ---------------------------------------------------------------------------


@dataclass
class ConversationTrace:
    """Live handle on the canonical ``Message[]`` built by one agent run.

    :meth:`~nova.core.agent.NovaAgent.stream` attaches the *actual* message list
    it feeds the provider, so the caller observes every user/assistant/tool
    message the run produced without re-deriving anything from events.
    """

    messages: list[Message] = field(default_factory=list)
    raw_task: str = ""
    user_message: Message | None = None

    def attach(self, messages: list[Message], task: str) -> None:
        """Bind the trace to the run's live message list."""
        self.messages = messages
        self.raw_task = task
        # ``build_messages`` appends the new user turn last.
        self.user_message = messages[-1] if messages else None

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return bool(self.messages)


def canonical_history(trace: ConversationTrace) -> list[Message]:
    """The persisted conversation produced by ``trace``'s run.

    * ``system`` messages are dropped: the system prompt and project context are
      rebuilt from the workspace on every run, so storing them would duplicate
      (and freeze) them.
    * The run's user message is stored as the **raw task**, not the
      context-augmented prompt the provider saw, so a continuation does not
      replay a stale context blob.
    """
    history: list[Message] = []
    for message in trace.messages:
        if message.role == "system":
            continue
        if message is trace.user_message and trace.raw_task:
            history.append(Message.user(trace.raw_task))
            continue
        history.append(message)
    return history


__all__ = [
    "CANONICAL_ROLES",
    "CONVERSATION_SCHEMA_VERSION",
    "ConversationTrace",
    "canonical_history",
    "is_canonical_history",
    "is_legacy_history",
    "message_from_dict",
    "message_to_dict",
    "messages_from_dicts",
    "messages_to_dicts",
    "trim_history",
]
