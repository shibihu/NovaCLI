"""Response parsing utilities for legacy text JSON responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class AgentDecision:
    """A parsed model turn."""

    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final: str | None = None
    raw: str = ""
    parse_error: str | None = None

    @property
    def is_action(self) -> bool:
        return self.action is not None and self.final is None

    @property
    def is_final(self) -> bool:
        return self.final is not None


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_ACTION_KEYS = ("action", "tool", "tool_name", "name", "command")
_INPUT_KEYS = ("action_input", "input", "args", "arguments", "parameters", "params")
_FINAL_KEYS = ("final_answer", "final", "answer", "response", "done", "result", "message")
_THOUGHT_KEYS = ("thought", "reasoning", "thinking", "plan")


def extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` object in ``text``, or ``None``."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _first_present(payload: dict[str, Any], keys: Sequence[str]) -> tuple[str, Any]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return key, payload[key]
    return "", None


def parse_agent_response(text: str) -> AgentDecision:
    """Parse a text model reply into an :class:`AgentDecision`."""
    raw = (text or "").strip()
    if not raw:
        return AgentDecision(raw=text, parse_error="empty response")

    candidate = _FENCE_RE.sub("", raw).strip()
    blob = extract_json_object(candidate)

    payload: dict[str, Any] | None = None
    if blob:
        try:
            loaded = json.loads(blob)
            if isinstance(loaded, dict):
                payload = loaded
        except (json.JSONDecodeError, ValueError):
            payload = None

    if payload is None:
        return AgentDecision(final=raw, raw=text, thought="")

    _, final_value = _first_present(payload, _FINAL_KEYS)
    _, action_value = _first_present(payload, _ACTION_KEYS)
    _, input_value = _first_present(payload, _INPUT_KEYS)
    _, thought_value = _first_present(payload, _THOUGHT_KEYS)

    thought = str(thought_value).strip() if isinstance(thought_value, (str, int, float)) else ""
    action_input = input_value if isinstance(input_value, dict) else {}

    action: str | None = None
    if isinstance(action_value, str) and action_value.strip():
        action = action_value.strip()
        if " " in action and not action_input:
            action, _, remainder = action.partition(" ")
            action_input = {"command": remainder.strip()}
        elif action == "run_command" and not action_input and isinstance(payload.get("command"), str):
            action_input = {"command": payload["command"]}

    if final_value is not None and not action:
        if isinstance(final_value, (dict, list)):
            final_text = json.dumps(final_value, indent=2)
        else:
            final_text = str(final_value).strip()
        if final_text:
            return AgentDecision(thought=thought, final=final_text, raw=text)

    if action:
        return AgentDecision(
            thought=thought, action=action, action_input=action_input, raw=text
        )

    return AgentDecision(
        thought=thought,
        raw=text,
        parse_error="JSON object contained neither an action nor a final_answer",
    )
