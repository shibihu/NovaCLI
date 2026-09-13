"""Agent tool declarations and OpenAI/Groq schema conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of an agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "read_file",
        "Read a UTF-8 text file from the project.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative file path to read"}},
            "required": ["path"],
        },
    ),
    ToolSpec(
        "write_file",
        "Create or overwrite a project file.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path to write"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        "list_files",
        "List a directory in the project.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path (default '.')"},
                "depth": {"type": "integer", "description": "Recursion depth (default 1)"},
            },
            "required": [],
        },
    ),
    ToolSpec(
        "search",
        "Search file contents.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text or regex query to search"},
                "glob": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        "run_command",
        "Run a shell command in the project root.",
        {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
            "required": ["command"],
        },
    ),
    ToolSpec(
        "project_summary",
        "Describe the project: languages, entry points, commands.",
        {"type": "object", "properties": {}, "required": []},
    ),
)

TOOL_NAMES: frozenset[str] = frozenset(spec.name for spec in TOOL_SPECS)


def render_tool_catalog() -> str:
    """Tool list rendered for legacy text fallback."""
    return "\n".join(f"- {s.name}: {s.description}" for s in TOOL_SPECS)


def get_native_tools_schema() -> list[dict[str, Any]]:
    """Convert TOOL_SPECS into native OpenAI/Groq tool schema format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in TOOL_SPECS
    ]
