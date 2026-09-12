"""Project understanding / prompt assembly.

:class:`ContextBuilder` decides *what the model gets to see*. It assembles a
compact brief — project shape, file tree, and the handful of files most likely
relevant to the task — and redacts secrets on the way out.

Hard guarantees:

* The API key and any credential material are redacted before inclusion.
* ``.env``-style files are never read (the :class:`Workspace` refuses them).
* Total size is bounded so a large repo cannot blow past the context window or
  a phone's memory.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from nova.workspace.files import BINARY_EXTENSIONS, Workspace, detect_language
from nova.workspace.projects import ProjectAnalyzer

from .models import ProjectSummary
from .safety import SafetyError, SafetyPolicy

#: Words too generic to be useful for ranking file relevance.
STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "from", "into", "your",
        "you", "our", "are", "was", "were", "will", "would", "should", "could",
        "can", "may", "might", "must", "have", "has", "had", "does", "did",
        "doing", "make", "made", "use", "using", "used", "add", "adding",
        "fix", "fixes", "fixing", "bug", "bugs", "please", "help", "need",
        "want", "like", "just", "then", "than", "when", "where", "what",
        "which", "how", "why", "who", "all", "any", "some", "not", "but",
        "about", "code", "file", "files", "project", "nova", "nova cli",
        "change", "changes", "update", "updates", "new", "old", "run", "runs",
        "test", "tests", "testing",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def keywords(text: str) -> set[str]:
    """Extract meaningful lowercase keywords from free text."""
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in STOPWORDS
    }


@dataclass
class ContextBundle:
    """The assembled context plus the files that were chosen."""

    text: str
    files: list[str] = field(default_factory=list)
    summary: ProjectSummary | None = None
    truncated: bool = False
    estimated_tokens: int = 0

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, object]:
        return {
            "files": self.files,
            "truncated": self.truncated,
            "estimated_tokens": self.estimated_tokens,
            "chars": len(self.text),
        }


class ContextBuilder:
    """Builds the project brief handed to the model."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        analyzer: ProjectAnalyzer | None = None,
        safety: SafetyPolicy | None = None,
        max_chars: int = 14_000,
        max_files: int = 6,
        max_chars_per_file: int = 2_200,
        max_scan_bytes: int = 256_000,
        tree_depth: int = 3,
        tree_entries: int = 120,
    ) -> None:
        self.workspace = workspace
        self.analyzer = analyzer or ProjectAnalyzer(workspace)
        self.safety = safety or workspace.safety
        self.max_chars = max_chars
        self.max_files = max_files
        self.max_chars_per_file = max_chars_per_file
        self.max_scan_bytes = max_scan_bytes
        self.tree_depth = tree_depth
        self.tree_entries = tree_entries

    # -- Relevance ranking -----------------------------------------------

    def rank_files(self, task: str, *, limit: int | None = None) -> list[str]:
        """Rank workspace files by relevance to ``task``.

        Scoring favours filename hits over path hits over content hits, so a
        file called ``auth.py`` beats one that merely mentions "auth". Content
        is always scanned (bounded) because some tasks - "where is login
        defined?" - can only be answered by looking inside the files.
        """
        limit = limit if limit is not None else self.max_files
        task_keywords = keywords(task)
        if not task_keywords:
            return self._fallback_files(limit)

        scored: Counter[str] = Counter()
        for candidate in self.workspace.iter_files(limit=2_000):
            relative = self.workspace.relative(candidate)
            relative_lower = relative.lower()
            name_lower = candidate.name.lower()
            stem_lower = candidate.stem.lower()

            score = 0
            for keyword in task_keywords:
                if keyword in (name_lower, stem_lower):
                    score += 12
                elif keyword in name_lower:
                    score += 6
                elif keyword in relative_lower:
                    score += 4

            score += 2 * self._content_hits(candidate, task_keywords)
            if score:
                scored[relative] = score

        if not scored:
            return self._fallback_files(limit)
        return [path for path, _ in scored.most_common(limit)][:limit]

    def _content_hits(self, candidate: Any, task_keywords: set[str]) -> int:
        """How many task keywords appear in a file's body (0 when unreadable)."""
        if candidate.suffix.lower() in BINARY_EXTENSIONS:
            return 0
        if candidate.suffix.lower() in {".json", ".lock", ".map"}:
            return 0
        try:
            if candidate.stat().st_size > self.max_scan_bytes:
                return 0
            body = self.workspace.read_text(candidate, max_bytes=self.max_scan_bytes)
        except (OSError, ValueError, SafetyError):
            return 0
        lowered = body.lower()
        return sum(1 for keyword in task_keywords if keyword in lowered)

    def _fallback_files(self, limit: int) -> list[str]:
        """When no keywords match, lead with entry points and tests."""
        summary = self.analyzer.summarize()
        ordered = list(summary.entry_points) + list(summary.key_files)
        return ordered[:limit]

    # -- Assembly --------------------------------------------------------

    def build(self, task: str = "", *, extra_files: list[str] | None = None) -> ContextBundle:
        """Assemble the full project brief."""
        summary = self.analyzer.summarize()
        chosen: list[str] = []

        for path in (extra_files or []):
            relative = self.workspace.relative(path)
            if relative not in chosen:
                chosen.append(relative)
        for path in self.rank_files(task, limit=self.max_files):
            if path not in chosen:
                chosen.append(path)

        truncated = False
        sections: list[str] = []

        # 1. Project shape -------------------------------------------------
        languages = ", ".join(
            f"{name} ({count})" for name, count in list(summary.languages.items())[:6]
        ) or "unknown"
        head = [
            "# Project brief",
            f"Name: {summary.name}",
            f"Root: {summary.root}",
            f"Languages: {languages}",
            f"Files: {summary.total_files} ({summary.total_bytes / 1024:.1f} KiB)",
            f"Git repository: {'yes' if summary.has_git else 'no'}",
        ]
        if summary.entry_points:
            head.append("Entry points: " + ", ".join(summary.entry_points[:6]))
        if summary.commands:
            head.append(
                "Likely commands: "
                + "; ".join(f"{key}={value}" for key, value in summary.commands.items())
            )
        if summary.tests:
            head.append(f"Test files: {len(summary.tests)} (e.g. {', '.join(summary.tests[:3])})")
        sections.append("\n".join(head))

        # 2. Structure -----------------------------------------------------
        try:
            tree = self.workspace.tree(".", max_depth=self.tree_depth, max_entries=self.tree_entries)
        except (OSError, ValueError):
            tree = "(unavailable)"
        sections.append("# Structure\n```\n" + tree + "\n```")

        # 3. README --------------------------------------------------------
        if summary.readme:
            sections.append("# README (excerpt)\n" + summary.readme[:1_200])

        # 4. Relevant files ------------------------------------------------
        file_blocks: list[str] = []
        for relative in chosen:
            block = self._render_file(relative)
            if block:
                file_blocks.append(block)
        if file_blocks:
            sections.append("# Possibly relevant files\n\n" + "\n\n".join(file_blocks))

        sections.append(
            "# Rules\n"
            "- Paths are relative to the project root.\n"
            "- Secrets are redacted; credential files are unavailable by design.\n"
            "- Always read a file before editing it."
        )

        text = "\n\n".join(sections)
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n\n... [context truncated]"
            truncated = True

        text = self.safety.redact(text)
        return ContextBundle(
            text=text,
            files=chosen,
            summary=summary,
            truncated=truncated,
            estimated_tokens=max(1, len(text) // 4),
        )

    def _render_file(self, relative: str) -> str:
        """Render one file as a fenced block, or a note if unavailable."""
        language = detect_language(relative) or ""
        fence = language.lower().replace(" ", "") if language else "text"
        try:
            # head_text, not read_text: a big generated file should still
            # contribute its opening lines instead of being dropped entirely.
            content = self.workspace.head_text(
                relative, max_bytes=self.max_chars_per_file * 4
            )
        except (OSError, ValueError, SafetyError) as exc:
            return f"### {relative}\n(unavailable: {exc})"

        if len(content) > self.max_chars_per_file:
            content = content[: self.max_chars_per_file].rstrip() + "\n... [truncated]"
        return f"### {relative}\n```{fence}\n{content}\n```"

    # -- Convenience -----------------------------------------------------

    def summarize_text(self) -> str:
        """The rendered project summary, for ``nova summary``."""
        return self.analyzer.render_summary()

    def language_breakdown(self) -> dict[str, str]:
        """Map detected language -> ``"<count> files"`` for UI display."""
        return {
            name: f"{count} file{'s' if count != 1 else ''}"
            for name, count in self.analyzer.detect_languages().items()
        }


__all__ = [
    "ContextBuilder",
    "ContextBundle",
    "keywords",
]
