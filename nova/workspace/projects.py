"""Project understanding.

:class:`ProjectAnalyzer` turns a directory into a compact, structured picture of
the codebase: what languages it uses, where the entry points are, how to run the
tests, and what the README says. This is what lets NovaCLI answer "what is this
project?" without reading every file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nova.core.models import ProjectSummary

from .files import Workspace, detect_language

#: Files that reveal how a project is built and tested.
KEY_FILE_NAMES: tuple[str, ...] = (
    "README.md", "README.rst", "README.txt", "README",
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "Pipfile", "poetry.lock", "uv.lock",
    "package.json", "tsconfig.json", "deno.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "composer.json", "pubspec.yaml", "mix.exs",
    "Makefile", "Dockerfile", "docker-compose.yml", "compose.yaml",
    ".env.example", "Procfile", "justfile", "Taskfile.yml",
)

ENTRY_POINT_NAMES: tuple[str, ...] = (
    "nova.py", "main.py", "app.py", "__main__.py", "cli.py", "manage.py",
    "server.py", "run.py", "index.js", "index.mjs", "main.js", "main.ts",
    "server.js", "server.ts", "app.js", "app.ts", "index.ts",
    "main.go", "main.rs", "Main.java", "Program.cs", "index.php",
)

TEST_DIR_NAMES: frozenset[str] = frozenset({"tests", "test", "spec", "specs", "__tests__"})


class ProjectAnalyzer:
    """Derives a structured summary of the workspace."""

    def __init__(self, workspace: Workspace, *, max_files: int = 5_000) -> None:
        self.workspace = workspace
        self.max_files = max_files

    # -- Pieces ----------------------------------------------------------

    def source_files(self) -> list[Path]:
        return self.workspace.iter_files(limit=self.max_files)

    def detect_languages(self) -> dict[str, int]:
        """Map language name -> file count, most used first."""
        counts: Counter[str] = Counter()
        for path in self.source_files():
            language = detect_language(path)
            if language:
                counts[language] += 1
        return dict(counts.most_common())

    def find_key_files(self) -> list[str]:
        """Build/config/doc files, in the canonical order above."""
        present: dict[str, str] = {}
        for path in self.source_files():
            if path.name in KEY_FILE_NAMES and path.name not in present:
                present[path.name] = self.workspace.relative(path)
        ordered = [present[name] for name in KEY_FILE_NAMES if name in present]
        # Anything else that looks important and is not already listed.
        for path in self.source_files():
            relative = self.workspace.relative(path)
            if relative in ordered:
                continue
            if len(path.parts) - len(self.workspace.root.parts) == 1 and path.suffix in {".md", ".toml", ".cfg", ".ini", ".txt"}:
                ordered.append(relative)
        return ordered[:25]

    def find_tests(self) -> list[str]:
        """Test files, discovered by directory and filename conventions."""
        found: list[str] = []
        for path in self.source_files():
            relative = self.workspace.relative(path)
            parts = set(Path(relative).parts)
            name = path.name
            is_test = (
                bool(parts & TEST_DIR_NAMES)
                or name.startswith("test_")
                or name.endswith("_test.py")
                or name.endswith(".test.js")
                or name.endswith(".test.ts")
                or name.endswith(".spec.js")
                or name.endswith(".spec.ts")
                or name.endswith("_test.go")
            )
            if is_test:
                found.append(relative)
            if len(found) >= 60:
                break
        return found

    def find_entry_points(self) -> list[str]:
        """Likely program entry points, shallowest first."""
        candidates: list[tuple[int, str]] = []
        for path in self.source_files():
            if path.name not in ENTRY_POINT_NAMES:
                continue
            relative = self.workspace.relative(path)
            candidates.append((len(Path(relative).parts), relative))
        candidates.sort(key=lambda item: item[0])

        unique: list[str] = []
        for _, relative in candidates:
            if relative not in unique:
                unique.append(relative)
        return unique[:10]

    def detect_commands(self) -> dict[str, str]:
        """Best-effort test / install / run commands for this project."""
        commands: dict[str, str] = {}
        names = {Path(p).name for p in self.source_files()}

        if "package.json" in names:
            scripts = self._package_scripts()
            if "test" in scripts:
                commands["test"] = self._node_command(scripts["test"])
            if "build" in scripts:
                commands["build"] = self._node_command(scripts["build"])
            if "dev" in scripts:
                commands["run"] = self._node_command(scripts["dev"])
            commands.setdefault("install", "npm install")

        if {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"} & names:
            has_tests = bool(self.find_tests())
            if has_tests:
                commands.setdefault("test", "python -m pytest -q")
            commands.setdefault("install", "python -m pip install -r requirements.txt")

        if "go.mod" in names:
            commands.setdefault("test", "go test ./...")
            commands.setdefault("build", "go build ./...")
            commands.setdefault("run", "go run .")

        if "Cargo.toml" in names:
            commands.setdefault("test", "cargo test")
            commands.setdefault("build", "cargo build")

        if "Makefile" in names:
            targets = self._make_targets()
            for target in ("test", "build", "lint", "run"):
                if target in targets:
                    commands.setdefault(target, f"make {target}")

        return commands

    @staticmethod
    def _node_command(script: str) -> str:
        """Use the fastest available JS runner that matches the lockfile."""
        return f"npm run {script}" if script else "npm test"

    def _package_scripts(self) -> dict[str, str]:
        for path in self.source_files():
            if path.name == "package.json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                except (OSError, ValueError):
                    return {}
                scripts = data.get("scripts")
                if isinstance(scripts, dict):
                    return {str(k): str(v) for k, v in scripts.items()}
        return {}

    def _make_targets(self) -> set[str]:
        for path in self.source_files():
            if path.name in {"Makefile", "makefile", "GNUmakefile"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return set()
                targets: set[str] = set()
                for line in text.splitlines():
                    if line.startswith("\t") or "=" in line.split(":")[0]:
                        continue
                    head, sep, _ = line.partition(":")
                    if sep and head and not head.startswith("."):
                        targets.add(head.strip().split()[0] if head.strip() else "")
                return targets
        return set()

    def readme_excerpt(self, *, max_chars: int = 1_200) -> str:
        """First meaningful chunk of the README, redacted by the caller."""
        for candidate in ("README.md", "README.rst", "README.txt", "README"):
            path = self.workspace.root / candidate
            if not path.is_file():
                continue
            try:
                text = self.workspace.read_text(path, max_bytes=64_000)
            except (OSError, ValueError):
                continue
            text = "\n".join(line for line in text.splitlines() if line.strip())
            return text[:max_chars]
        return ""

    def has_git(self) -> bool:
        return (self.workspace.root / ".git").exists()

    # -- Summary ---------------------------------------------------------

    def summarize(self) -> ProjectSummary:
        """Build the complete :class:`ProjectSummary`."""
        stats = self.workspace.stats(limit=self.max_files)
        return ProjectSummary(
            name=self.workspace.root.name or "workspace",
            root=str(self.workspace.root),
            languages=self.detect_languages(),
            total_files=stats.files,
            total_bytes=stats.bytes,
            key_files=self.find_key_files(),
            tests=self.find_tests(),
            entry_points=self.find_entry_points(),
            commands=self.detect_commands(),
            has_git=self.has_git(),
            readme=self.readme_excerpt(),
        )

    def render_summary(self, summary: ProjectSummary | None = None) -> str:
        """Human-readable summary used by ``nova summary`` and prompts."""
        summary = summary or self.summarize()
        primary = (
            ", ".join(
                f"{name} ({count})" for name, count in list(summary.languages.items())[:5]
            )
            or "unknown"
        )

        lines = [
            f"Project: {summary.name}",
            f"Root: {summary.root}",
            f"Git repository: {'yes' if summary.has_git else 'no'}",
            f"Files: {summary.total_files}  |  Size: {summary.total_bytes / 1024:.1f} KiB",
            f"Languages: {primary}",
        ]

        if summary.entry_points:
            lines.append("Entry points: " + ", ".join(summary.entry_points[:5]))
        if summary.tests:
            lines.append(f"Test files: {len(summary.tests)}")
        if summary.commands:
            rendered = "  |  ".join(f"{key}: {value}" for key, value in summary.commands.items())
            lines.append(f"Commands: {rendered}")
        if summary.key_files:
            lines.append("Key files: " + ", ".join(summary.key_files[:8]))
        if summary.readme:
            first_line = summary.readme.strip().splitlines()[0][:120]
            lines.append(f"README: {first_line}")

        return "\n".join(lines)
