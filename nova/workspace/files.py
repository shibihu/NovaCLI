"""Jailed filesystem access.

:class:`Workspace` is the only way NovaCLI touches the user's files. Every path
is resolved against the workspace root and validated by the safety policy, so a
model that asks to read ``../../etc/passwd`` or ``.env`` simply gets refused.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from nova.core.models import FileEntry, SearchHit
from nova.core.safety import (
    SENSITIVE_FILENAMES,
    SENSITIVE_PATH_PARTS,
    SENSITIVE_READ_ALLOWLIST,
    SENSITIVE_SUFFIXES,
    SafetyError,
    SafetyMode,
    SafetyPolicy,
    SafetyVerdict,
)

#: Directories that are noise for an LLM and expensive to walk.
DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
        "env", ".env.d", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox",
        ".idea", ".vscode", "dist", "build", ".next", ".nuxt", "target",
        ".gradle", ".nova", ".nova-cache", ".cache", "coverage", "htmlcov",
        ".terraform", "bower_components", ".dart_tool", "Pods",
    }
)

DEFAULT_IGNORED_PATTERNS: tuple[str, ...] = (
    "*.pyc", "*.pyo", "*.so", "*.o", "*.a", "*.class", "*.jar", "*.log",
    "*.tmp", "*.swp", "*.lock", ".DS_Store", "Thumbs.db", "*.min.js",
    "*.min.css", "*.map",
)

#: Extensions treated as binary and therefore never read as text.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svgz",
        ".pdf", ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tar",
        ".mp3", ".mp4", ".wav", ".ogg", ".avi", ".mov", ".mkv", ".webm",
        ".ttf", ".otf", ".woff", ".woff2", ".eot", ".exe", ".dll", ".dylib",
        ".bin", ".dat", ".db", ".sqlite", ".sqlite3", ".pyc", ".class",
        ".jar", ".apk", ".aab", ".iso", ".img", ".dex",
    }
)

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "Python", ".pyi": "Python", ".ipynb": "Jupyter",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".jsx": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin", ".scala": "Scala",
    ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".hpp": "C++",
    ".cs": "C#", ".swift": "Swift", ".m": "Objective-C", ".mm": "Objective-C++",
    ".lua": "Lua", ".pl": "Perl", ".r": "R", ".jl": "Julia", ".dart": "Dart",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".fish": "Shell",
    ".ps1": "PowerShell", ".bat": "Batch", ".cmd": "Batch",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".sass": "Sass", ".less": "Less", ".vue": "Vue", ".svelte": "Svelte",
    ".sql": "SQL", ".graphql": "GraphQL", ".proto": "Protocol Buffers",
    ".md": "Markdown", ".rst": "reStructuredText", ".txt": "Text",
    ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
    ".ini": "INI", ".cfg": "Config", ".xml": "XML", ".csv": "CSV",
    ".dockerfile": "Docker", ".tf": "Terraform", ".gradle": "Gradle",
}

FILENAME_LANGUAGES: dict[str, str] = {
    "Dockerfile": "Docker",
    "Makefile": "Makefile",
    "makefile": "Makefile",
    "GNUmakefile": "Makefile",
    "Jenkinsfile": "Groovy",
    "Procfile": "Procfile",
    ".gitignore": "Git Config",
    ".dockerignore": "Docker",
}

# Guard rails for a phone: never read more than a few MB into memory.
MAX_READ_BYTES = 2_000_000
MAX_WRITE_BYTES = 5_000_000


def is_sensitive_name(name: str) -> bool:
    """True when a filename suggests credential material.

    Used to keep such files out of directory listings, trees and searches —
    they are refused on read anyway, so surfacing them is pure noise.
    """
    if name in SENSITIVE_READ_ALLOWLIST:
        return False
    return (
        name in SENSITIVE_FILENAMES
        or name.startswith(".env.")
        or Path(name).suffix.lower() in SENSITIVE_SUFFIXES
    )


def detect_language(path: str | Path) -> str | None:
    """Guess the language of a file from its name."""
    p = Path(path)
    if p.name in FILENAME_LANGUAGES:
        return FILENAME_LANGUAGES[p.name]
    if p.suffix.lower() == ".example":
        p = p.with_suffix("")
    return LANGUAGE_BY_EXTENSION.get(p.suffix.lower())


def looks_binary(path: str | Path, sample: bytes | None = None) -> bool:
    """True when a file is (probably) binary and should not be read as text."""
    p = Path(path)
    if p.suffix.lower() in BINARY_EXTENSIONS:
        return True
    if sample is None:
        return False
    if b"\x00" in sample:
        return True
    # Heuristic: mostly non-text bytes in the sample.
    if not sample:
        return False
    text_chars = bytes(range(0x20, 0x7F)) + b"\n\r\t\f\b"
    non_text = sum(1 for byte in sample if byte not in text_chars)
    return non_text / len(sample) > 0.30


@dataclass(frozen=True)
class WorkspaceStats:
    """Cheap aggregate numbers about the workspace."""

    files: int
    directories: int
    bytes: int

    def to_dict(self) -> dict[str, int]:
        return {"files": self.files, "directories": self.directories, "bytes": self.bytes}


class Workspace:
    """Root-jailed view of the user's project."""

    def __init__(
        self,
        root: str | Path,
        *,
        safety: SafetyPolicy | None = None,
        extra_ignore: tuple[str, ...] = (),
        create_root: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if create_root:
            self.root.mkdir(parents=True, exist_ok=True)
        self.safety = safety or SafetyPolicy(self.root, SafetyMode.SMART)
        self.ignored_dirs = DEFAULT_IGNORED_DIRS
        self.ignored_patterns = DEFAULT_IGNORED_PATTERNS + tuple(extra_ignore)

    # -- Path handling ---------------------------------------------------

    def resolve(self, path: str | Path, *, for_write: bool = False) -> Path:
        """Resolve ``path`` and enforce the safety policy.

        Raises:
            SafetyError: if the path is outside the workspace or protected.
        """
        raw = Path(path).expanduser()
        if not raw.is_absolute():
            raw = self.root / raw
        try:
            candidate = raw.resolve()
        except OSError as exc:
            raise SafetyError(f"cannot resolve path {path!r}: {exc}", self._deny("unresolvable path")) from exc

        verdict = self.safety.check_path(candidate, write=for_write)
        if not verdict.allowed:
            raise SafetyError(
                f"Refused: {verdict.reason} ({self.relative(candidate)})", verdict
            )
        return candidate

    def relative(self, path: str | Path) -> str:
        """Workspace-relative display path, using ``/`` separators."""
        candidate = Path(path)
        try:
            resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        except OSError:  # pragma: no cover - defensive
            return str(path)
        try:
            return resolved.relative_to(self.root).as_posix() or "."
        except ValueError:
            return str(path)

    def _deny(self, reason: str) -> SafetyVerdict:
        from nova.core.models import RiskLevel

        return SafetyVerdict(RiskLevel.FORBIDDEN, False, False, reason, "path")

    def is_ignored(self, path: str | Path) -> bool:
        """True when a path should be skipped by tree walks and searches.

        Covers build noise, editor/OS junk, and credential files (which the
        safety layer would refuse to read anyway).
        """
        p = Path(path)
        if any(part in self.ignored_dirs for part in p.parts):
            return True
        if is_sensitive_name(p.name):
            return True
        lowered = p.as_posix().lower()
        if any(part in lowered for part in SENSITIVE_PATH_PARTS):
            return True
        return any(fnmatch.fnmatch(p.name, pattern) for pattern in self.ignored_patterns)

    # -- Reading ---------------------------------------------------------

    def exists(self, path: str | Path) -> bool:
        try:
            return self.resolve(path).exists()
        except SafetyError:
            return False

    def read_text(self, path: str | Path, *, max_bytes: int = MAX_READ_BYTES) -> str:
        """Read a UTF-8 text file, refusing binaries and oversized files."""
        target = self.resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"{self.relative(target)} does not exist")
        if target.is_dir():
            raise IsADirectoryError(f"{self.relative(target)} is a directory")

        size = target.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"{self.relative(target)} is {size} bytes, larger than the {max_bytes} byte limit"
            )

        raw = target.read_bytes()
        if looks_binary(target, raw[:4096]):
            raise ValueError(f"{self.relative(target)} looks like a binary file")

        return raw.decode("utf-8", errors="replace")

    def head_text(self, path: str | Path, *, max_bytes: int = 8_000) -> str:
        """Read at most ``max_bytes`` from the start of a text file.

        Unlike :meth:`read_text`, an oversized file is not an error: the
        leading chunk is returned. This is what context assembly wants, since
        a large generated file should still contribute its opening lines.
        """
        target = self.resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"{self.relative(target)} does not exist")
        if target.is_dir():
            raise IsADirectoryError(f"{self.relative(target)} is a directory")

        with open(target, "rb") as handle:
            raw = handle.read(max_bytes)
        if looks_binary(target, raw[:4096]):
            raise ValueError(f"{self.relative(target)} looks like a binary file")
        return raw.decode("utf-8", errors="replace")

    def read_many(self, paths: list[str], *, max_bytes_each: int = 20_000) -> dict[str, str]:
        """Best-effort read of several files; failures are reported inline."""
        out: dict[str, str] = {}
        for path in paths:
            try:
                out[self.relative(path)] = self.read_text(path, max_bytes=max_bytes_each)
            except (OSError, ValueError, SafetyError) as exc:
                out[self.relative(path)] = f"<unavailable: {exc}>"
        return out

    # -- Writing ---------------------------------------------------------

    def write_text(
        self, path: str | Path, content: str, *, create_dirs: bool = True, overwrite: bool = True
    ) -> FileEntry:
        """Write text to a workspace file, creating parent directories."""
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ValueError(
                f"refusing to write {len(encoded)} bytes (limit {MAX_WRITE_BYTES})"
            )
        target = self.resolve(path, for_write=True)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"{self.relative(target)} is a directory")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{self.relative(target)} already exists")

        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return self.entry(target)

    def delete(self, path: str | Path) -> bool:
        """Delete a single file (never a directory tree)."""
        target = self.resolve(path, for_write=True)
        if not target.exists():
            return False
        if target.is_dir():
            raise IsADirectoryError(
                f"{self.relative(target)} is a directory; refusing to delete directories"
            )
        target.unlink()
        return True

    # -- Listing ---------------------------------------------------------

    def entry(self, path: str | Path) -> FileEntry:
        target = Path(path)
        if not target.is_absolute():
            target = self.root / target
        try:
            stat = target.stat()
            size = stat.st_size
        except OSError:
            size = 0
        return FileEntry(
            path=self.relative(target),
            is_dir=target.is_dir(),
            size=size,
            language=None if target.is_dir() else detect_language(target),
        )

    def list_dir(self, path: str | Path = ".", *, include_ignored: bool = False) -> list[FileEntry]:
        """List one directory level, directories first, then alphabetical."""
        target = self.resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"{self.relative(target)} is not a directory")

        entries: list[FileEntry] = []
        try:
            children = sorted(target.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return entries

        for child in children:
            if not include_ignored and self.is_ignored(child):
                continue
            entries.append(self.entry(child))
        entries.sort(key=lambda e: (not e.is_dir, e.path.lower()))
        return entries

    def iter_files(
        self, path: str | Path = ".", *, limit: int = 5_000, max_depth: int = 12
    ) -> list[Path]:
        """Walk the workspace collecting non-ignored regular files."""
        start = self.resolve(path)
        if start.is_file():
            return [start]
        found: list[Path] = []
        base_depth = len(start.parts)
        for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
            current = Path(dirpath)
            if len(current.parts) - base_depth >= max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in self.ignored_dirs]
            for name in filenames:
                candidate = current / name
                if self.is_ignored(candidate):
                    continue
                found.append(candidate)
                if len(found) >= limit:
                    return found
        return found

    def stats(self, *, limit: int = 20_000) -> WorkspaceStats:
        """File/directory/byte totals, ignoring noise directories."""
        files = directories = total_bytes = 0
        seen_dirs: set[Path] = set()
        for path in self.iter_files(limit=limit):
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
            files += 1
            parent = path.parent
            while parent != self.root and parent not in seen_dirs:
                seen_dirs.add(parent)
                parent = parent.parent
        directories = len(seen_dirs)
        return WorkspaceStats(files=files, directories=directories, bytes=total_bytes)

    def tree(self, path: str | Path = ".", *, max_depth: int = 3, max_entries: int = 200) -> str:
        """Render an ASCII tree, depth- and entry-limited for small screens."""
        start = self.resolve(path)
        if not start.exists():
            raise FileNotFoundError(f"{self.relative(start)} does not exist")
        if start.is_file():
            return self.relative(start)

        lines: list[str] = [f"{self.relative(start) or '.'}/"]
        counter = {"n": 0, "truncated": False}

        def walk(directory: Path, prefix: str, depth: int) -> None:
            if counter["truncated"] or depth > max_depth:
                return
            try:
                children = sorted(
                    (c for c in directory.iterdir() if not self.is_ignored(c)),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except OSError:
                return
            for index, child in enumerate(children):
                if counter["n"] >= max_entries:
                    counter["truncated"] = True
                    return
                counter["n"] += 1
                last = index == len(children) - 1
                lines.append(f"{prefix}{'└── ' if last else '├── '}{child.name}{'/' if child.is_dir() else ''}")
                if child.is_dir():
                    walk(child, prefix + ("    " if last else "│   "), depth + 1)

        walk(start, "", 1)
        if counter["truncated"]:
            lines.append(f"... [tree truncated at {max_entries} entries]")
        return "\n".join(lines)

    # -- Searching -------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        glob: str | None = None,
        max_results: int = 50,
        case_sensitive: bool = False,
        regex: bool = False,
        max_file_bytes: int = 1_000_000,
    ) -> list[SearchHit]:
        """Search file contents for ``query``.

        A dependency-free equivalent of ``grep -rn``. Handles both plain
        substring and regular-expression modes.
        """
        if not query:
            return []
        flags = 0 if case_sensitive else __import__("re").IGNORECASE
        try:
            matcher = __import__("re").compile(query if regex else __import__("re").escape(query), flags)
        except __import__("re").error as exc:
            raise ValueError(f"invalid search pattern: {exc}") from exc

        hits: list[SearchHit] = []
        for path in self.iter_files():
            relative = self.relative(path)
            if glob and not fnmatch.fnmatch(relative, glob) and not fnmatch.fnmatch(path.name, glob):
                continue
            if path.suffix.lower() in BINARY_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in text[:2048]:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if matcher.search(line):
                    hits.append(SearchHit(path=relative, line=number, text=line[:300]))
                    if len(hits) >= max_results:
                        return hits
        return hits
