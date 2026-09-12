"""Tests for :mod:`nova.workspace.files` and :mod:`nova.workspace.projects`."""

from __future__ import annotations

from pathlib import Path

import pytest

from nova.core.safety import SafetyError
from nova.workspace.files import (
    Workspace,
    detect_language,
    looks_binary,
)
from nova.workspace.projects import ProjectAnalyzer


# --- Language detection -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("main.py", "Python"),
        ("app.tsx", "TypeScript"),
        ("style.css", "CSS"),
        ("Dockerfile", "Docker"),
        ("Makefile", "Makefile"),
        ("unknown.zzz", None),
    ],
)
def test_detect_language(name: str, expected: str | None) -> None:
    assert detect_language(name) == expected


def test_looks_binary_by_extension() -> None:
    assert looks_binary("image.png") is True
    assert looks_binary("script.py") is False


def test_looks_binary_by_content() -> None:
    assert looks_binary("data.bin", b"\x00\x01\x02\x03") is True
    assert looks_binary("data.txt", b"hello world") is False


# --- Path jailing -----------------------------------------------------------


def test_resolve_returns_root_relative_path(workspace: Workspace) -> None:
    assert workspace.resolve("main.py") == workspace.root / "main.py"


def test_resolve_refuses_traversal(workspace: Workspace) -> None:
    with pytest.raises(SafetyError):
        workspace.resolve("../escape.py")


def test_resolve_refuses_absolute_outside_path(workspace: Workspace) -> None:
    with pytest.raises(SafetyError):
        workspace.resolve("/etc/passwd")


def test_read_env_file_is_refused(workspace: Workspace) -> None:
    with pytest.raises(SafetyError) as excinfo:
        workspace.read_text(".env")
    assert "secret" in str(excinfo.value).lower() or ".env" in str(excinfo.value)


def test_read_env_example_is_allowed(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("GROQ_API_KEY=\n", encoding="utf-8")
    workspace = Workspace(tmp_path)
    assert "GROQ_API_KEY" in workspace.read_text(".env.example")


def test_relative_output(workspace: Workspace) -> None:
    assert workspace.relative(workspace.root / "sub" / "helper.py") == "sub/helper.py"


def test_relative_for_root_is_dot(workspace: Workspace) -> None:
    assert workspace.relative(".") == "."


# --- Reading ----------------------------------------------------------------


def test_read_text(workspace: Workspace) -> None:
    assert "def greet" in workspace.read_text("main.py")


def test_read_missing_file_raises(workspace: Workspace) -> None:
    with pytest.raises(FileNotFoundError):
        workspace.read_text("nope.py")


def test_read_directory_raises(workspace: Workspace) -> None:
    with pytest.raises(IsADirectoryError):
        workspace.read_text("sub")


def test_read_respects_size_limit(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="larger than"):
        workspace.read_text("main.py", max_bytes=5)


def test_read_refuses_binary(tmp_path: Path) -> None:
    (tmp_path / "blob.dat").write_bytes(b"\x00" * 64)
    workspace = Workspace(tmp_path)
    with pytest.raises(ValueError, match="binary"):
        workspace.read_text("blob.dat")


def test_read_many_reports_failures_inline(workspace: Workspace) -> None:
    result = workspace.read_many(["main.py", ".env", "ghost.py"])
    assert "def greet" in result["main.py"]
    assert result[".env"].startswith("<unavailable")
    assert result["ghost.py"].startswith("<unavailable")


# --- Writing ----------------------------------------------------------------


def test_write_creates_file(workspace: Workspace) -> None:
    entry = workspace.write_text("new.txt", "hello")
    assert entry.path == "new.txt"
    assert entry.size == 5
    assert (workspace.root / "new.txt").read_text() == "hello"


def test_write_creates_parent_directories(workspace: Workspace) -> None:
    workspace.write_text("deep/nested/file.py", "x = 1\n")
    assert (workspace.root / "deep" / "nested" / "file.py").exists()


def test_write_refuses_outside_workspace(workspace: Workspace) -> None:
    with pytest.raises(SafetyError):
        workspace.write_text("../escape.txt", "nope")


def test_write_refuses_env_file(workspace: Workspace) -> None:
    with pytest.raises(SafetyError):
        workspace.write_text(".env", "GROQ_API_KEY=stolen")


def test_write_refuses_git_internals(workspace: Workspace) -> None:
    with pytest.raises(SafetyError):
        workspace.write_text(".git/config", "[core]")


def test_write_refuses_when_no_overwrite(workspace: Workspace) -> None:
    with pytest.raises(FileExistsError):
        workspace.write_text("main.py", "replaced", overwrite=False)


def test_write_overwrites_by_default(workspace: Workspace) -> None:
    workspace.write_text("main.py", "replaced = True\n")
    assert workspace.read_text("main.py") == "replaced = True\n"


def test_write_refuses_oversized_content(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="refusing to write"):
        workspace.write_text("huge.txt", "x" * 6_000_000)


def test_delete_file(workspace: Workspace) -> None:
    workspace.write_text("temp.txt", "x")
    assert workspace.delete("temp.txt") is True
    assert workspace.delete("temp.txt") is False


def test_delete_refuses_directory(workspace: Workspace) -> None:
    with pytest.raises(IsADirectoryError):
        workspace.delete("sub")


# --- Listing / tree ---------------------------------------------------------


def test_list_dir_skips_ignored_entries(workspace: Workspace) -> None:
    names = [entry.path for entry in workspace.list_dir(".")]
    assert "main.py" in names
    assert "__pycache__" not in names
    assert ".git" not in names
    assert ".env" not in names


def test_list_dir_puts_directories_first(workspace: Workspace) -> None:
    entries = workspace.list_dir(".")
    first_file = next(i for i, e in enumerate(entries) if not e.is_dir)
    assert all(e.is_dir for e in entries[:first_file])


def test_list_dir_reports_language(workspace: Workspace) -> None:
    entries = {entry.path: entry for entry in workspace.list_dir(".")}
    assert entries["main.py"].language == "Python"


def test_list_dir_on_a_file_raises(workspace: Workspace) -> None:
    with pytest.raises(NotADirectoryError):
        workspace.list_dir("main.py")


def test_iter_files_excludes_noise(workspace: Workspace) -> None:
    names = {path.name for path in workspace.iter_files()}
    assert "main.py" in names
    assert "junk.pyc" not in names


def test_iter_files_respects_limit(workspace: Workspace) -> None:
    assert len(workspace.iter_files(limit=2)) == 2


def test_tree_renders_structure(workspace: Workspace) -> None:
    rendered = workspace.tree(".", max_depth=2)
    assert "main.py" in rendered
    assert "sub/" in rendered
    assert "__pycache__" not in rendered


def test_tree_truncates(workspace: Workspace) -> None:
    rendered = workspace.tree(".", max_entries=2)
    assert "truncated" in rendered


def test_tree_of_a_single_file(workspace: Workspace) -> None:
    assert workspace.tree("main.py") == "main.py"


def test_stats_counts_files(workspace: Workspace) -> None:
    stats = workspace.stats()
    assert stats.files >= 6
    assert stats.bytes > 0
    assert stats.directories >= 1


# --- Searching --------------------------------------------------------------


def test_search_finds_matches(workspace: Workspace) -> None:
    hits = workspace.search("def greet")
    assert any(hit.path == "main.py" for hit in hits)


def test_search_is_case_insensitive_by_default(workspace: Workspace) -> None:
    assert workspace.search("DEF GREET")


def test_search_case_sensitive_option(workspace: Workspace) -> None:
    assert workspace.search("DEF GREET", case_sensitive=True) == []


def test_search_respects_glob(workspace: Workspace) -> None:
    hits = workspace.search("greet", glob="**/*.md")
    assert all(hit.path.endswith(".md") for hit in hits)


def test_search_max_results(workspace: Workspace) -> None:
    assert len(workspace.search("e", max_results=1)) <= 1


def test_search_regex_mode(workspace: Workspace) -> None:
    hits = workspace.search(r"def\s+\w+", regex=True)
    assert hits


def test_search_invalid_regex_raises(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="invalid search pattern"):
        workspace.search("([", regex=True)


def test_search_empty_query_returns_nothing(workspace: Workspace) -> None:
    assert workspace.search("") == []


def test_search_reports_line_numbers(workspace: Workspace) -> None:
    hits = [h for h in workspace.search("def greet") if h.path == "main.py"]
    assert hits[0].line == 1


def test_search_hit_render(workspace: Workspace) -> None:
    hit = workspace.search("def greet")[0]
    assert hit.render().startswith(f"{hit.path}:{hit.line}:")


# --- Project analysis -------------------------------------------------------


def test_summary_detects_languages(analyzer: ProjectAnalyzer) -> None:
    summary = analyzer.summarize()
    assert summary.languages.get("Python", 0) >= 3


def test_summary_finds_key_files(analyzer: ProjectAnalyzer) -> None:
    summary = analyzer.summarize()
    assert "README.md" in summary.key_files
    assert "requirements.txt" in summary.key_files


def test_summary_finds_tests(analyzer: ProjectAnalyzer) -> None:
    assert "tests/test_main.py" in analyzer.summarize().tests


def test_summary_finds_entry_points(analyzer: ProjectAnalyzer) -> None:
    assert "main.py" in analyzer.summarize().entry_points


def test_summary_detects_pytest_command(analyzer: ProjectAnalyzer) -> None:
    assert "pytest" in analyzer.summarize().commands.get("test", "")


def test_summary_reads_readme(analyzer: ProjectAnalyzer) -> None:
    assert "Demo Project" in analyzer.summarize().readme


def test_summary_reports_git(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert ProjectAnalyzer(Workspace(tmp_path)).summarize().has_git is True


def test_summary_counts_files(analyzer: ProjectAnalyzer) -> None:
    assert analyzer.summarize().total_files >= 6


def test_render_summary_is_readable(analyzer: ProjectAnalyzer) -> None:
    rendered = analyzer.render_summary()
    assert "Project:" in rendered
    assert "Languages:" in rendered


def test_package_scripts_are_detected(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts": {"test": "jest", "build": "tsc"}}', encoding="utf-8"
    )
    commands = ProjectAnalyzer(Workspace(tmp_path)).detect_commands()
    assert commands["test"] == "npm run jest"
    assert commands["build"] == "npm run tsc"


def test_make_targets_are_detected(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("test:\n\tpytest\n\nbuild:\n\tgcc x.c\n", encoding="utf-8")
    commands = ProjectAnalyzer(Workspace(tmp_path)).detect_commands()
    assert commands["test"] == "make test"
