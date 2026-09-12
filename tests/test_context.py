"""Tests for :mod:`nova.core.context`."""

from __future__ import annotations

from pathlib import Path

import pytest

from nova.core.context import ContextBuilder, ContextBundle, keywords
from nova.core.safety import SafetyPolicy
from nova.workspace.files import Workspace

SECRET = "gsk_supersecretkeyvalue_abcdefghijklmnop"


@pytest.fixture
def builder(tmp_path: Path) -> ContextBuilder:
    (tmp_path / "auth.py").write_text(
        f'API_KEY = "{SECRET}"\n\n\ndef login(user):\n    return user\n',
        encoding="utf-8",
    )
    (tmp_path / "unrelated.py").write_text("def unrelated():\n    return 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Auth service\n", encoding="utf-8")
    workspace = Workspace(tmp_path, safety=SafetyPolicy(tmp_path))
    return ContextBuilder(workspace, max_chars=8_000, max_files=3)


# --- keywords ---------------------------------------------------------------


def test_keywords_extracts_significant_tokens() -> None:
    assert "authentication" in keywords("Fix the authentication flow")


def test_keywords_drops_stopwords() -> None:
    found = keywords("please fix the bug in the code")
    assert "please" not in found
    assert "the" not in found


def test_keywords_ignores_short_tokens() -> None:
    assert "ab" not in keywords("ab cd ef")


def test_keywords_empty_input() -> None:
    assert keywords("") == set()


# --- Ranking ----------------------------------------------------------------


def test_rank_files_prefers_a_name_match(builder: ContextBuilder) -> None:
    assert builder.rank_files("update auth")[0] == "auth.py"


def test_rank_files_matches_content(builder: ContextBuilder) -> None:
    assert "auth.py" in builder.rank_files("where is login defined")


def test_rank_files_falls_back_to_entry_points(builder: ContextBuilder) -> None:
    assert builder.rank_files("zzzzz nothing matches") != []


def test_rank_files_respects_limit(builder: ContextBuilder) -> None:
    assert len(builder.rank_files("login auth readme", limit=2)) <= 2


def test_rank_files_returns_unique_paths(builder: ContextBuilder) -> None:
    ranked = builder.rank_files("auth login readme", limit=10)
    assert len(ranked) == len(set(ranked))


# --- Assembly ---------------------------------------------------------------


def test_build_contains_project_brief(builder: ContextBuilder) -> None:
    bundle = builder.build("explain auth.py")
    assert "# Project brief" in bundle.text
    assert "Languages:" in bundle.text


def test_build_contains_structure(builder: ContextBuilder) -> None:
    assert "# Structure" in builder.build("anything").text


def test_build_includes_relevant_file_content(builder: ContextBuilder) -> None:
    bundle = builder.build("fix auth.py")
    assert "auth.py" in bundle.files
    assert "def login" in bundle.text


def test_build_never_leaks_the_api_key(tmp_path: Path) -> None:
    (tmp_path / "creds.py").write_text(f'KEY = "{SECRET}"\n', encoding="utf-8")
    workspace = Workspace(tmp_path, safety=SafetyPolicy(tmp_path, secret_values=[SECRET]))
    builder = ContextBuilder(workspace)

    bundle = builder.build("show me creds.py")
    assert SECRET not in bundle.text
    assert "REDACTED" in bundle.text


def test_build_never_reads_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_dotenv_secret_value_here\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    workspace = Workspace(tmp_path, safety=SafetyPolicy(tmp_path))
    bundle = ContextBuilder(workspace).build("read the env file")
    assert "gsk_dotenv_secret_value_here" not in bundle.text


def test_build_truncates_to_max_chars(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("value = 1\n" * 5_000, encoding="utf-8")
    workspace = Workspace(tmp_path, safety=SafetyPolicy(tmp_path))
    bundle = ContextBuilder(workspace, max_chars=1_000).build("show big.py")
    assert len(bundle.text) <= 1_100
    assert bundle.truncated is True


def test_build_estimates_tokens(builder: ContextBuilder) -> None:
    assert builder.build("hello").estimated_tokens > 0


def test_build_reports_chosen_files(builder: ContextBuilder) -> None:
    bundle = builder.build("auth.py")
    assert isinstance(bundle.files, list)
    assert bundle.summary is not None


def test_build_accepts_extra_files(builder: ContextBuilder) -> None:
    bundle = builder.build("unrelated task", extra_files=["unrelated.py"])
    assert bundle.files[0] == "unrelated.py"


def test_build_handles_unreadable_file_gracefully(builder: ContextBuilder) -> None:
    bundle = builder.build("task", extra_files=[".env"])
    assert "# Project brief" in bundle.text  # still produces a brief


def test_build_includes_rules_section(builder: ContextBuilder) -> None:
    assert "# Rules" in builder.build("task").text


def test_bundle_to_dict(builder: ContextBuilder) -> None:
    data = builder.build("task").to_dict()
    assert set(data) == {"files", "truncated", "estimated_tokens", "chars"}


def test_bundle_str_returns_text(builder: ContextBuilder) -> None:
    bundle = builder.build("task")
    assert str(bundle) == bundle.text


def test_summarize_text_delegates_to_analyzer(builder: ContextBuilder) -> None:
    assert "Project:" in builder.summarize_text()


def test_language_breakdown(builder: ContextBuilder) -> None:
    breakdown = builder.language_breakdown()
    assert "Python" in breakdown


def test_empty_task_still_builds(builder: ContextBuilder) -> None:
    bundle = builder.build("")
    assert isinstance(bundle, ContextBundle)
    assert "# Project brief" in bundle.text
