"""Tests for :mod:`nova.config`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova.config import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    API_KEY_HINT,
    ConfigError,
    Settings,
    load_settings,
    mask_secret,
    parse_dotenv,
    read_dotenv,
    read_user_config,
)

TEST_KEY = "gsk_abcdefghijklmnopqrstuvwxyz012345"


# --- .env parsing -----------------------------------------------------------


def test_parse_dotenv_basic_pairs() -> None:
    parsed = parse_dotenv("A=1\nB=two\n")
    assert parsed == {"A": "1", "B": "two"}


def test_parse_dotenv_handles_comments_blanks_and_export() -> None:
    text = "\n# a comment\n  \nexport TOKEN=abc\nOTHER = spaced \n"
    assert parse_dotenv(text) == {"TOKEN": "abc", "OTHER": "spaced"}


def test_parse_dotenv_strips_matching_quotes() -> None:
    parsed = parse_dotenv('SINGLE=\'one\'\nDOUBLE="two"\nMISMATCH="three\n')
    assert parsed["SINGLE"] == "one"
    assert parsed["DOUBLE"] == "two"
    assert parsed["MISMATCH"] == '"three'


def test_parse_dotenv_ignores_lines_without_equals() -> None:
    assert parse_dotenv("JUST_A_WORD\n=novalue\n") == {}


def test_read_dotenv_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path / "nope.env") == {}


def test_read_dotenv_reads_real_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("KEY=value\n", encoding="utf-8")
    assert read_dotenv(path) == {"KEY": "value"}


def test_read_user_config_tolerates_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_user_config(path) == {}


def test_read_user_config_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[1, 2]", encoding="utf-8")
    assert read_user_config(path) == {}


def test_read_user_config_missing_returns_empty(tmp_path: Path) -> None:
    assert read_user_config(tmp_path / "absent.json") == {}


# --- mask_secret ------------------------------------------------------------


def test_mask_secret_hides_the_middle() -> None:
    masked = mask_secret(TEST_KEY)
    assert TEST_KEY not in masked
    assert masked.startswith("gsk_")
    assert "*" in masked


@pytest.mark.parametrize("value", [None, "", "ab"])
def test_mask_secret_handles_short_values(value: str | None) -> None:
    result = mask_secret(value)
    assert TEST_KEY not in result


# --- API key priority -------------------------------------------------------


def test_environment_wins_over_dotenv_and_user_config(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_from_dotenv_aaaaaaaaaaaa\n", encoding="utf-8")
    user_config = tmp_path / "config.json"
    user_config.write_text(json.dumps({"groq_api_key": "gsk_from_config_bbbbbbbbbbbb"}), encoding="utf-8")

    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": TEST_KEY},
        user_config_path=user_config,
    )
    assert settings.groq_api_key == TEST_KEY
    assert settings.api_key_source == "environment"


def test_dotenv_wins_over_user_config(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_from_dotenv_aaaaaaaaaaaa\n", encoding="utf-8")
    user_config = tmp_path / "config.json"
    user_config.write_text(json.dumps({"groq_api_key": "gsk_from_config_bbbbbbbbbbbb"}), encoding="utf-8")

    settings = load_settings(project_root=tmp_path, env={}, user_config_path=user_config)
    assert settings.groq_api_key == "gsk_from_dotenv_aaaaaaaaaaaa"
    assert settings.api_key_source == ".env"


def test_user_config_used_as_last_resort(tmp_path: Path) -> None:
    user_config = tmp_path / "config.json"
    user_config.write_text(json.dumps({"groq_api_key": TEST_KEY}), encoding="utf-8")

    settings = load_settings(project_root=tmp_path, env={}, user_config_path=user_config)
    assert settings.groq_api_key == TEST_KEY
    assert settings.api_key_source == "user-config"


def test_no_key_anywhere(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path, env={}, user_config_path=tmp_path / "absent.json"
    )
    assert settings.groq_api_key is None
    assert settings.has_api_key is False
    assert settings.api_key_source == "none"


def test_missing_key_error_explains_how_to_configure(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path, env={}, user_config_path=tmp_path / "absent.json"
    )
    with pytest.raises(ConfigError) as excinfo:
        settings.require_api_key()

    message = str(excinfo.value)
    assert "GROQ_API_KEY" in message
    assert ".env" in message
    assert "console.groq.com" in message
    assert message.count("1.") == 1  # the numbered setup list is present


def test_require_api_key_returns_key_when_present(settings: Settings) -> None:
    assert settings.require_api_key() == settings.groq_api_key


def test_load_settings_can_require_key_up_front(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(
            project_root=tmp_path,
            env={},
            user_config_path=tmp_path / "absent.json",
            require_api_key=True,
        )


# --- Defaults & overrides ---------------------------------------------------


def test_defaults_are_applied(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": TEST_KEY},
        user_config_path=tmp_path / "absent.json",
    )
    assert settings.groq_model == DEFAULT_MODEL
    assert settings.command_timeout == DEFAULT_TIMEOUT
    assert settings.project_root == tmp_path.resolve()
    assert settings.safety_mode == "smart"
    assert settings.port == 8000


def test_model_can_be_overridden_by_environment(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": TEST_KEY, "GROQ_MODEL": "llama-3.1-8b-instant"},
        user_config_path=tmp_path / "absent.json",
    )
    assert settings.groq_model == "llama-3.1-8b-instant"


def test_invalid_safety_mode_falls_back_to_smart(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": TEST_KEY, "NOVA_SAFETY_MODE": "reckless"},
        user_config_path=tmp_path / "absent.json",
    )
    assert settings.safety_mode == "smart"


def test_timeout_is_clamped(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": TEST_KEY, "NOVA_COMMAND_TIMEOUT": "999999"},
        user_config_path=tmp_path / "absent.json",
    )
    assert settings.command_timeout <= 1800


def test_non_numeric_timeout_uses_default(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": TEST_KEY, "NOVA_COMMAND_TIMEOUT": "soon"},
        user_config_path=tmp_path / "absent.json",
    )
    assert settings.command_timeout == DEFAULT_TIMEOUT


def test_project_root_from_environment(tmp_path: Path) -> None:
    other = tmp_path / "nested"
    other.mkdir()
    settings = load_settings(
        env={"GROQ_API_KEY": TEST_KEY, "NOVA_PROJECT_ROOT": str(other)},
        user_config_path=tmp_path / "absent.json",
    )
    assert settings.project_root == other.resolve()


def test_with_overrides_returns_new_object(settings: Settings) -> None:
    updated = settings.with_overrides(groq_model="other-model")
    assert updated.groq_model == "other-model"
    assert settings.groq_model == "test-model"


# --- Secret hygiene ---------------------------------------------------------


def test_public_dict_never_contains_the_key(settings: Settings) -> None:
    public = settings.to_public_dict()
    assert TEST_KEY not in json.dumps(public)
    assert public["has_api_key"] is True
    assert "*" in str(public["api_key_preview"])


def test_repr_redacts_the_key(settings: Settings) -> None:
    rendered = repr(settings)
    assert TEST_KEY not in rendered
    assert "Settings(" in rendered


def test_api_key_hint_mentions_all_three_sources() -> None:
    assert "export GROQ_API_KEY" in API_KEY_HINT
    assert ".env" in API_KEY_HINT
    assert "config.json" in API_KEY_HINT


def test_settings_is_frozen(settings: Settings) -> None:
    with pytest.raises(Exception):
        settings.groq_model = "mutated"  # type: ignore[misc]
