"""Tests for NovaConfigStore and global credential resolution independent of CWD."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from nova.config import (
    NovaConfigStore,
    Settings,
    load_settings,
)

GLOBAL_KEY = "gsk_global_key_0123456789abcdef"
DOTENV_KEY = "gsk_dotenv_key_9876543210fedcba"
ENV_KEY = "gsk_env_key_11223344556677889900"


def test_nova_config_store_paths(tmp_path: Path) -> None:
    nova_dir = tmp_path / ".nova"
    store = NovaConfigStore(nova_dir=nova_dir)
    assert store.config_path == nova_dir / "config.json"
    assert store.credentials_path == nova_dir / "credentials.json"


def test_save_and_load_credentials(tmp_path: Path) -> None:
    nova_dir = tmp_path / ".nova"
    store = NovaConfigStore(nova_dir=nova_dir)

    assert store.load_credentials() == {}

    store.save_credentials({"groq_api_key": GLOBAL_KEY})
    assert store.credentials_path.exists()

    # Verify permissions on POSIX systems
    if os.name == "posix":
        mode = store.credentials_path.stat().st_mode
        assert mode & stat.S_IRWXG == 0
        assert mode & stat.S_IRWXO == 0

    creds = store.load_credentials()
    assert creds.get("groq_api_key") == GLOBAL_KEY


def test_global_credentials_work_across_different_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nova_dir = tmp_path / ".nova"
    store = NovaConfigStore(nova_dir=nova_dir)
    store.save_credentials({"groq_api_key": GLOBAL_KEY})

    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    tmp_dir = tmp_path / "tmp_dir"
    for d in (project_a, project_b, tmp_dir):
        d.mkdir()

    # Run load_settings from project_a
    monkeypatch.chdir(project_a)
    settings_a = load_settings(
        env={},
        user_config_path=store.config_path,
        user_credentials_path=store.credentials_path,
    )
    assert settings_a.groq_api_key == GLOBAL_KEY
    assert settings_a.api_key_source == "global-credentials"

    # Run load_settings from project_b
    monkeypatch.chdir(project_b)
    settings_b = load_settings(
        env={},
        user_config_path=store.config_path,
        user_credentials_path=store.credentials_path,
    )
    assert settings_b.groq_api_key == GLOBAL_KEY
    assert settings_b.api_key_source == "global-credentials"

    # Run load_settings from /tmp directory
    monkeypatch.chdir(tmp_dir)
    settings_tmp = load_settings(
        env={},
        user_config_path=store.config_path,
        user_credentials_path=store.credentials_path,
    )
    assert settings_tmp.groq_api_key == GLOBAL_KEY
    assert settings_tmp.api_key_source == "global-credentials"


def test_credential_resolution_priority(tmp_path: Path) -> None:
    nova_dir = tmp_path / ".nova"
    store = NovaConfigStore(nova_dir=nova_dir)
    store.save_credentials({"groq_api_key": GLOBAL_KEY})

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dotenv_path = project_dir / ".env"
    dotenv_path.write_text(f"GROQ_API_KEY={DOTENV_KEY}\n", encoding="utf-8")

    # Priority 1: Environment variable
    settings_env = load_settings(
        project_root=project_dir,
        env={"GROQ_API_KEY": ENV_KEY},
        user_config_path=store.config_path,
        user_credentials_path=store.credentials_path,
    )
    assert settings_env.groq_api_key == ENV_KEY
    assert settings_env.api_key_source == "environment"

    # Priority 2: Project .env
    settings_dotenv = load_settings(
        project_root=project_dir,
        env={},
        user_config_path=store.config_path,
        user_credentials_path=store.credentials_path,
    )
    assert settings_dotenv.groq_api_key == DOTENV_KEY
    assert settings_dotenv.api_key_source == ".env"

    # Priority 3: Global credentials (remove .env)
    dotenv_path.unlink()
    settings_global = load_settings(
        project_root=project_dir,
        env={},
        user_config_path=store.config_path,
        user_credentials_path=store.credentials_path,
    )
    assert settings_global.groq_api_key == GLOBAL_KEY
    assert settings_global.api_key_source == "global-credentials"


def test_no_credential_leaks_in_dict_or_repr(tmp_path: Path) -> None:
    nova_dir = tmp_path / ".nova"
    store = NovaConfigStore(nova_dir=nova_dir)
    store.save_credentials({"groq_api_key": GLOBAL_KEY})

    settings = load_settings(
        project_root=tmp_path,
        env={},
        user_config_path=store.config_path,
        user_credentials_path=store.credentials_path,
    )

    public_dict = settings.to_public_dict()
    assert GLOBAL_KEY not in json.dumps(public_dict)
    assert public_dict["has_api_key"] is True
    assert repr(settings).find(GLOBAL_KEY) == -1
