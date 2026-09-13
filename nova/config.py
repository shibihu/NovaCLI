"""Configuration loading for NovaCLI.

Resolution order for the API key (highest priority first):

1. ``GROQ_API_KEY`` environment variable
2. ``.env`` file in the project root
3. ``~/.nova/credentials.json`` (``{"groq_api_key": "gsk_..."}``)
4. ``~/.nova/config.json`` (``{"groq_api_key": "gsk_..."}``)

Secrets are never logged, never returned to the browser and never placed in
the project context handed to the model. Every external representation of a
:class:`Settings` object goes through :meth:`Settings.to_public_dict`, which
replaces the key with a boolean + masked preview.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

# --- Defaults ---------------------------------------------------------------

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_STEPS = 8
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_CONTEXT_CHARS = 14_000
DEFAULT_APPROVAL_TIMEOUT = 300.0

MIN_TIMEOUT = 1
MAX_TIMEOUT = 1800
MIN_MAX_STEPS = 1
MAX_MAX_STEPS = 50

ENV_FILE_NAME = ".env"
DEFAULT_USER_CONFIG_PATH = Path.home() / ".nova" / "config.json"
DEFAULT_USER_CREDENTIALS_PATH = Path.home() / ".nova" / "credentials.json"

VALID_SAFETY_MODES = ("smart", "strict", "permissive")

API_KEY_HINT = f"""No Groq API key configured.

NovaCLI needs a key to talk to the model. Set it in any ONE of these places
(highest priority first):

  1. Environment variable:
         export GROQ_API_KEY=gsk_your_key_here

  2. A {ENV_FILE_NAME} file in your project root:
         cp .env.example {ENV_FILE_NAME}
         # then edit {ENV_FILE_NAME} and set GROQ_API_KEY=gsk_...

  3. Global credentials file ({DEFAULT_USER_CREDENTIALS_PATH}):
         {{"groq_api_key": "gsk_your_key_here"}}

  4. User-level config file ({DEFAULT_USER_CONFIG_PATH}):
         {{"groq_api_key": "gsk_your_key_here"}}

Get a free key at https://console.groq.com/keys
"""


class ConfigError(Exception):
    """Raised when NovaCLI cannot build a usable configuration."""


# --- Credential & Config Store ---------------------------------------------


class NovaConfigStore:
    """Storage for user-level global configuration and credentials under ~/.nova/."""

    def __init__(self, nova_dir: str | Path | None = None) -> None:
        if nova_dir is None:
            self.nova_dir = Path.home() / ".nova"
        else:
            self.nova_dir = Path(nova_dir).expanduser().resolve()

    @property
    def config_path(self) -> Path:
        return self.nova_dir / "config.json"

    @property
    def credentials_path(self) -> Path:
        return self.nova_dir / "credentials.json"

    def load_config(self) -> dict[str, object]:
        return read_user_config(self.config_path)

    def save_config(self, data: dict[str, object]) -> None:
        self.nova_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_credentials(self) -> dict[str, object]:
        return read_user_config(self.credentials_path)

    def save_credentials(self, data: dict[str, object]) -> None:
        self.nova_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2)
        if os.name == "posix":
            fd = os.open(self.credentials_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with open(fd, "w", encoding="utf-8") as f:
                f.write(text)
            try:
                os.chmod(self.credentials_path, 0o600)
            except OSError:
                pass
        else:
            self.credentials_path.write_text(text, encoding="utf-8")


# --- Primitive helpers ------------------------------------------------------


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` text into a mapping."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def read_dotenv(path: Path) -> dict[str, str]:
    """Read a ``.env`` file, returning ``{}`` if it is missing or unreadable."""
    try:
        return parse_dotenv(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeDecodeError):
        return {}


def read_user_config(path: Path) -> dict[str, object]:
    """Read JSON config/credential file, tolerating a missing or corrupt file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def mask_secret(secret: str | None, *, keep: int = 4) -> str:
    """Return a display-safe version of a secret, e.g. ``gsk_…a1b2``."""
    if not secret:
        return ""
    if len(secret) <= keep:
        return "*" * len(secret)
    return f"{secret[:keep]}{'*' * 6}{secret[-2:]}"


def _as_int(value: object, default: int, *, low: int, high: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


# --- Settings ---------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Resolved NovaCLI configuration."""

    groq_api_key: str | None
    groq_model: str
    project_root: Path
    command_timeout: int

    # Additional (non-required) knobs.
    max_steps: int = DEFAULT_MAX_STEPS
    safety_mode: str = "smart"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT
    api_key_source: str = "none"
    extra_ignore: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    # -- Derived state -----------------------------------------------------

    @property
    def has_api_key(self) -> bool:
        return bool(self.groq_api_key)

    def require_api_key(self) -> str:
        """Return the API key or raise :class:`ConfigError` with setup steps."""
        if not self.groq_api_key:
            raise ConfigError(API_KEY_HINT)
        return self.groq_api_key

    @property
    def masked_api_key(self) -> str:
        return mask_secret(self.groq_api_key)

    def with_overrides(self, **changes: object) -> "Settings":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def to_public_dict(self) -> dict[str, object]:
        """A browser/CLI-safe view. Contains no secret material."""
        return {
            "groq_model": self.groq_model,
            "project_root": str(self.project_root),
            "command_timeout": self.command_timeout,
            "max_steps": self.max_steps,
            "safety_mode": self.safety_mode,
            "host": self.host,
            "port": self.port,
            "has_api_key": self.has_api_key,
            "api_key_preview": self.masked_api_key,
            "api_key_source": self.api_key_source,
        }

    def __repr__(self) -> str:  # pragma: no cover
        """Redacted repr so tracebacks and logs never leak the key."""
        return (
            f"Settings(groq_api_key={self.masked_api_key!r}, "
            f"groq_model={self.groq_model!r}, "
            f"project_root={str(self.project_root)!r}, "
            f"command_timeout={self.command_timeout!r}, "
            f"safety_mode={self.safety_mode!r})"
        )


# --- Loading ----------------------------------------------------------------


def load_settings(
    *,
    project_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
    user_config_path: str | Path | None = None,
    user_credentials_path: str | Path | None = None,
    require_api_key: bool = False,
    **overrides: object,
) -> Settings:
    """Build :class:`Settings` using the documented priority order."""
    environ: Mapping[str, str] = os.environ if env is None else env
    user_config = read_user_config(
        Path(user_config_path) if user_config_path else DEFAULT_USER_CONFIG_PATH
    )
    user_credentials = read_user_config(
        Path(user_credentials_path) if user_credentials_path else DEFAULT_USER_CREDENTIALS_PATH
    )

    # -- Workspace root ----------------------------------------------------
    raw_root = (
        project_root
        or environ.get("NOVA_PROJECT_ROOT")
        or _as_str(user_config.get("project_root"))
        or Path.cwd()
    )
    root = Path(raw_root).expanduser()
    try:
        root = root.resolve()
    except OSError:
        root = root.absolute()

    # -- .env layer --------------------------------------------------------
    env_file = Path(dotenv_path) if dotenv_path else root / ENV_FILE_NAME
    dotenv = read_dotenv(env_file)

    def layered(key: str) -> str | None:
        """Environment variable > .env > user credentials > user config."""
        from_env = environ.get(key)
        if from_env:
            return _as_str(from_env)
        if key in dotenv and dotenv[key]:
            return dotenv[key]
        from_creds = user_credentials.get(key) or user_credentials.get(key.lower())
        if from_creds:
            return _as_str(from_creds)
        return _as_str(user_config.get(key) or user_config.get(key.lower())) or None

    # -- API key -----------------------------------------------------------
    api_key = layered("GROQ_API_KEY")
    if environ.get("GROQ_API_KEY"):
        key_source = "environment"
    elif dotenv.get("GROQ_API_KEY"):
        key_source = ENV_FILE_NAME
    elif user_credentials.get("groq_api_key") or user_credentials.get("GROQ_API_KEY"):
        key_source = "global-credentials"
    elif user_config.get("groq_api_key") or user_config.get("GROQ_API_KEY"):
        key_source = "user-config"
    else:
        key_source = "none"

    # -- Remaining values --------------------------------------------------
    model = layered("GROQ_MODEL") or DEFAULT_MODEL
    timeout = _as_int(
        layered("NOVA_COMMAND_TIMEOUT") or DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT,
        low=MIN_TIMEOUT,
        high=MAX_TIMEOUT,
    )
    max_steps = _as_int(
        layered("NOVA_MAX_STEPS") or DEFAULT_MAX_STEPS,
        DEFAULT_MAX_STEPS,
        low=MIN_MAX_STEPS,
        high=MAX_MAX_STEPS,
    )
    port = _as_int(
        layered("NOVA_PORT") or DEFAULT_PORT, DEFAULT_PORT, low=1, high=65535
    )

    safety_mode = (layered("NOVA_SAFETY_MODE") or "smart").lower()
    if safety_mode not in VALID_SAFETY_MODES:
        safety_mode = "smart"

    settings = Settings(
        groq_api_key=api_key,
        groq_model=_as_str(model, DEFAULT_MODEL),
        project_root=root,
        command_timeout=timeout,
        max_steps=max_steps,
        safety_mode=safety_mode,
        host=layered("NOVA_HOST") or DEFAULT_HOST,
        port=port,
        max_context_chars=_as_int(
            layered("NOVA_MAX_CONTEXT_CHARS") or DEFAULT_MAX_CONTEXT_CHARS,
            DEFAULT_MAX_CONTEXT_CHARS,
            low=2_000,
            high=200_000,
        ),
        approval_timeout=float(
            _as_int(
                layered("NOVA_APPROVAL_TIMEOUT") or int(DEFAULT_APPROVAL_TIMEOUT),
                int(DEFAULT_APPROVAL_TIMEOUT),
                low=5,
                high=3600,
            )
        ),
        api_key_source=key_source,
        environment=dict(environ),
    )

    if overrides:
        settings = settings.with_overrides(**overrides)

    if require_api_key:
        settings.require_api_key()
    return settings
