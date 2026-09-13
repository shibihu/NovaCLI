"""Configuration loading for NovaCLI.

Resolution order for configuration and credentials (highest priority first):

1. Explicit CLI arguments / overrides
2. Environment variables (NOVA_PROVIDER, NOVA_MODEL, OLLAMA_BASE_URL, GROQ_API_KEY, GROQ_MODEL)
3. Project .env file
4. Global user credentials (~/.nova/credentials.json)
5. Global user config (~/.nova/config.json)
6. Defaults
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

# --- Defaults ---------------------------------------------------------------

DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

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
VALID_PROVIDERS = ("groq", "ollama")

API_KEY_HINT = f"""No Groq API key configured.

NovaCLI needs a key to talk to Groq. Set it in any ONE of these places
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
        current = self.load_config()
        current.update(data)
        self.config_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def load_credentials(self) -> dict[str, object]:
        return read_user_config(self.credentials_path)

    def save_credentials(self, data: dict[str, object]) -> None:
        self.nova_dir.mkdir(parents=True, exist_ok=True)
        current = self.load_credentials()
        current.update(data)
        text = json.dumps(current, indent=2)
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


def prompt_and_save_api_key(store: NovaConfigStore | None = None) -> str:
    """Prompt the user for a missing Groq API key interactively in TTY and save it globally."""
    store = store or NovaConfigStore()
    if not sys.stdin.isatty():
        raise ConfigError(API_KEY_HINT)

    print("\nGroq API key is not configured.")
    try:
        import getpass
        key = getpass.getpass("Enter your Groq API key (gsk_...): ").strip()
    except Exception:
        key = input("Enter your Groq API key (gsk_...): ").strip()

    if not key:
        raise ConfigError("API key cannot be empty.")

    store.save_credentials({"groq_api_key": key})
    print(f"✓ Saved Groq API key globally to {store.credentials_path}\n")
    return key


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

    provider: str
    groq_api_key: str | None
    groq_model: str
    ollama_model: str
    ollama_base_url: str
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
    web_token: str | None = None
    reasoning_effort: str | None = None
    extra_ignore: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    # -- Derived state -----------------------------------------------------

    @property
    def model(self) -> str:
        """Return the active model name based on selected provider."""
        if self.provider == "ollama":
            return self.ollama_model
        return self.groq_model

    @property
    def has_api_key(self) -> bool:
        if self.provider == "ollama":
            return True
        return bool(self.groq_api_key)

    def require_api_key(self) -> str:
        """Return the API key or raise :class:`ConfigError` with setup steps."""
        if self.provider == "ollama":
            return ""
        if not self.groq_api_key:
            raise ConfigError(API_KEY_HINT)
        return self.groq_api_key

    @property
    def masked_api_key(self) -> str:
        return mask_secret(self.groq_api_key)

    @property
    def has_web_token(self) -> bool:
        return bool(self.web_token)

    def with_overrides(self, **changes: object) -> "Settings":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def to_public_dict(self) -> dict[str, object]:
        """A browser/CLI-safe view. Contains no secret material."""
        return {
            "provider": self.provider,
            "model": self.model,
            "groq_model": self.groq_model,
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "project_root": str(self.project_root),
            "command_timeout": self.command_timeout,
            "max_steps": self.max_steps,
            "safety_mode": self.safety_mode,
            "host": self.host,
            "port": self.port,
            "has_api_key": self.has_api_key,
            "api_key_preview": self.masked_api_key,
            "api_key_source": self.api_key_source,
            "has_web_token": self.has_web_token,
            "reasoning_effort": self.reasoning_effort,
        }

    def __repr__(self) -> str:  # pragma: no cover
        """Redacted repr so tracebacks and logs never leak the key."""
        return (
            f"Settings(provider={self.provider!r}, model={self.model!r}, "
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

    # -- Provider ----------------------------------------------------------
    provider = (layered("NOVA_PROVIDER") or _as_str(user_config.get("provider")) or DEFAULT_PROVIDER).lower()
    if provider not in VALID_PROVIDERS:
        provider = DEFAULT_PROVIDER

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

    web_token = layered("NOVA_WEB_TOKEN")

    # -- Models & Ollama Endpoint ------------------------------------------
    generic_model = layered("NOVA_MODEL")
    groq_model = layered("GROQ_MODEL") or (generic_model if provider == "groq" else None) or DEFAULT_GROQ_MODEL
    ollama_model = layered("OLLAMA_MODEL") or (generic_model if provider == "ollama" else None) or DEFAULT_OLLAMA_MODEL
    ollama_base_url = layered("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL

    # -- Remaining values --------------------------------------------------
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

    reasoning_effort = layered("NOVA_REASONING_EFFORT")
    if reasoning_effort and reasoning_effort.lower() in {"low", "medium", "high"}:
        reasoning_effort = reasoning_effort.lower()
    else:
        reasoning_effort = None

    settings = Settings(
        provider=provider,
        groq_api_key=api_key,
        groq_model=_as_str(groq_model, DEFAULT_GROQ_MODEL),
        ollama_model=_as_str(ollama_model, DEFAULT_OLLAMA_MODEL),
        ollama_base_url=_as_str(ollama_base_url, DEFAULT_OLLAMA_BASE_URL),
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
        web_token=web_token,
        reasoning_effort=reasoning_effort,
        environment=dict(environ),
    )

    if overrides:
        settings = settings.with_overrides(**overrides)

    if require_api_key and settings.provider == "groq":
        settings.require_api_key()
    return settings
