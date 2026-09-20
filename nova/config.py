"""Configuration loading for NovaCLI.

Resolution order for configuration and credentials (highest priority first):

1. Explicit CLI arguments / overrides
2. Environment variables (NOVA_PROVIDER, NOVA_MODEL, GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY, OLLAMA_BASE_URL)
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

from nova.ai import normalize_provider_name, provider_names

# --- Defaults ---------------------------------------------------------------

DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b"
DEFAULT_CEREBRAS_MODEL = "llama3.1-8b"

DEFAULT_TIMEOUT = 30
DEFAULT_LLM_TIMEOUT = 1800
DEFAULT_MAX_STEPS = 8
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_CONTEXT_CHARS = 14_000
DEFAULT_APPROVAL_TIMEOUT = 300.0

MIN_TIMEOUT = 1
MAX_COMMAND_TIMEOUT = 1800
MAX_LLM_TIMEOUT = 86400
MIN_MAX_STEPS = 1
MAX_MAX_STEPS = 50

ENV_FILE_NAME = ".env"
DEFAULT_USER_CONFIG_PATH = Path.home() / ".nova" / "config.json"
DEFAULT_USER_CREDENTIALS_PATH = Path.home() / ".nova" / "credentials.json"

VALID_SAFETY_MODES = ("smart", "strict", "permissive")
VALID_PROVIDERS = provider_names()


def get_api_key_hint(provider: str) -> str:
    prov = normalize_provider_name(provider)
    if prov == "ollama":
        return "Ollama does not require an API key. Ensure Ollama server is running at your configured base URL."

    names = {
        "groq": ("Groq", "GROQ_API_KEY", "gsk_your_key_here", "https://console.groq.com/keys"),
        "gemini": ("Gemini", "GEMINI_API_KEY", "your_gemini_api_key", "https://aistudio.google.com/app/apikey"),
        "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", "sk-or-v1-your_key", "https://openrouter.ai/keys"),
        "cerebras": ("Cerebras", "CEREBRAS_API_KEY", "csk-your_key", "https://cloud.cerebras.ai"),
    }
    display_name, env_var, example, url = names.get(prov, ("Groq", "GROQ_API_KEY", "gsk_your_key_here", "https://console.groq.com/keys"))

    return f"""No {display_name} API key configured.

NovaCLI needs a key to talk to {display_name}. Set it in any ONE of these places
(highest priority first):

  1. Environment variable:
         export {env_var}={example}

  2. A {ENV_FILE_NAME} file in your project root:
         cp .env.example {ENV_FILE_NAME}
         # then edit {ENV_FILE_NAME} and set {env_var}={example}

  3. Global credentials file ({DEFAULT_USER_CREDENTIALS_PATH}):
         {{"{env_var.lower()}": "{example}"}}

  4. User-level config file ({DEFAULT_USER_CONFIG_PATH}):
         {{"{env_var.lower()}": "{example}"}}

Get a key at {url}
"""


API_KEY_HINT = get_api_key_hint("groq")


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


def prompt_and_save_api_key(store: NovaConfigStore | None = None, provider: str = "groq") -> str:
    """Prompt the user for a missing provider API key interactively in TTY and save it globally."""
    store = store or NovaConfigStore()
    prov = normalize_provider_name(provider)
    if prov == "ollama":
        print("Ollama does not require an API key.")
        return ""

    hint = get_api_key_hint(prov)
    if not sys.stdin.isatty():
        raise ConfigError(hint)

    key_field = f"{prov}_api_key"
    disp_name = prov.capitalize()
    print(f"\n{disp_name} API key is not configured.")
    try:
        import getpass
        key = getpass.getpass(f"Enter your {disp_name} API key: ").strip()
    except Exception:
        key = input(f"Enter your {disp_name} API key: ").strip()

    if not key:
        raise ConfigError("API key cannot be empty.")

    store.save_credentials({key_field: key})
    print(f"✓ Saved {disp_name} API key globally to {store.credentials_path}\n")
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
    gemini_api_key: str | None
    gemini_model: str
    ollama_model: str
    ollama_base_url: str
    openrouter_api_key: str | None
    openrouter_model: str
    cerebras_api_key: str | None
    cerebras_model: str
    project_root: Path
    command_timeout: int
    llm_timeout: int = DEFAULT_LLM_TIMEOUT

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
    #: Retry a provider call that failed with a temporary rate limit.
    rate_limit_retry: bool = True
    #: Seconds to wait when the provider sent no usable Retry-After hint.
    rate_limit_fallback_seconds: int = 60
    execution_backend: str = "local"
    docker_image: str = "python:3.12-slim"
    extra_ignore: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    # -- Derived state -----------------------------------------------------

    @property
    def active_api_key(self) -> str | None:
        prov = normalize_provider_name(self.provider)
        if prov == "groq":
            return self.groq_api_key
        if prov == "gemini":
            return self.gemini_api_key
        if prov == "openrouter":
            return self.openrouter_api_key
        if prov == "cerebras":
            return self.cerebras_api_key
        return None

    @property
    def active_secrets(self) -> list[str]:
        keys = [self.groq_api_key, self.gemini_api_key, self.openrouter_api_key, self.cerebras_api_key, self.web_token]
        return [k for k in keys if k]

    @property
    def model(self) -> str:
        """Return the active model name based on selected provider."""
        prov = normalize_provider_name(self.provider)
        if prov == "ollama":
            return self.ollama_model
        if prov == "gemini":
            return self.gemini_model
        if prov == "openrouter":
            return self.openrouter_model
        if prov == "cerebras":
            return self.cerebras_model
        return self.groq_model

    @property
    def has_api_key(self) -> bool:
        prov = normalize_provider_name(self.provider)
        if prov == "ollama":
            return True
        return bool(self.active_api_key)

    def require_api_key(self) -> str:
        """Return the active API key or raise :class:`ConfigError` with setup steps."""
        prov = normalize_provider_name(self.provider)
        if prov == "ollama":
            return ""
        key = self.active_api_key
        if not key:
            raise ConfigError(get_api_key_hint(prov))
        return key

    @property
    def masked_api_key(self) -> str:
        prov = normalize_provider_name(self.provider)
        if prov == "ollama":
            return "not required"
        return mask_secret(self.active_api_key)

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
            "gemini_model": self.gemini_model,
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "openrouter_model": self.openrouter_model,
            "cerebras_model": self.cerebras_model,
            "project_root": str(self.project_root),
            "command_timeout": self.command_timeout,
            "llm_timeout": self.llm_timeout,
            "max_steps": self.max_steps,
            "safety_mode": self.safety_mode,
            "host": self.host,
            "port": self.port,
            "has_api_key": self.has_api_key,
            "api_key_preview": self.masked_api_key,
            "api_key_source": self.api_key_source,
            "has_web_token": self.has_web_token,
            "reasoning_effort": self.reasoning_effort,
            "execution_backend": self.execution_backend,
            "docker_image": self.docker_image,
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
    raw_prov = layered("NOVA_PROVIDER") or _as_str(user_config.get("provider")) or DEFAULT_PROVIDER
    provider = normalize_provider_name(raw_prov)
    if provider not in VALID_PROVIDERS:
        provider = DEFAULT_PROVIDER

    # -- Provider Keys -----------------------------------------------------
    groq_key = layered("GROQ_API_KEY")
    gemini_key = layered("GEMINI_API_KEY")
    openrouter_key = layered("OPENROUTER_API_KEY")
    cerebras_key = layered("CEREBRAS_API_KEY")

    active_env_var = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }.get(provider)

    key_source = "none"
    if active_env_var:
        field_name = active_env_var.lower()
        if environ.get(active_env_var):
            key_source = "environment"
        elif dotenv.get(active_env_var):
            key_source = ENV_FILE_NAME
        elif user_credentials.get(field_name) or user_credentials.get(active_env_var):
            key_source = "global-credentials"
        elif user_config.get(field_name) or user_config.get(active_env_var):
            key_source = "user-config"
    elif provider == "ollama":
        key_source = "not-required"

    web_token = layered("NOVA_WEB_TOKEN")

    # -- Models & Ollama Endpoint ------------------------------------------
    generic_model = layered("NOVA_MODEL")
    groq_model = layered("GROQ_MODEL") or (generic_model if provider == "groq" else None) or DEFAULT_GROQ_MODEL
    gemini_model = layered("GEMINI_MODEL") or (generic_model if provider == "gemini" else None) or DEFAULT_GEMINI_MODEL
    ollama_model = layered("OLLAMA_MODEL") or (generic_model if provider == "ollama" else None) or DEFAULT_OLLAMA_MODEL
    openrouter_model = layered("OPENROUTER_MODEL") or (generic_model if provider == "openrouter" else None) or DEFAULT_OPENROUTER_MODEL
    cerebras_model = layered("CEREBRAS_MODEL") or (generic_model if provider == "cerebras" else None) or DEFAULT_CEREBRAS_MODEL
    ollama_base_url = layered("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL

    # -- Remaining values --------------------------------------------------
    command_timeout = _as_int(
        layered("NOVA_COMMAND_TIMEOUT") or DEFAULT_TIMEOUT,
        DEFAULT_TIMEOUT,
        low=MIN_TIMEOUT,
        high=MAX_COMMAND_TIMEOUT,
    )
    llm_timeout = _as_int(
        layered("NOVA_LLM_TIMEOUT") or DEFAULT_LLM_TIMEOUT,
        DEFAULT_LLM_TIMEOUT,
        low=MIN_TIMEOUT,
        high=MAX_LLM_TIMEOUT,
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
        groq_api_key=groq_key,
        groq_model=_as_str(groq_model, DEFAULT_GROQ_MODEL),
        gemini_api_key=gemini_key,
        gemini_model=_as_str(gemini_model, DEFAULT_GEMINI_MODEL),
        ollama_model=_as_str(ollama_model, DEFAULT_OLLAMA_MODEL),
        ollama_base_url=_as_str(ollama_base_url, DEFAULT_OLLAMA_BASE_URL),
        openrouter_api_key=openrouter_key,
        openrouter_model=_as_str(openrouter_model, DEFAULT_OPENROUTER_MODEL),
        cerebras_api_key=cerebras_key,
        cerebras_model=_as_str(cerebras_model, DEFAULT_CEREBRAS_MODEL),
        project_root=root,
        command_timeout=command_timeout,
        llm_timeout=llm_timeout,
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
        execution_backend=layered("NOVA_EXECUTION_BACKEND") or "local",
        docker_image=layered("NOVA_DOCKER_IMAGE") or "python:3.12-slim",
        environment={**dotenv, **environ},
    )

    if overrides:
        settings = settings.with_overrides(**overrides)

    if require_api_key:
        settings.require_api_key()
    return settings
