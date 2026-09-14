"""Smart Mode — NovaCLI's safety layer.

Three responsibilities:

1. **Risk classification** of shell commands (safe / moderate / dangerous /
   forbidden) via an explicit, readable rule table.
2. **Path jailing** so file operations can never escape the workspace root or
   touch credential files, even if the model asks them to.
3. **Secret redaction** for everything that leaves the process: logs, terminal
   output and the project context sent to the model.

Design rule: this module fails closed. Anything unrecognised is at least
``MODERATE``, and anything the model tries to do outside the workspace is
``FORBIDDEN``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .models import RiskLevel, SafetyMode

# --- Rule tables ------------------------------------------------------------

# Compiled once. Each entry is (pattern, reason). Ordered most-severe first.
FORBIDDEN_COMMANDS: Sequence[tuple[re.Pattern[str], str]] = tuple(
    (re.compile(pattern), reason)
    for pattern, reason in (
        (r"\brm\s+(?:-[\w-]+\s+)*/(?:\s|$)", "recursive delete of the filesystem root"),
        (r"\brm\s+-[\w-]*r[\w-]*\s+~(?:/\s*)?(?:\s|$)", "recursive delete of the home directory"),
        (r"\brm\s+-[\w-]*r[\w-]*\s+/(?:etc|usr|bin|sbin|system|data)\b", "recursive delete of a system directory"),
        (r"\bmkfs(?:\.\w+)?\b", "filesystem format"),
        (r"\bdd\b[^\n]*\bof=/dev/", "raw write to a block device"),
        (r">\s*/dev/(?:sd|mmcblk|nvme|hda|block)", "raw write to a block device"),
        (r":\(\)\s*\{.*\}\s*;\s*:", "fork bomb"),
        (r"\b(?:shutdown|reboot|halt|poweroff)\b", "system power control"),
        (r"\b(?:init|telinit)\s+[06]\b", "system runlevel change"),
        (r"\bchmod\s+-R\s+777\s+/(?:\s|$)", "world-writable root"),
        (r"\bchown\s+-R\b[^\n]*\s+/(?:\s|$)", "recursive ownership change of root"),
        (r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|k)?sh\b", "pipe remote script into a shell"),
        (r"\bgit\s+push\b[^\n]*--force(?!-with-lease)", "force push can destroy remote history"),
        (r"\bgit\s+reset\s+--hard\b", "hard reset discards uncommitted work"),
        (r"\bgit\s+clean\s+-[\w-]*[fx][\w-]*", "removes untracked files permanently"),
        (r"\b(?:shred|wipefs)\b", "irreversible data destruction"),
        (r"\bkill(?:all)?\s+-9\s+-1\b", "kill every process"),
        (r"\b(?:apt|apt-get|pkg)\s+(?:remove|purge)\b", "uninstalls system packages"),
        (r"\brm\s+-[\w-]*r[\w-]*\s+\$HOME(?:\s|$)", "recursive delete of the home directory"),
        (r"\btruncate\s+-s\s*0\b", "truncates file contents irreversibly"),
        (r"\b>+\s*/(?:etc|usr|bin|sbin)/", "overwrite of a system file"),
        (r"\b(?:su|sudo)\b", "privilege escalation"),
        (r"\b:>\s*/dev/sd", "raw write to a block device"),
        (
            r"\$(?:GROQ_API_KEY|NOVA_SECRET\w*)",
            "reads NovaCLI's own credentials from the environment",
        ),
        (
            r"\b(?:printenv|env)\b[^\n]*\b(?:GROQ_API_KEY|NOVA_SECRET\w*)",
            "reads NovaCLI's own credentials from the environment",
        ),
    )
)

MODERATE_COMMANDS: Sequence[tuple[re.Pattern[str], str]] = tuple(
    (re.compile(pattern), reason)
    for pattern, reason in (
        (r"\brm\b", "deletes files"),
        (r"\brmdir\b", "removes directories"),
        (r"\bmv\b", "moves or overwrites files"),
        (r"\bchmod\b", "changes permissions"),
        (r"\bchown\b", "changes ownership"),
        (r"\bkill\b", "terminates processes"),
        (r"\bpkill\b", "terminates processes"),
        (r"\b(?:pip|pip3)\s+(?:install|uninstall)", "modifies the Python environment"),
        (r"\bpkg\s+(?:install|upgrade)\b", "installs system packages"),
        (r"\b(?:npm|pnpm|yarn)\s+(?:install|add|remove)\b", "modifies the Node environment"),
        (r"\bgit\s+(?:push|reset|checkout\s+--|restore|rebase|filter-branch|branch\s+-D)\b", "rewrites or publishes history"),
        (r"\bgit\s+clean\b", "removes untracked files"),
        (r"\b(?:docker|podman|systemctl|service)\b", "controls services or containers"),
        (r"\b(?:make)\s+\w*(?:clean|distclean)\b", "destructive build target"),
        (r"(?<![0-9<>])>(?!>)\s*\S", "redirects output, overwriting a file"),
        (r"\b(?:curl|wget)\b", "network access"),
        (r"\bshutil\.rmtree\b", "recursive delete"),
        (r"\bos\.remove\b|\bos\.unlink\b|\bos\.rmdir\b", "deletes files"),
        (r"\bos\.system\b|\bsubprocess\b", "spawns further processes"),
        (r"\beval\b|\bexec\b", "dynamic code execution"),
        (r"\bcrontab\b", "modifies scheduled jobs"),
        (r"\b(?:tee)\b\s+/", "writes to a file"),
    )
)

# Commands known to be read-only in practice. Used so STRICT mode stays usable.
SAFE_COMMAND_PREFIXES: frozenset[str] = frozenset(
    {
        "ls", "cat", "pwd", "echo", "printf", "head", "tail", "wc", "grep",
        "rg", "find", "fd", "tree", "file", "stat", "du", "df", "which",
        "whoami", "uname", "date", "env", "sort", "uniq", "cut", "tr", "diff",
        "less", "more", "basename", "dirname", "realpath", "readlink", "sed",
        "awk", "python", "python3", "pytest", "pydoc", "black", "ruff", "mypy",
        "flake8", "isort", "pip", "npm", "node", "git", "jq", "tsort", "xxd",
        "sha256sum", "md5sum", "sleep", "true", "false", "pydoc3",
    }
)

SAFE_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {"status", "diff", "log", "show", "branch", "remote", "describe", "blame",
     "rev-parse", "ls-files", "config", "tag", "shortlog", "stash"}
)

SAFE_PIP_SUBCOMMANDS: frozenset[str] = frozenset({"list", "show", "freeze", "--version"})

# --- Secret protection ------------------------------------------------------

SENSITIVE_FILENAMES: frozenset[str] = frozenset(
    {
        ".env", ".netrc", ".npmrc", ".pypirc", ".git-credentials",
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
        "credentials", "credentials.json", "secrets.json", "secrets.yaml",
        "secrets.yml", ".htpasswd", "keystore.jks",
    }
)

SENSITIVE_SUFFIXES: tuple[str, ...] = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")

SENSITIVE_READ_ALLOWLIST: frozenset[str] = frozenset({".env.example", ".env.sample", ".env.template"})

SENSITIVE_PATH_PARTS: tuple[str, ...] = (
    ".git/config",
    ".git-credentials",
    ".ssh/",
    ".aws/",
    ".gnupg/",
    ".nova/config.json",
)

# Patterns redacted from any outbound text.
_SECRET_PATTERNS: Sequence[tuple[re.Pattern[str], str]] = (
    # Real keys are alphanumeric, but test keys and rotated formats may
    # contain underscores, so allow them to avoid a silent miss.
    (re.compile(r"\bgsk_[A-Za-z0-9_\-]{16,}"), "[REDACTED_GROQ_KEY]"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAIza[A-Za-z0-9\-_]{20,}\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|password|passwd)\b"
            r"(\s*[:=]\s*)(['\"]?)([^\s'\"]{6,})\3"
        ),
        r"\1\2\3[REDACTED]\3",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]{16,}=*"), "Bearer [REDACTED]"),
)


def redact_secrets(text: str, *extra_secrets: str | None) -> str:
    """Strip credential-shaped strings from ``text``.

    Any ``extra_secrets`` (e.g. the live API key) are removed verbatim first,
    which covers keys that match none of the generic patterns.
    """
    if not text:
        return text
    for secret in extra_secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def contains_secret(text: str, *extra_secrets: str | None) -> bool:
    """True when ``text`` appears to contain credential material."""
    return redact_secrets(text, *extra_secrets) != text


# --- Verdicts ---------------------------------------------------------------


@dataclass(frozen=True)
class SafetyVerdict:
    """The safety layer's judgement about one action."""

    level: RiskLevel
    allowed: bool
    requires_approval: bool
    reason: str = ""
    rule: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "level": str(self.level),
            "allowed": self.allowed,
            "requires_approval": self.requires_approval,
            "reason": self.reason,
            "rule": self.rule,
        }


class SafetyError(Exception):
    """Raised when a workspace operation is refused by the safety layer."""

    def __init__(self, message: str, verdict: SafetyVerdict) -> None:
        super().__init__(message)
        self.verdict = verdict


# --- Policy -----------------------------------------------------------------


class SafetyPolicy:
    """Classifies commands and file paths against a :class:`SafetyMode`."""

    def __init__(
        self,
        project_root: str | Path,
        mode: SafetyMode | str = SafetyMode.SMART,
        *,
        extra_deny_patterns: Iterable[str] = (),
        secret_values: Iterable[str | None] = (),
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.mode = SafetyMode(mode) if not isinstance(mode, SafetyMode) else mode
        self.secret_values = tuple(s for s in secret_values if s)
        self._extra_deny = tuple(
            (re.compile(p), "custom denylist rule") for p in extra_deny_patterns
        )

    # -- Command analysis ------------------------------------------------

    def check_command(self, command: str) -> SafetyVerdict:
        """Classify a shell command string."""
        raw = (command or "").strip()
        if not raw:
            return SafetyVerdict(
                RiskLevel.FORBIDDEN, False, False, "empty command", "empty"
            )

        for pattern, reason in self._extra_deny:
            if pattern.search(raw):
                return SafetyVerdict(
                    RiskLevel.FORBIDDEN, False, False, reason, "custom-denylist"
                )

        # A shell can reach any file the path jail protects, so scan the
        # command text for credential targets too. Without this,
        # `cat .env` would read straight past the workspace policy.
        target = self._sensitive_target(raw)
        if target is not None:
            return SafetyVerdict(
                RiskLevel.FORBIDDEN,
                False,
                False,
                f"command references protected credential path {target!r}",
                "sensitive-target",
            )

        for pattern, reason in FORBIDDEN_COMMANDS:
            if pattern.search(raw):
                return self._apply_mode(
                    RiskLevel.FORBIDDEN, reason, f"forbidden:{pattern.pattern}"
                )

        # Note: `rm -rf` style single targets are MODERATE, not forbidden.
        level, reason, rule = RiskLevel.MODERATE, "unrecognised command", "default"

        for pattern, why in MODERATE_COMMANDS:
            if pattern.search(raw):
                level, reason, rule = RiskLevel.MODERATE, why, f"moderate:{pattern.pattern}"
                break

        if self._is_known_safe(raw):
            level, reason, rule = RiskLevel.SAFE, "read-only or local development command", "allowlist"

        return self._apply_mode(level, reason, rule)

    @staticmethod
    def _sensitive_target(command: str) -> str | None:
        """Return the name of a credential path referenced by a command.

        Matches whole tokens rather than substrings, so ``.env.example`` is
        allowed while ``.env``, ``id_rsa`` and ``server.pem`` are not.
        """
        for token in re.split(r"[\s;|&()<>'\"`=,]+", command):
            token = token.strip()
            if not token or token.startswith("-"):
                continue
            name = token.rsplit("/", 1)[-1]
            if not name or name in SENSITIVE_READ_ALLOWLIST:
                continue
            if name in SENSITIVE_FILENAMES or name.startswith(".env."):
                return name
            if Path(name).suffix.lower() in SENSITIVE_SUFFIXES:
                return name

        lowered = command.lower()
        for part in SENSITIVE_PATH_PARTS:
            if part in lowered:
                return part
        return None

    def _is_known_safe(self, command: str) -> bool:
        """Heuristic: a single, allow-listed program with no shell metachars."""
        if re.search(r"[;&|`$><\n]", command):
            return False
        tokens = command.split()
        if not tokens:
            return False
        program = tokens[0].rsplit("/", 1)[-1]
        if program not in SAFE_COMMAND_PREFIXES:
            return False
        if program == "git" and len(tokens) > 1:
            return tokens[1] in SAFE_GIT_SUBCOMMANDS
        if program in {"pip", "pip3"} and len(tokens) > 1:
            return tokens[1] in SAFE_PIP_SUBCOMMANDS
        if program in {"npm", "node"} and len(tokens) > 1:
            return tokens[1] in {"test", "--version", "-v"}
        return True

    # -- Path analysis ---------------------------------------------------

    def is_inside_workspace(self, path: str | Path) -> bool:
        """True when ``path`` resolves inside the workspace root."""
        try:
            candidate = self._resolve(path)
        except OSError:
            return False
        return candidate == self.root or self.root in candidate.parents

    def check_path(
        self, path: str | Path, *, write: bool = False
    ) -> SafetyVerdict:
        """Classify access to a filesystem path."""
        try:
            candidate = self._resolve(path)
        except (OSError, ValueError):
            return SafetyVerdict(
                RiskLevel.FORBIDDEN, False, False, "unresolvable path", "invalid-path"
            )

        name = candidate.name
        lowered = candidate.as_posix().lower()

        if name in SENSITIVE_READ_ALLOWLIST:
            return SafetyVerdict(
                RiskLevel.SAFE, True, False, "example template file", "allowlist"
            )

        for part in SENSITIVE_PATH_PARTS:
            if part in lowered:
                return SafetyVerdict(
                    RiskLevel.FORBIDDEN,
                    False,
                    False,
                    f"credential or tooling path is protected ({part})",
                    "sensitive-path",
                )

        if name in SENSITIVE_FILENAMES or name.startswith(".env."):
            return SafetyVerdict(
                RiskLevel.FORBIDDEN,
                False,
                False,
                f"{name!r} may contain secrets and is never exposed to the model",
                "sensitive-file",
            )

        if candidate.suffix.lower() in SENSITIVE_SUFFIXES:
            return SafetyVerdict(
                RiskLevel.FORBIDDEN, False, False, f"{candidate.suffix} files are protected", "sensitive-suffix"
            )

        if not self.is_inside_workspace(candidate):
            return SafetyVerdict(
                RiskLevel.FORBIDDEN,
                False,
                False,
                "path is outside the workspace root",
                "outside-workspace",
            )

        if write:
            for protected in (".git", ".nova"):
                if protected in candidate.parts:
                    return SafetyVerdict(
                        RiskLevel.FORBIDDEN,
                        False,
                        False,
                        f"writing inside {protected}/ is not allowed",
                        "protected-directory",
                    )
            # NOTE: whether the target is a file or a directory is the
            # caller's concern — this method only answers "is it permitted?".
            return SafetyVerdict(
                RiskLevel.MODERATE, True, True, "writing to a workspace file", "write"
            )

        return SafetyVerdict(RiskLevel.SAFE, True, False, "read inside workspace", "read")

    # -- Mode application ------------------------------------------------

    def _apply_mode(self, level: RiskLevel, reason: str, rule: str) -> SafetyVerdict:
        """Map a raw risk level onto the active policy profile.

        ``allowed`` means "not refused outright"; ``requires_approval`` means
        "a human must confirm before this runs". A non-forbidden action is
        therefore always allowed, and the approval flag alone decides whether
        it proceeds unattended. Keeping those two concerns separate is what
        lets the runner accept a strict-mode action once it has been approved.
        """
        if level == RiskLevel.FORBIDDEN:
            return SafetyVerdict(level, False, False, reason, rule)

        if self.mode == SafetyMode.PERMISSIVE:
            needs_approval = False
        elif self.mode == SafetyMode.STRICT:
            needs_approval = level != RiskLevel.SAFE
        else:  # SMART
            needs_approval = level == RiskLevel.DANGEROUS

        return SafetyVerdict(level, True, needs_approval, reason, rule)

    # -- Tool-call gateway -----------------------------------------------

    def check_tool_call(
        self, tool: str, arguments: Mapping[str, object]
    ) -> SafetyVerdict:
        """Pre-flight check for one agent tool invocation.

        Used to decide whether the user must approve before the tool runs. The
        workspace/runner enforce the same rules again at execution time.
        """
        if tool == "run_command":
            return self.check_command(str(arguments.get("command", "")))
        if tool == "write_file":
            return self.check_path(str(arguments.get("path", "")), write=True)
        if tool == "read_file":
            return self.check_path(str(arguments.get("path", "")))
        if tool in {"list_files", "search", "project_summary"}:
            target = str(arguments.get("path") or ".")
            return self.check_path(target)
        return SafetyVerdict(RiskLevel.SAFE, True, False, "no side effects", "meta")

    # -- Redaction -------------------------------------------------------

    def redact(self, text: str) -> str:
        """Remove credential material, including this session's live key."""
        return redact_secrets(text, *self.secret_values)

    # -- Internals -------------------------------------------------------

    def _resolve(self, path: str | Path) -> Path:
        raw = str(path).replace("\\", "/")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.resolve()
