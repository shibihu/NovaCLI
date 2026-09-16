# ⚡ NovaCLI v2.0

**An AI-powered developer environment** — an autonomous coding agent, a command-line
interface, Project Intelligence 2.0, and a mobile-first web IDE, all driven by one shared core.

Built to run on **Android + Termux + Linux**, with **Python 3.11+ (3.13/3.14 ready)**.

```
                    ⚡ NovaCLI
                         │
                  ┌──────┴──────┐
                  │  Nova Core  │
                  │   Python    │
                  └──────┬──────┘
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       AI Agent     Workspace    🛡️ Safety
          │              │              │
       Groq API       Filesystem    Smart Mode
                         │
                         ↓
                     Runner
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
          🖥️ CLI                FastAPI
                                  │
                                  ↓
                              IDE Web UI
```

**The CLI and the Web IDE call the same `NovaAgent`.** They differ only in how they
render the event stream and how they answer approval prompts — so the two surfaces
can never drift apart.

---

## Table of contents

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start](#quick-start)
- [Configuration & Global Credentials](#configuration)
- [Project Intelligence 2.0](#project-intelligence-20)
- [Native Tool Calling](#native-tool-calling)
- [CLI reference](#cli-reference)
- [Web IDE](#web-ide)
- [Safety — Smart Mode](#safety--smart-mode)
- [Architecture](#architecture)
- [HTTP API](#http-api)
- [Testing](#testing)
- [Termux / Android notes](#termux--android-notes)
- [Troubleshooting](#troubleshooting)
- [Extending](#extending)

---

## What it does

| Goal | How NovaCLI delivers it |
|---|---|
| **AI coding agent** | A reasoning loop that plans, executes native tool calls, observes results and reports every step. |
| **Native Tool Calling** | Full native Groq/OpenAI tool-calling support (`openai/gpt-oss-20b`, `llama-3.3-70b-versatile`) with structured schema definitions, eliminating text JSON parsing issues. |
| **Global Credentials** | API keys are stored globally in `~/.nova/credentials.json` and persist regardless of changing directories (`cwd`). |
| **Project Intelligence 2.0** | Ecosystem detection (Python, Node.js, Luau/Roblox, Godot, Web), entry points, dependencies, test frameworks, Git state, and safe caching without rescan loops. |
| **Mobile-first Web IDE** | Four tabs — Chat, Files, Terminal, Project — with 44px touch targets, safe-area insets and a dark theme. No build step. |
| **CLI** | `nova ask`, `nova chat`, `nova serve`, `nova config`, `nova project` plus workspace commands for files, search and project understanding. |
| **Safe code/file operations** | Every path is resolved against the workspace root. Credential files are refused at read, write *and* shell level. |
| **Command execution** | Timeout-bounded, process-group-killed, output-truncated, secret-scrubbed shell execution. |
| **Testing** | The agent is told to verify its own changes; `nova summary` infers the project's test command. |
| **Error analysis** | Provider errors, policy refusals and tool failures all become readable explanations, never crashes. |
| **Agent progress reporting** | A typed event stream (`thought`, `tool_call`, `tool_result`, `progress`, `final`, …) rendered live in the terminal and streamed over SSE to the browser. |

---

## Install

### Termux / Android

```bash
pkg update && pkg upgrade
pkg install python git

git clone <your-repo-url> NovaCLI
cd NovaCLI
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional — install as a command (`nova ...` instead of `python nova.py ...`):

```bash
pip install -e .
```

> **Do not use a virtualenv on `/sdcard`.** Android's emulated storage is mounted
> `noexec`, so `python -m venv` fails on the `lib64` symlink and compiled wheels
> cannot be loaded. Keep the checkout in the Termux home directory
> (`~/NovaCLI`), or install directly with `pip install --target` and set
> `PYTHONPATH`.

Get a free API key at **<https://console.groq.com/keys>**.

---

## Quick start

```bash
# 1. Set global credential once (persists across all directory changes)
nova config set-key gsk_your_groq_api_key_here

# 2. Or initialize a project-specific .env file
nova init

# 3. Check everything is wired up from any directory
cd ~/Projects/App
nova doctor

# 4. Ask something
nova ask "what does this project do?"

# 5. Or open the mobile IDE
nova serve --host 0.0.0.0
```

Then browse to `http://<your-device-ip>:8000` on your phone.

---

## Configuration

### Global Credentials & Priority

NovaCLI separates global user configuration and credentials from project-specific configuration. Your Groq API key is stored globally under `~/.nova/credentials.json` (with `0600` permissions on POSIX systems) so changing directory never breaks AI agent functionality.

Credential resolution priority (first match wins):

1. **Environment variable** — `export GROQ_API_KEY=gsk_...`
2. **Project `.env`** in the workspace root — `GROQ_API_KEY=gsk_...`
3. **Global credentials** — `~/.nova/credentials.json` (`{"groq_api_key": "gsk_..."}`)
4. **Global user config** — `~/.nova/config.json` (`{"groq_api_key": "gsk_..."}`)

If no key is found, NovaCLI prints an actionable explanation and exits cleanly with code `3`.

```bash
# Example: Changing working directory retains your global key
cd ~/project-a
nova ask "hello"

cd ~/project-b
nova ask "hello"

cd /tmp
nova doctor
```

### Settings

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `groq_api_key` | `GROQ_API_KEY` | — | Provider credential |
| `groq_model` | `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model id (GPT-OSS 20B recommended) |
| `project_root` | `NOVA_PROJECT_ROOT` | current directory | Workspace root the agent may touch |
| `command_timeout` | `NOVA_COMMAND_TIMEOUT` | `30` | Seconds before a command is killed |
| `max_steps` | `NOVA_MAX_STEPS` | `8` | Agent reasoning steps per task |
| `safety_mode` | `NOVA_SAFETY_MODE` | `smart` | `smart` \| `strict` \| `permissive` |
| `reasoning_effort` | `NOVA_REASONING_EFFORT` | `None` | `low` \| `medium` \| `high` (for GPT-OSS models) |
| `host` / `port` | `NOVA_HOST` / `NOVA_PORT` | `127.0.0.1` / `8000` | Web IDE bind address |

---

## Native Tool Calling

NovaCLI uses native Groq/OpenAI tool calling. Rather than forcing models to generate raw text JSON, NovaCLI formats tools using standard JSON schema definitions and passes them via `tools` with `tool_choice="auto"`.

```
    Model (e.g. gpt-oss-20b, llama-3.3-70b)
                      │
           native tool call (id & args)
                      │
                      ▼
               Nova ToolBox
                      │
           Safety & execution result
                      │
                      ▼
            tool result message (id)
                      │
                      ▼
            Model → final answer
```

This ensures reliable execution with models such as `openai/gpt-oss-20b` while retaining backward-compatible text parsing as a fallback.

---

## Project Intelligence 2.0

NovaCLI includes Project Intelligence 2.0, a subsystem that analyzes project context before the AI Agent executes a task.

- **Ecosystems**: Python, Node.js, Roblox/Luau, Godot, Web (HTML/CSS/JS).
- **Metadata**: Package managers (`pip`, `npm`, `yarn`, `pnpm`, `poetry`, `uv`, `wally`), dependencies, frameworks (`FastAPI`, `React`, `Flask`, `Pydantic`, etc.).
- **Structure**: Entry points (`main.py`, `app.py`, `index.js`, `project.godot`), tests, and Git repository state (`branch`, modified files, untracked files).
- **Caching**: Cached safely at `<project_root>/.nova/intelligence.json`.
- **Secret Protection**: Secret files (`.env`, `.netrc`, `.pem`, `.key`, `id_rsa`) are strictly excluded from scanner reads and caches.

Commands:
```bash
nova project info     # View project intelligence
nova project scan     # Force a re-scan and refresh cache
```

---

## CLI reference

Global options apply to every subcommand:

```
-C, --project-root PATH   Workspace root (default: current directory)
--model MODEL             Override the Groq model id
--timeout SECONDS         Per-command timeout
--safety {smart,strict,permissive}
--json                    Machine-readable output
-v, --verbose             Show full tool output
--no-color                Disable ANSI colours
-y, --yes                 Auto-approve actions that would otherwise prompt
```

| Command | Purpose |
|---|---|
| `nova ask "…"` | One-shot agent task with live progress |
| `nova chat` | Interactive REPL (`/exit`, `/new`, `/files`, `/help`) |
| `nova serve` | Start the mobile web IDE |
| `nova run <cmd>` | Run a shell command through the safety layer |
| `nova project` | Show or refresh Project Intelligence 2.0 (`info`, `scan`) |
| `nova summary` | What NovaCLI understands about the project |
| `nova tree [path]` | Print a project tree |
| `nova ls [path]` | List a directory |
| `nova read <path>` | Print a file (credential files refused) |
| `nova search <query>` | Search file contents (`-g` glob, `--regex`) |
| `nova config` | Effective configuration and status (`status`, `set-key <key>`) |
| `nova doctor` | Diagnose Python, dependencies, API key, workspace |
| `nova init [path]` | Create a `.env` for the project |
| `nova version` | Version and interpreter info |

---

---

## Interactive PTY Web Terminal

NovaCLI features a **Full Interactive PTY Web Terminal** with cross-platform support in the Web IDE.

### Supported Platforms

- **Linux** (Debian, Ubuntu, Fedora, etc.) — Unix PTY backend
- **macOS / Darwin** — Unix PTY backend  
- **WSL** (Windows Subsystem for Linux) — Unix PTY backend
- **Termux / Android** — Unix PTY backend
- **Windows** (Windows 10 build 17763+) — Windows ConPTY backend

### Architecture

NovaCLI supports two distinct command execution paths:

1. **Non-Interactive Execution (`POST /api/run`)**:
   - Executes single commands via `CommandRunner` with `LocalBackend` or `DockerBackend`.
   - Ideal for script runs (`pytest`, `git status`, `pip install`).

2. **Interactive PTY Terminal (`/ws/terminal`)**:
   - Connects browser to a real OS Pseudo-Terminal (PTY) over WebSocket using [xterm.js](https://xtermjs.org/).

   **On Linux, WSL, and Termux:**
   - Spawns an interactive shell (`/bin/bash`, `/usr/bin/bash`, or `/bin/sh`) attached to a POSIX PTY master/slave pair (`openpty`).
   - Supports interactive TTY applications: `python`, `nano`, `vim`, `bash`, `top`, `htop`, `less`, `watch`.
   - Interprets ANSI escape sequences so `clear` and terminal colors render cleanly.
   - Synchronizes browser terminal dimensions with the OS PTY via `TIOCSWINSZ` ioctl (`stty size`).

   **On Windows:**
   - Uses Windows ConPTY (Windows 10+ pseudo-console API) for true terminal emulation.
   - Spawns an interactive shell (default: `cmd.exe`, configurable via `NOVA_TERMINAL_SHELL`).
   - Supports interactive applications like `python`, `powershell`, batch scripts.
   - Synchronizes browser terminal dimensions with ConPTY via `ResizePseudoConsole` API.
   - Can use `powershell.exe` or `pwsh.exe` if configured: `NOVA_TERMINAL_SHELL=powershell.exe`.

### WebSocket Protocol

- **Endpoint**: `/ws/terminal`
- **Authentication**: Protected by NovaCLI's Web API token (`Authorization: Bearer <token>` or `X-Nova-Web-Token: <token>`).

**Client -> Server Messages:**
```json
{"type": "input", "data": "ls -la\n"}
{"type": "resize", "cols": 120, "rows": 30}
```

**Server -> Client Messages:**
```json
{"type": "output", "data": "..."}
{"type": "exit", "code": 0}
{"type": "error", "message": "..."}
```

### Mobile UI & Keyboards

Designed for mobile devices (phones/tablets and Termux/Android):
- Touch-friendly layout with mobile quick control bar (`Ctrl`, `Esc`, `Tab`, `↑`, `↓`, `←`, `→`).
- Auto-fits terminal dimensions to device screen or orientation changes.

### Diagnostics

Check your terminal backend:

```bash
nova doctor
```

Example output on Windows:
```
terminal PTY: Windows ConPTY
    shell: C:\Windows\System32\cmd.exe
```

Example output on Linux:
```
terminal PTY: Unix PTY
    shell: /bin/bash
```

> **Security Warning**:
> An interactive local PTY executes commands with the privileges of the NovaCLI host environment. It provides direct shell access to the workspace directory and should be treated as a local terminal, not an isolated security sandbox.

---

## Web IDE

```bash
nova serve                 # localhost only
nova serve --host 0.0.0.0  # reachable from your phone on the LAN
```

The IDE is a single page with a bottom tab bar:

| Tab | What it does |
|---|---|
| **💬 Chat** | Ask the agent to build, fix or explain. Streams thoughts, tool calls, results and progress bars live. |
| **📁 Files** | Browse the project, open a file, edit and save. Credential files never appear. |
| **▸_ Terminal** | Run commands with the same safety gates; risky commands raise a confirm dialog. |
| **📊 Project** | Language breakdown, file counts, entry points, detected commands, and Project Intelligence 2.0 details. |

---

## Safety — Smart Mode

Every action passes through `SafetyPolicy` before it happens. This module fails
closed: anything unrecognised needs approval, anything outside the workspace is
refused.

### Risk levels and modes

| Level | Examples | `smart` (default) | `strict` | `permissive` |
|---|---|---|---|---|
| **Safe** | `ls`, `cat`, `git status`, `pytest` | runs | runs | runs |
| **Moderate** | `rm`, `mv`, `pip install`, `curl`, `>` redirect | runs | **asks** | runs |
| **Dangerous** | writes to workspace files | **asks** | **asks** | runs |
| **Forbidden** | `rm -rf /`, `mkfs`, `dd of=/dev/…`, fork bombs, `sudo`, `curl … \| sh`, `git push --force`, `git reset --hard`, `shutdown` | **refused** | **refused** | **refused** |

Forbidden is forbidden in *every* mode. `--yes` / *Always allow* can never
override it.

---

## Testing

```bash
python -m pytest -q
```

Full unit, integration, and regression test suite covering native tool calling, global credential resolution, Project Intelligence 2.0, workspace safety, agent controls, and Web API. An optional integration test running against real Groq APIs (`openai/gpt-oss-20b`) is included in `tests/test_groq_integration.py` and runs automatically whenever `GROQ_API_KEY` is set in the environment.

---

---

## Execution Backends & Security Model

NovaCLI supports two command execution backends:

1. **Local Host Execution (`LocalBackend`)**:
   - Executes commands directly on the host machine.
   - Guarded by `SafetyPolicy` path and command checks, process-group timeouts, environment secret scrubbing, and interactive human approval for risky operations.

2. **Docker Isolation (`DockerBackend`)**:
   - Optional sandboxed execution inside a Docker container.
   - Applies container safety flags: `--cap-drop=ALL`, `--security-opt=no-new-privileges`, memory/CPU limits (`--memory 512m`, `--cpus 1.5`), `--pids-limit 256`, and isolated `/tmp` tmpfs.
   - Automatically terminates hanging containers on command timeout to prevent orphaned processes.

## Agent Checkpoints, Diff Review & Undo

NovaCLI automatically captures recovery checkpoints before/during Agent tasks:
- **Checkpoints**: Stored under `.nova/checkpoints/` (ignored by Git). Snapshots text files without copying sensitive credentials (`.env`, keys).
- **Inviolable User Ownership Protection**: Pre-existing user uncommitted changes are tracked and strictly preserved during rollbacks. `confirm` authorizes the rollback operation but never overwrites or deletes pre-existing user-owned files.
- **Snapshot Symlink & Storage-Root Hardening**: Checkpoint snapshot creation rejects `.nova` or `.nova/checkpoints` symlink/junction/reparse-point redirects, skips file symlinks, enforces canonical workspace containment before reads, and strictly validates manifest schema and field types.
- **Session & Workspace Context Enforcement**: Checkpoint operations verify session ID and workspace root context to prevent cross-session or cross-workspace checkpoint tampering.
- **Centralized Execution**: All Git operations in CheckpointManager use centralized `GitService` and `CommandRunner.run_args()` without direct `subprocess` calls or shell invocation.
- **Diff Review**: Inspect changed files, added/removed lines, and unified diffs before accepting or undoing changes.
- **Undo / Rollback**: Safely restores Agent-modified files and removes Agent-created files without destroying user-owned work.
- **Git Workflow**: Structured Git operations (`status`, `diff`, `commit_preview`) executed safely via `CommandRunner.run_args()`.

### Developer API & Command Execution Security

All developer APIs and Web IDE endpoints enforce strict security boundaries:
- **Project Test Execution (`POST /api/tests/run`)**: Commands detected from project metadata (`package.json`, `Makefile`, etc.) are treated as untrusted code execution targets and strictly evaluated through `SafetyPolicy`. High-risk or forbidden commands are blocked or require explicit approval.
- **Git Path Jailing (`GET /api/git/diff`)**: File paths passed to Git APIs are validated against the workspace root (`Workspace.resolve`) and safely escaped (`shlex.quote`) to prevent path traversal or shell command injection.
- **Workspace Path Traversal Protection**: File operations (`/api/file`, `/api/tree`, `/api/files`, `/api/search`, `/api/file/rename`, `/api/file/mkdir`) reject escape attempts (`../`, `..\`, absolute paths, Windows drive paths, UNC paths, and symlinks pointing outside the workspace).
- **Interactive Terminal**: Authenticated WebSocket terminal connections provide direct interactive shell access guarded by `NOVA_WEB_TOKEN` authentication.

> **Security Threat Model & Limitations**:
> `SafetyPolicy` provides application-level policy enforcement and credential filtering. It is not an unbreakable OS sandbox.
> Optional Docker isolation adds container-level confinement, but workspace files remain mounted for tool usability.

---

## License

MIT

---

## Provider & Model Switching

NovaCLI supports 5 AI model providers:

1. **Groq** (`groq`) — Default model `openai/gpt-oss-20b` (Requires `GROQ_API_KEY`)
2. **Gemini** (`gemini`) — Default model `gemini-2.5-flash` (Requires `GEMINI_API_KEY`)
3. **Ollama** (`ollama`) — Default model `qwen3:4b` at `http://localhost:11434` (No API key required)
4. **OpenRouter** (`openrouter`) — Default model `openai/gpt-oss-20b` (Requires `OPENROUTER_API_KEY`)
5. **Cerebras** (`cerebras`) — Default model `llama3.1-8b` (Requires `CEREBRAS_API_KEY`)

### Configuring Provider & Model via CLI

```bash
# Switch active provider
nova config provider groq
nova config provider ollama
nova config provider gemini
nova config provider openrouter
nova config provider cerebras

# Set model for active provider
nova config model openai/gpt-oss-20b
nova config model qwen3:8b
nova config model gemini-2.5-flash

# View current configuration and available providers
nova config
```

### Provider Environment Variables

- `NOVA_PROVIDER`: Active provider (`groq`, `gemini`, `ollama`, `openrouter`, `cerebras`)
- `NOVA_MODEL`: Active model override
- `GROQ_API_KEY` / `GROQ_MODEL`
- `GEMINI_API_KEY` / `GEMINI_MODEL`
- `OLLAMA_MODEL` / `OLLAMA_BASE_URL`
- `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`
- `CEREBRAS_API_KEY` / `CEREBRAS_MODEL`
