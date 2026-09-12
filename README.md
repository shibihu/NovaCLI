# ⚡ NovaCLI v1.0

**An AI-powered developer environment** — an autonomous coding agent, a command-line
interface, and a mobile-first web IDE, all driven by one shared core.

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
- [Configuration](#configuration)
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
| **AI coding agent** | A reasoning loop that plans, calls tools, observes results and reports every step. |
| **Mobile-first Web IDE** | Four tabs — Chat, Files, Terminal, Project — with 44px touch targets, safe-area insets and a dark theme. No build step. |
| **CLI** | `nova ask`, `nova chat`, `nova serve` plus workspace commands for files, search and project understanding. |
| **Project understanding** | Language detection, entry points, test discovery, inferred build/test commands, README excerpts, and relevance-ranked file context. |
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
# 1. Create a .env for this project
python nova.py init

# 2. Paste your key into .env
#    GROQ_API_KEY=gsk_...

# 3. Check everything is wired up
python nova.py doctor

# 4. Ask something
python nova.py ask "what does this project do?"

# 5. Or open the mobile IDE
python nova.py serve --host 0.0.0.0
```

Then browse to `http://<your-device-ip>:8000` on your phone.

---

## Configuration

### API key priority

The key is resolved in this order — first match wins:

1. **Environment variable** — `export GROQ_API_KEY=gsk_...`
2. **`.env`** in the project root — `GROQ_API_KEY=gsk_...` (recommended)
3. **`~/.nova/config.json`** — `{"groq_api_key": "gsk_..."}`

If no key is found, NovaCLI prints an explanation of exactly these three options
and exits with code `3` — it never fails with a bare traceback.

### Settings

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `groq_api_key` | `GROQ_API_KEY` | — | Provider credential |
| `groq_model` | `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `project_root` | `NOVA_PROJECT_ROOT` | current directory | Workspace root the agent may touch |
| `command_timeout` | `NOVA_COMMAND_TIMEOUT` | `30` | Seconds before a command is killed |
| `max_steps` | `NOVA_MAX_STEPS` | `8` | Agent reasoning steps per task |
| `safety_mode` | `NOVA_SAFETY_MODE` | `smart` | `smart` \| `strict` \| `permissive` |
| `host` / `port` | `NOVA_HOST` / `NOVA_PORT` | `127.0.0.1` / `8000` | Web IDE bind address |

### Secret handling

NovaCLI guarantees, and the test suite asserts, that the API key:

- is **never** sent to the browser (`/api/config` returns only `has_api_key` and a masked preview),
- is **never** printed in logs, errors or tracebacks (`Settings.__repr__` is redacted; provider errors are sanitised),
- is **never** placed in the project context sent to the model,
- is **removed from the environment** of every child process it spawns,
- is **redacted** from all tool output alongside generic patterns for Groq, OpenAI, Anthropic, Google, GitHub, Slack and AWS keys, PEM private keys, `Bearer` tokens and `key=value` credentials.

`.env` is git-ignored; `.env.example` is the committed template.

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
| `nova summary` | What NovaCLI understands about the project |
| `nova tree [path]` | Print a project tree |
| `nova ls [path]` | List a directory |
| `nova read <path>` | Print a file (credential files refused) |
| `nova search <query>` | Search file contents (`-g` glob, `--regex`) |
| `nova config` | Effective configuration, secrets masked |
| `nova doctor` | Diagnose Python, dependencies, API key, workspace |
| `nova init [path]` | Create a `.env` for the project |
| `nova version` | Version and interpreter info |

### Examples

```bash
# Understand an unfamiliar repo
python nova.py summary
python nova.py ask "explain the request flow end to end"

# Make a change and have the agent verify it
python nova.py ask "add tests for calculator.py and run them" --yes

# Interactive session with a persistent conversation
python nova.py chat

# Inspect before you let it act
python nova.py search "def handle_" -g "**/*.py"
python nova.py ask "refactor handle_ into smaller functions"      # will ask before writing

# Script it
python nova.py ask "list every TODO" --json | jq -r .answer
```

### `nova run` and flags

`nova run` uses a remainder argument, so flags after the command belong to the
command:

```bash
python nova.py run -C ~/app pytest -q        # -C goes to NovaCLI
python nova.py run -C ~/app ls -1            # -1 goes to ls
```

---

## Web IDE

```bash
python nova.py serve                 # localhost only
python nova.py serve --host 0.0.0.0  # reachable from your phone on the LAN
```

The IDE is a single page with a bottom tab bar:

| Tab | What it does |
|---|---|
| **💬 Chat** | Ask the agent to build, fix or explain. Streams thoughts, tool calls, results and progress bars live. |
| **📁 Files** | Browse the project, open a file, edit and save. Credential files never appear. |
| **▸_ Terminal** | Run commands with the same safety gates; risky commands raise a confirm dialog. |
| **📊 Project** | Language breakdown, file counts, entry points, detected commands, and a project tree. |

**Mobile-friendly by design:**

- `viewport-fit=cover` plus `env(safe-area-inset-*)` so nothing hides behind the
  status bar or gesture pill.
- `--tap: 44px` minimum touch target on every button.
- `dvh` units so the composer stays visible when the keyboard opens.
- `overflow-x: hidden` — no horizontal scrolling at any width.
- Dark theme by default, light theme via `prefers-color-scheme`.
- Honours `prefers-reduced-motion`.
- Zero build step: plain HTML, CSS and JS served straight from disk.

**Agent controls in the UI:**

- **Stop** — cancels a running agent immediately (the same controller the CLI's
  `Ctrl+C` uses).
- **Approval dialog** — when the agent wants to write a file or run a risky
  command, you get *Approve* / *Always allow* / *Deny*. "Always allow" is
  remembered per tool for that session. If you never answer, it times out and
  fails closed.

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

### Path jailing

All file access goes through `Workspace.resolve()`, which:

- resolves relative paths against the workspace root and refuses anything outside it,
- refuses credential files in both directions — `.env`, `id_rsa`, `*.pem`, `*.key`, `credentials.json`, `.git/config`, `.ssh/`, `.aws/`, `.nova/config.json`,
- refuses writes into `.git/` and `.nova/`,
- allows `.env.example` and friends, which are meant to be read.

### The shell is not a side door

Because a shell can reach any file, command strings are also scanned for
credential targets and for reads of NovaCLI's own environment:

```
cat .env                 → refused (sensitive-target)
echo $GROQ_API_KEY       → refused
cat .env.example         → allowed (template)
cat notes.env.md         → allowed (not a credential file)
```

Child processes additionally run with `GROQ_API_KEY` scrubbed from their
environment, so even a successful read returns nothing.

### Execution hardening

- Every command runs with its own **process group**; on timeout the whole tree is
  `SIGTERM`-ed, then `SIGKILL`-ed after a grace period. No orphaned `sleep 300`.
- Output is capped (200 KB default, head+tail preserved) so a runaway command
  cannot exhaust memory or the context window.
- The working directory is clamped to the workspace root.

---

## Architecture

```
nova/
├── config.py              Settings, priority resolution, secret masking
├── core/
│   ├── models.py          Shared dataclasses, enums, event vocabulary
│   ├── safety.py          Smart Mode: risk rules, path jail, redaction
│   ├── runner.py          Async, timeout-bounded, env-scrubbed execution
│   ├── context.py         Project brief assembly + relevance ranking
│   └── agent.py           The reasoning loop, tools, and controls
├── ai/
│   ├── __init__.py        AIProvider protocol + factory
│   └── groq.py            AsyncGroq implementation
├── workspace/
│   ├── files.py           Jailed read/write/list/tree/search
│   └── projects.py        Language, entry point, test and command detection
├── cli/commands.py        argparse app — rendering only
└── web/
    ├── app.py             FastAPI factory
    ├── routes.py          HTTP + SSE endpoints
    ├── events.py          Session registry and event bus
    ├── templates/         index.html
    └── static/            app.css, app.js
```

### The agent loop

The model speaks one strict JSON protocol per turn:

```json
{"thought": "why", "action": "read_file", "action_input": {"path": "nova/config.py"}}
{"thought": "why", "final_answer": "the answer"}
```

Using text instead of vendor tool-calling APIs is deliberate: it keeps
`nova.ai.groq` swappable for any backend that can return text.

Each step:

1. **Check cancellation** — stop immediately if the user asked.
2. **Call the model** — provider errors end the run with an explanation.
3. **Parse** — fenced JSON, prose, and alias keys (`tool`, `args`, `answer`) are all
   tolerated; a reply naming neither an action nor an answer gets one nudge and a retry.
4. **Safety gate** — forbidden actions are refused and reported back to the model so it
   changes approach; approvable ones raise an approval event.
5. **Execute** the tool and redact its output.
6. **Observe** — the result is fed back and the loop continues.

The loop always ends with exactly one terminal event: `final`, `error` or
`cancelled`.

### Tools

| Tool | Arguments | Notes |
|---|---|---|
| `read_file` | `{"path": "..."}` | Binary and credential files refused |
| `write_file` | `{"path": "...", "content": "..."}` | Creates parent dirs; needs approval |
| `list_files` | `{"path": ".", "depth": 2}` | Depth 1 lists, deeper renders a tree |
| `search` | `{"query": "...", "glob": "**/*.py"}` | Substring or regex, capped results |
| `run_command` | `{"command": "pytest -q"}` | Full safety gate + timeout |
| `project_summary` | `{}` | Languages, entry points, commands |

### Event stream

`AgentEvent` is the single contract between core and both front ends:

| Event | Meaning |
|---|---|
| `agent_start` | Run began; includes model, workspace, safety mode |
| `step_start` | A reasoning step began |
| `thought` | The model's stated reasoning |
| `tool_call` | A tool is about to run |
| `approval_request` | Waiting on the user; includes a request id and a diff/command preview |
| `approval_resolved` | The user answered |
| `tool_result` | Outcome, duration and output |
| `blocked` | Refused by policy or denied by the user |
| `progress` | Percentage complete, for the progress bar |
| `final` / `error` / `cancelled` | Terminal |

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The IDE shell |
| `GET` | `/api/health` | Liveness, model, `has_api_key` |
| `GET` | `/api/config` | Effective config — **no secrets** |
| `GET` | `/api/project` | Summary, skills and tree |
| `GET` | `/api/tree?path=&depth=` | Subtree |
| `GET` | `/api/files?path=` | Directory listing |
| `GET` | `/api/file?path=` | File contents (403 for credential files) |
| `PUT` | `/api/file` | Write a file |
| `POST` | `/api/run` | Run a command; returns `requires_approval` rather than failing |
| `POST` | `/api/agent` | Start a session → `{session_id}` |
| `GET` | `/api/agent/stream?session_id=` | **Server-Sent Events** stream |
| `GET` | `/api/agent/result?session_id=` | Poll state (for clients without SSE) |
| `GET` | `/api/agent/sessions` | Recent sessions |
| `POST` | `/api/agent/approve` | Approve or deny a pending request |
| `POST` | `/api/agent/cancel` | Cancel a running session |

Interactive docs at `/api/docs`.

```bash
curl -s localhost:8000/api/agent -H 'content-type: application/json' \
  -d '{"task":"what does this project do?"}'
# {"session_id":"ab12…","status":"pending"}

curl -N localhost:8000/api/agent/stream?session_id=ab12…
# data: {"type":"agent_start","data":{…}}
# data: {"type":"tool_call","data":{"tool":"read_file",…}}
# data: {"type":"final","data":{"answer":"…"}}
```

---

## Testing

```bash
python -m pytest -q
```

**470 tests** across 12 modules plus shared fixtures:

| Module | Covers |
|---|---|
| `test_config.py` | `.env` parsing, key priority, defaults, clamping, secret masking |
| `test_groq.py` | Provider success and every failure mode, via a fake SDK client |
| `test_workspace.py` | Path jailing, read/write, ignore rules, tree, search, project analysis |
| `test_context.py` | Ranking, assembly, truncation, redaction, `.env` never read |
| `test_safety.py` | Every risk rule, mode behaviour, credential paths, redaction |
| `test_runner.py` | Stdout/stderr, timeouts, process-group kill, env scrubbing, output caps |
| `test_agent.py` | JSON parsing, the loop, all tools, step limits, redaction |
| `test_cli.py` | Every subcommand, exit codes, masked secrets, `--json` |
| `test_web.py` | All endpoints, SSE, approvals, validation, error codes |
| `test_web_ui.py` | Viewport, safe-area, touch targets, tabs, dialog, client behaviour |
| `test_agent_controls.py` | Cancel, approve, deny, always, timeouts, session registry |
| `test_e2e.py` | Full read→write→verify runs through the agent, CLI and HTTP |

No test requires network access or a real API key — the provider is always faked.

### What the tests genuinely caught

The suite is not decoration; it found three real defects during development:

1. **A shell-level jail bypass** — `cat .env` sails past path checks because the
   shell reads the file, not the `Workspace`. Fixed by scanning command text for
   credential targets.
2. **A `dest` collision** — the `run` subcommand's positional shadowed the
   subparser's `command` dest, so `nova run -C path` silently printed help.
3. **Approval semantics inverted** — `allowed` and `requires_approval` were
   conflated, so a strict-mode command could not run even after approval.

---

## Termux / Android notes

- **Use plain `uvicorn`, not `uvicorn[standard]`.** The standard extra pulls in
  `uvloop` and `httptools`, which need a C toolchain and routinely fail to build on
  Android. Plain uvicorn uses the pure-Python asyncio loop, which is ample for a
  single-user phone IDE.
- Every dependency is pure Python, so `pip install` needs no compiler.
- Keep the checkout under `~/` (Termux home). `/sdcard` is `noexec`, which breaks
  virtualenvs and compiled wheels.
- ANSI colours switch off automatically when `TERM=dumb`, `NO_COLOR` is set, or
  stdout is not a TTY — so the CLI stays readable in Termux's default terminal.
- To open the IDE from another device, bind to all interfaces
  (`--host 0.0.0.0`) and visit `http://<phone-ip>:8000`. NovaCLI has no
  authentication; only do this on a network you trust.

---

## Troubleshooting

Run `python nova.py doctor` first — it checks Python, dependencies, the API key,
workspace writability and shell availability.

| Symptom | Fix |
|---|---|
| `No Groq API key configured` | `python nova.py init`, then set `GROQ_API_KEY` in `.env`. |
| `Groq rejected the API key` | Regenerate at <https://console.groq.com/keys>. |
| `Groq rejected model …` | Set `GROQ_MODEL` to a supported id. |
| `Refused: path is outside the workspace root` | Use `-C <path>` (or `NOVA_PROJECT_ROOT`) to widen the workspace. |
| Command `timed out` | Raise `--timeout`. |
| `Address already in use` | `python nova.py serve --port 8001`. |
| `module 'groq' missing` | `python -m pip install -r requirements.txt`. |
| Nothing happens on `nova chat` | The agent needs a key; the REPL itself works offline. |

---

## Extending

### Add a provider

Implement four members, then register it:

```python
class MyProvider:
    model_name = "my-model"
    configured = True

    async def complete(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        return "..."

    async def aclose(self) -> None:
        ...
```

Add it to `provider_names()` and `get_provider()` in `nova/ai/__init__.py`. Nothing
in `nova/core` changes.

### Add a tool

1. Append a `ToolSpec` to `TOOL_SPECS` in `nova/core/agent.py` (this is what the
   model is told about).
2. Implement `_tool_<name>(self, args: dict) -> ToolOutcome` on `ToolBox`
   (async is fine).
3. If it needs a permission gate, handle it in `SafetyPolicy.check_tool_call`.

The agent loop needs no edits — it validates tools against `TOOL_NAMES`.

---

## License

MIT
