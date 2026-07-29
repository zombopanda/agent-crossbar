# Agent Crossbar

[![npm version](https://img.shields.io/npm/v/agent-crossbar)](https://www.npmjs.com/package/agent-crossbar)
[![PyPI version](https://img.shields.io/pypi/v/agent-crossbar)](https://pypi.org/project/agent-crossbar/)

Delegate review, advice, text, and dev work to local coding agents — Codex, Claude, OpenCode — through a single MCP server. One `agent_start` call, one `job_result` answer.

**Experimental developer preview (v0.3).** APIs may change. Provider guarantees are qualified by live gates.

Expose to MCP clients with the server key `agents`.

## Ten-Minute Quickstart

### 1. Install

```bash
# Canonical: uvx pulls the latest PyPI release
uvx agent-crossbar

# Or via npm (thin launcher → delegates to uvx)
npx agent-crossbar
```

Prerequisites: uv is required for both launch paths because the npm package is
only a thin launcher around `uvx`; install it from the
[uv documentation](https://docs.astral.sh/uv/getting-started/installation/).
The npm path additionally requires
[Node.js](https://nodejs.org/) ≥ 20.

### 2. Check Readiness (doctor)

```bash
uvx agent-crossbar doctor

# Optional: check one provider and emit machine-readable output
uvx agent-crossbar doctor --profile codex --json
```

Verifies that supported provider CLIs are installed, authenticated, and runnable. A provider must be `ready` before jobs can be created.

### 3. Configure Your MCP Client

#### Codex

For a user-wide installation shared by the Codex app, CLI, and IDE extension:

```bash
codex mcp add agents -- uvx agent-crossbar
codex mcp list
```

This writes the native Codex MCP configuration to `~/.codex/config.toml`.
The equivalent explicit TOML is:

```toml
[mcp_servers.agents]
command = "uvx"
args = ["agent-crossbar"]
```

For a trusted-project-only installation, put the same TOML table in
`.codex/config.toml` inside that repository. Codex does **not** use
Claude Code's `.mcp.json` format.

#### Claude Code

Claude Code uses the native `claude_bg` backend (`claude` profile). Interactive follow-ups attach a harness-owned terminal to the same background session; print mode remains disabled because `claude -p` uses separate Agent SDK credit/metered billing — read [Claude Billing](#claude-subscription-vs-print-sdk-billing) below.

For a user-wide installation:

```bash
claude mcp add --scope user agents -- uvx agent-crossbar
claude mcp get agents
```

Use `--scope project` instead to create a shareable project-root `.mcp.json`,
or omit `--scope` for Claude Code's private local-project scope.

**Claude prerequisite**: authenticate with `claude auth login`. The doctor will report `needs_auth` until you do.

#### OpenCode

Add this to the global `~/.config/opencode/opencode.json` or to a project-root
`opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "agents": {
      "type": "local",
      "command": ["uvx", "agent-crossbar"],
      "enabled": true
    }
  }
}
```

Then verify it with `opencode mcp list`.

### 4. First Review Flow

With the MCP server running, from any MCP client:

```
1. profiles_list                     → see available profiles and their tiers
2. profile_health                    → verify readiness before creating jobs
3. agent_start(
     profile="codex",
     model="gpt-5.6-sol",
     prompt="Review my uncommitted changes for security issues.",
     task="review"
   )                                 → creates a review job
4. job_tail(job_id="<id>")           → stream real-time output
5. job_result(job_id="<id>")         → get final structured result
```

## Tools (8)

| # | Tool | Description |
|---|------|-------------|
| 1 | `agent_start` | Start an agent task (ask, review, or dev) in one call |
| 2 | `profiles_list` | List available agent profiles with support tiers and capabilities |
| 3 | `profile_health` | Run live readiness probes for all configured profiles |
| 4 | `job_tail` | Stream incremental job output by sequence number |
| 5 | `job_result` | Get final structured result, exit code, and summary |
| 6 | `job_send` | Send follow-up input to a running interactive job (not available for Claude bg) |
| 7 | `job_stop` | Stop a running job gracefully |
| 8 | `job_list` | List jobs scoped to the current client session |

**Exact 8-tool MCP surface.** No hidden tools, no deprecated aliases.

### Session Isolation

By default, all job-access tools (`job_tail`, `job_result`, `job_stop`,
`job_send`, `job_list`) are scoped to the requesting `client_session_id`.
A client session can only see and operate on its own jobs; foreign jobs
return `"error": "job_not_found"` with a `cross_session_note` hint:

```json
{
  "ok": false,
  "error": "job_not_found",
  "cross_session_note": "pass client_session_id=\"*\" for explicit local cross-session access"
}
```

#### Explicit Local Cross-Session Access

Pass `client_session_id="*"` to `job_tail`, `job_result`, `job_stop`,
`job_send`, or `job_list` for explicit local cross-session access. This
permits any local client to see and operate on all jobs regardless of the
owning session. No environment variable or token setup is required —
`"*"` is the literal opt-in string.

## Support Matrix

### Supported Profiles

| Profile | Tasks | Backend | OS | Model selection |
|---------|-------|---------|-----|-----------------|
| `codex` | ask, review, dev | ACP one-shot (including explicit effort) | macOS, Linux | Required on every call |
| `claude` | ask, review, dev | Native `claude_bg`; interactive follow-ups via `job_send` (no `claude -p`) | macOS, Linux | Required on every call |
| `opencode` | ask, review, dev | ACP one-shot (including explicit effort) | macOS, Linux | Required on every call |

`model` is mandatory for every `agent_start` request. Agent Crossbar never
chooses or falls back to a default model. Use `profiles_list` to inspect the
currently available model IDs before starting a job.

### Experimental (Installed, Not Guaranteed)

| Profile | Tasks | Interactive | Notes |
|---------|-------|-------------|-------|
| `reasonix` | ask, review, dev | both | Supports noninteractive and interactive modes; results use heuristic TUI parsing |
| `chatgpt_pro` | ask, review | false | Experimental macOS browser adapter; requires an open, signed-in ChatGPT window in Helium, Chrome, or Safari plus `cua-driver` |

### Provider Prerequisites

| Provider | Binary | Auth Check |
|----------|--------|-----------|
| Codex | `codex` CLI + `pnpm` | `codex login status` |
| Claude | `claude` CLI | `claude auth status --json` |
| OpenCode | `opencode` CLI | `opencode auth list` |
| Reasonix | `reasonix` CLI | `reasonix doctor --json` |

## Claude Subscription vs Print SDK Billing

Agent Crossbar uses Claude's native `claude --bg` subscription path. This uses your ordinary Claude plan — no separate API billing.

- `claude -p` (print/SDK mode) is **disabled** — it uses separate Agent SDK metered billing
- With `interactive: true`, Agent Crossbar opens a harness-owned `claude attach <session-id>` tmux session; `job_send` writes the next turn into that same native session
- Profile `claude` maps to `claude_bg` for one-shot calls and `claude_bg_pty` internally for attach-backed interactive calls
- Readiness is validated via `claude auth status --json` before job creation

## Timeouts

| Layer | Default | Notes |
|-------|---------|-------|
| External MCP read timeout | Client-dependent | Set in your MCP client. A client-side timeout does **not** cancel the durable background job — it continues executing and results remain available via `job_tail`/`job_result`. |
| Internal preflight probe | Profile-dependent | Sequential read-only checks are individually bounded: up to 35s for Codex, 25s for OpenCode, 15s for Claude, and 30s for Reasonix. Results are cached for 60s. A failure blocks job creation before a job is written. |
| ACP startup and model selection | 30s | `initialize`, `session/new`, and explicit model selection are separately bounded. Provider quota/rate-limit diagnostics terminate the job immediately when detected. |
| `max_runtime_sec` (agent_start) | 1800s (30 min) | Server-side job deadline, configurable per job. When exceeded, the job terminates with a terminal `timeout` result. |
| `job_tail` / `job_result` | — | Available any time after the initial `agent_start` response. No deadline is enforced on result polling. |

The `doctor` CLI reports readiness and preflight failures only. It does **not** report active job deadlines or running-job state.

## Local State and Retention

- **State directory**: `~/.local/state/agent-crossbar` (override with `AGENT_CROSSBAR_STATE_DIR`)
- **Job storage**: one directory per job under `jobs/`
- **Retention**: no automatic cleanup in v0.3 — jobs persist until manually deleted — jobs persist until manually deleted
- **Local audit logs**: full MCP request and response payloads, including
  prompts and results, are written under `telemetry/` with owner-only
  permissions. They follow the same no-cleanup policy in v0.3.
- **No remote telemetry**: these audit logs are not sent remotely; Agent
  Crossbar does not phone home.

## Troubleshooting by Error Code

| Error Code | Meaning | Action |
|-----------|---------|--------|
| `codex_missing` | Codex CLI not on PATH | Install the Codex CLI |
| `codex_not_authenticated` | Not logged into Codex | Run `codex login` |
| `pnpm_missing` | pnpm not installed | Install pnpm (https://pnpm.io/installation) |
| `claude_missing` | Claude CLI not on PATH | Install Claude Code |
| `not_authenticated` | Claude not logged in | Run `claude auth login` |
| `opencode_missing` | OpenCode CLI not on PATH | Install the OpenCode CLI |
| `reasonix_missing` | Reasonix CLI not on PATH | Install the Reasonix CLI |
| `unsupported_os` | Provider requires different OS | Use a supported OS or different provider |
| `chatgpt_pro_cua_driver_missing` | `cua-driver` is not installed or not on PATH | Install `cua-driver` and grant it accessibility permission |
| `chatgpt_pro_browser_not_running` | No supported ChatGPT browser is running | Open https://chatgpt.com in Helium, Chrome, or Safari |
| `chatgpt_pro_window_not_found` | A supported browser runs but has no ChatGPT window | Open a ChatGPT tab in that browser on the current desktop |
| `chatgpt_pro_not_authenticated` | A ChatGPT window is open but signed out | Sign in with a ChatGPT Pro account |
| `chatgpt_pro_readiness_unverified` | The ChatGPT window exists but its composer could not be read | Bring the window onto the current desktop and retry |
| `chatgpt_pro_browser_probe_failed` | `cua-driver` could not inspect the desktop | Check its accessibility permissions, then retry |
| `model_not_available` | The requested ChatGPT model is not offered by the visible picker | Pick a model shown in the ChatGPT UI (see `diagnostics.selection.available_choices`) |
| `effort_not_available` | The requested effort is not offered by the visible picker | Omit `effort` or pick one the UI exposes |
| `composer_not_empty` | An unrelated ChatGPT draft is open | Clear that draft; Agent Crossbar never overwrites it |
| `prompt_verification_failed` | The composer did not contain the exact prompt before Send | Retry; the prompt was never submitted |
| `session_mismatch` / `session_window_unavailable` | The owned ChatGPT window changed, closed, or became ambiguous | Retry without moving or closing that window mid-turn |
| `generation_status_unavailable` | The prompt was submitted but its response could not be read safely | Check the ChatGPT window; the turn is never retried in another browser |
| `generation_timed_out` | The prompt was submitted but did not finish within `max_runtime_sec` | Increase `max_runtime_sec` or shorten the request |
| `cancelled` | `job_stop` cancelled the turn | Check `provider_stop_confirmed` in the `cancelled` event for whether the visible Stop action was clicked |
| `context_path_missing` / `context_path_symlink` / `context_path_outside_cwd` | A `scope` path is missing, symlinked, or escapes `cwd` | Pass explicit, real paths inside `cwd` |
| `attachment_missing` / `attachment_too_large` / `attachment_symlink` | A `scope.attachments` entry is missing, oversized, or symlinked | Pass a real file inside `cwd` under the size budget |
| `missing_model` | `agent_start` omitted or passed an empty `model` | Call `profiles_list`, choose a model, and pass it explicitly |
| `provider_limit_exhausted` | Provider quota, credits, or rate limit is exhausted | Wait for reset or choose another explicitly available model |
| `provider_unavailable` | No backend is currently available for the selected model | Choose another model or retry after the provider recovers |
| `acp_launch_error` | ACP agent process failed to launch (binary missing, dependency error) | Check provider CLI installation, run `agent-crossbar doctor` |
| `acp_protocol_error` | ACP protocol handshake or message error (version mismatch, invalid request) | Check provider and protocol logs; provider CLI may need upgrade |
| `acp_timeout` | ACP job exceeded `max_runtime_sec` while awaiting an already-delivered prompt's response | Follow `failure.next_action`: normally increase `max_runtime_sec`; for OpenCode, `check_provider_limits_or_retry_with_free_model` |
| `acp_prompt_delivery_timeout` | ACP startup did not finish within its bounded startup window, before the prompt was dispatched | Check provider availability, quota, CLI installation, and selected model |

`job_stop` is idempotent. ACP jobs persist a terminal result even when the
provider process has already exited; running ACP child processes receive
SIGTERM and then SIGKILL after a bounded grace period when necessary.
GUI (browser) jobs are marked terminal first, then their registered run handle
is cancelled: the worker stops polling, clicks ChatGPT's visible Stop action
when one is exposed, retires the browser session, and records whether the
provider stop was actually confirmed. A late provider completion can never
overwrite a stopped result.

Stable error codes are guaranteed across patch versions. The `next_action` field in job results provides exact remediation.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_CROSSBAR_STATE_DIR` | `~/.local/state/agent-crossbar` | State root directory |
| `AGENT_CROSSBAR_CLIENT_NAME` | `agent-crossbar` | Client name in telemetry |
| `AGENT_CROSSBAR_CLIENT_VERSION` | `unknown` | Optional client version recorded in local audit logs |
| `AGENT_CROSSBAR_DEFAULT_CWD` | `PWD` | Default working directory for dev jobs |

**Migration note**: The old `AGENT_HARNESS_*` env var names still work but emit a `FutureWarning`. Rename them to `AGENT_CROSSBAR_*`. The compat shim will be removed in v0.4.0.

## Architecture

```
MCP Client (Codex / Claude / OpenCode)
        │
        ▼
  FastMCP("agents")  ← 8-tool MCP surface
        │
   ┌────┼────┐
   ▼    ▼    ▼
  Codex Claude OpenCode  ← provider adapters
   │    │     │
   ▼    ▼     ▼
  ACP / claude_bg / tmux / GUI  ← provider backends
```

One Python package (`agent-crossbar` on PyPI). Bounded provider adapters under `agent_crossbar.adapters`. No separate plugin packages in v0.3.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick rules for contributors in [AGENTS.md](AGENTS.md).

## License

MIT — see [LICENSE](LICENSE).
