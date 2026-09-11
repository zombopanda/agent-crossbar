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
| 6 | `job_send` | Send follow-up input, or a structured owner decision, to a running interactive job |
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
| `claude` | ask, review, dev | Native `claude_bg`; **interactive-only** — non-interactive `agent_start` is rejected with `interactive_required` | macOS, Linux | Required on every call |
| `opencode` | ask, review, dev | ACP one-shot (including explicit effort); owner permission decisions held through `job_send` | macOS, Linux | Required on every call |

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
- Claude is **interactive-only**: `agent_start(profile="claude")` rejects an explicit or defaulted `interactive=false` with `interactive_required` before readiness, admission, lease, or job creation
- With `interactive: true`, Agent Crossbar opens a harness-owned `claude attach <session-id>` tmux session; `job_send` writes the next turn into that same native session
- The background monitor stays alive across `awaiting_input` under one monotonic `max_runtime_sec` deadline, so a `job_send` reply is observed without restarting the monitor or resetting the deadline
- When the native session pauses mid-turn, the `awaiting_input` detail carries the full question text recovered from the tmux transcript (redacted, not cut to the 2 KiB diagnostics prefix) alongside the native `waiting_for` label
- Profile `claude` maps to `claude_bg` for one-shot calls and `claude_bg_pty` internally for attach-backed interactive calls
- Readiness is validated via `claude auth status --json` before job creation

## Owner-Mediated Permissions and Continuation

Two provider pause points are surfaced to the owner through the existing
`job_send` tool — no new tool and no new `agent_start` field.

**Claude questions.** When a Claude interactive turn pauses mid-turn, the job
becomes `awaiting_input` and `job_tail` reports the native `waiting_for` label
plus a `question` field carrying the full, redacted question text. Reply with
plain text via `job_send`; the same monitor thread observes the resume on its
next poll.

**ACP permissions (OpenCode/DeepSeek).** A `session/request_permission` call
that is not resolved by the existing bounded local-edit auto-allow policy is
held pending instead of being auto-rejected. The job becomes `awaiting_input`
with a durable `pending_request` in `job_tail`:

```json
{
  "pending_request": {
    "request_id": "perm-1a2b3c4d5e6f",
    "kind": "other",
    "decisions": ["allow", "reject"],
    "command": "rm -rf /tmp/x"
  }
}
```

`request_id`, `kind`, bounded tool/command/path(s), and generic `allow`/`reject`
choices are exposed; raw ACP option ids are never surfaced, and no decision
can escalate to `allow_always`. Resolve it with a JSON object as `job_send`'s
`text`:

```json
{"request_id": "perm-1a2b3c4d5e6f", "decision": "allow"}
```

Any `text` that is not a JSON object carrying string `request_id` and
`decision` keeps its exact plain-text meaning. A decision for a stale,
unknown, or already-resolved request is rejected with `request_not_pending`
(never reinterpreted as plain text). Resolution enforces the same
owner/`client_session_id` policy as any other `job_send`, including the `"*"`
wildcard. Pending requests settle (expire/cancel) in the same operation that
terminalizes a job by `job_stop` or deadline, so a late decision can never
resurrect a terminal job; a durable pending request with no live provider
callback fails closed and cleans up.

OpenCode does not support free-text interactive continuation (`interactive` is
false), but its ACP permission flow advertises `owner_permission_decisions`
separately in `profiles_list`; structured decisions still use `job_send` while
the ACP connection remains alive.

**ACP completion honesty.** `ok=true, status="completed"` now requires native
completion evidence: a prompt response whose `stop_reason` is not `refusal`,
`max_turn_requests`, `cancelled`, or `max_tokens`. Non-empty progress output
alone no longer counts as success.

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

## Controller terminal waiter

When a controller starts an asynchronous job, wait for the durable terminal
result with the CLI waiter. If the Agents MCP server uses a non-default state
root, pass that exact root explicitly; a job ID alone is not enough to locate a
job across MCP processes:

```bash
uv run --directory <agent-crossbar-repo> \
  python -m agent_crossbar.cli wait-job \
  --job-id "<job-id>" \
  --state-dir "<the-state-root-used-by-Agents-MCP>" \
  --timeout-sec 1815
```

The waiter treats `result_not_ready` as an expected intermediate response and
never calls `job_stop`. Exit codes are stable: `0` successful completion, `2`
observed deadline, `3` terminal failure/cancellation, and `4` missing or
inaccessible job. Do not infer a stall from a quiet `job_tail`, no workspace
diff, or absent output.

The waiter intentionally never cancels a job. If an unhandleable blocking
prompt is visible in `job_tail`, or the declared runtime deadline has elapsed,
use the explicit stop-then-collect wrapper (never for silence alone):

```bash
python3 ${CODEX_HOME}/skills/quota-aware-delegation/scripts/terminalize_job.py \
  "<job-id>" blocking_prompt
# or, after the runtime deadline:
python3 ${CODEX_HOME}/skills/quota-aware-delegation/scripts/terminalize_job.py \
  "<job-id>" runtime_deadline
```

It calls the existing provider-neutral `job_stop` lifecycle and then waits for
terminal `job_result`. `result_not_ready`, `job_tail`, and durable metadata do
not authorize replacement; if terminalization itself times out, retain the
original job and report the unresolved state.

`runtime_deadline` is accepted only when the job metadata records a positive
`max_runtime_sec` and `started_at`, and the current time is past that deadline
plus the documented 15-second result grace. `blocking_prompt` is accepted only
for the durable `awaiting_input` state.

Development jobs are serialized by a durable per-canonical-cwd writer lease.
`agent_start(task="dev")` acquires it before provider launch and releases it
only after a terminal result (or stop). Controller-local fallback must use the
same configured state through the quota-aware wrapper, which avoids guessing
the Agents MCP state root:

```bash
python3 ${CODEX_HOME}/skills/quota-aware-delegation/scripts/writer_lease.py \
  acquire --cwd "<cwd>" --owner-id "<controller-id>" --owner-kind local
# hold the returned token for the full edit/test window, then:
python3 ${CODEX_HOME}/skills/quota-aware-delegation/scripts/writer_lease.py \
  release --token "<token>"
```

The MCP surface remains exactly eight tools; writer-lease commands are an
internal CLI/controller path, not MCP tools.

An external job lease is never reclaimed by age while its job is nonterminal,
missing, or corrupt. If a job record is truly missing or corrupt, an operator
may use the explicit recovery path after preserving the state directory:

```bash
python3 ${CODEX_HOME}/skills/quota-aware-delegation/scripts/writer_lease.py \
  recover --cwd "<cwd>" --acknowledgement recover-missing-or-corrupt-job
```

Recovery refuses nonterminal jobs and corrupt lease files; ordinary
controllers must wait for `job_result` or call the existing provider-neutral
`job_stop` lifecycle.

The credential-free `scripts/acp_quiet_live_harness.py` is a lifecycle
regression harness, not a provider E2E. Maintainers can run the real provider
surface gate separately:

```bash
uv run --directory <agent-crossbar-repo> \
  python scripts/provider_surface_gate.py \
  --profile opencode \
  --model opencode-go/deepseek-v4-flash \
  --task dev \
  --max-runtime-sec 1800 \
  --artifact-dir /tmp/agent-crossbar-live-opencode-<timestamp>
```

For the full real chain, Codex root runs `gpt-5.6-sol` at `low`, starts the
native Luna coder with `gpt-5.6-luna` at `xhigh`, and that coder captures route
output from
`uv run --directory <codexbar-mcp-repo> delegation-route --lane implementation`, then start the exact routed profile/model/effort
through Agents MCP and run the source-path waiter above with the same Agents
MCP state root. This covers Codex root -> native coder -> `delegation_router`
-> Agents MCP -> OpenCode. It is a maintainer-only live gate and is not run in
provider-credential-free CI.

Reasonix is retained for credential-free compatibility tests but is excluded
from live gates because its paid quota is exhausted. Use the OpenCode Go
`opencode-go/*` namespace for DeepSeek live verification.

### Maintainer live gate prerequisites

The provider gate is a maintainer-local check. Run it from a checkout with a
fresh `uv` environment and authenticated provider CLIs already installed:

- Codex: `codex login` and a discovered model from `profiles_list`.
- Claude: Claude Code authenticated through its supported subscription login.
- OpenCode Go: an authenticated OpenCode Go account and a live
  `opencode-go/*` model from `profiles_list`.

The repository does not install provider CLIs, handle credentials, or claim a
GitHub-hosted runner can execute these gates. The removed workflow was
unusable on stock `ubuntu-latest` runners. For a semantic dev gate, use a
fixture that creates a file and runs its tests, then retain the job directory
and `job_result` evidence for review.

### Optional local admission policy

Admission is disabled by default. A configured local Agents process can enable
the private callback protocol through its `mcp_servers.agents.env` settings:

```toml
[mcp_servers.agents.env]
AGENT_CROSSBAR_ADMISSION_MODE = "strict"
AGENT_CROSSBAR_ADMISSION_COMMAND = '["uv", "run", "--directory", "/path/to/codexbar-mcp", "python", "-m", "codexbar_mcp.admission"]'
AGENT_CROSSBAR_ADMISSION_TIMEOUT_SEC = "5"
```

Strict mode fails closed when the command, timeout, response schema, or exact
profile/model/task candidate is missing or malformed. The callback receives
canonical request data on stdin and returns a versioned allow/deny decision.
It runs in a fixed inherited working directory; the requested workspace is
data, not callback process authority. This policy applies to every client in
the configured Agents process and does not trust client or session metadata.

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
| `writer_busy` | Another external or controller-local dev writer holds the canonical-cwd lease | Wait for its terminal result/release, or retry after stale reconciliation; do not edit concurrently |
| `writer_lease_corrupt` | Lease or associated external-job state is unreadable | Preserve the state directory, restore the record, or use the explicit acknowledged recovery path only for a missing/corrupt job |
| `writer_recovery_confirmation_required` / `writer_recovery_unsafe` | Explicit lease recovery was missing its acknowledgement or targeted a nonterminal/unsupported owner | Wait for terminal `job_result`/`job_stop`; recovery never overrides a nonterminal external job |
| `terminal_wait_timeout` | Explicit terminal result was not observed before the bounded waiter deadline | Do not replace the writer; inspect the retained job and use explicit `terminalize_job.py` only for a blocking prompt or elapsed runtime deadline |
| `terminalize_reason_not_permitted` | `blocking_prompt` was requested for a job that is not durably `awaiting_input` | Do not stop it; wait for the provider-required input state or use ordinary `job_stop` only when explicitly requested |
| `runtime_deadline_not_reached` / `runtime_deadline_unavailable` | Deadline recovery was requested before `max_runtime_sec` + 15s grace, or required metadata was missing/invalid | Do not stop or replace the job; retain it until the recorded deadline is proven |
| `acp_launch_error` | ACP agent process failed to launch (binary missing, dependency error) | Check provider CLI installation, run `agent-crossbar doctor` |
| `acp_protocol_error` | ACP protocol handshake or message error (version mismatch, invalid request) | Check provider and protocol logs; provider CLI may need upgrade |
| `acp_timeout` | ACP job exceeded `max_runtime_sec` while awaiting an already-delivered prompt's response | Follow `failure.next_action`: normally increase `max_runtime_sec`; for OpenCode, `check_provider_limits_or_retry_with_free_model` |
| `owner_input_timeout` | ACP execution deadline expired while an owner-mediated permission request was still pending | Retry the job and answer the surfaced permission request via `job_send` before the deadline |
| `acp_prompt_delivery_timeout` | ACP startup did not finish within its bounded startup window, before the prompt was dispatched | Check provider availability, quota, CLI installation, and selected model |
| `acp_empty_result` | An ACP `dev` task returned whitespace-only or entirely absent output — treated as a failed no-op rather than `completed`, since a real dev turn should produce observable text even when `changes` stays empty (Agents MCP does not inventory the workspace) | Retry, or inspect the prompt and provider session mode |
| `acp_incomplete` | An ACP turn stopped on an incomplete native `stop_reason` (`refusal`, `max_turn_requests`, `cancelled`, or `max_tokens`) even though it produced output — reported as failed, not `completed` | Adjust the prompt, grant a pending permission, or retry |
| `interactive_required` | `agent_start(profile="claude")` was called with `interactive=false` (explicitly or by default) | Pass `interactive=true` |
| `request_not_pending` | A `job_send` structured decision referenced an unknown, already-resolved, or terminal `request_id`, or a durable pending request had no live provider callback | Re-read `job_tail` for the current `pending_request`, or use plain text for an interactive reply |
| `invalid_decision` | A `job_send` structured decision used a value not offered in `pending_request.decisions` (for example `allow_always`) | Use one of the surfaced `decisions`; the request stays pending |

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
| `AGENT_CROSSBAR_ADMISSION_MODE` | `off` | `strict` enables the configured local exact-candidate admission callback |
| `AGENT_CROSSBAR_ADMISSION_COMMAND` | unset | JSON argv array for the private admission callback |
| `AGENT_CROSSBAR_ADMISSION_TIMEOUT_SEC` | `5` | Bounded callback timeout in strict mode |

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
