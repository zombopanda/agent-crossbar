# Owner-mediated job permissions and continuation

## Why

Two provider-owned pause points — a Claude interactive turn asking a
follow-up question, and an OpenCode/DeepSeek ACP agent asking permission to
run a tool/command or touch a path — are not honestly represented today.

Claude's `agent_start` silently accepts `interactive=false` and runs a fully
unattended `claude --bg` session even though the only way to answer a
mid-turn question is `job_send`, which requires `interactive=true`; the
background monitor thread also exits the moment a job goes
`awaiting_input`, so nothing resumes polling the native session after the
owner replies, and the surfaced "question" is a bare native status label
(e.g. `"permission prompt"`), never the actual text Claude asked.

ACP permission requests (`session/request_permission`) are auto-decided
in-process — bounded local edits are auto-allowed once, everything else
(including `kind="other"`) is auto-rejected — with zero owner visibility,
no durable record, and no way for an owner to say yes to a specific,
legitimate request. Task completion for ACP jobs also ignores the native
`stop_reason` field: a refused (`stop_reason="refusal"`) or truncated
(`"max_turn_requests"`, `"cancelled"`) turn that still produced non-empty
text is reported as `ok=True, status="completed"`.

This change makes both pause points owner-mediated through one
provider-neutral mechanism layered on the existing `job_send` tool, without
adding a new public MCP tool or new public `agent_start` fields.

## What Changes

- Claude's `agent_start` rejects an explicit or defaulted
  `interactive=false` with a stable `interactive_required` error before any
  job/lease/provider state mutation; `claude_lifecycle.start_claude_job`
  enforces the same rule for any caller that bypasses the public tool.
- The Claude background monitor no longer exits its polling loop when a job
  becomes `awaiting_input`. It keeps running under the one monotonic
  `max_runtime_sec` deadline established at job start, so a `job_send`
  reply is picked up on the next poll with no restart and no new deadline.
- The `awaiting_input` detail Claude publishes carries the full,
  untruncated question text recovered from the tmux transcript (via the
  existing screen-reader extraction over the complete `tmux-output.log`),
  not a bare native label or a diagnostics-sanitizer-truncated log prefix.
- ACP `request_permission` holds a genuinely pending request instead of
  auto-deciding: it publishes a durable `awaiting_input` detail with a
  unique `request_id`, the bounded tool/command/path(s), the available
  decisions, and an honest `kind` (including `"other"`), then awaits an
  owner decision without tearing down the ACP connection/coroutine. No
  decision path can produce `allow_always`; only the existing
  generic `allow`/`reject` decisions are reachable; ACP option IDs remain
  private to the adapter and no `allow_always` escalation is possible.
- `rawInput` path extraction recognizes OpenCode's `filepath`/`parentDir`
  field names (and their snake_case equivalents) instead of silently
  missing them.
- ACP job completion is gated on the native `stop_reason` and protocol outcome:
  `refusal`, `max_turn_requests`, `cancelled`, and owner-rejected permission
  turns are reported as truthfully incomplete/refused, never folded into
  `ok=True, status="completed"` merely because output is non-empty.
- `job_send`'s existing `text` argument gains a provider-neutral, opt-in
  structured form: a JSON object `{"request_id": ..., "decision": ...}`
  that resolves one specific pending permission/question. Any other `text`
  (including malformed JSON) is treated as plain interactive input, exactly
  as today. Resolution enforces the existing owner/`"*"`-wildcard
  cross-session policy, rejects stale/foreign/already-resolved
  `request_id`s with a stable error, and applies a decision exactly once.
- A pending request is settled (marked expired/superseded) at the same
  moment its job becomes terminal via timeout or `job_stop`, so a
  late-arriving decision can never resurrect an already-terminal job. The
  single job-level `max_runtime_sec` deadline is never paused or extended
  while a request is pending.
- `job_tail` exposes the concrete pending request (`request_id`, `kind`,
  bounded detail, available decisions) via `meta`/event data instead of a
  bare `waiting_for` string, so a waiter/CLI can act without scraping raw
  terminal text.
- Profile capability metadata, README, and CHANGELOG are updated to match;
  the package version moves to 0.5.0 (Claude's `interactive=false`
  rejection is a breaking behavior change on a currently-successful path).

## Capabilities

### New Capabilities

- `claude-interactive-lifecycle`: Claude's `agent_start`/monitor/awaiting_input
  contract — interactive-only enforcement, in-place monitor continuation
  across `job_send`, and full-fidelity question text.
- `acp-owner-mediated-permissions`: ACP permission-request pending/hold,
  surfacing, decision mapping, and native-evidence completion gating.
- `job-owner-decision-protocol`: the shared, provider-neutral `job_send`
  structured-decision contract (request correlation, ownership/wildcard
  policy, staleness/duplicate/race handling, settlement on
  terminal/timeout).

### Modified Capabilities

None — `admission-contract` is untouched; this change does not alter quota
or admission policy.

## Impact

- Affected modules: `server.py` (agent_start gate only — no new
  provider-specific branching), `adapters/base.py`, `adapters/claude.py`,
  `adapters/claude_lifecycle.py`, `agent_runner.py`, `jobs.py`,
  `acp_client.py`, `acp_runtime.py`, `profiles/claude.py`,
  `profiles/opencode.py`, `cli.py` (read-only exposure only).
- Public contract: the locked `agent_start`/`job_send` field sets are
  unchanged. `job_send`'s `text` argument gains a documented, backward
  compatible structured form; plain text keeps its exact current meaning.
  Claude's `interactive=false` behavior changes from "succeeds
  unattended" to "rejected before job creation" — a minor-version-worthy
  breaking change per this project's pre-1.0 compatibility policy.
- No new public MCP tool. No new `agent_start` field. No change to the
  8-tool surface.
- Dependency/infra note: no new third-party dependency; the request_id
  correlation inbox is durable job metadata, with an in-process Future only
  as a wake-up optimization. Provider liveness uses the recorded process
  identity, and the original single runtime deadline remains in force.
