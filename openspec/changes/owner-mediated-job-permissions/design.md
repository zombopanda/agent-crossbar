# Design: Owner-mediated job permissions and continuation

## Context

Two independent pause mechanisms exist today and both under-serve the
owner:

- **Claude (`claude_bg`/tmux)**: a native background session polled by a
  daemon thread in `agent_runner._run_adapter_job`. The thread already
  transitions the job to `awaiting_input` for both "turn complete, awaiting
  follow-up" (`native_state == "done"`) and "mid-turn question"
  (`native_state == "blocked"`), but only the `done` branch keeps polling
  (`continue`); the `blocked` branch `return`s, ending the thread. Nothing
  else ever restarts it. `job_send` (`jobs.py:send_user_input`) delivers
  keystrokes into the tmux pane and flips the job status back to
  `"running"`, but with no live poller, that status update is inert until
  some other call (`job_tail`/`job_result`/`list_jobs`) happens to trigger
  the separate, pattern-matching-based `_finalize_completed_tmux_job` path
  — a second, disconnected notion of "done" from the native-state monitor.
- **ACP (`codex`/`opencode`)**: `acp_client._OneShotClient.request_permission`
  decides synchronously in-process. It never surfaces a request, never
  blocks on an owner, and never records that a request happened. Task
  completion in `acp_runtime.run_acp_job` only checks
  "is `result.output` non-empty" — `result.stop_reason` (a native,
  protocol-level field: `end_turn | max_tokens | max_turn_requests |
  refusal | cancelled`) is captured but never consulted for the
  `ok`/`status` decision.

Both mechanisms need the same three things: (1) a durable, request-scoped
pending record the owner can see and act on, (2) a way for `job_send` to
resolve that specific request without becoming a new tool, and (3) settle
semantics so a late decision can never resurrect a job that already went
terminal by timeout or `job_stop`.

## Goals / Non-Goals

**Goals:**

- Make Claude interactive-only at the public contract boundary, rejected
  before any state mutation, with the existing `_tool_error` shape.
- Keep the Claude monitor thread alive across `awaiting_input` so the same
  one monotonic `max_runtime_sec` deadline governs the whole job, including
  time spent waiting on the owner — never paused, never extended, never
  restarted from a fresh clock.
- Recover and surface the actual full question text Claude asked, not a
  native one-word label or a diagnostics-sanitizer-truncated prefix.
- Hold ACP permission requests genuinely pending, with a `request_id`,
  bounded/honest detail (including `kind="other"` and OpenCode's
  `filepath`/`parentDir` raw-input fields), and no path to `allow_always`.
- Keep the ACP process/coroutine alive while a decision is pending — the
  same single job deadline still applies; expiry while pending fails
  closed exactly like expiry while running.
- Gate ACP completion on native `stop_reason`, not just non-empty output.
- Layer request/decision correlation onto `job_send`'s existing `text`
  argument via an opt-in JSON shape, preserving the existing plain-text
  interactive-input meaning as the fallback.
- Preserve the existing `"*"` cross-session wildcard policy unchanged.
- Guarantee settle-on-terminal: a pending request outlives its usefulness
  the instant its job is stopped or reaped for timeout, and a decision
  arriving afterward is rejected, not silently absorbed.

**Non-Goals:**

- No new public MCP tool and no new `agent_start`/`job_send` field — the
  8-tool surface and locked field lists stay exactly as documented in
  AGENTS.md.
- No `allow_always`/blanket permission escalation of any kind — that
  remains a deliberately unreachable outcome.
- No change to quota/admission policy (`admission-contract` is untouched).
- No redesign of the tmux-pattern-matching completion path
  (`_finalize_completed_tmux_job`/`tmux_output.py`) beyond what's needed so
  it no longer disagrees with a monitor that now stays alive; a full
  unification of the two completion signal sources is out of scope.
- No pausing/extension of `max_runtime_sec` for either provider while a
  request is pending — one monotonic bounded deadline, full stop.

## Decisions

### 1. Fix the Claude monitor by not exiting, not by restarting it

**Decision:** change the interactive `blocked` branch in
`_run_adapter_job` to behave like the interactive `done` branch already
does — check whether the job is already `awaiting_input` (skip a duplicate
transition/event if so), then `continue` the same `while True` loop instead
of `return`ing. The thread that was started once by `start_agent_job` keeps
running, using the same `deadline` local computed once at thread entry,
for the entire life of the job, across any number of `awaiting_input` ↔
`running` cycles.

**Alternative rejected:** have `jobs.py:send_user_input` re-invoke
`start_agent_job` for the same `job_id` when it flips `awaiting_input` back
to `running`. This was the initially obvious fix, but it requires either
(a) recomputing `started_monotonic`/`deadline` fresh — silently resetting
the runtime budget, which is exactly the bug class this change must not
introduce — or (b) threading the original `started_at`/`max_runtime_sec`
back into a second call, plus reconciling `run_handles.register` being
called twice for one job. Never letting the thread die is strictly
simpler, keeps one deadline variable in one closure for the job's whole
life, and requires no change to `send_user_input`'s job-resume logic at
all beyond what request-decision settlement needs (Decision 4).

### 2. Question text: extract from the full transcript, not a native label

**Decision:** when the interactive `blocked` branch enters (or re-enters)
`awaiting_input`, read the complete `tmux-output.log` file
(`meta["tmux_output_path"]`), normalize it with the existing
`tmux_output.normalize_tmux_output`, and extract the current on-screen text
with `tmux_output.interactive_tmux_output_summary` — the same tail-anchored,
completion-marker window already used to summarize a completed interactive
job — using a bounded window well above the 2 KiB diagnostics limit
(`_QUESTION_MAX_CHARS`). Publish the result as a `question` field alongside
the existing `waiting_for` native label, redacted with `redact_secrets` but
**not** routed through `sanitize_diagnostic_text`'s 2 KiB prefix truncation.

`adapters.claude._screen_reader_final_response` is deliberately not used here:
it keys off `claude:` markers in the `claude logs` output, which the tmux
attach transcript does not contain (it renders `⏺` markers instead), so it
would return an empty string for the very transcript this branch reads.

**Alternative rejected:** raise `DIAGNOSTICS_MAX_BYTES` globally, or add a
provider-specific carve-out inside `envelope.py`. Both touch a core module
that AGENTS.md rule 8 forbids from carrying provider-specific behavior;
extracting in the Claude-specific `agent_runner`/`adapters/claude` call
site and publishing an already-bounded, purpose-built field keeps the
core sanitizer generic and untouched.

### 3. ACP pending permission: durable inbox plus an in-process wake-up keyed by (job_id, request_id)

**Decision:** publish the pending request into job metadata/events before
registering a small in-process wake-up registry. `job_send` performs an
owner-checked compare-and-set under the job metadata lock, changing the
record to `state=resolved` with generic `decision=allow|reject`. The ACP
coroutine polls this durable inbox while also awaiting its local Future, so a
controller in a different process can resolve the request without relying on
`call_soon_threadsafe`. The registry only wakes the local waiter and is never
used as a provider liveness test. A live provider PID permits durable
cross-process resolution; a missing/dead PID fails closed as restart/death.
ACP option ids remain private to `permission_response_for_decision`, and
`allow_always` is unreachable.

**Alternative rejected:** rely only on an in-process Future. That breaks when
the owner calls `job_send` from a CLI/controller process, which is the normal
cross-session case. The bounded metadata poll is intentionally inside the
existing `asyncio.wait_for(timeout)` and does not create a second deadline or
an unbounded wait.

### 4. `job_send`'s `text` gains an opt-in structured decision shape

**Decision:** in `JobStore.send_user_input`, attempt `json.loads(text)` first.
Any JSON object containing string `request_id` and `decision` is a structured
decision attempt. Ownership is validated before mutation; the exact request
id, pending state, generic decision, and provider liveness/restart fence are
checked under the metadata lock. Unknown, stale, duplicate, terminal, and
provider-gone attempts return `request_not_pending` and never fall through as
plain text. A successful attempt atomically records `state=resolved` and
`decision=allow|reject` before waking any local Future. JSON that does not
match the structured shape, malformed JSON, and ordinary text retain today's
plain-input path.

**Alternative rejected:** a new `job_decide`/`job_permission_respond` MCP
tool. Rejected per the task's explicit constraint (no new tool) and per
AGENTS.md rule 2 (the 8-tool surface is locked and any change needs a
separate design discussion); layering on `job_send` also means CLI/waiter
callers get the new capability without any new client-side wiring.

### 5. Settlement: mark pending requests terminal exactly when their job does

**Decision:** `_reap_deadline_expired_job` and `stop_job` — the two
existing places that make a job terminal — also settle (mark
`expired`/`cancelled`) any pending request recorded in that job's meta, in
the same critical section, before releasing the job-meta lock. A
structured decision arriving after that point fails the "still pending"
check in Decision 4 with a stable `request_not_pending`/`request_stale`
error, mirroring the existing `job_already_terminal` guard pattern
(`transition_job_status`, `set_result`) that already prevents a late
`awaiting_input` transition from resurrecting a stopped job
(`tests/test_jobs.py::test_stopped_job_publishes_envelope_and_cannot_resurrect_awaiting_input`).
No new lock is introduced; this reuses the existing `_job_meta_lock`
critical sections.

### 6. Deadline stays single, monotonic, and un-pausable

**Decision:** the ACP run and the Claude monitor both keep exactly the
budget they have today (`asyncio.wait_for(..., timeout=effective_timeout)`
for ACP; the `deadline` local for Claude) — awaiting a permission/question
resolution happens *inside* that same budget, not alongside a second one.
If the deadline fires while a request is pending, the existing timeout
path runs unchanged (ACP: `AcpTimeoutError` → `acp_timeout` classification;
Claude: the `deadline` check at the top of the polling loop →
`max_runtime_exceeded`), and Decision 5's settlement fires as part of that
same terminalization.

**Alternative rejected (explicitly, per reviewer correction):** pausing or
extending the deadline while `awaiting_input`/pending so a slow owner
doesn't cause a timeout. Rejected — a paused/extended deadline is an
unbounded liveness commitment disguised as a bounded one, and reintroduces
exactly the "which clock is authoritative" ambiguity this change exists to
remove. One monotonic bounded deadline, always.

## Risks / Trade-offs

- **[Risk]** Keeping the Claude monitor thread alive across many
  `awaiting_input` cycles means a slow-answering owner keeps a daemon
  thread and a writer lease held for the job's full `max_runtime_sec`. →
  Mitigation: the single monotonic deadline (Decision 6) still bounds the
  lifecycle, and a native `done` state is terminal once the current turn has
  positive evidence.
- **[Risk]** An ACP `asyncio.Future`-based pending registry adds new
  concurrency surface (future resolved twice, future never resolved,
  future resolved after the owning task is already cancelled). →
  Mitigation: resolution is guarded by an idempotency check
  (`future.done()`) mirroring `run_handles`' existing idempotent
  register/release pattern; cancellation always resolves/cancels any
  outstanding future in the same `finally` block that already tears down
  the process on timeout/cancel, while the durable metadata inbox handles a
  controller in another process. The registry is never treated as liveness
  proof.
- **[Risk]** Widening `_PATH_KEYS` to catch `filepath`/`parentDir` could
  over-match unrelated fields. → Mitigation: the added keys are exact,
  narrowly-named (`filepath`, `filePath`, `parentDir`, `parent_dir`), not a
  generic case-insensitive substring match.
- **[Trade-off]** Version bump to 0.5.0 for what is, mechanically, a
  narrow gate addition — but Claude's `interactive=false` path is
  currently a *successful, tested, documented* code path becoming a
  rejection, which is a breaking behavior change on the public contract by
  this project's own compatibility policy (README: "Minor versions may
  change APIs").

## Migration Plan

1. Land the generic `adapter.requires_interactive` capability flag and the
   `server.py`/`claude_lifecycle.py` gate; update the Claude profile and
   every test that launches Claude non-interactively to pass
   `interactive=true` (or move that assertion to the new rejection).
2. Land the Claude monitor continuation + question-text fixes; add
   regression coverage for multi-cycle `awaiting_input` → `job_send` →
   `running` → `awaiting_input` within one unmoved deadline.
3. Land the ACP pending-permission registry, `request_id` surfacing, and
   `stop_reason` completion gating; add regression coverage for
   allow/deny/stale/kind=other/rawInput cases and refusal/cancelled
   completion.
4. Land the shared `job_send` structured-decision parsing in
   `send_user_input`, plus settlement in `_reap_deadline_expired_job` and
   `stop_job`; add regression coverage for stale/duplicate/foreign/wildcard
   and timeout/cancel-then-late-decision races.
5. Expose `pending_request` via `job_tail`; update CLI/README examples.
6. Bump `package.json`/`pyproject.toml` to 0.5.0, add the `CHANGELOG.md`
   entry, update profile capability metadata.
7. Rollback: every change is additive/gated behind the new
   `requires_interactive`/pending-request fields; reverting is a plain
   revert of this change's commits with no data migration, since no
   on-disk job-meta schema for prior jobs is read differently.

## Open Questions

- Whether `_finalize_completed_tmux_job`'s independent tmux-pattern-match
  completion path should eventually be retired now that the native-state
  monitor never abandons a job — left as a follow-up; this change only
  ensures the two paths stop disagreeing about `awaiting_input` jobs
  in-flight, not a full unification.
