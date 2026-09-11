# Tasks

All checkboxes remain open until implementation and the named verification
evidence are complete.

## 1. Claude interactive-only contract

- [x] 1.1 Add the provider capability that requires interactive launch and
  set it only on Claude.
- [x] 1.2 Reject resolved `interactive=false` before readiness, admission,
  lease, or job creation; aliases must not bypass the guard.
- [x] 1.3 Enforce the same stable `interactive_required` error for direct
  Claude lifecycle callers.
- [x] 1.4 Update mocked Claude launch tests and prove no job/session is
  created on explicit false.

## 2. Claude monitor continuation and question fidelity

- [x] 2.1 Keep the same monitor alive across repeated
  `awaiting_input` -> `job_send` -> `running` cycles under one monotonic
  deadline.
- [x] 2.2 Surface the full redacted owner question from the bounded transcript
  without the diagnostics prefix truncation.
- [x] 2.3 Do not convert every interactive native `done` into follow-up input;
  naturally collect final turns, including fast replies.
- [x] 2.4 Use a durable input/reply generation or equivalent positive marker so
  stale done readings cannot finalize the previous turn without requiring an
  intermediate working observation.
- [x] 2.5 Add deterministic tests for two awaits, full question text, and the
  fast-reply path.

## 3. ACP owner-mediated permissions

- [x] 3.1 Hold non-wildcard permission requests pending and keep the ACP
  coroutine/connection alive inside the original bounded deadline.
- [x] 3.2 Surface a durable bounded request with generic `allow`/`reject`,
  honest `kind`, unique `request_id`, and redacted tool/command/path detail.
- [x] 3.3 Recognize OpenCode `rawInput.filepath` and `parentDir` shapes and
  preserve all existing cwd/path checks for auto-allow.
- [x] 3.4 Keep provider option IDs private and prevent `allow_always` or other
  scope escalation.
- [x] 3.5 Implement a durable inter-process decision inbox; local registry
  absence alone must not expire a request while the provider is live.
- [x] 3.6 Fail closed after restart/dead provider with no callback and never
  replay persisted decisions into a replacement session.
- [x] 3.7 Bound the live registry explicitly: reject/settle overflow and bind
  per-request cancellation callbacks correctly.
- [x] 3.8 Add protocol-shaped allow/deny, stale, duplicate, kind/path, IPC,
  and restart tests.

## 4. ACP completion honesty

- [x] 4.1 Require native completion evidence and latch refusal, unresolved
  permission, or incomplete tool outcomes independently of output narration.
- [x] 4.2 Reproduce the original progress-only + `end_turn` false-success
  shape and assert a failed/incomplete result.
- [x] 4.3 Add non-empty refusal, cancellation, truncation, and valid end-turn
  regression tests.

## 5. Shared job_send decision protocol

- [x] 5.1 Parse structured JSON syntax-first; any object with string
  `request_id` and `decision` is a decision attempt.
- [x] 5.2 Atomically validate owner, exact request, state, generic decision,
  and provider liveness under the same metadata lock before callback delivery.
- [x] 5.3 Reject unknown/stale/duplicate/terminal decisions with
  `request_not_pending`; never fall through to plain text.
- [x] 5.4 Preserve plain text and explicit `"*"` wildcard behavior.
- [x] 5.5 Settle pending requests in the same terminalization critical section
  as timeout/cancel and prove late decisions cannot resurrect.
- [x] 5.6 Add competing-send, owner, stale, duplicate, timeout/cancel, and
  lease-retention tests.

## 6. Waiter/CLI exposure

- [x] 6.1 Expose the concrete pending request/question immediately through
  `job_tail` and waiter exit status `5` without releasing the lease.
- [x] 6.2 Document sending a validated decision/reply and resuming the waiter.
- [x] 6.3 Test needs-input output, lease retention, and terminal timeout.

## 7. Docs, profiles, version

- [x] 7.1 Keep profile capabilities accurate: Claude interactive-only; expose
  any OpenCode permission side channel without claiming unsupported free-text
  interactive transport.
- [x] 7.2 Update README/CHANGELOG and the global quota-aware waiter/controller
  guidance.
- [x] 7.3 Bump the public package version to `0.5.0` for the Claude behavior
  change and verify package hygiene.

## 8. Required verification

- [x] 8.1 Full deterministic credential-free suite green.
- [x] 8.2 Lint, type, build, and package smoke checks green.
- [x] 8.3 Fresh model discovery/profile health and runtime provenance recorded.
- [x] 8.4 LIVE OpenCode allow -> continue -> natural final, reject -> truthful
  result, and no reply -> bounded timeout with no orphan process/lease.
- [x] 8.5 LIVE Claude equivalents, including two awaits, are required before
  claiming complete when Claude quota is available; the live equivalent is
  now covered by the recorded evidence.

## Evidence

- Implementation and deterministic protocol coverage are present in the
  provider adapters, ACP runtime, job decision protocol, lifecycle monitor,
  CLI/waiter, profile, and contract test changes above.
- Final verification: `uv run pytest -q` passed twice with `1371 passed,
  2 skipped`; focused lifecycle/parser/cleanup tests and repeated parser
  stress also passed; Ruff, format, and `git diff --check` passed.
- Review-blocker regressions now also cover all ACP `args`/`arguments`/`argv`
  forms with fail-closed malformed/truncated handling, bounded ANSI cursor
  rendering, and a durable single-owner deadline terminalization CAS.
- Updated final verification: `uv run pytest -q` passed twice with `1384
  passed, 2 skipped`; the 10x focused permission/parser/monitor stress,
  quota-aware skill suite (`53 passed`), Ruff, format, and `git diff --check`
  all passed.
- Timeout terminalization now persists its envelope/result even when timeout
  telemetry `send_event` or underlying event I/O fails; bounded telemetry error
  evidence and cleanup/lease disposition remain truthful (`115` focused
  lifecycle regressions, plus the full suite at `1384 passed, 2 skipped`).
- Fresh profile discovery/readiness recorded Claude and OpenCode as ready,
  with discovered model IDs and runtime provenance.
- Live evidence was recorded for OpenCode allow/continue, reject, and bounded
  timeout cleanup, plus Claude two-await continuation, stale-state recovery,
  cursor-redraw extraction, and terminal cleanup/lease disposition.
