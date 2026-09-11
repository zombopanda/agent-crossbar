## Purpose

Define how OpenCode/DeepSeek ACP permission requests are held pending,
surfaced honestly to the owner, resolved without escalation, and how ACP
task completion is gated on native protocol evidence rather than output
presence alone.

## ADDED Requirements

### Requirement: ACP permission requests MUST be held pending, not auto-decided

Any ACP `session/request_permission` call that is not resolved by the
existing bounded local-edit auto-allow policy SHALL be held as a genuinely
pending request rather than auto-rejected in-process. The ACP
coroutine/connection SHALL remain alive while the request is pending; the
existing single per-job execution deadline still applies and a timeout
firing while pending fails the job closed exactly as a timeout firing
while running does.

#### Scenario: A permission request pauses for an owner decision

- **WHEN** the ACP agent process sends a `session/request_permission` call
  that is not within the bounded local-edit auto-allow policy
- **THEN** the job transitions to `awaiting_input` with a durable pending
  record and no decision is returned to the agent process until the owner
  resolves it

#### Scenario: The ACP process is not torn down while pending

- **WHEN** a permission request is pending
- **THEN** the underlying agent subprocess and ACP connection remain
  running (not cancelled/terminated) unless the job's single execution
  deadline expires

#### Scenario: Deadline expiry while pending fails closed

- **WHEN** the job's `max_runtime_sec` deadline elapses while a permission
  request is still pending
- **THEN** the job is terminalized as timed out exactly as it would be for
  any other in-progress execution, and the pending request is settled
  (never left dangling, never silently granted)

### Requirement: Pending permission details MUST be bounded, honest, and non-escalating

The pending-request detail surfaced to the owner SHALL include a unique
`request_id`, the request's honest `kind` (including `"other"` and
`"switch_mode"` — never relabeled as a known kind), a bounded
tool/command/path description, all materially relevant bounded targets, and
the available decision options. If bounding would hide part of an action, the
record SHALL indicate truncation and SHALL offer `reject` only. No decision
path SHALL be able to produce an `allow_always`/blanket-scope grant, and no
secret-looking value SHALL appear in the surfaced detail.

#### Scenario: kind=other is surfaced honestly

- **WHEN** an ACP tool call reports `kind="other"`
- **THEN** the pending-request detail records `kind: "other"` verbatim,
  not folded into a known kind and not silently dropped

#### Scenario: filepath and parentDir raw-input fields are recognized

- **WHEN** a tool call's `rawInput` contains `filepath` or `parentDir`
  (or their snake_case equivalents) instead of the previously recognized
  path keys
- **THEN** the surfaced pending-request detail includes that path, honestly
  bounded, rather than omitting it

#### Scenario: Truncated details cannot be approved

- **WHEN** a command or target exceeds the bounded detail size
- **THEN** the pending record marks `details_truncated: true` and offers only
  generic `reject` until a complete reviewable action is available

#### Scenario: Decisions are generic and one-shot

- **WHEN** an owner resolves a pending request via `job_send`
- **THEN** the owner sees and sends only generic `allow` or `reject`; the
  adapter maps that decision privately to a single non-escalating option from
  the ACP request's own options, and `allow_always` is never reachable

### Requirement: Owner decisions use a durable inter-process handoff

The pending request record SHALL remain a durable inbox until the live ACP
coroutine consumes the decision. A missing in-process callback SHALL NOT by
itself expire a request while the recorded provider process is live. A
restart or dead/unidentifiable provider SHALL fail closed and settle the
request; a persisted decision SHALL never be replayed into a replacement
provider session.

#### Scenario: Owner process differs from provider process

- **WHEN** a controller sends a valid decision while the ACP callback is in a
  different process
- **THEN** `job_send` atomically records the generic decision and the live ACP
  coroutine observes it through the durable inbox and continues

#### Scenario: Restart with no live callback fails closed

- **WHEN** a durable pending record exists but no live callback or provider
  process can be proven
- **THEN** the request is settled as expired and a later decision is rejected
  without starting or resuming any provider session

#### Scenario: Process identity prevents PID-reuse replay

- **WHEN** a pending record's PID is live but its recorded process-start
  identity does not match the live process
- **THEN** the request is treated as dead/restarted, settled, and no decision
  is delivered

### Requirement: ACP completion MUST require native completion evidence

`acp_runtime.run_acp_job` SHALL NOT report `ok=True, status="completed"`
based solely on non-empty output. It SHALL consult the native
`stop_reason` field returned by the ACP prompt response; `refusal`,
`max_turn_requests`, and `cancelled` SHALL be reported as truthfully
incomplete/refused execution, never as success, regardless of output
content.

Any permission refusal, unresolved owner request, or incomplete tool call
latched by the protocol SHALL also force a failed/incomplete result even when
the provider follows it with progress text and `stop_reason="end_turn"`.

#### Scenario: A refusal with output is not reported as success

- **WHEN** an ACP prompt completes with `stop_reason="refusal"` and
  non-empty output text
- **THEN** the job result reports the execution as refused/incomplete, not
  `ok=True, status="completed"`

#### Scenario: A genuinely complete turn is unaffected

- **WHEN** an ACP prompt completes with `stop_reason="end_turn"` and
  non-empty output
- **THEN** the job result reports `ok=True, status="completed"` as today

#### Scenario: Refusal followed by progress text and end_turn is not success

- **WHEN** the protocol records a refused or incomplete permission/tool event,
  then emits progress text and finally reports `stop_reason="end_turn"`
- **THEN** the job result remains failed/incomplete and never claims success
