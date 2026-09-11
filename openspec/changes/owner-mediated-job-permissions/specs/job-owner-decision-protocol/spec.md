## Purpose

Define the shared, provider-neutral protocol layered on the existing
`job_send` tool's `text` argument for resolving a pending owner-mediated
request (Claude question or ACP permission), including ownership policy,
staleness/duplicate/race handling, and settlement on job termination.

## ADDED Requirements

### Requirement: job_send text MAY carry a structured decision, plain text is unaffected

`JobStore.send_user_input` SHALL attempt to parse the `text` argument as
JSON. Any object containing both `request_id` (string) and `decision`
(string) is a structured decision attempt. The exact request id, pending
state, owner, provider liveness, and generic decision are validated under one
metadata-lock compare-and-set; unknown, stale, duplicate, terminal, or
provider-gone attempts return `request_not_pending` and never become plain
input. JSON that does not match this shape, malformed JSON, and ordinary text
retain today's plain interactive conversation behavior. No new MCP tool is
introduced.

#### Scenario: Structured decision resolves a pending request

- **WHEN** `job_send(job_id, text='{"request_id": "<id>", "decision":
  "allow"}')` is called and `<id>` matches the job's current pending
  request
- **THEN** the durable pending inbox is resolved with generic `allow` exactly
  once and the live provider coroutine resumes

#### Scenario: Plain text remains ordinary interactive input

- **WHEN** `job_send(job_id, text="continue please")` is called
- **THEN** the text is delivered as interactive input exactly as before
  this change, regardless of whether a request happens to be pending

#### Scenario: Malformed or non-decision JSON falls back to plain text

- **WHEN** `job_send` is called with `text` that is not valid JSON, or is
  valid JSON missing `request_id`/`decision`
- **THEN** it is treated as plain interactive text, not rejected as a
  malformed decision

### Requirement: Decision resolution MUST enforce existing ownership and wildcard policy

Resolving a structured decision SHALL be subject to the same job-ownership
check as any other `job_send` call, including the existing `"*"`
cross-session wildcard sentinel for explicit local-controller/operator
access. No new ownership or scope model is introduced.

#### Scenario: Foreign session is denied by default

- **WHEN** a structured decision is sent for a job owned by a different
  `client_session_id`, without the wildcard sentinel
- **THEN** it is rejected exactly as a foreign-session `job_send` is
  rejected today (`job_not_found`)

#### Scenario: Wildcard sentinel is preserved

- **WHEN** `client_session_id="*"` is used to send a structured decision
- **THEN** cross-session resolution is permitted, exactly matching today's
  documented wildcard behavior for `job_send`/`job_tail`/`stop_job`

### Requirement: Stale, unknown, and duplicate decisions MUST be rejected, never silently accepted or replayed

A structured decision referencing a `request_id` that does not exist, no
longer exists for that job, or has already been resolved/expired/
superseded SHALL be rejected with a stable, distinct error
(`request_not_pending`), never silently accepted and never treated as a
fresh request. A second decision for a `request_id` that was already
resolved SHALL NOT re-apply its side effect (no double-delivery, no
replay-escalation to a different decision than the one first applied).

#### Scenario: Unknown request_id is rejected

- **WHEN** a structured decision references a `request_id` the job has
  never issued
- **THEN** the call fails with `request_not_pending`, and no job state
  changes

#### Scenario: Duplicate decision does not double-apply

- **WHEN** the same `request_id` receives two structured decisions in
  sequence, the first while still pending and the second after it resolved
- **THEN** the first is applied exactly once and the second fails with
  `request_not_pending` without re-triggering the resolved side effect

### Requirement: Pending requests MUST settle exactly when their job becomes terminal

Whenever a job is terminalized by deadline-reap (`_reap_deadline_expired_job`)
or explicit stop (`stop_job`), any pending request recorded for that job
SHALL be settled (marked expired or cancelled) in the same critical
section that makes the job terminal. A structured decision arriving after
that point SHALL be rejected by the staleness check above; it SHALL NOT be
able to resurrect an already-terminal job.

#### Scenario: Timeout settles a pending request

- **WHEN** a job's `max_runtime_sec` deadline elapses while a request is
  pending
- **THEN** the job is terminalized as timed out and the pending request is
  marked expired in the same operation

#### Scenario: A late decision cannot resurrect a stopped job

- **WHEN** `job_stop` terminalizes a job with a pending request, and a
  structured decision for that request arrives afterward
- **THEN** the decision is rejected (`request_not_pending`) and the job
  remains stopped

### Requirement: Waiters MUST see a concrete pending request, not a vague waiting state

`job_tail` SHALL expose the concrete pending request (its `request_id`,
`kind`, bounded detail, and available decision options) for an
`awaiting_input` job, sourced from durable job metadata, so a waiter/CLI
can act on it without reconstructing it from raw event text or terminal
capture. The writer lease for the job SHALL remain held by its current
owner while a request is pending.

#### Scenario: job_tail exposes the pending request

- **WHEN** `job_tail` is called on a job that is `awaiting_input` with an
  open pending request
- **THEN** the response includes the pending request's `request_id`,
  `kind`, bounded detail, and available decisions

#### Scenario: Writer lease is retained while pending

- **WHEN** a job has an open pending request
- **THEN** its writer lease is not released or reassigned until the job
  reaches a terminal state
