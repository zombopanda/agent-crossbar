## Purpose

Define Claude's `agent_start`/background-monitor/`awaiting_input` contract:
interactive-only launch, in-place monitor continuation across `job_send`
under one unmoved deadline, and full-fidelity question text.

## ADDED Requirements

### Requirement: Claude launch MUST be interactive-only

The Claude adapter SHALL reject any `agent_start` request that resolves to
`interactive=false`, before readiness probing, admission checks, writer
lease acquisition, or job creation. The rejection SHALL use a stable error
code distinct from `interactive_not_supported` (which describes a provider
that never supports interactive mode).

#### Scenario: Explicit interactive=false is rejected before job creation

- **WHEN** `agent_start` is called with `profile="claude"` and
  `interactive=false` (explicitly or via the parameter default)
- **THEN** the response is `{"ok": false, "error": "interactive_required", ...}`
- **AND** no job directory, writer lease, or native Claude session is created

#### Scenario: Direct lifecycle callers get the same rejection

- **WHEN** `claude_lifecycle.start_claude_job` is invoked directly with
  `interactive=False`, bypassing the public `agent_start` tool
- **THEN** it returns the same `interactive_required` error before calling
  `store.create_job`

#### Scenario: interactive=true is unaffected

- **WHEN** `agent_start` is called with `profile="claude"` and
  `interactive=true`
- **THEN** the existing tmux-attach lifecycle proceeds unchanged

### Requirement: The Claude monitor MUST remain alive across awaiting_input

The background monitor thread for a Claude job SHALL NOT exit its polling
loop when the job transitions to `awaiting_input`. It SHALL continue
polling the native session under the single monotonic deadline established
when the thread started, so that a subsequent `job_send` is observed on
the monitor's next poll with no thread restart and no deadline reset or
extension.

#### Scenario: job_send reply is observed without restarting the monitor

- **WHEN** a Claude job is `awaiting_input` and the owner calls `job_send`
  with plain-text follow-up input
- **THEN** the same background monitor thread (never exited) observes the
  native session's subsequent state change on its next poll
- **AND** no new monitor thread is started and no writer-lease/run-handle
  re-registration occurs

#### Scenario: The original deadline governs the whole job

- **WHEN** a Claude job cycles between `running` and `awaiting_input`
  multiple times via repeated `job_send` calls
- **THEN** the job's terminal timeout (`max_runtime_exceeded`) fires, if at
  all, based on the single deadline computed from the job's original
  `started_at` and `max_runtime_sec` — never a value recomputed from a
  later `job_send` or resumed-monitor start time

#### Scenario: A concurrent stop remains terminal

- **WHEN** `job_stop` marks the job terminal while the monitor is about to
  transition it to `awaiting_input`
- **THEN** the transition is rejected (`job_already_terminal`) and the
  monitor does not resurrect or continue polling a terminal job

### Requirement: awaiting_input MUST carry the full owner question text

When a Claude job enters `awaiting_input` because the native session is
mid-turn and blocked on a question, the published detail SHALL include the
actual question text recovered from the complete tmux transcript, not
merely the native CLI's short status label, and SHALL NOT be limited to the
2 KiB diagnostics-sanitizer truncation applied to other provider log
output.

#### Scenario: Full question text is preserved

- **WHEN** Claude's native session reports `blocked` with a question longer
  than 2048 bytes
- **THEN** the `awaiting_input` event/job-meta detail includes the full
  question text (redacted for secrets, not truncated to a byte-limited
  prefix)

#### Scenario: Native label is still available for compatibility

- **WHEN** a Claude job enters `awaiting_input`
- **THEN** the existing `waiting_for` native-label field is still present
  alongside the new full-text field, preserving current consumers
