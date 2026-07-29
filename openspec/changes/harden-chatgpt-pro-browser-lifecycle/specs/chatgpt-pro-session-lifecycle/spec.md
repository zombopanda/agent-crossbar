## ADDED Requirements

### Requirement: Every GUI job owns a verified ChatGPT session

The system SHALL associate each ChatGPT Pro GUI job with an internal session handle containing a job identity, browser candidate, window identity, lifecycle state, and cancellation callback. The handle SHALL be internal and SHALL NOT add provider-specific fields to `agent_start` or any shipped tool schema.

#### Scenario: A job acquires a usable session

- **WHEN** a ChatGPT Pro job starts and a supported browser window can be verified
- **THEN** the session manager records the owned identity before prompt delivery and serializes CUA access to that session

#### Scenario: The discovered window does not match the owned session

- **WHEN** a later AX snapshot identifies a different, closed, or ambiguous window
- **THEN** the runner fails closed, records a session-mismatch diagnostic, and does not type or submit into that window

### Requirement: Foreground CUA access is serialized and bounded

The ChatGPT Pro session manager SHALL serialize foreground CUA and clipboard mutations. The initial implementation SHALL keep capacity at one active GUI turn and SHALL not enable parallel jobs merely by removing the existing global lock.

#### Scenario: A second job arrives while a GUI turn is active

- **WHEN** the session capacity is occupied
- **THEN** the second job receives the stable busy/queued behavior selected by the implementation without touching CUA

#### Scenario: A session is retired after contamination

- **WHEN** prompt state, window identity, or response state cannot be trusted
- **THEN** the session is retired before another job can use it

### Requirement: `job_stop` cancels provider execution as well as durable state

The system SHALL mark a job terminal before provider cleanup, invoke its registered cancellation handle when present, and preserve the existing late-result guard. Cancellation SHALL be idempotent.

#### Scenario: Stop is requested during GUI generation

- **WHEN** `job_stop` is called after prompt submission and before completion
- **THEN** the runner requests the visible ChatGPT stop action when available, stops polling, retires the session, and leaves the durable job status stopped

#### Scenario: Stop races GUI startup

- **WHEN** `job_stop` is called before browser/session setup completes
- **THEN** the worker observes the stopped state before focus, prompt delivery, or session publication and performs any needed cleanup without submitting the prompt

#### Scenario: Provider completion races with stop

- **WHEN** the provider returns after the job has been stopped
- **THEN** no late success overwrites the stopped result or terminal metadata

### Requirement: Fallback is forbidden after prompt submission

The runner SHALL permit browser-candidate fallback only while no prompt has been submitted. Once submission is durable, it SHALL return the original generation/status failure and SHALL not submit the same request to another candidate.

#### Scenario: Pre-submit browser failure

- **WHEN** a candidate fails before prompt submission
- **THEN** the next eligible candidate may be attempted under the existing safety checks

#### Scenario: Post-submit status failure

- **WHEN** a candidate has submitted the prompt but response status cannot be safely read
- **THEN** the runner returns a generation-status failure and does not fall back

### Requirement: Every turn runs in a fresh temporary ChatGPT conversation

The runner SHALL open a dedicated ChatGPT Temporary Chat for each turn and SHALL confirm the visible temporary-chat indicator before attaching the prompt. It SHALL NOT reuse an existing conversation, SHALL NOT append a turn to a thread that already contains unrelated messages, and SHALL NOT rely on the user's default chat window.

#### Scenario: A turn acquires its own temporary conversation

- **WHEN** a ChatGPT Pro turn starts and a supported Chromium-family browser is available
- **THEN** the runner opens `https://chatgpt.com/?temporary-chat=true` in a window it owns, confirms the visible `Temporary Chat` indicator, and only then prepares the composer

#### Scenario: The temporary indicator cannot be confirmed

- **WHEN** the opened window does not expose the temporary-chat indicator
- **THEN** the turn fails closed with a stable session error before the prompt is attached, and no message is sent into a persistent conversation

#### Scenario: Prior turns never leak into a later turn

- **WHEN** two turns run in sequence
- **THEN** the second turn's conversation contains none of the first turn's messages, and neither turn appends to a conversation the user already had open

### Requirement: Owned turn windows are closed after the turn

The runner SHALL close the temporary window it opened once the turn reaches a terminal state, including cancellation and failure, and SHALL never accumulate ChatGPT windows across turns.

#### Scenario: A completed turn releases its window

- **WHEN** a turn reaches `complete`, `cancelled`, or `failed`
- **THEN** the window the runner opened for that turn is closed and the session is retired

#### Scenario: An unrelated user window is never closed

- **WHEN** the turn did not open the window it used
- **THEN** the runner leaves that window open and only retires its own session state

### Requirement: Input never takes the user's foreground or keyboard focus

Prompt delivery, model selection, submission, and cancellation SHALL be performed through background accessibility actions that do not raise the browser, move the visible cursor, or mutate the clipboard. A foreground fallback SHALL be attempted only when the background write cannot be verified, SHALL be recorded in diagnostics, and SHALL never run while another turn is live.

#### Scenario: A normal turn leaves the user's focus untouched

- **WHEN** a turn runs against a rendered ChatGPT window
- **THEN** the frontmost application never changes and the system clipboard is never written

#### Scenario: Background delivery cannot be verified

- **WHEN** the background write does not produce the exact expected composer content
- **THEN** the runner clears only its own partial text and records the fallback it used, without typing over user content

### Requirement: Concurrent GUI turns are excluded host-wide

Mutual exclusion for ChatGPT GUI turns SHALL be host-global, not scoped to a state directory, because foreground automation, the clipboard, and the browser UI are process-global. While a turn has a submitted prompt that is still generating, the runner SHALL NOT start another turn and SHALL NOT launch or fall back to another browser candidate.

#### Scenario: A second process attempts a turn with a different state directory

- **WHEN** two agent-crossbar processes with different state directories request a ChatGPT turn
- **THEN** the second one observes the busy state instead of driving the same browser concurrently

#### Scenario: A candidate fails while an earlier turn is still generating

- **WHEN** any submitted turn is still generating in a browser
- **THEN** no other browser candidate is launched or driven until that turn reaches a terminal state
