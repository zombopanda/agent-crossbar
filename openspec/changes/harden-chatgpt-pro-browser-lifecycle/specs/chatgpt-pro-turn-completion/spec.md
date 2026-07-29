## ADDED Requirements

### Requirement: GUI turns expose explicit internal lifecycle stages

The runner SHALL track the stages `bootstrap`, `authenticated`, `model_selected`, `composer_ready`, `prompt_verified`, `submitted`, `streaming`, `complete`, `cancelled`, and `failed`. Stage events SHALL be bounded, redactable job evidence and SHALL NOT become provider-specific public request fields.

#### Scenario: A successful turn advances through submission and completion

- **WHEN** ChatGPT accepts the prompt and produces a valid answer
- **THEN** the recorded lifecycle reaches `submitted`, `streaming`, and `complete` in order

#### Scenario: A turn fails before submission

- **WHEN** authentication, model selection, composer preparation, or prompt verification fails
- **THEN** the lifecycle records the failing stage and no candidate after a failed pre-submit attempt receives the prompt unless fallback remains safe

### Requirement: Prompt attachment is exact and verified before send

The runner SHALL refuse to overwrite a non-empty unrelated composer, SHALL attach the complete expected prompt, and SHALL verify exact visible composer content from a fresh snapshot before clicking Send.

#### Scenario: Prompt is preserved exactly

- **WHEN** the composer accepts the wrapped prompt
- **THEN** the fresh verification matches the expected content and the runner may submit

#### Scenario: Paste or insertion is truncated or altered

- **WHEN** the observed composer content differs from the expected prompt
- **THEN** the runner fails before Send and records expected length, actual length, common-prefix length, and browser/stage diagnostics without overwriting the composer blindly

### Requirement: Completion requires stable visible response evidence

The completion tracker SHALL require a present response, non-running state, non-empty final text, a visible completion action, and unchanged final text for the configured stability interval. The existing nonce marker SHALL remain a correlation check but SHALL not be the sole completion criterion.

#### Scenario: Final response becomes stable

- **WHEN** the visible response meets all completion predicates and remains unchanged for the stability interval
- **THEN** the runner extracts the correlated answer and records `complete`

#### Scenario: Marker appears in an unstable or stale page

- **WHEN** the nonce is visible but the response is still running, empty, changing, or belongs to an unverified window
- **THEN** the runner continues tracking or fails closed and does not report success

### Requirement: DOM health fails closed on vanished or empty responses

The runner SHALL detect when a response DOM never appears, disappears while active, or reaches a visible terminal state without final answer text. Each condition SHALL produce a stable failure category and bounded diagnostics.

#### Scenario: Response DOM disappears during generation

- **WHEN** an observed response disappears for longer than the configured grace period
- **THEN** the turn fails with a status-unavailable diagnostic and the session is retired

#### Scenario: ChatGPT completes without answer text

- **WHEN** the UI exposes completion controls but no final response text for longer than the empty-response grace period
- **THEN** the turn fails without returning an empty success

### Requirement: Progress and terminal events are single-owner and bounded

The runner SHALL emit bounded stage, heartbeat, and visible-progress events through the existing job event stream, SHALL avoid duplicate terminal events, and SHALL never persist hidden chain-of-thought.

#### Scenario: Long generation remains active

- **WHEN** no terminal state has been reached within the heartbeat interval
- **THEN** the job emits a bounded generation-progress heartbeat containing elapsed time and stage only

#### Scenario: Client stops or disconnects while the job continues

- **WHEN** the durable job is stopped or the provider runner exits
- **THEN** exactly one terminal state is retained and subsequent polling observes the persisted result rather than a second completion
