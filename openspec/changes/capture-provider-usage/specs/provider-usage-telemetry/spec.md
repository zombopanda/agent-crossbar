# Delta: provider usage telemetry

## ADDED Requirements

### Requirement: Terminal results expose native provider usage

Terminal result envelopes and persisted `result.json` MUST expose a structured
usage block with input, cache creation/write, cache read, output, reasoning,
total, source/provenance, availability/completeness, and nested provider
subagent attribution when native evidence exists.

#### Scenario: Claude tmux transcript

- **WHEN** a Claude tmux job has a native transcript keyed by its recorded
  session id
- **THEN** its terminal result reports deduplicated provider/model token counts
  and nested Task-tool subagent counts

#### Scenario: OpenCode ACP usage

- **WHEN** an OpenCode ACP prompt response supplies structured usage
- **THEN** the terminal result preserves the exact native counts and reports
  the ACP provenance

### Requirement: Streamed duplicate usage is not double-counted

The extractor MUST deduplicate repeated streamed messages for one provider API
turn before summing usage.

#### Scenario: Repeated request id

- **WHEN** several assistant JSONL rows share one `requestId` and repeat the
  same usage object
- **THEN** that usage contributes exactly once

### Requirement: Unknown telemetry remains explicit

The system MUST report `partial` or `unavailable` with null unknown fields and
a reason when native evidence is absent or incomplete; it MUST NOT coerce
unknown counts to zero or estimate them.

#### Scenario: ACP omits usage

- **WHEN** an ACP provider returns no usage object or invalid native counts
- **THEN** the result exposes structured unavailable/partial provenance and
  does not claim exact total usage

### Requirement: Existing MCP surface remains stable

Adding usage MUST NOT change the existing public MCP tool names or argument
schemas.

#### Scenario: Existing result consumer

- **WHEN** a caller invokes the existing `job_result` tool
- **THEN** it receives the prior result fields plus the additive structured
  usage block
