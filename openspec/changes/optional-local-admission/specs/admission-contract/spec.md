## Purpose

Defines the bounded local admission boundary that applies exact quota decisions
before Crossbar mutates jobs, leases, or provider execution state.

## ADDED Requirements

### Requirement: Provider-neutral local admission
The configured local Agents process MAY run one provider-neutral admission
callback after canonical request validation and readiness, and before any job,
writer-lease, or provider-launch mutation. Admission is disabled by default.
Strict mode MUST fail closed when the callback configuration, request, response,
quota evidence, or callback execution is missing, malformed, stale, timed out,
or otherwise unavailable.

#### Scenario: Default mode preserves the existing route
- **WHEN** admission mode is unset or `off`
- **THEN** Crossbar skips the callback and applies its existing provider-neutral
  validation and workflow routing without changing the requested profile, model,
  task, or effort

#### Scenario: Strict mode denies before state mutation
- **WHEN** strict admission denies, times out, exits nonzero, or returns an
  invalid response
- **THEN** Crossbar returns a stable admission error before creating a job,
  acquiring a writer lease, or launching a provider

#### Scenario: Exact request is preserved
- **WHEN** the callback receives a canonical request
- **THEN** it evaluates that exact profile, model, task, effort, and canonical
  cwd; Crossbar performs no fallback, model substitution, or provider rewrite

### Requirement: Bounded callback protocol
The callback protocol SHALL use a JSON request and the exact versioned response
shape `{version: 1, decision: "allow"|"deny", reason: string}`. The configured
command SHALL be an argv array executed with the inherited Agents controller
cwd; the requested user cwd is data, not callback process cwd. Crossbar SHALL
bound streamed stdin/stdout, runtime, and callback process-group cancellation,
and SHALL close callback pipes on every success and failure path.

#### Scenario: Callback cancellation is bounded
- **WHEN** the callback exceeds its deadline, emits excessive output, or exits
  through a bounded execution path
- **THEN** Crossbar applies bounded cancellation to its owned process group,
  waits for the recorded leader as implemented, and denies the request without
  creating admission state; the cancellation result is not treated as proof
  that every descendant has disappeared

### Requirement: Quota policy remains outside Crossbar core
The admission callback SHALL own quota policy. All Claude subscription models
use the Claude quota bucket; only live `opencode-go/*` models use the OpenCode
Go bucket. Other OpenCode namespaces and Codex, Reasonix, and GUI profiles are
not applicable and MUST NOT trigger an unrelated quota fetch. Malformed
OpenCode model namespaces are denied. Admission is a quota gate only; it does
not authenticate caller roles or claim provider ordering.

#### Scenario: Provider namespace mapping is explicit
- **WHEN** a request names Claude, `opencode-go/*`, another OpenCode namespace,
  or a non-OpenCode profile
- **THEN** the callback evaluates the matching quota bucket or returns
  not-applicable according to the mapping without rewriting the request

### Requirement: Local process activation does not change the public contract
The strict callback SHALL be configured only in the local Agents process
environment and SHALL apply to every client using that process. Client or
session metadata MUST NOT be authorization. The eight public MCP tools and
provider-neutral `agent_start` schema MUST remain unchanged.

#### Scenario: Client metadata cannot bypass admission
- **WHEN** a client supplies a different client name, session identifier, or
  other metadata
- **THEN** Crossbar applies the same configured admission mode and exact-request
  decision
