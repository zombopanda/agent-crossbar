## ADDED Requirements

### Requirement: Browser-based readiness is truthful and non-mutating

The ChatGPT Pro readiness probe SHALL inspect the existing macOS browser/CUA surface rather than require the native ChatGPT desktop application. It SHALL not launch, focus, navigate, click, type, change the clipboard, or otherwise mutate browser or ChatGPT state.

#### Scenario: Existing authenticated Pro browser is inspectable

- **WHEN** the supported browser surface exposes a ChatGPT window with authenticated Pro evidence
- **THEN** readiness reports `ready` with bounded, redacted evidence identifying the inspected browser surface

#### Scenario: Browser is absent or authentication cannot be proven

- **WHEN** no supported ChatGPT browser window exists or the visible state does not prove authentication
- **THEN** readiness reports an actionable non-ready state and remediation without claiming that the native desktop app is required

#### Scenario: Readiness is probed on a non-macOS host

- **WHEN** the readiness probe runs on an unsupported operating system
- **THEN** it reports the stable unsupported-platform result without attempting browser automation

### Requirement: Readiness results remain cached and truthful

The provider readiness result SHALL use the existing bounded cache policy and SHALL never report `ready` solely because the profile is registered or because macOS is present.

#### Scenario: Cached readiness is reused within the TTL

- **WHEN** the same profile is probed again before the configured readiness TTL expires
- **THEN** the cached result is returned without a second browser inspection

#### Scenario: Registration alone is insufficient

- **WHEN** the profile exists but no browser/authentication evidence is available
- **THEN** readiness is not `ready`

### Requirement: Requested model and effort are confirmed before submission

The ChatGPT Pro turn setup SHALL verify the requested `model` and optional `effort` against live visible UI/capability evidence before attaching or submitting the prompt. It SHALL fail closed when the requested selection cannot be confirmed and SHALL not select a default model.

#### Scenario: Requested selection is visible and confirmed

- **WHEN** the live picker exposes the requested model/effort and the rendered control confirms it
- **THEN** the turn may proceed to composer preparation

#### Scenario: Requested selection is unavailable

- **WHEN** the picker does not expose or confirm the requested model/effort
- **THEN** the turn fails before prompt submission with a stable capability error and bounded available-choice diagnostics
