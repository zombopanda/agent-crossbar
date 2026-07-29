# Changelog

All notable changes to Agent Crossbar.

## [Unreleased]

## [0.3.7] — 2026-07-29

### Added
- Generic in-process run-handle registry (`run_handles`): `job_stop` marks a job
  terminal first and then cancels the registered provider work. The interface is
  transport-neutral — `jobs.py` never branches on a provider name.
- `chatgpt_pro` GUI turns now expose internal lifecycle stages (`bootstrap`,
  `authenticated`, `model_selected`, `composer_ready`, `prompt_verified`,
  `submitted`, `streaming`, `complete`, `cancelled`, `failed`) as bounded job
  events, with exactly one terminal stage per turn.
- Stable completion tracking: a turn completes only with a present, non-running
  response, non-empty final text, a visible completion action, and text that is
  unchanged across the stability window. The nonce marker remains a correlation
  check, not the sole completion signal. Configurable via
  `AGENT_CROSSBAR_CHATGPT_STABILITY_SEC`.
- DOM-health tracking for responses that never appear, vanish mid-generation, or
  complete without answer text; each fails closed and retires the session.
- Job-owned browser sessions with verified window identity, serialized to one
  active GUI turn, retired after completion, cancellation, or contamination.
- Bounded, deterministic context packing from the existing generic `cwd`/`scope`
  inputs, with symlink and generated-directory exclusion, file/chunk/total
  budgets, secret redaction, and explicit attachment validation.
- ChatGPT GUI diagnostics are written under the owning job's directory with
  realpath containment, symlink/hardlink rejection, and redaction of context
  envelopes, credentials, and visible hidden-reasoning markers.
- `scripts/provider_surface_gate.py --chatgpt-lifecycle`: maintainer-only live
  gate for browser readiness, cancellation after submit, and sequential
  isolation.
- `JobStore.job_status()` so a worker can observe a stop that raced its startup.

### Changed
- ChatGPT Pro prompt delivery and submission now use a **background accessibility
  write** (`set_value` + AX `press`) as the primary path: a turn no longer takes
  the user's foreground window or keyboard focus and no longer writes the system
  clipboard. Verified live — the frontmost application stayed unchanged across a
  full turn. The clipboard/foreground path remains a recorded fallback; live
  testing showed it interleaves with the user's own typing and corrupts both the
  prompt and the user's draft.
- The Stop control is matched by label family (`Stop`, `Stop answering`, …). The
  previous fixed list missed the live `Stop answering` label, which made a
  running turn look idle and prevented a confirmed provider stop.
- A failed delivery or failed pre-Send verification now clears only the partial
  text that turn wrote. A corrupted leftover draft previously blocked every
  later turn with `composer_not_empty`.
- A window holding an unrelated draft is no longer a hard failure: the turn opens
  a fresh ChatGPT window instead and never clears or types over user content.
- The browser window is only raised when its page is not already exposed over
  accessibility.
- Requested `scope.attachments` fail closed with `attachment_upload_unsupported`:
  no supported browser candidate exposes a usable CDP file-input route (Safari has
  no attachable endpoint; the Chromium route requires an isolated profile that is
  by definition not the user's signed-in profile).
- ChatGPT Pro readiness now inspects the running browser surface instead of
  requiring the native ChatGPT desktop app. It stays strictly non-mutating
  (no launch, focus, navigation, click, typing, or clipboard writes), keeps the
  60-second cache, and reports `ready` only with visible signed-in evidence.
  New error codes replace `chatgpt_pro_manual_gate`.
- The requested `model` (and `effort`, when the UI exposes it) is confirmed
  against the live visible picker before the prompt is attached; an unconfirmable
  selection fails closed with the redacted list of available choices. No default
  model is ever selected.
- The composer is re-verified for the exact prompt from a fresh snapshot
  immediately before Send; a mismatch reports expected/actual/common-prefix
  lengths and never submits.
- `agent_start`'s generic `scope` is threaded into the internal request. The
  public eight-tool surface, `agent_start` fields, and `model` requiredness are
  unchanged.
- Codex model validation now uses the same live-discovered catalog that
  `profiles_list` publishes; a selected model can no longer be advertised and
  then rejected before launch.
- The live gate recognizes the current labelled Claude transcript and a
  Reasonix terminal footer after noisy MCP startup output, while still rejecting
  prompt echoes as answers.

## [0.3.0] — 2026-07-24

### Changed
- `profile_health` model discovery output is dramatically more compact for
  large catalogs: effort names are replaced with small int codes resolved via
  a top-level `effort_legend`, the redundant flat `models[]` list is dropped
  in favor of `model_info` alone, and `model_info` entries are grouped by
  their `provider/` id prefix (stripped from each entry). A catalog with
  hundreds of models (e.g. `opencode`) now serializes at roughly a tenth of
  its previous size with no loss of information needed to select a model.

## [0.2.0] — 2026-07-23

### Added
- Public product identity: `agent-crossbar` (repo, PyPI, npm, CLI).
- Environment variable migration: `AGENT_CROSSBAR_*` replaces `AGENT_HARNESS_*` with backward-compatible deprecation shim.
- Python 3.11, 3.12, and 3.13 support.
- Public README with quickstart, tool table, support matrix, troubleshooting codes.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, MIT `LICENSE`.
- `AGENTS.md` contributor code contract.
- npm package reduced to logic-free `uvx` launcher.
- Public GitHub Actions CI: Python matrix, package audits, secret scan, CodeQL, Dependabot.
- Protected maintainer workflows for live provider gates and trusted release publishing.
- Package content audits excluding internal results, benchmarks, private paths, and telemetry.

### Changed
- Public identity renamed from the rejected `Agent Relay MCP` /
  `agent-relay-mcp` name to `Agent Crossbar` / `agent-crossbar`.
- Import package renamed: `agent_harness_mcp` → `agent_crossbar`.
- Distribution name: `agent-harness-mcp` → `agent-crossbar`.
- State directory default: `~/.local/state/agent-crossbar`.
- All env vars use `AGENT_CROSSBAR_` prefix.
- npm package: `@pandenko/agent-harness-mcp` → `agent-crossbar` (no scope).
- Removed private registry metadata from public package/docs.
- Removed `@pandenko` branding from public-facing files.

### Fixed
- `requires-python` lowered from `>=3.13` to `>=3.11` after compatibility verification.

### Migration
- The accidentally published `agent-relay-mcp==0.1.3` release is superseded by
  `agent-crossbar==0.2.0`. It remains available only as a yanked historical
  release so the abandoned namespace cannot be silently reused.

## [0.1.2] and earlier — Internal

Pre-release internal versions. Not published publicly. See private repository history.
