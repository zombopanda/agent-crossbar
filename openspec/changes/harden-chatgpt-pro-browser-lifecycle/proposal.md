## Why

The `chatgpt_pro` profile already executes through browser-based CUA/AX automation, but its readiness contract still describes the native desktop app and can reject the public start path before a job is created. Once a browser turn has started, `job_stop` changes durable job state without cancelling the GUI generation, while completion, session identity, context handling, and diagnostics remain weaker than the proven patterns in Agentify Desktop and `codex-chatgpt-web`.

This change makes the existing browser transport truthful, cancellable, isolated, and fail-closed without adding provider-specific fields to the public MCP contract or introducing a second browser gateway.

## What Changes

- Make ChatGPT Pro readiness browser-based and non-mutating, with explicit authentication/model evidence and actionable failure states.
- Add an internal, generic cancellation handle so `job_stop` can stop an active GUI turn and prevent late completion from resurfacing as success.
- Add a job-scoped ChatGPT session/window manager with serialized access, identity checks, and safe session retirement.
- Replace marker-only completion polling with stable visible-response and DOM-health tracking, structured lifecycle stages, and bounded heartbeats.
- Verify the requested model and effort against the live ChatGPT UI; do not add static or default model selections.
- Add deterministic, bounded context packing from the existing generic `cwd`/`scope` inputs and secure job-local GUI diagnostics/artifacts.
- Extend unit, mocked-CUA, and maintainer-only live provider gates for readiness, cancellation, isolation, completion, context, and artifact safety.
- Preserve the locked eight-tool surface, provider-neutral `agent_start` schema, current no-fallback-after-submit rule, and `chatgpt_pro` non-interactive capability until a separate interactive lifecycle is proven.

## Capabilities

### New Capabilities

- `chatgpt-pro-readiness`: truthful browser-based readiness and live model/effort evidence.
- `chatgpt-pro-session-lifecycle`: job-scoped browser session ownership, cancellation, serialization, and retirement.
- `chatgpt-pro-turn-completion`: fail-closed prompt delivery, visible completion tracking, DOM health, and lifecycle events.
- `chatgpt-pro-context-artifacts`: bounded context preparation and secure job-local diagnostics/artifact registration.

### Modified Capabilities

None.

## Impact

- Affected modules: `readiness.py`, `jobs.py`, `runner.py`, `server.py`, `context.py`, the ChatGPT Pro adapter/profile, and provider/lifecycle tests.
- The public MCP contract remains unchanged; all session, browser, cancellation, context, and artifact metadata stays internal or in durable job evidence.
- No Agentify or `codex-chatgpt-web` runtime dependency is introduced. Their bounded implementation patterns are ported with license/notice review where source code is reused.
- The live gate will require an authenticated macOS browser session and a Pro-capable account; provider-dependent tests remain skipped when credentials/session state are absent.
