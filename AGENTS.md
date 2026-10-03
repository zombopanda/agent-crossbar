# Agent Crossbar — Contributor Code Rules

These rules apply to code contributions. They are intentionally concise and focused on the public contract.

## Public Contract (Provider-Neutral)

1. **No top-level provider-specific fields.** The public API (`agent_start`, `job_result`, tool schemas) MUST NOT expose fields that only apply to a single provider. Provider-specific metadata lives in per-profile capability entries.

2. **The 8-tool MCP surface is locked.** Adding or removing a tool requires a design discussion and version bump. No hidden tools. No deprecated aliases.

3. **Stable error codes.** Error codes documented in the README troubleshooting table MUST NOT change semantics in a patch version. New codes may be added in minor versions.

4. **`agent_start` schema is provider-neutral and minimal.** The only public fields are: `profile`, `prompt`, `task`, `interactive`, `model`, `effort`, `cwd`, `scope`, `max_runtime_sec`, plus standard client metadata (`client`, `client_name`, `client_version`, `client_session_id`). No transport enum, no provider-specific routing hints.

   `model` is required. No profile, adapter, discovery result, convenience
   wrapper, or test may silently select a default model.

5. **`interactive` is a boolean.** The public schema exposes `interactive` as a single boolean. No transport enum, no public transport field. Transport selection is an internal adapter concern.

6. **Forbidden (dead) public fields.** The following fields MUST NOT appear in the `agent_start` signature or any shipped schema: `external_context`, `budget_usd`, `text_subtype`, `review_target`, `context_target`, `full_local`. No compatibility shims, no deprecated aliases for these.

## Architecture

7. **Per-profile adapter modules.** Each provider gets one bounded module under `agent_crossbar/adapters/` implementing `ProviderAdapter` from `base.py`. Adapter modules MUST NOT import from each other.

8. **Core modules are provider-agnostic.** `server.py`, `jobs.py`, `validation.py`, `envelope.py`, `readiness.py` MUST NOT contain provider-specific branching. Provider behavior is injected through adapter lookups.

9. **Per-profile capabilities/models live in separate profile modules.** Each profile under `agent_crossbar/profiles/` owns its capabilities and models. Model lists MUST come from CLI discovery (`discovery.py`), not hardcoded. Profiles MUST NOT publish or apply default models. Profile modules MUST NOT import from each other.

10. **No compatibility aliases for removed fields.** Removed fields (`transport`, `autonomy`, `sensitivity`, `sanitized_context_only`, `timeout_sec`, etc.) MUST NOT have runtime compatibility shims, aliases, or warning-based remapping. They are simply absent.

11. **Readiness probes are non-mutating and cached.** Probes inspect state, never change it. Results cache for 60 seconds. Registration alone never produces `ready`.

## Testing

12. **TDD for provider changes.** Any new provider adapter, capability, or transport MUST include:
    - Unit tests proving validation, schema conformance, and error paths (no live provider).
    - A passing live provider gate (maintainer-only workflow) on every claimed operation/transport combination.

13. **No fake capabilities or models.** Model lists and capability declarations MUST come from live provider discovery or documentation references. Do not invent capabilities.

14. **Deterministic CI.** All tests in public PR CI MUST pass without provider credentials. Provider-dependent tests use `pytest.skip` when credentials are absent.

## Package Hygiene

15. **Public artifacts exclude internal content.** Benchmark results, `.reasonix/`, `.memsearch/`, personal paths, private URLs, and handoff files MUST NOT appear in wheel/sdist/npm tarball.

<!-- openspec-canonical-workflow-v2 -->
## OpenSpec Canonical Workflow

- `openspec/specs/` is the canonical contract for verified current behavior. `openspec/changes/<name>/specs/` contains temporary deltas for work that needs a durable decision record.
- OpenSpec is a complexity gate, not a requirement for every feature, bug, retry, or normative edit. Use it for explicit OpenSpec requests, cross-cutting or architectural work, non-obvious trade-offs, risky migrations, public contracts, security boundaries, deployment architecture, or work likely to span sessions or agents.
- Before changing an existing capability, read its main spec and record one result: `spec impact: none`, `spec impact: restores <requirement>`, `spec impact: canonical correction — <evidence>`, `spec impact: changes contract — direct canonical update`, or `spec impact: changes contract — OpenSpec <change-name>`.
- A fix that restores an existing canonical requirement proceeds without a delta. Add regression evidence and reference the restored requirement.
- A direct canonical correction is allowed only when implementation, tests, configuration, or verified runtime evidence establish that the spec is stale. Do not use this path to introduce an undecided policy.
- A small, well-understood contract adjustment with obvious acceptance criteria may update the canonical spec directly. A contract change requires OpenSpec only when it also passes the complexity gate.
- Retry code does not trigger OpenSpec by itself. Distinguish restoration of an existing recovery guarantee, an internal mechanism with no external semantic change, a small direct contract adjustment, and a complex retry policy with non-obvious trade-offs.
- Keep exact cron expressions, polling intervals, queue names, internal paths, and similar mechanisms out of normative specs unless the exact value is itself a required external or operator contract. Put operational values in config, contract tests, and runbook documentation.
- Before creating a change, inspect active changes for the affected capability. Reuse one only for the same bounded outcome; otherwise group related work by coherent capability or release slice, never one change per bug.
- The project task tracker (if any) owns individual defects and implementation work. OpenSpec is not an issue tracker.
- When unsure whether a small change deserves OpenSpec, default to direct work. When the intended contract itself is ambiguous, resolve that ambiguity before implementation.
- Sync verified deltas into main specs and archive completed changes in the same delivery lifecycle. Do not leave completed changes active or use long-lived changes as competing canonical specifications.
- When all tasks and required verification gates are complete, report the completion and delta-sync state and ask Bo for explicit current-thread approval before archiving. Never archive a completed change automatically; a direct instruction to close or archive that named change counts as approval.
<!-- end-openspec-canonical-workflow-v2 -->

<!-- memsearch-agent-instructions-v3 -->
<!-- AUTO-GENERATED BLOCK - do not edit manually. Source: /Users/bo/.codex/memsearch/agents.md -->

## Memsearch Recall (MANDATORY WHEN RELEVANT)

Use MemSearch when prior decisions can change the answer: debugging regressions or repeated runtime failures, resuming earlier work, requests about past decisions, or nontrivial work involving project conventions or shared infrastructure. Skip it for self-contained exact lookups, tiny edits, and pure formatting unless Bo asks.

Use the existing `memory-recall` skill for search, expansion, and transcript retrieval. Use `memsearch-global` for cross-project recall and `memsearch search` in the owning project for literal strings. State which project or collections you searched, then verify relevant facts against current files or runtime.

The project's `.memsearch.toml` owns one collection across its worktrees and submodules. Never guess collection names or add nested `.memsearch.toml` files. Use the installed wrapper; do not add another lock or install a second MemSearch/Milvus runtime. Follow stricter project-specific executable rules.

<!-- end-memsearch-agent-instructions -->
