# Optional local admission callback

## Why

Agent Crossbar needs a bounded way for the configured local Agents process to
apply exact quota policy before any job, writer lease, or provider launch. The
policy must remain outside Crossbar's provider-neutral core and must never
rewrite a requested profile or model.

## What changes

- Add a private, versioned JSON subprocess callback that is disabled by default.
- In strict mode, invoke it after canonical validation/readiness and before
  lease/job/provider state mutation.
- Fail closed for missing configuration, malformed requests/responses,
  timeout, nonzero exit, stale or unavailable quota evidence.
- Keep quota mapping in `codexbar-mcp`: Claude subscription models share the
  Claude bucket; only live `opencode-go/*` uses the OpenCode Go bucket. Other
  namespaces and profiles are not applicable. Task maps to ask/review/dev
  semantics without changing the requested model.
- Keep the eight public MCP tools and public `agent_start` schema unchanged.

## Spec impact

changes contract — OpenSpec optional-local-admission. This is a new security
and lifecycle boundary; do not archive this change without explicit approval.
