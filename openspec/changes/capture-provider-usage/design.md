# Design

## Canonical usage block

Every envelope has a structured `usage` object. Native evidence reports
`available: true`, `status: complete|partial`, `source`/`provenance`,
`input_tokens`, `cache_creation_input_tokens` (and `cache_write_tokens`),
`cache_read_input_tokens`, `output_tokens`, `reasoning_tokens`,
`total_tokens`, and nested `subagents`. Unknown values are `null`; no number is
estimated. No evidence reports `available: false`, `status: unavailable`, a
machine-readable `reason`, the same fields as `null`, and an empty subagent
list.

## Claude transcript replay

Claude Code writes one assistant JSONL row for each streamed content block.
The extractor keeps one usage object per `requestId`, sums the distinct API
turns, and computes `total_tokens` only when the required components are valid.
It reads `<projects>/<slug>/<session>/subagents/agent-*.jsonl` for nested
workers, preserving `agent_id`, `agent_type`, description, model, counts, and
partial status. Historical replay fixtures use a copy under `tests/fixtures`;
the source job directories are never mutated.

## ACP pass-through

`AcpResult` carries the protocol's optional `PromptResponse.usage`. The runtime
normalizes snake/camel ACP names and validates non-negative integer counts.
Provider cumulative totals are preserved as reported rather than summed from
stream events. Missing optional fields produce `partial`; absent usage is
`unavailable` with a reason.

## Persistence and public surface

All adapter, ACP, stop, timeout, and lazy tmux finalization paths call the same
envelope builder. `result.json` therefore stores usage, and `JobStore.get_result`
surfaces it at the existing top-level alongside the envelope. The eight MCP
tools and their request schemas are unchanged.
