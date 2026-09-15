# Proposal: Capture provider-native token usage

Agent Crossbar currently publishes `usage: {available: false}` for terminal
jobs even when Claude Code's transcript or an ACP `PromptResponse.usage`
contains native counts. This hides provider/model cost and makes streamed
duplicate messages easy to over-count in downstream audits.

Add a provider-neutral usage block to result envelopes and `result.json`.
Claude tmux jobs read their native JSONL transcript, deduplicating repeated
stream rows by `requestId` and retaining nested Task-tool subagent attribution.
OpenCode ACP jobs pass through the structured usage object when supplied.
Missing or malformed fields remain explicit `partial`/`unavailable` evidence;
the implementation never estimates or turns unknown values into zero.

## Scope

- Claude Code tmux transcript extraction, including nested subagents.
- OpenCode ACP native usage extraction and persistence.
- Result-envelope and public `job_result` pass-through compatibility.
- Replay fixtures and focused regression coverage for the two historical job
  evidence shapes without modifying the historical records.

No MCP tool names, arguments, quota policy, or provider lifecycle semantics
change.
