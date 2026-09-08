# Design

Crossbar serializes the canonical request as JSON on stdin to a configured
argv array. The callback process inherits a fixed controller cwd; user `cwd`
is data only. Crossbar starts a new process group, bounds input/output and
runtime, and kills the owned group on timeout or malformed execution.

The response is exactly `{version: 1, decision: "allow"|"deny", reason: str}`.
An allow is accepted only for that request. There is no fallback candidate or
model substitution. The callback is enabled through the local Agents process
environment (`mcp_servers.agents.env`), never through client/session metadata.

`codexbar_mcp.admission` calls the existing `evaluate_claude` and
`evaluate_opencode` evaluators directly. It does not call the workflow router
or `build_live_report`, which would select a fallback. Claude uses one shared
subscription bucket for all requested Claude models. OpenCode Go is selected
only for a qualified `opencode-go/*` model; other OpenCode namespaces,
Codex, Reasonix, and GUI are outside this quota policy and are admitted as
not-applicable after request-shape validation.
