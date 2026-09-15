# Tasks

- [x] Add provider-neutral usage extraction and structured unavailable/partial
      envelopes.
- [x] Wire Claude transcript and nested subagent usage into tmux lifecycle
      finalization paths.
- [x] Wire OpenCode ACP native usage through `AcpResult` and runtime envelopes.
- [x] Preserve usage in `result.json` and public `job_result`, including lazy
      tmux finalization.
- [x] Add historical-shape replay fixtures and regression tests without
      mutating historical job records.
- [x] Run focused and full test suites (`299 focused passed`; `1396 passed,
      2 skipped` via `uv run --offline pytest -q`).
- [x] Validate with `openspec validate capture-provider-usage --strict`.
- [ ] Add verification evidence comment to bead `agent-crossbar-abe`; do not
      archive this change without Bo's approval.
