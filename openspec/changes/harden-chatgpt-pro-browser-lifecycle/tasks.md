## 1. Contract and readiness foundation

- [x] 1.1 Add unit fixtures for browser-based ChatGPT readiness states, redacted evidence, unsupported OS, and the existing 60-second cache policy. (`tests/test_readiness.py::TestChatGPTProBrowserReadiness`, `FakeCuaRunner`)
- [x] 1.2 Replace the native-desktop readiness wording with a non-mutating browser/CUA probe that never launches, focuses, clicks, types, or changes the clipboard. (`readiness.check_chatgpt_pro_readiness`; the probe only calls `list_apps`, `list_windows`, `get_window_state`, asserted by `test_probe_only_uses_read_only_tools`)
- [x] 1.3 Define the live evidence needed to report authenticated Pro readiness and return actionable non-ready remediation without publishing a default model. (`ready` requires a signed-in composer; new codes `chatgpt_pro_browser_not_running`, `chatgpt_pro_window_not_found`, `chatgpt_pro_not_authenticated`, `chatgpt_pro_readiness_unverified`, `chatgpt_pro_browser_probe_failed`, `chatgpt_pro_cua_driver_missing`)
- [x] 1.4 Add contract tests proving `agent_start` fields, the eight-tool surface, `model` requiredness, and provider-neutral boundaries remain unchanged. (`tests/test_minimal_public_contract.py`, incl. new no-provider-fields / generic-scope / transport-neutral-stop tests)

## 2. Generic cancellation and terminal lifecycle

- [x] 2.1 Introduce a generic in-process job run-handle registry with registration, lookup, cancellation, idempotent release, and bounded cleanup. (`src/agent_crossbar/run_handles.py`, `tests/test_run_handles.py`)
- [x] 2.2 Integrate `JobStore.stop_job` with the generic run handle after durable terminal-first metadata write, without adding a `chatgpt_pro` branch to `jobs.py`.
- [x] 2.3 Pass a cancellation event/handle from `start_gui_job` through `run_gui_job` and `run_gui_request` to every CUA action and polling wait.
- [x] 2.4 Implement GUI stop cleanup: detect and click the visible ChatGPT Stop action when available, stop polling, retire the session, and record whether provider stop was confirmed. (`_chatgpt_request_stop`, `provider_stop_confirmed`)
- [x] 2.5 Add race tests for stop-before-start, stop-during-submit, stop-during-generation, repeated stop, and late provider completion. (`tests/test_gui_cancellation.py`)

## 3. Job-scoped browser session manager

- [x] 3.1 Add an internal `ChatGptSessionManager` with job/session identity, browser candidate, window identity, lifecycle state, per-session lock, and retirement callback. (`src/agent_crossbar/gui_lifecycle.py`)
- [x] 3.2 Replace arbitrary foreground-window reuse with explicit window identity verification before model picker, composer, paste, Send, Stop, and response reads. (`owned()` guard before authentication/selection and again before Send; an unreadable window list is recorded, a real mismatch/closure/ambiguity fails closed)
- [x] 3.3 Keep initial GUI capacity at one and preserve safe busy behavior; add tests for serialization, closed-window detection, mismatch, and contaminated-session retirement.
- [~] 3.4 Implement fresh-turn preparation where the supported CUA surface permits it, otherwise retire the session after a completed or untrusted turn. (Partial: every turn ends with `sessions.retire`, and a window holding an unrelated draft is replaced with a freshly opened one. **Superseded by 8.5** — live testing disproved the original assumption: `?temporary-chat=true` gives a controlled fresh conversation, so retirement alone is not enough and turns still shared one conversation.)
- [x] 3.5 Preserve the existing no-fallback-after-submit rule and add mocked-CUA tests proving no duplicate submission to another browser candidate.

## 4. Turn state machine and prompt/model verification

- [x] 4.1 Define internal turn stages and bounded event payloads for bootstrap, authentication, model selection, composer readiness, prompt verification, submission, streaming, completion, cancellation, and failure. (`TurnLifecycle`; emitted as bounded `turn_stage` job events, single-owner terminal stage)
- [x] 4.2 Add exact live model/effort selection and post-selection confirmation using the visible ChatGPT picker; fail closed with redacted available-choice diagnostics. (`_chatgpt_verify_selection`; `model_not_available` / `effort_not_available`. When `effort` is requested but the UI exposes no effort control the diagnostics record `effort=control_absent` — the published support matrix declares `effort_support=False`, so this is not turned into a failure.)
- [x] 4.3 Keep the current safe composer behavior and add structured prompt mismatch diagnostics with expected length, actual length, and common-prefix length. (`prompt_mismatch_diagnostics`, `prompt_verification_failed`)
- [x] 4.4 Add unit and mocked-CUA tests for exact prompt preservation, non-empty draft protection, model mismatch, effort mismatch, picker disappearance, and pre-submit fallback.

## 5. Stable completion and DOM health

- [x] 5.1 Implement a completion tracker requiring response presence, non-running state, non-empty final text, visible completion action, and a configurable text stability interval. (`CompletionTracker`, `AGENT_CROSSBAR_CHATGPT_STABILITY_SEC`)
- [x] 5.2 Implement a DOM-health tracker for missing response DOM, vanished response DOM, and terminal empty response with bounded grace periods. (`DomHealthTracker`)
- [x] 5.3 Replace marker-only success polling with the completion/DOM-health trackers plus nonce correlation and explicit timeout/status-unavailable outcomes. (When an AX snapshot is momentarily unreadable the turn falls back to the correlated closed nonce marker plus the stability window and records `ax_unreadable_polls`, instead of failing every such turn.)
- [x] 5.4 Emit bounded heartbeat and visible progress events without persisting hidden chain-of-thought or unredacted UI envelopes. (`redact_gui_evidence` strips context envelopes and visible reasoning markers from every persisted artifact)
- [x] 5.5 Add deterministic tests for stable completion, changing text, stale marker, vanished DOM, empty completion, long generation heartbeat, timeout, and duplicate terminal events. (`tests/test_gui_lifecycle.py`, `tests/test_runner.py`)

## 6. Context preparation and secure artifacts

- [x] 6.1 Extend the generic context module with deterministic path walking, symlink/generated-directory exclusion, file/chunk/total budgets, priority ordering, and inclusion/omission summaries. (`context.pack_context`, `tests/test_context_pack.py`)
- [x] 6.2 Thread the existing generic `cwd`/`scope` semantics into the internal GUI request without adding provider-specific public fields. (`server.agent_start` sets `req["scope"]`; packing only happens when the scope explicitly names paths/attachments, so no automatic repo inclusion was introduced.)
- [~] 6.3 Add safe explicit attachment validation and bounded binary upload handling for browser surfaces that support it; fail before Send on missing or over-budget inputs. (Validation and fail-closed budgets are implemented — `attachment_missing`, `attachment_too_large`, `attachment_symlink`, `attachment_outside_cwd`, all rejected before any browser work. The actual upload is **not** implemented: the current CUA surface has no verified file-picker path, so no attachment is ever handed to ChatGPT and no upload acceptance is claimed.)
- [x] 6.4 Move ChatGPT GUI diagnostics under the owning job artifact directory and enforce realpath containment plus symlink/hardlink rejection. (`JobArtifactStore`; `run_gui_job` passes the job directory)
- [x] 6.5 Add redaction and artifact-manifest tests for clipboard data, credentials, context envelopes, hidden reasoning markers, unsafe paths, and bounded AX/page evidence.

## 7. Provider gate and integration verification

- [x] 7.1 Extend `scripts/provider_surface_gate.py` with ChatGPT Pro cases for browser readiness, exact ask/review sentinel, model/effort confirmation, and artifact evidence. (`--chatgpt-lifecycle`)
- [x] 7.2 Add a maintainer-only cancellation gate that submits a long-running prompt, calls `job_stop`, verifies provider stop/retirement evidence, and confirms no late success after a grace period. (`_run_chatgpt_cancellation_case`)
- [x] 7.3 Add a maintainer-only isolation gate with an existing unrelated draft/window and sequential jobs, proving no draft overwrite or transcript cross-contamination. (`_run_chatgpt_isolation_case`; the draft-overwrite refusal itself is covered by the existing `composer_not_empty` unit test.)
- [~] 7.4 Run the full unit/lint/package verification, then the authenticated macOS ChatGPT Pro gate; record exact commands and durable evidence. (Done: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q` → 1049 passed, 11 skipped. **Not done:** the live authenticated macOS gate — it needs a signed-in ChatGPT Pro browser window and a maintainer desktop session, so no live `job_result`/artifact evidence is recorded yet. Run `uv run --extra test python scripts/provider_surface_gate.py --profile chatgpt_pro --model pro --task ask --chatgpt-lifecycle` before claiming the provider hardened.)
- [x] 7.5 Update stable error-code documentation and release notes for any newly introduced lifecycle/capability errors, then review the change for public package hygiene and external-license notices. (README error table + CHANGELOG `[Unreleased]`. No external code was copied from Agentify Desktop or `codex-chatgpt-web` — every ported pattern is a clean-room implementation, so no third-party license notice is required.)

## 8. Temporary per-turn sessions and background-only input (from live findings)

- [x] 8.1 Match the live `Stop answering` control family so a running turn is never read as idle and cancellation can confirm a provider stop.
- [x] 8.2 Make background accessibility input (`set_value` + AX `press`) the primary delivery path; keep the clipboard/foreground path as a recorded fallback only.
- [x] 8.3 Clear only the partial text this turn wrote when delivery or pre-Send verification fails, so a corrupted draft cannot block every later turn.
- [x] 8.4 Stop raising the browser window when its page is already exposed over accessibility (window discovery and model picking).
- [x] 8.5 Open a dedicated ChatGPT Temporary Chat (`?temporary-chat=true`) per turn, confirm the visible `Temporary Chat` indicator before attaching the prompt, and fail closed when it cannot be confirmed.
- [x] 8.6 Close the window the runner opened when the turn reaches any terminal state; never close a window the runner did not open; add a window-leak regression check.
- [x] 8.7 Move GUI mutual exclusion to a host-global lock path so two processes with different state directories cannot drive the same browser.
- [x] 8.8 Forbid launching or falling back to another browser candidate while any submitted turn is still generating.
- [ ] 8.9 Add live gates for: conversation isolation between two sequential turns, zero frontmost-application change during a turn, clipboard untouched, and no window accumulation.
