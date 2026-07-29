## Context

`chatgpt_pro` is an existing macOS browser-based provider. The runner uses CUA/AX snapshots, clipboard-backed prompt delivery, a nonce response marker, and a Helium/Chrome/Safari fallback chain. The public MCP contract is provider-neutral and locked: `agent_start` must keep its current fields, `model` remains required, the eight-tool surface cannot change, and provider-specific routing/session fields must remain internal.

The current implementation has four lifecycle weaknesses. Readiness still describes the native desktop app and returns a degraded result that can reject `agent_start`; GUI jobs have no registered cancellation handle; browser selection is based on the currently discoverable window rather than a job-owned session identity; and completion relies on finding a nonce in page text rather than a stable visible response with DOM-health checks. Context packing and artifact registration are also not yet integrated with the generic `cwd`/`scope` inputs and durable job evidence.

The design ports bounded patterns from Agentify Desktop's keyed tab manager, stop-aware ChatGPT controller, context packer, and artifact store, plus `codex-chatgpt-web`'s fresh-turn isolation, explicit model/effort confirmation, prompt equality assertion, completion tracker, DOM-health tracker, and cancellable turn registry. It does not add either project as a runtime dependency or create a second browser gateway.

## Goals / Non-Goals

**Goals:**

- Make readiness truthful for the existing browser/CUA transport and keep the probe non-mutating and cached.
- Give every GUI job a generic internal cancellation handle that is effective before submit, during generation, and during cleanup.
- Bind a job to a verified browser window/session, serialize foreground CUA access, and retire a contaminated or ambiguous session.
- Make prompt delivery, model/effort selection, response completion, DOM health, fallback, and terminal result behavior explicit and fail-closed.
- Preserve the current public contract, current stable error semantics, current no-fallback-after-submit safety rule, and non-interactive ChatGPT Pro capability.
- Add deterministic bounded context preparation and secure job-local diagnostics that can be verified from `job_result` and persisted artifacts.
- Provide unit, mocked-CUA, and maintainer-only live evidence for every claimed ChatGPT Pro operation.

**Non-Goals:**

- No new MCP tools, public provider-specific fields, transport enum, browser profile selector, tab key, or storage-state field.
- No replacement of CUA/AX with Playwright, CDP, Electron, Responses/SSE, or an external Agentify daemon in this change.
- No ChatGPT Pro `interactive=true`/`job_send` support; that requires a separate session protocol and live gate.
- No hidden chain-of-thought capture; only visible status/commentary and final answer text may become job evidence.
- No static/default model catalog copied from another repository. Model/effort availability must be proven by live discovery or documented provider capability.

## Decisions

### 1. Use a generic cancellation registry, not a GUI branch in `JobStore`

`JobStore.stop_job` will keep its existing terminal-first behavior, then invoke a generic in-process run-handle registry keyed by `job_id`. The GUI runner registers a handle containing a cancellation event and a provider cleanup callback before execution becomes observable. Other transports can use the same interface; `jobs.py` must not branch on `chatgpt_pro`.

The runner checks cancellation between every CUA action and polling interval. After submission it attempts the visible ChatGPT Stop action, records whether it was clicked, stops reading the page, retires the session, and lets the existing `set_result` guard reject any late provider completion. A stop racing startup must be handled by checking durable job state before browser focus, prompt delivery, and session publication.

Alternative rejected: only marking metadata stopped. That prevents a late success from being persisted but leaves the provider generating and can leave the next job exposed to stale UI.

### 2. Start with one job-owned browser session at a time

Introduce an internal `ChatGptSessionManager` with a stable job/session identity, candidate/browser identity, window identity, lifecycle state, per-session lock, and last-used timestamp. Initial capacity remains one because foreground CUA and the system clipboard are process-global. The manager rejects an ambiguous or mismatched window rather than selecting an arbitrary foreground ChatGPT window.

Each turn gets a fresh conversation/page when the active browser can support it; otherwise the session is retired after a completed or contaminated turn. Parallel jobs are not enabled by removing the existing global lock. A future capacity increase requires isolated browser profiles or CDP sessions and its own live cross-contamination gate.

Alternative rejected: embedding Agentify's whole persistent tab service. That would duplicate browser ownership, authentication, rate limits, and MCP surfaces.

### 3. Keep readiness non-mutating and separate it from turn setup

The readiness probe only inspects macOS/browser process and visible window/auth/model evidence. It must not launch a browser, focus a window, click a picker, write the clipboard, or change ChatGPT state. It returns a truthful state and redacted evidence, cached for the existing 60-second TTL. Turn setup performs the mutating-but-in-scope actions after the job exists and owns the session.

If a browser session is absent or signed out, readiness returns an actionable auth/manual-gate state; it must not claim the native desktop app is required. If the session is ready but the requested model/effort is unavailable, the turn fails closed with a stable capability error rather than silently selecting another mode.

Alternative rejected: treating every macOS machine as ready. That hides browser/auth failures and violates the readiness contract.

### 4. Use an explicit turn state machine and two completion trackers

The runner records internal stages: `bootstrap`, `authenticated`, `model_selected`, `composer_ready`, `prompt_verified`, `submitted`, `streaming`, `complete`, `cancelled`, and `failed`. Stage changes become bounded job events and diagnostics, not new public fields.

Completion requires a visible response, non-running state, non-empty final text, visible completion action, and unchanged text for a short stability window. A DOM-health tracker fails if the response never appears, disappears for the configured grace period, or completes empty. Polling snapshots capture only visible text/status controls and redact diagnostic envelopes. Existing nonce extraction remains the final correlation check, not the sole completion signal.

Fallback remains allowed only before prompt submission. Once `prompt_submitted` is durable, the runner returns a status-unavailable or timeout failure and never sends the same prompt to another browser candidate.

Alternative rejected: accepting the first nonce found in page text. Echoed prompts, stale transcripts, partial markers, and a vanished response DOM can otherwise produce false completion.

### 5. Verify model and effort explicitly, without publishing defaults

The runner resolves the requested public `model` and optional `effort` against live visible picker labels/capabilities, clicks only when necessary, and confirms the rendered selection after the UI settles. It reports available choices in redacted diagnostics on mismatch. The profile continues to publish no model until discovery is implemented and backed by a live gate; no profile or test may silently choose a model.

Alternative rejected: copying a model ID or effort map from `codex-chatgpt-web`, because that catalog is not authoritative for this provider surface.

### 6. Add a generic bounded context packer and job-local artifact store

`cwd` and the existing generic `scope` input remain the only public context controls. A provider-neutral packer deterministically walks approved paths, skips symlinks and generated directories, sorts files, applies total/file/chunk limits, prioritizes useful repository metadata and source files, and produces a compact summary. Explicit missing paths fail before prompt submission; binary files are attached only when the CUA surface supports safe upload and size limits permit it.

GUI diagnostics are written below the job's artifact directory, with realpath containment and symlink/hardlink checks. The artifact manifest records kind, stage, browser/session identity, and redacted metadata. Clipboard contents, authentication material, raw context envelopes, and hidden reasoning are never persisted.

Alternative rejected: dumping the entire `cwd` into the prompt or storing artifacts under a provider-global directory without registration. Both increase leakage and make evidence difficult to attribute to a job.

## Risks / Trade-offs

- **[Risk]** ChatGPT UI labels or AX trees change. → Keep selectors layered, record bounded diagnostics, fail closed on missing confirmation, and require a live gate after selector changes.
- **[Risk]** Foreground CUA can still be disrupted by user interaction. → Serialize access, record the owned window identity, check it before every destructive action, and retire on mismatch.
- **[Risk]** A stop action may not be exposed by the current browser surface. → Return a distinct cancellation cleanup result, stop polling immediately, retire the session, and never claim that provider generation was stopped unless the UI action was confirmed.
- **[Risk]** Context packing can leak sensitive files. → Require explicit `cwd`/scope, skip symlinks, enforce budgets, redact diagnostics, and test path containment.
- **[Risk]** A live browser session is unavailable in CI. → Keep provider tests skip-safe and run the full surface gate only on a maintainer macOS session.
- **[Risk]** More lifecycle state increases compatibility surface. → Keep it internal/durable job evidence, preserve existing public error meanings, and add only documented minor-version error codes.
- **[Risk]** Directly copying external source brings license obligations. → Prefer clean-room ports of bounded algorithms; if exact code is copied, preserve MPL/MIT/third-party notices in the package review.

## Migration Plan

1. Land readiness and generic cancellation plumbing behind unit/mocked-CUA tests; keep `chatgpt_pro` non-interactive and serialized.
2. Add the session manager and explicit state machine without enabling parallel browser jobs.
3. Add completion/DOM-health trackers and replace marker-only success in the runner.
4. Add live model/effort confirmation, bounded context packing, and job-local artifact registration.
5. Run the ChatGPT Pro provider-surface gate on macOS: readiness variants, ask/review sentinel, cancellation after submit, timeout/challenge, model mismatch, draft isolation, and artifact evidence.
6. Roll back by reverting the change artifacts and implementation commit if the live gate fails; do not hot-patch production browser state.

## Open Questions

- Which exact ChatGPT UI label set and model identifiers are currently exposed by the maintainer's Pro account? This must be answered by live discovery before publishing profile models.
- Can the supported CUA browser surface reliably create a fresh ChatGPT conversation without opening an uncontrolled window? If not, session retirement must be the default isolation mechanism.
- Does the visible Stop action remain available and detectable in every supported browser candidate? The live gate must establish the supported cancellation claim.
- Should context packing be enabled for `ask`/`review` immediately, or first land as an internal helper with no automatic repo inclusion until scope semantics are finalized?

## Live findings (2026-07-28, authenticated macOS session)

These were measured on a real signed-in ChatGPT Pro session with
`cua-driver 0.10.0`, Helium as candidate one, and Safari also running. They
change several earlier assumptions and are the basis for decisions 7–10.

- **Background input works.** `set_value` on the ChatGPT composer element
  succeeds on a backgrounded Chromium window, and ChatGPT re-renders its
  `Send prompt` control, proving its editor state accepted the text. An AX
  `press` on that control submits the turn. A full turn (read → confirm model →
  write prompt → send → poll → complete) ran with the frontmost application
  sampled 50/50 times as the terminal: **zero focus theft, no clipboard use.**
- **The foreground path actively corrupts input.** Runs that used the
  clipboard/`delivery_mode: foreground` path produced mangled composer content
  (`You arre the aChatGPT Pro advisor…`) and left fragments of the user's own
  typing (`aс оту`) in the composer. Foreground key delivery interleaves with
  whatever the user types, damaging both sides.
- **A leftover draft blocks every later turn.** Because the runner refuses to
  overwrite a non-empty composer, one corrupted draft made every subsequent
  turn fail with `composer_not_empty` — the observed cause of a failed
  cancellation gate and a failed isolation gate.
- **The Stop control is labelled `Stop answering`.** A fixed label list missed
  it, so a running turn looked idle and cancellation could not confirm a
  provider stop.
- **All turns shared one conversation.** The runner reused whichever ChatGPT
  window it found, so every turn appended to the same thread — prior prompts
  and answers leaked into later turns' context.
- **Temporary Chat is reachable and detectable.** `https://chatgpt.com/?temporary-chat=true`
  opens a conversation that is not saved to history and exposes
  `AXHeading "Temporary Chat"` plus `AXButton (Turn off temporary chat)` in the
  AX tree — a verifiable per-turn isolation primitive.
- **Windows leaked.** Successive turns left three on-screen ChatGPT windows
  behind; nothing closed the windows the runner itself opened.
- **The GUI lock is not host-global.** `_acquire_chatgpt_lock` lives under the
  state directory, so two processes with different `AGENT_CROSSBAR_STATE_DIR`
  values (for example the provider gate and an ordinary server) can drive the
  same browser at once. That is how a second browser candidate was launched
  while an earlier turn was still generating.
- **No CDP route exists for attachments.** `get_browser_state` refuses Helium
  (`pid is not a recognized browser process`) and Safari
  (`no_attachable_runtime_endpoint`, WebKit), and `browser_prepare` demands an
  isolated profile — which by construction is not the user's signed-in profile.
  File attachment through `browser_set_input_files` is therefore unavailable on
  every supported candidate.

### 7. Every turn gets its own temporary conversation

Each turn opens `?temporary-chat=true` in a window the runner owns, confirms the
visible temporary indicator, uses it, and closes it. This removes conversation
carry-over, removes the "unrelated draft" class of failure entirely, and keeps
the user's own ChatGPT window untouched. Safari cannot open a controlled extra
window, so it stays a read/fallback-only candidate for this path.

Alternative rejected: clicking "New chat" in the existing window. It still
writes into the user's persistent history and depends on sidebar state.

### 8. Background accessibility input is the primary path

`set_value` + AX `press` become the default delivery and submission path. The
clipboard/foreground path stays only as a verified fallback, because it demonstrably
corrupts concurrent user input. Any partial text this turn wrote is cleared by
the turn itself; user content is never cleared or typed over.

### 9. Mutual exclusion must be host-global

The GUI turn lock moves out of the state directory to a fixed host-global path,
and no candidate fallback may launch another browser while a submitted turn is
still generating anywhere.

### 10. Attachments are refused, not silently dropped

Requested attachments fail closed before submission with
`attachment_upload_unsupported` and the detected capability reason. Inline text
context through `scope.paths` remains the supported route.
