"""Agent job lifecycle runner using provider adapters.

Background monitoring, result normalization, and finalization for jobs
launched through the adapter registry (Claude bg, future providers).
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_crossbar.adapters.base import LifecycleAdapter
from agent_crossbar.adapters.claude_model_probe import strip_ansi
from agent_crossbar.envelope import build_result_envelope, sanitize_diagnostic_text
from agent_crossbar.redaction import redact_secrets
from agent_crossbar.run_handles import run_handles
from agent_crossbar.subprocess_runner import LocalSubprocessRunner
from agent_crossbar.tmux_output import (
    interactive_tmux_output_complete_since,
    interactive_tmux_output_summary,
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PROVIDER_LIMIT_MARKERS = (
    "monthly spend limit",
    "usage limit",
    "usage-credits",
    "quota exceeded",
    "rate limit",
)
_PROVIDER_AUTH_MARKERS = ("not logged in", "please run /login", "authentication required")
_PROVIDER_API_RETRY_RE = re.compile(r"\bapi\s+error\b.*\bretrying\b", re.IGNORECASE | re.DOTALL)
_HEARTBEAT_INTERVAL_SEC = 30.0
# A provider status probe is allowed to block independently of the lifecycle
# deadline (for example, a CLI subprocess whose descendants keep stdout open).
# Keep that probe bounded so the deadline watchdog can claim the job even when
# the monitor thread is stuck inside provider code.
_STATUS_PROBE_TIMEOUT_SEC = 15.0
_CLEANUP_PROBE_TIMEOUT_SEC = 15.0
# The owner question is recovered from the tmux transcript at full fidelity,
# bounded well above the 2 KiB diagnostics-sanitizer truncation so a long
# question is not cut to a prefix.
_QUESTION_MAX_CHARS = 16384


def _clean_provider_logs(logs: str) -> str:
    """Remove terminal control data, redact secrets, and bound public output."""
    plain = strip_ansi(logs).replace("\r", "\n")
    plain = _CONTROL_CHAR_RE.sub("", plain)
    return sanitize_diagnostic_text(plain.strip())


def _incremental_log_delta(previous: str, current: str) -> tuple[str, str]:
    """Return unseen text and the accumulated observation cursor.

    Provider log commands normally return a growing transcript, but some
    providers return a bounded tail. Comparing only prefixes turns a sliding
    tail into repeated output. Preserve the longest suffix/prefix overlap so
    every byte observed by the monitor is emitted once.
    """
    if not current or current == previous:
        return "", previous
    if current.startswith(previous):
        return current[len(previous) :], current
    if previous.endswith(current):
        return "", previous

    for overlap in range(min(len(previous), len(current)), 0, -1):
        if previous.endswith(current[:overlap]):
            delta = current[overlap:]
            return delta, previous + delta

    # A log reset has no reliable shared cursor. Emit the new snapshot once.
    return current, current


def _provider_limit_detected(logs: str) -> bool:
    lowered = strip_ansi(logs).lower()
    return any(marker in lowered for marker in _PROVIDER_LIMIT_MARKERS)


def _provider_auth_failure_detected(logs: str) -> bool:
    lowered = strip_ansi(logs).lower()
    return any(marker in lowered for marker in _PROVIDER_AUTH_MARKERS)


def _provider_api_retry_detected(logs: str) -> bool:
    """Recognize a provider retry marker without treating it as completion."""
    return bool(_PROVIDER_API_RETRY_RE.search(strip_ansi(logs)))


def _cancel_provider(
    adapter: LifecycleAdapter, runner: LocalSubprocessRunner, session_id: str
) -> dict[str, Any]:
    """Request native cleanup and return bounded, truthful evidence."""
    try:
        confirmed = bool(adapter.cancel(runner, session_id))
        return {"adapter_cancel_requested": True, "adapter_cancel_confirmed": confirmed}
    except Exception as exc:
        return {
            "adapter_cancel_requested": True,
            "adapter_cancel_confirmed": False,
            "adapter_cancel_error": type(exc).__name__,
        }


def _bounded_call(call: Any, timeout_sec: float) -> tuple[Any, BaseException | None, bool]:
    """Run an adapter call without letting it block its lifecycle watchdog.

    The worker is deliberately a daemon: a provider CLI can outlive a timed
    out monitor, but it must not prevent durable terminalization.  Exceptions
    are returned to the caller so existing monitor-failure semantics remain
    intact; a timeout is represented separately and never guessed as a
    provider result.
    """
    result: list[Any] = []
    error: list[BaseException] = []
    finished = threading.Event()

    def invoke() -> None:
        try:
            result.append(call())
        except Exception as exc:  # preserve provider exception type
            error.append(exc)
        finally:
            finished.set()

    threading.Thread(target=invoke, name="agents-provider-probe", daemon=True).start()
    if not finished.wait(max(float(timeout_sec), 0.001)):
        return None, None, True
    if error:
        return None, error[0], False
    return (result[0] if result else None), None, False


def _terminalize_runtime_deadline(
    store: Any,
    job_id: str,
    adapter: LifecycleAdapter,
    *,
    session_id: str,
    effective_max_runtime: int,
    watchdog: bool,
) -> bool:
    """Claim and publish a runtime timeout, independent of monitor progress."""
    job = store.get_job(job_id)
    if job is None:
        return False
    terminalization_owner = "deadline_watchdog" if watchdog else "deadline_monitor"
    claim = store.claim_terminalization(
        job_id,
        reason="max_runtime_exceeded",
        owner=terminalization_owner,
    )
    if not claim.get("ok"):
        return False
    meta = claim.get("meta") or store._read_job_meta(job.path)
    started_at = meta.get("started_at") or meta.get("created")
    if not started_at:
        started_at = datetime.now(timezone.utc).isoformat()
    created_at = meta.get("created", started_at)

    cleanup, cleanup_error, cleanup_timed_out = _bounded_call(
        lambda: _cancel_provider(adapter, LocalSubprocessRunner(), session_id),
        _CLEANUP_PROBE_TIMEOUT_SEC,
    )
    # Keep the callback evidence transport-neutral and truthful even if the
    # provider's stop command itself hangs.
    if cleanup_timed_out:
        cleanup = {
            "adapter_cancel_requested": True,
            "adapter_cancel_confirmed": False,
            "adapter_cancel_error": "timeout",
        }
    elif cleanup_error is not None:
        cleanup = {
            "adapter_cancel_requested": True,
            "adapter_cancel_confirmed": False,
            "adapter_cancel_error": type(cleanup_error).__name__,
        }
    if not isinstance(cleanup, dict):
        cleanup = {
            "adapter_cancel_requested": True,
            "adapter_cancel_confirmed": False,
            "adapter_cancel_error": "invalid_cleanup_result",
        }
    cleanup_confirmed = bool(cleanup.get("adapter_cancel_confirmed"))
    cleanup["watchdog"] = bool(watchdog)
    if cleanup_timed_out:
        cleanup["cleanup_probe_timeout_sec"] = _CLEANUP_PROBE_TIMEOUT_SEC
    lease_disposition = "released" if cleanup_confirmed else "retained_cleanup_unconfirmed"
    finished_at = datetime.now(timezone.utc).isoformat()
    event_data = {
        "max_runtime_sec": effective_max_runtime,
        "watchdog": bool(watchdog),
        "provider_cleanup": cleanup,
        "cleanup_confirmed": cleanup_confirmed,
        "lease_disposition": lease_disposition,
    }
    telemetry_failure: dict[str, Any] | None = None
    try:
        timeout_event = store.send_event(
            job_id,
            level="error",
            type="timeout",
            message=f"Job exceeded max runtime of {effective_max_runtime}s",
            data=event_data,
        )
        if isinstance(timeout_event, dict) and not timeout_event.get("ok", True):
            telemetry_failure = {
                "event": "timeout",
                "error_type": "event_write_failed",
            }
    except Exception as exc:
        # The durable claim must never strand a job when telemetry storage is
        # unavailable.  Keep only the exception class as bounded diagnostics;
        # envelope/result persistence and lease disposition remain authoritative.
        telemetry_failure = {
            "event": "timeout",
            "error_type": type(exc).__name__,
        }
    envelope = build_result_envelope(
        status="failed",
        stop_reason="max_runtime_exceeded",
        output=f"max_runtime_sec ({effective_max_runtime}s) exceeded",
        created_at=created_at,
        started_at=started_at,
        finished_at=finished_at,
        requested={
            "profile": meta.get("profile"),
            "model": meta.get("model"),
            "effort": meta.get("effort"),
            "task": meta.get("task"),
            "interactive": meta.get("interactive", False),
            "cwd": meta.get("cwd"),
        },
        resolved={
            "profile": meta.get("profile"),
            "model": meta.get("model"),
            "effort": meta.get("effort"),
            "task": meta.get("task"),
            "interactive": meta.get("interactive", False),
            "backend": meta.get("backend"),
            "cwd": meta.get("cwd"),
        },
        failure={
            "stage": "execution",
            "code": "max_runtime_exceeded",
            "retryable": True,
            "next_action": "retry_with_higher_timeout",
            "diagnostics": {
                "layer": "timeout",
                "max_runtime_sec": effective_max_runtime,
                "watchdog": bool(watchdog),
                "provider_cleanup": cleanup,
                **({"telemetry_failure": telemetry_failure} if telemetry_failure else {}),
            },
        },
        technical={
            "lifecycle_events": _count_lifecycle_events(store, job_id),
            "native_session_id": session_id,
            "native_full_session_id": meta.get("native_full_session_id"),
            "watchdog": bool(watchdog),
            "provider_cleanup": cleanup,
            "cleanup_confirmed": cleanup_confirmed,
            "cleanup_pending": not cleanup_confirmed,
            "lease_disposition": lease_disposition,
            **({"telemetry_failure": telemetry_failure} if telemetry_failure else {}),
        },
    )
    stored = store.set_result(
        job_id,
        ok=False,
        summary=f"max_runtime_sec ({effective_max_runtime}s) exceeded",
        envelope=envelope,
        release_writer_lease=cleanup_confirmed,
        terminalization_owner=terminalization_owner,
    )
    if not stored.get("ok"):
        return False
    # set_result marks cleanup_pending for an unconfirmed stop.  Keep the
    # positive/negative evidence and lease disposition explicit in metadata
    # for independent readers and later retry reconciliation.
    store.update_job_meta(
        job_id,
        {
            "cleanup_confirmed": cleanup_confirmed,
            "cleanup_pending": not cleanup_confirmed,
            "lease_disposition": lease_disposition,
            "cleanup_last_result": cleanup,
        },
    )
    return True


def _native_turn_evidence(status: dict[str, Any]) -> str | None:
    """Return an adapter-provided per-turn identity, when available.

    ``state=done`` is a session state and can remain unchanged while a native
    client dispatches a follow-up.  Adapters that expose a turn/update
    generation can prove the new turn directly; absent that evidence, the
    runner requires a completion marker appended after the durable input
    boundary and never promotes a repeated ``done`` observation by timeout.
    """
    for key in (
        "turn_generation",
        "turn_id",
        "generation",
        "updated_at_ms",
        "updated_at",
        "activity_id",
    ):
        value = status.get(key)
        if value is not None and str(value):
            return f"{key}:{value}"
    return None


def _extract_full_question(meta: dict[str, Any]) -> str | None:
    """Recover the current on-screen prompt from the untruncated tmux
    transcript — the actual question/permission text Claude is showing,
    not a bare native status label or a diagnostics-sanitizer-truncated
    prefix of provider logs.

    Uses the same tail-anchored transcript window
    (``interactive_tmux_output_summary``) already relied on to summarize a
    completed interactive job, so a long question is preserved from the
    end of the visible screen rather than cut off at a byte-limited prefix.
    """
    tmux_output_path = meta.get("tmux_output_path")
    if not tmux_output_path:
        return None
    try:
        raw = Path(tmux_output_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not raw.strip():
        return None
    summary = interactive_tmux_output_summary(raw, profile="claude", max_chars=_QUESTION_MAX_CHARS)
    redacted, _ = redact_secrets(summary)
    redacted = redacted.strip()
    return redacted or None


def _count_lifecycle_events(store: Any, job_id: str) -> int:
    """Count events in a job's events.jsonl — cheap derivation from disk."""
    job = store.get_job(job_id)
    if job is None:
        return 0
    try:
        return job.events.last_seq
    except Exception:
        return 0


def _run_adapter_job(
    store: Any,
    job_id: str,
    adapter: LifecycleAdapter,
    *,
    session_id: str,
    poll_interval_sec: float = 2.0,
    max_runtime_sec: int | None = None,
    deadline_monotonic: float | None = None,
) -> None:
    """Background worker: poll adapter status until terminal, then finalize."""
    runner = LocalSubprocessRunner()

    started_at = datetime.now(timezone.utc).isoformat()
    meta = store._read_job_meta(store.get_job(job_id).path)
    created_at = meta.get("created", started_at)
    store.update_job_meta(job_id, {"started_at": started_at})

    _DEFAULT_MAX_RUNTIME_SEC = 1800
    effective_max_runtime = (
        max_runtime_sec if max_runtime_sec is not None else _DEFAULT_MAX_RUNTIME_SEC
    )
    started_monotonic = time.monotonic()
    deadline: float = deadline_monotonic or (started_monotonic + effective_max_runtime)

    last_logs = ""
    last_heartbeat_at: float | None = None
    last_lease_heartbeat_at = started_monotonic
    _consecutive_idle = 0  # counter for working+idle detection
    api_retry_reported = False

    # Stale-done guard: after a job_send reply resumes an interactive job
    # from awaiting_input, the native CLI can briefly still report the
    # previous turn's ``done`` state.  A repeated state is never promoted by
    # elapsed polls or transcript growth alone.  Completion requires either a
    # native per-turn/update identity change or a provider completion marker
    # appended after the byte boundary recorded before job_send.
    saw_activity_since_resume = True
    resume_transcript_baseline: int | None = None
    resume_native_evidence: str | None = None
    last_native_evidence: str | None = None
    try:
        last_input_generation = int(meta.get("input_generation", 0))
    except (TypeError, ValueError):
        last_input_generation = 0

    try:
        while True:
            current_job = store.get_job(job_id)
            if current_job is None:
                return
            lifecycle_snapshot = store._read_job_meta(current_job.path)
            if (
                lifecycle_snapshot.get("status") not in {"running", "awaiting_input", None, ""}
                or lifecycle_snapshot.get("terminalization_state") == "claimed"
            ):
                # A watchdog/reaper has claimed terminalization, or another
                # lifecycle owner already published a result.  Do not probe,
                # cancel, or emit a second terminal event.
                return
            if deadline and time.monotonic() > deadline:
                _terminalize_runtime_deadline(
                    store,
                    job_id,
                    adapter,
                    session_id=session_id,
                    effective_max_runtime=effective_max_runtime,
                    watchdog=False,
                )
                return

            # Keep the probe bound independent of the remaining runtime.  A
            # call may begin just before the deadline and remain blocked; the
            # watchdog handles the deadline while this probe reports a useful
            # provider-stall event when its own interval expires.
            status_probe_timeout = _STATUS_PROBE_TIMEOUT_SEC
            status, status_error, status_timed_out = _bounded_call(
                lambda: adapter.status(runner, session_id),
                status_probe_timeout,
            )
            if status_timed_out:
                store.send_event(
                    job_id,
                    level="error",
                    type="provider_status_timeout",
                    message="Provider status probe exceeded its bounded interval",
                    data={
                        "timeout_sec": status_probe_timeout,
                        "max_runtime_sec": effective_max_runtime,
                    },
                )
                # The independent watchdog owns deadline terminalization. If
                # it has not won yet, continue with a fresh bounded probe.
                if store.job_status(job_id) not in {"running", "awaiting_input"}:
                    return
                time.sleep(min(poll_interval_sec, max(deadline - time.monotonic(), 0.001)))
                continue
            if status_error is not None:
                store.send_event(
                    job_id,
                    level="error",
                    type="provider_status_error",
                    message="Provider status probe failed",
                    data={"error_type": type(status_error).__name__},
                )
                raise status_error
            if not isinstance(status, dict):
                raise TypeError("Provider status probe returned a non-object")
            native_state = status.get("state", "unknown")
            native_evidence = _native_turn_evidence(status)

            now = time.monotonic()
            if now - last_lease_heartbeat_at >= _HEARTBEAT_INTERVAL_SEC:
                store.heartbeat_writer_lease(job_id)
                last_lease_heartbeat_at = now

            # Persist full session ID on every poll, not just terminal
            full_session_id = status.get("session_id")
            if full_session_id:
                store.update_job_meta(job_id, {"native_full_session_id": full_session_id})

            current_meta_snapshot = store._read_job_meta(store.get_job(job_id).path)
            current_durable_status = current_meta_snapshot.get("status")
            if (
                current_durable_status not in {"running", "awaiting_input", None, ""}
                or current_meta_snapshot.get("terminalization_state") == "claimed"
            ):
                return
            try:
                current_input_generation = int(current_meta_snapshot.get("input_generation", 0))
            except (TypeError, ValueError):
                current_input_generation = last_input_generation
            generation_changed_this_poll = current_input_generation != last_input_generation
            if generation_changed_this_poll:
                # ``send_user_input`` always increments ``input_generation``
                # before (in a separate lock acquisition) transitioning the
                # status away from ``awaiting_input``.  Detecting the resume
                # boundary here — rather than also independently off the
                # status transition itself — avoids observing the same
                # boundary twice from two different signals that can land on
                # different polls: a second, delayed reset from a stale
                # ``last_seen_status`` read would wipe out a legitimate
                # ``working`` observation already recorded for this resume
                # and strand the job on every subsequent ``done`` reading.
                last_input_generation = current_input_generation
                saw_activity_since_resume = False
                raw_baseline = current_meta_snapshot.get("input_transcript_baseline")
                try:
                    resume_transcript_baseline = int(raw_baseline)
                except (TypeError, ValueError):
                    resume_transcript_baseline = None
                resume_native_evidence = last_native_evidence

            if native_state == "working":
                saw_activity_since_resume = True

            if native_state == "done" and not saw_activity_since_resume:
                if (
                    resume_native_evidence is not None
                    and native_evidence is not None
                    and native_evidence != resume_native_evidence
                ):
                    saw_activity_since_resume = True
                elif resume_transcript_baseline is not None:
                    output_path = current_meta_snapshot.get("tmux_output_path")
                    if output_path:
                        try:
                            current_output = Path(output_path).read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except OSError:
                            current_output = ""
                        if interactive_tmux_output_complete_since(
                            current_output,
                            baseline_bytes=resume_transcript_baseline,
                            profile=str(
                                current_meta_snapshot.get("profile") or meta.get("profile") or ""
                            ),
                        ):
                            saw_activity_since_resume = True

            if native_state == "done" and not saw_activity_since_resume:
                # The native CLI has not yet caught up to the owner's reply,
                # and the transcript has not grown since the resume; this
                # "done" reading almost certainly describes the prior
                # (already-answered) turn, not the new one. Keep polling
                # instead of finalizing the wrong turn's output.
                store.send_event(
                    job_id,
                    level="info",
                    type="stale_done_ignored",
                    message=(
                        "Native session reported done immediately after a "
                        "resume with no transcript growth; treating as a "
                        "stale pre-reply reading"
                    ),
                    data={"native_state": native_state},
                )
                time.sleep(poll_interval_sec)
                continue

            # A native ``done`` state while the durable job is still waiting
            # for owner input belongs to the previous turn.  Do not finalize
            # before the owner reply's generation is durably recorded.
            if native_state == "done" and current_durable_status == "awaiting_input":
                time.sleep(poll_interval_sec)
                continue

            last_native_evidence = native_evidence

            if native_state in ("done", "failed", "stopped"):
                break

            # Log output and source changes are not a liveness signal: a
            # provider can spend minutes reasoning without producing either.
            # Emit a sparse, provider-neutral heartbeat from its native status
            # so a supervising client never has to infer a stall from silence.
            if native_state == "working" and (
                last_heartbeat_at is None or now - last_heartbeat_at >= _HEARTBEAT_INTERVAL_SEC
            ):
                store.send_event(
                    job_id,
                    level="info",
                    type="provider_heartbeat",
                    message="Provider still reports working",
                    data={
                        "native_state": native_state,
                        "process_status": status.get("process_status"),
                        "elapsed_sec": int(now - started_monotonic),
                    },
                )
                last_heartbeat_at = now

            # ── Claude idle-after-completion detection ──────────────────
            # Claude may report state=working, process_status=idle even
            # after producing a complete final response in the logs.
            # Track consecutive unchanged idle polls and safely finalize
            # when the logs contain a recoverable final response.
            if native_state == "working" and status.get("process_status") == "idle":
                try:
                    idle_logs = adapter.get_logs(runner, session_id)
                except Exception:
                    idle_logs = last_logs
                if idle_logs == last_logs:
                    # Logs haven't changed since the previous poll → truly idle
                    _consecutive_idle += 1
                else:
                    _consecutive_idle = 0
                    last_logs = idle_logs

                if _consecutive_idle >= 2:
                    # Three consecutive idle polls with unchanged logs and a final
                    # response in logs → treat as terminal completion.
                    from agent_crossbar.adapters.claude import _screen_reader_final_response

                    final_text = _screen_reader_final_response(_clean_provider_logs(idle_logs))
                    if final_text:
                        store.send_event(
                            job_id,
                            level="info",
                            type="idle_finalized",
                            message=(
                                "Claude session idle with complete final response; "
                                "finalizing as completed"
                            ),
                            data={
                                "consecutive_idle_polls": _consecutive_idle,
                                "native_state": native_state,
                                "process_status": status.get("process_status"),
                            },
                        )
                        # Mutate status dict so the terminal block below sees
                        # this as a completed session.
                        status = dict(status)
                        status["state"] = "done"
                        break
            elif native_state != "working" or status.get("process_status") != "idle":
                _consecutive_idle = 0

            if native_state == "blocked":
                meta_blocked = store._read_job_meta(store.get_job(job_id).path)
                # Claude can keep reporting the previous native ``blocked``
                # state after a valid owner reply.  A post-reply transcript
                # with the provider's terminal footer is stronger evidence
                # than that stale state, but only when the durable job has
                # resumed and the completion parser rejects later busy text.
                if meta_blocked.get("interactive") and current_durable_status != "awaiting_input":
                    raw_baseline = current_meta_snapshot.get("input_transcript_baseline")
                    try:
                        transcript_baseline = int(raw_baseline)
                    except (TypeError, ValueError):
                        transcript_baseline = None
                    output_path = current_meta_snapshot.get("tmux_output_path")
                    current_output = ""
                    if transcript_baseline is not None and output_path:
                        try:
                            current_output = Path(output_path).read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except OSError:
                            current_output = ""
                    if transcript_baseline is not None and interactive_tmux_output_complete_since(
                        current_output,
                        baseline_bytes=transcript_baseline,
                        profile=str(
                            current_meta_snapshot.get("profile") or meta.get("profile") or ""
                        ),
                    ):
                        store.send_event(
                            job_id,
                            level="info",
                            type="blocked_output_finalized",
                            message=(
                                "Native session remained blocked, but the post-reply "
                                "transcript contained a complete provider response"
                            ),
                            data={
                                "native_state": native_state,
                                "transcript_baseline_bytes": transcript_baseline,
                                "evidence": "post_reply_transcript_and_terminal_footer",
                            },
                        )
                        status = dict(status)
                        status["state"] = "done"
                        break
                if generation_changed_this_poll:
                    # ``status()`` ran before the owner reply's durable
                    # generation became visible.  Do not turn that stale
                    # blocked reading into a second awaiting_input state;
                    # re-poll once so a real prompt raised by the new turn
                    # can still be surfaced on the next observation.
                    store.send_event(
                        job_id,
                        level="info",
                        type="stale_blocked_ignored",
                        message=(
                            "Native session reported blocked while the owner reply "
                            "generation changed; treating it as stale pre-reply state"
                        ),
                        data={"native_state": native_state},
                    )
                    time.sleep(poll_interval_sec)
                    continue
                if not meta_blocked.get("interactive"):
                    logs = adapter.get_logs(runner, session_id)
                    clean_logs = _clean_provider_logs(logs)
                    provider_limited = _provider_limit_detected(logs)
                    provider_auth_failed = _provider_auth_failure_detected(logs)
                    waiting_for = status.get("waiting_for")
                    if not waiting_for and not provider_auth_failed and not provider_limited:
                        # A provider can briefly report ``blocked`` while its
                        # TUI/session is still starting. Without a concrete
                        # waiting prompt or auth/quota evidence, terminating
                        # it would turn harmless startup noise into a false
                        # blocked failure.
                        store.send_event(
                            job_id,
                            level="info",
                            type="blocked_unconfirmed",
                            message="Provider reported blocked without a prompt or failure evidence",
                            data={"native_state": native_state},
                        )
                        time.sleep(poll_interval_sec)
                        continue
                    if provider_auth_failed:
                        public_summary = "Claude is not authenticated. Start Claude and run /login."
                        failure_code = "provider_needs_auth"
                        failure_stage = "auth"
                        retryable = False
                        next_action = "authenticate_provider"
                        stop_reason = "provider_needs_auth"
                        provider_diagnostic = "provider reported that authentication is required"
                    elif provider_limited:
                        public_summary = (
                            "Claude is unavailable because its subscription or organization "
                            "usage limit is exhausted. Check Claude /usage or retry after reset."
                        )
                        failure_code = "provider_limit_exhausted"
                        failure_stage = "execution"
                        retryable = True
                        next_action = "check_provider_limits_or_retry_after_reset"
                        stop_reason = "provider_limit_exhausted"
                        provider_diagnostic = (
                            "provider reported a subscription or organization usage limit"
                        )
                    else:
                        public_summary = clean_logs or "Provider is waiting for interactive input."
                        failure_code = "blocked_noninteractive"
                        failure_stage = "execution"
                        retryable = False
                        next_action = "retry_with_interactive_mode"
                        stop_reason = "blocked"
                        provider_diagnostic = clean_logs
                    finished_at = datetime.now(timezone.utc).isoformat()
                    store.send_event(
                        job_id,
                        level="error",
                        type="blocked",
                        message=f"Non-interactive job blocked: {status.get('waiting_for', 'unknown')}",
                        data={
                            "waiting_for": status.get("waiting_for"),
                            "native_state": native_state,
                        },
                    )
                    blocked_cleanup = _cancel_provider(adapter, runner, session_id)
                    envelope = build_result_envelope(
                        status="failed",
                        stop_reason=stop_reason,
                        output=public_summary,
                        created_at=created_at,
                        started_at=started_at,
                        finished_at=finished_at,
                        requested={
                            "profile": meta_blocked.get("profile"),
                            "model": meta_blocked.get("model"),
                            "effort": meta_blocked.get("effort"),
                            "task": meta_blocked.get("task"),
                            "interactive": meta_blocked.get("interactive", False),
                            "cwd": meta_blocked.get("cwd"),
                        },
                        resolved={
                            "profile": meta_blocked.get("profile"),
                            "model": meta_blocked.get("model"),
                            "effort": meta_blocked.get("effort"),
                            "task": meta_blocked.get("task"),
                            "interactive": meta_blocked.get("interactive", False),
                            "backend": meta_blocked.get("backend"),
                            "cwd": meta_blocked.get("cwd"),
                        },
                        failure={
                            "stage": failure_stage,
                            "code": failure_code,
                            "retryable": retryable,
                            "next_action": next_action,
                            "diagnostics": {
                                "waiting_for": status.get("waiting_for"),
                                "native_state": native_state,
                                "provider_output": provider_diagnostic,
                                "provider_cleanup": blocked_cleanup,
                            },
                        },
                        technical={
                            "lifecycle_events": _count_lifecycle_events(store, job_id),
                            "native_session_id": session_id,
                            "native_full_session_id": meta_blocked.get("native_full_session_id"),
                            "provider_cleanup": blocked_cleanup,
                        },
                    )
                    store.set_result(
                        job_id,
                        ok=False,
                        summary=public_summary,
                        envelope=envelope,
                        release_writer_lease=bool(blocked_cleanup.get("adapter_cancel_confirmed")),
                    )
                    return
                else:
                    # Interactive job blocked → awaiting_input. The monitor
                    # never exits here: it keeps polling under the one
                    # deadline established at thread start, so a job_send
                    # reply is observed on the very next poll with no
                    # restart and no deadline reset. Re-entering this
                    # branch while already awaiting_input (still blocked on
                    # the same prompt) is idempotent — no duplicate
                    # transition/event.
                    already_awaiting = meta_blocked.get("status") == "awaiting_input"
                    waiting_for = status.get("waiting_for")
                    if not waiting_for and not already_awaiting:
                        # A provider can briefly report blocked while its
                        # session is still starting, with no concrete prompt
                        # evidence yet. Without that evidence, declaring
                        # awaiting_input would turn harmless startup noise
                        # into a false pause.
                        store.send_event(
                            job_id,
                            level="info",
                            type="blocked_unconfirmed",
                            message="Provider reported blocked without a prompt or failure evidence",
                            data={"native_state": native_state},
                        )
                        time.sleep(poll_interval_sec)
                        continue
                    if not already_awaiting:
                        question = _extract_full_question(meta_blocked)
                        updates: dict[str, Any] = {"waiting_for": waiting_for}
                        if question:
                            updates["question"] = question
                        transitioned = store.transition_job_status(
                            job_id,
                            "awaiting_input",
                            allowed_from={"running", "awaiting_input"},
                            updates=updates,
                            # ``native_state`` came from ``adapter.status()``
                            # at the top of this iteration; a concurrent
                            # ``job_send`` can durably record a reply (and
                            # resume the job) in the gap between that call and
                            # this write.  The generation CAS guard makes the
                            # comparison atomic with the write itself: it
                            # rejects the transition rather than overwriting
                            # a legitimate resume with stale "blocked"
                            # evidence for an already-answered turn, which
                            # would otherwise strand the job in
                            # ``awaiting_input`` with no pending question left
                            # for the owner to reply to.
                            #
                            # The guard compares the *durable* value read at
                            # the top of this iteration (``None`` for a job
                            # that has never been sent input) rather than the
                            # ``0``-normalized local, so an absent key still
                            # compares equal to itself instead of spuriously
                            # failing every transition with
                            # ``stale_transition``.
                            require_meta={
                                "input_generation": current_meta_snapshot.get("input_generation")
                            },
                        )
                        if not transitioned.get("ok"):
                            if transitioned.get("error") == "stale_transition":
                                time.sleep(poll_interval_sec)
                                continue
                            # A concurrent stop is terminal and must remain so.
                            return
                        event_data: dict[str, Any] = {"waiting_for": waiting_for}
                        if question:
                            event_data["question"] = question
                        store.send_event(
                            job_id,
                            level="info",
                            type="awaiting_input",
                            message=f"Job awaiting input: {waiting_for or 'unknown'}",
                            data=event_data,
                        )
                    time.sleep(poll_interval_sec)
                    continue

            # Provider logs are append-only in the normal case.  Polling them
            # here makes output available through the existing job event
            # cursor, without changing the public MCP surface.
            try:
                current_logs = adapter.get_logs(runner, session_id)
                if current_logs:
                    delta, last_logs = _incremental_log_delta(last_logs, current_logs)
                    clean_delta = _clean_provider_logs(delta)
                    if clean_delta:
                        if not api_retry_reported and _provider_api_retry_detected(clean_delta):
                            store.send_event(
                                job_id,
                                level="warn",
                                type="provider_api_retry",
                                message="Provider reported an API retry; monitoring continues",
                                data={"native_state": native_state},
                            )
                            api_retry_reported = True
                        store.send_event(
                            job_id,
                            level="info",
                            type="log_delta",
                            message="Incremental provider output",
                            data={"text": clean_delta},
                        )
            except Exception:
                # A transient logs failure must not interrupt the job monitor.
                # Keep the previous cursor so the next successful poll emits
                # every log byte it can recover.
                pass

            time.sleep(poll_interval_sec)

        # Terminal state reached
        terminal_snapshot = store._read_job_meta(store.get_job(job_id).path)
        if (
            terminal_snapshot.get("status") not in {"running", "awaiting_input", None, ""}
            or terminal_snapshot.get("terminalization_state") == "claimed"
        ):
            return
        logs = adapter.get_logs(runner, session_id)
        normalized = adapter.normalize_result(status, logs)
        finished_at = datetime.now(timezone.utc).isoformat()
        meta_terminal = store._read_job_meta(store.get_job(job_id).path)

        # Map adapter status to envelope status
        env_status = normalized.status  # completed | failed | cancelled | waiting

        # Build failure for native failures and adapter-level finalization
        # failures (for example, a native "done" session whose final response
        # cannot be recovered from provider logs).
        failure: dict[str, Any] | None = None
        if native_state == "failed":
            failure = {
                "stage": "execution",
                "code": "native_failed",
                "retryable": True,
                "next_action": "inspect_logs",
                "diagnostics": {
                    "output": logs[-2048:] if logs else "",
                    "native_state": native_state,
                },
            }
        elif normalized.status == "failed":
            failure = {
                "stage": normalized.error_stage or "finalization",
                "code": normalized.stop_reason or "provider_result_invalid",
                "retryable": True,
                "next_action": "inspect_logs_and_retry",
                "diagnostics": {
                    "output": logs[-2048:] if logs else "",
                    "native_state": native_state,
                },
            }

        envelope = build_result_envelope(
            status=env_status,
            stop_reason=normalized.stop_reason or native_state,
            output=normalized.output,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            requested={
                "profile": meta_terminal.get("profile"),
                "model": meta_terminal.get("model"),
                "effort": meta_terminal.get("effort"),
                "task": meta_terminal.get("task"),
                "interactive": meta_terminal.get("interactive", False),
                "cwd": meta_terminal.get("cwd"),
            },
            resolved={
                "profile": meta_terminal.get("profile"),
                "model": meta_terminal.get("model"),
                "effort": meta_terminal.get("effort"),
                "task": meta_terminal.get("task"),
                "interactive": meta_terminal.get("interactive", False),
                "backend": meta_terminal.get("backend"),
                "cwd": meta_terminal.get("cwd"),
            },
            failure=failure,
            technical={
                "lifecycle_events": _count_lifecycle_events(store, job_id),
                "native_session_id": session_id,
                "native_full_session_id": meta_terminal.get("native_full_session_id"),
            },
        )

        stored = store.set_result(
            job_id,
            ok=env_status in ("completed", "waiting"),
            summary=normalized.output,
            envelope=envelope,
        )
        if not stored.get("ok"):
            return

        if normalized.error:
            store.send_event(
                job_id,
                level="error",
                type=normalized.stop_reason or "execution_error",
                message=sanitize_diagnostic_text(normalized.error),
                data={"stop_reason": normalized.stop_reason, "error_stage": normalized.error_stage},
            )

    except Exception as exc:
        meta_exc = store._read_job_meta(store.get_job(job_id).path)
        if (
            meta_exc.get("status") not in {"running", "awaiting_input", None, ""}
            or meta_exc.get("terminalization_state") == "claimed"
        ):
            return
        cleanup = _cancel_provider(adapter, runner, session_id)
        finished_at = datetime.now(timezone.utc).isoformat()
        store.send_event(
            job_id,
            level="error",
            type="monitor_failure",
            message=f"Monitor thread failed: {exc}",
        )
        envelope = build_result_envelope(
            status="failed",
            stop_reason="monitor_failure",
            output=str(exc),
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            requested={
                "profile": meta_exc.get("profile"),
                "model": meta_exc.get("model"),
                "effort": meta_exc.get("effort"),
                "task": meta_exc.get("task"),
                "interactive": meta_exc.get("interactive", False),
                "cwd": meta_exc.get("cwd"),
            },
            resolved={
                "profile": meta_exc.get("profile"),
                "model": meta_exc.get("model"),
                "effort": meta_exc.get("effort"),
                "task": meta_exc.get("task"),
                "interactive": meta_exc.get("interactive", False),
                "backend": meta_exc.get("backend"),
                "cwd": meta_exc.get("cwd"),
            },
            failure={
                "stage": "finalization",
                "code": "monitor_failure",
                "retryable": False,
                "next_action": "inspect_monitor_logs",
                "diagnostics": {
                    "exception_type": type(exc).__name__,
                    "provider_cleanup": cleanup,
                },
            },
            technical={
                "lifecycle_events": _count_lifecycle_events(store, job_id),
                "native_session_id": session_id,
                "native_full_session_id": meta_exc.get("native_full_session_id"),
                "provider_cleanup": cleanup,
            },
        )
        store.set_result(
            job_id,
            ok=False,
            summary=str(exc),
            envelope=envelope,
            release_writer_lease=bool(cleanup.get("adapter_cancel_confirmed")),
        )


def start_agent_job(
    store: Any,
    job_id: str,
    adapter: LifecycleAdapter,
    *,
    session_id: str,
    poll_interval_sec: float = 2.0,
    max_runtime_sec: int | None = None,
) -> threading.Thread:
    """Start background monitoring for an adapter-launched job."""

    def on_cancel() -> dict[str, Any]:
        return _cancel_provider(adapter, LocalSubprocessRunner(), session_id)

    # Make native adapter cleanup available to provider-neutral job_stop and
    # deadline reaping. The callback is bounded and returns evidence only.
    run_handles.register(job_id, on_cancel=on_cancel)

    _DEFAULT_MAX_RUNTIME_SEC = 1800
    effective_max_runtime = (
        max_runtime_sec if max_runtime_sec is not None else _DEFAULT_MAX_RUNTIME_SEC
    )
    # This deadline belongs to the background job, not to the monitor's next
    # successful status poll.  A separate daemon watchdog therefore remains
    # able to terminalize a job while the monitor is blocked in provider code.
    deadline = time.monotonic() + effective_max_runtime
    watchdog_stop = threading.Event()

    def enforce_deadline() -> None:
        if watchdog_stop.wait(max(deadline - time.monotonic(), 0.0)):
            return
        _terminalize_runtime_deadline(
            store,
            job_id,
            adapter,
            session_id=session_id,
            effective_max_runtime=effective_max_runtime,
            watchdog=True,
        )

    watchdog_thread = threading.Thread(
        target=enforce_deadline,
        name=f"agents-deadline-watchdog-{job_id}",
        daemon=True,
    )
    watchdog_thread.start()

    def run_and_release() -> None:
        try:
            _run_adapter_job(
                store=store,
                job_id=job_id,
                adapter=adapter,
                session_id=session_id,
                poll_interval_sec=poll_interval_sec,
                max_runtime_sec=max_runtime_sec,
                deadline_monotonic=deadline,
            )
        finally:
            watchdog_stop.set()
            watchdog_thread.join(timeout=1.0)
            run_handles.release(job_id)

    thread = threading.Thread(
        target=run_and_release,
        name=f"agents-adapter-{job_id}",
        daemon=True,
    )
    thread.start()
    return thread


def monitor_agent_job(
    store: Any,
    job_id: str,
    adapter: LifecycleAdapter,
    poll_interval_sec: float = 2.0,
) -> None:
    """Synchronous monitor — polls until terminal, then finalizes.

    Used by tests; production code uses start_agent_job for background monitoring.
    """
    meta = store._read_job_meta(store.get_job(job_id).path)
    session_id = meta.get("native_session_id") or ""
    if not session_id:
        store.set_result(job_id, ok=False, summary="No native session id in job meta")
        return

    _run_adapter_job(
        store,
        job_id,
        adapter,
        session_id=session_id,
        poll_interval_sec=poll_interval_sec,
        max_runtime_sec=None,
    )
