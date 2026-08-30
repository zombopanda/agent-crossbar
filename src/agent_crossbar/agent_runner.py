"""Agent job lifecycle runner using provider adapters.

Background monitoring, result normalization, and finalization for jobs
launched through the adapter registry (Claude bg, future providers).
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

from agent_crossbar.adapters.base import LifecycleAdapter
from agent_crossbar.adapters.claude import LocalSubprocessRunner
from agent_crossbar.adapters.claude_model_probe import strip_ansi
from agent_crossbar.envelope import build_result_envelope, sanitize_diagnostic_text

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PROVIDER_LIMIT_MARKERS = (
    "monthly spend limit",
    "usage limit",
    "usage-credits",
    "quota exceeded",
    "rate limit",
)
_PROVIDER_AUTH_MARKERS = ("not logged in", "please run /login", "authentication required")
_HEARTBEAT_INTERVAL_SEC = 30.0


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
    deadline: float = started_monotonic + effective_max_runtime

    last_logs = ""
    last_heartbeat_at: float | None = None
    last_lease_heartbeat_at = started_monotonic
    _consecutive_idle = 0  # counter for working+idle detection

    try:
        while True:
            if deadline and time.monotonic() > deadline:
                finished_at = datetime.now(timezone.utc).isoformat()
                store.send_event(
                    job_id,
                    level="error",
                    type="timeout",
                    message=f"Job exceeded max runtime of {effective_max_runtime}s",
                )
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
                        },
                    },
                    technical={
                        "lifecycle_events": _count_lifecycle_events(store, job_id),
                        "native_session_id": session_id,
                        "native_full_session_id": meta.get("native_full_session_id"),
                    },
                )
                store.set_result(
                    job_id,
                    ok=False,
                    summary=f"max_runtime_sec ({effective_max_runtime}s) exceeded",
                    envelope=envelope,
                )
                return

            status = adapter.status(runner, session_id)
            native_state = status.get("state", "unknown")

            now = time.monotonic()
            if now - last_lease_heartbeat_at >= _HEARTBEAT_INTERVAL_SEC:
                store.heartbeat_writer_lease(job_id)
                last_lease_heartbeat_at = now

            # Persist full session ID on every poll, not just terminal
            full_session_id = status.get("session_id")
            if full_session_id:
                store.update_job_meta(job_id, {"native_full_session_id": full_session_id})

            if native_state == "done" and meta.get("interactive"):
                interactive_meta = store._read_job_meta(store.get_job(job_id).path)
                if interactive_meta.get("status") != "awaiting_input":
                    store.update_job_meta(
                        job_id,
                        {"status": "awaiting_input", "waiting_for": "follow_up"},
                    )
                    store.send_event(
                        job_id,
                        level="info",
                        type="awaiting_input",
                        message="Claude completed a turn and is awaiting follow-up input",
                        data={"waiting_for": "follow_up", "native_state": native_state},
                    )
                time.sleep(poll_interval_sec)
                continue

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
                if not meta_blocked.get("interactive"):
                    logs = adapter.get_logs(runner, session_id)
                    clean_logs = _clean_provider_logs(logs)
                    provider_limited = _provider_limit_detected(logs)
                    provider_auth_failed = _provider_auth_failure_detected(logs)
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
                            },
                        },
                        technical={
                            "lifecycle_events": _count_lifecycle_events(store, job_id),
                            "native_session_id": session_id,
                            "native_full_session_id": meta_blocked.get("native_full_session_id"),
                        },
                    )
                    store.set_result(
                        job_id,
                        ok=False,
                        summary=public_summary,
                        envelope=envelope,
                    )
                    return
                else:
                    # Interactive job blocked → awaiting_input
                    meta_i = store._read_job_meta(store.get_job(job_id).path)
                    meta_i["status"] = "awaiting_input"
                    meta_i["waiting_for"] = status.get("waiting_for")
                    store._write_job_meta(store.get_job(job_id).path, meta_i)
                    store.send_event(
                        job_id,
                        level="info",
                        type="awaiting_input",
                        message=f"Job awaiting input: {status.get('waiting_for', 'unknown')}",
                        data={"waiting_for": status.get("waiting_for")},
                    )
                    return

            # Provider logs are append-only in the normal case.  Polling them
            # here makes output available through the existing job event
            # cursor, without changing the public MCP surface.
            try:
                current_logs = adapter.get_logs(runner, session_id)
                if current_logs:
                    delta, last_logs = _incremental_log_delta(last_logs, current_logs)
                    clean_delta = _clean_provider_logs(delta)
                    if clean_delta:
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

        store.set_result(
            job_id,
            ok=env_status in ("completed", "waiting"),
            summary=normalized.output,
            envelope=envelope,
        )

        if normalized.error:
            store.send_event(
                job_id,
                level="error",
                type=normalized.stop_reason or "execution_error",
                message=sanitize_diagnostic_text(normalized.error),
                data={"stop_reason": normalized.stop_reason, "error_stage": normalized.error_stage},
            )

    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        store.send_event(
            job_id,
            level="error",
            type="monitor_failure",
            message=f"Monitor thread failed: {exc}",
        )
        meta_exc = store._read_job_meta(store.get_job(job_id).path)
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
                },
            },
            technical={
                "lifecycle_events": _count_lifecycle_events(store, job_id),
                "native_session_id": session_id,
                "native_full_session_id": meta_exc.get("native_full_session_id"),
            },
        )
        store.set_result(job_id, ok=False, summary=str(exc), envelope=envelope)


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
    thread = threading.Thread(
        target=_run_adapter_job,
        kwargs={
            "store": store,
            "job_id": job_id,
            "adapter": adapter,
            "session_id": session_id,
            "poll_interval_sec": poll_interval_sec,
            "max_runtime_sec": max_runtime_sec,
        },
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
