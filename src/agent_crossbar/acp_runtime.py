"""ACP runtime: direct official-SDK integration via acp_client."""

import asyncio
from datetime import datetime, timezone
from typing import Any

from agent_crossbar.acp_client import (
    NO_OUTPUT_SENTINEL,
    AcpError,
    AcpLaunchError,
    AcpProtocolError,
    AcpProviderUnavailableError,
    AcpResult,
    AcpTimeoutError,
    run_acp_prompt,
)
from agent_crossbar.envelope import build_result_envelope, sanitize_diagnostic_text
from agent_crossbar.models import Autonomy

DEFAULT_MAX_RUNTIME_SEC: int = 1800


def build_acp_agent_command(provider: str) -> list[str]:
    """Return the CLI command for a given ACP provider."""
    if provider == "opencode":
        return ["opencode", "acp"]
    if provider == "codex":
        return ["pnpm", "dlx", "@agentclientprotocol/codex-acp@1.1.7"]
    raise ValueError(f"Unknown ACP provider: {provider!r}")


def _safe_error(exc: Exception, prompt: str) -> str:
    """Return a safe error message with the prompt and any embedded secrets redacted.

    A provider exception can carry more than the prompt we sent it — its
    own stderr/env can leak into the message text — so this goes through
    the same secret-redaction pass as diagnostics, not just a prompt swap.
    """
    msg = str(exc)
    if prompt:
        msg = msg.replace(prompt, "[redacted]")
    return sanitize_diagnostic_text(msg)[:500]


def _count_events(store: Any, job_id: str) -> int:
    """Return highest event sequence number for a job; 0 on any error."""
    try:
        job = store.get_job(job_id)
        return int(job.events.last_seq) if job else 0
    except Exception:
        return 0


def _resolve_dev_session_mode(provider: str, task: str) -> str | None:
    """Return the provider-owned ACP session mode for *task*, if any.

    The decision of *which* mode value a "dev" task wants (e.g. OpenCode's
    "build") is owned by that provider's adapter (``dev_acp_mode``) — this
    stays provider-agnostic by only reading that declared preference. When an
    adapter declares a mode, ``run_acp_prompt`` requires live discovery and
    acceptance of that value before dispatching the prompt.
    """
    if task != "dev":
        return None
    from agent_crossbar.adapters.registry import get_adapter

    try:
        adapter = get_adapter(provider)
    except ValueError:
        return None
    return getattr(adapter, "dev_acp_mode", None)


def _is_empty_acp_output(output: str) -> bool:
    """Return True when *output* carries no observable assistant text."""
    return output == NO_OUTPUT_SENTINEL or not output.strip()


async def run_acp_job(
    store: Any,
    job_id: str,
    *,
    provider: str,
    prompt: str,
    cwd: str,
    model: str,
    task: str = "ask",
    effort: str | None = None,
    autonomy: str | Autonomy = Autonomy.READ_ONLY,
    max_runtime_sec: int | None = None,
) -> None:
    """Execute an ACP job via the official SDK and persist the result."""
    job = store.get_job(job_id)

    # -- creation timestamp --------------------------------------------------------
    meta = store._read_job_meta(job.path) if job else {}
    created_at = meta.get("created")

    started_at = datetime.now(timezone.utc).isoformat()

    # -- normalize autonomy --------------------------------------------------------
    if isinstance(autonomy, str):
        try:
            autonomy = Autonomy(autonomy)
        except ValueError:
            safe = _safe_error(AcpProtocolError(f"Invalid autonomy value: {autonomy!r}"), "")
            _fail(
                store=store,
                job_id=job_id,
                safe_output=safe,
                stop_reason="protocol_error",
                stage="preflight",
                code="acp_protocol_error",
                retryable=True,
                next_action="inspect_provider_and_protocol_logs",
                meta=meta,
                started_at=started_at,
                provider=provider,
                model=model,
                effort=effort,
                task=task,
                cwd=cwd,
                diagnostics={"error": safe},
            )
            return

    # -- build command & timeout ---------------------------------------------------
    command = build_acp_agent_command(provider)
    effective_timeout: int = max_runtime_sec or DEFAULT_MAX_RUNTIME_SEC

    # -- persist job meta (never prompt) -------------------------------------------
    assert job is not None
    store.update_job_meta(
        job_id,
        {
            **meta,
            "started_at": started_at,
            "backend": "acp",
            "acp_transport": "sdk_stdio",
            "provider": provider,
            "model": model,
            "effort": effort,
            "task": task,
            "autonomy": autonomy.value,
            "cwd": cwd,
            "max_runtime_sec": effective_timeout,
        },
    )

    # -- acp_command event (no prompt content) -------------------------------------
    store.send_event(
        job_id,
        level="info",
        type="acp_command",
        message="Starting ACP agent via SDK stdio",
        data={
            "argv": command,
            "prompt_bytes": len(prompt.encode("utf-8")),
            "timeout_sec": effective_timeout,
            "autonomy": autonomy.value,
        },
    )

    # -- run -----------------------------------------------------------------------
    try:

        def _record_acp_pid(pid: int) -> None:
            current = store._read_job_meta(job.path)
            store.update_job_meta(job_id, {**current, "acp_pid": pid})

        def _record_acp_text_delta(text: str) -> None:
            store.send_event(
                job_id,
                level="info",
                type="log_delta",
                message="ACP output received",
                data={"text": text},
            )

        def _record_acp_execution_heartbeat(data: dict[str, Any]) -> None:
            # This is deliberately transport/liveness evidence only.  ACP
            # does not expose a provider-native working state here.
            store.heartbeat_writer_lease(job_id)
            store.send_event(
                job_id,
                level="info",
                type="execution_heartbeat",
                message="ACP execution coroutine is still active",
                data=data,
            )

        result: AcpResult = await run_acp_prompt(
            command,
            prompt,
            cwd,
            timeout=effective_timeout,
            autonomy=autonomy,
            model=model,
            effort=effort,
            mode=_resolve_dev_session_mode(provider, task),
            on_process_start=_record_acp_pid,
            on_text_delta=_record_acp_text_delta,
            on_execution_heartbeat=_record_acp_execution_heartbeat,
        )
    except AcpProviderUnavailableError as exc:
        safe = _safe_error(exc, prompt)
        _fail(
            store=store,
            job_id=job_id,
            safe_output=safe,
            stop_reason="provider_unavailable",
            stage=getattr(exc, "stage", "prompt_delivery"),
            code=exc.code,
            retryable=True,
            next_action="choose_an_available_model_or_wait_for_quota_reset",
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"classification": exc.code},
        )
        return
    except AcpTimeoutError as exc:
        stage = getattr(exc, "stage", "execution")
        if stage == "prompt_delivery":
            safe = f"ACP prompt was not delivered before the {effective_timeout}s timeout"
            code = "acp_prompt_delivery_timeout"
            next_action = "inspect_provider_launch_and_retry"
        else:
            if provider == "opencode":
                safe = (
                    f"OpenCode did not complete within {effective_timeout}s. "
                    "The selected provider may be out of quota, rate-limited, "
                    "or temporarily unavailable; retry or choose an available free model."
                )
                next_action = "check_provider_limits_or_retry_with_free_model"
            else:
                safe = f"ACP job timed out after {effective_timeout}s"
                next_action = "retry_with_higher_timeout"
            code = "acp_timeout"
        _fail(
            store=store,
            job_id=job_id,
            safe_output=safe,
            stop_reason="timeout",
            stage=stage,
            code=code,
            retryable=True,
            next_action=next_action,
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"max_runtime_sec": effective_timeout},
        )
        return
    except AcpLaunchError as exc:
        safe = _safe_error(exc, prompt)
        _fail(
            store=store,
            job_id=job_id,
            safe_output=safe,
            stop_reason="launch_error",
            stage="launch",
            code="acp_launch_error",
            retryable=True,
            next_action=sanitize_diagnostic_text(f"install_or_repair_{provider}_acp"),
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"error": safe},
        )
        return
    except AcpProtocolError as exc:
        safe = _safe_error(exc, prompt)
        _fail(
            store=store,
            job_id=job_id,
            safe_output=safe,
            stop_reason="protocol_error",
            stage=getattr(exc, "stage", "execution"),
            code="acp_protocol_error",
            retryable=True,
            next_action="inspect_provider_and_protocol_logs",
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"error": safe},
        )
        return
    except AcpError as exc:
        safe = _safe_error(exc, prompt)
        _fail(
            store=store,
            job_id=job_id,
            safe_output=safe,
            stop_reason="execution_error",
            stage="execution",
            code="acp_error",
            retryable=False,
            next_action="inspect_logs",
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"error": safe},
        )
        return
    except asyncio.CancelledError:
        # Caller interruption (server shutdown, request cancellation) must not
        # leave a durable job orphaned. Persist a terminal failure before the
        # cancellation propagates; set_result is guarded so a concurrent stop or
        # a late provider completion still yields exactly one terminal result.
        # ``run_acp_prompt`` records its child PID through the callback after
        # this function's initial metadata snapshot, so refresh before trying
        # to clean up an interrupted provider process.
        current_meta = meta
        current_job = store.get_job(job_id)
        if current_job is not None:
            current_meta = store._read_job_meta(current_job.path)
        try:
            termination = safe_acp_termination(current_meta)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            # Persist the interruption even if child cleanup itself fails.
            termination = {
                "terminated": False,
                "reason": "termination_error",
                "error": type(exc).__name__,
                "pid": current_meta.get("acp_pid"),
            }
        _fail(
            store=store,
            job_id=job_id,
            safe_output=(
                "Job was interrupted before the provider produced a result. "
                "No partial result is claimed."
            ),
            stop_reason="cancelled",
            stage="execution",
            code="acp_interrupted",
            retryable=True,
            next_action="retry",
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"interrupted": True, "acp_stop": termination},
        )
        raise
    except Exception as exc:
        safe = _safe_error(exc, prompt)
        _fail(
            store=store,
            job_id=job_id,
            safe_output=safe,
            stop_reason="execution_error",
            stage="execution",
            code="acp_unexpected_error",
            retryable=False,
            next_action="inspect_logs",
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"error": safe},
        )
        return

    # -- fail closed: a "dev" job that produced no observable work ------------------
    # A successful dev turn may legitimately have concise nonempty text and an
    # externally empty `changes` field (Agents MCP does not inventory the
    # workspace) — that alone is not a failure signal. But whitespace-only or
    # entirely absent output from a dev task means nothing observable happened
    # at all, so it must never be reported as "completed".
    if task == "dev" and _is_empty_acp_output(result.output):
        safe = (
            "ACP dev task returned no observable output or changes. "
            "This is treated as a failure rather than a silent no-op completion."
        )
        _fail(
            store=store,
            job_id=job_id,
            safe_output=safe,
            stop_reason=result.stop_reason,
            stage="execution",
            code="acp_empty_result",
            retryable=True,
            next_action="retry_or_inspect_session_mode_and_prompt",
            meta=meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={
                "stop_reason": result.stop_reason,
                "native_session_id": getattr(result, "session_id", None),
                "output_len": len(result.output),
            },
        )
        return

    # -- success -------------------------------------------------------------------
    finished_at = datetime.now(timezone.utc).isoformat()

    store.send_event(
        job_id,
        level="info",
        type="acp_completed",
        message="ACP job completed successfully",
        data={
            "stop_reason": result.stop_reason,
            "session_id": getattr(result, "session_id", None),
        },
    )

    requested: dict[str, Any] = {
        "profile": provider,
        "model": model,
        "effort": effort,
        "task": task,
        "interactive": False,
        "cwd": cwd,
    }
    resolved: dict[str, Any] = {**requested, "backend": "acp"}

    envelope = build_result_envelope(
        status="completed",
        stop_reason=result.stop_reason,
        output=result.output,
        created_at=created_at or "",
        started_at=started_at,
        finished_at=finished_at,
        requested=requested,
        resolved=resolved,
        technical={
            "lifecycle_events": _count_events(store, job_id),
            "native_session_id": getattr(result, "session_id", None),
        },
    )

    store.set_result(job_id, ok=True, summary=result.output, envelope=envelope)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _fail(
    store: Any,
    job_id: str,
    safe_output: str,
    *,
    stop_reason: str,
    stage: str,
    code: str,
    retryable: bool,
    next_action: str,
    meta: dict[str, Any],
    started_at: str | None,
    provider: str,
    model: str | None,
    effort: str | None,
    task: str,
    cwd: str,
    diagnostics: dict[str, Any],
) -> None:
    """Persist a failure result.  Prompt is absent from all persisted data.

    Failure event data is kept minimal on the event (code, stop_reason,
    stage); diagnostics are only stored in the envelope for privacy.
    """
    finished_at = datetime.now(timezone.utc).isoformat()

    # Minimal event data — no diagnostics in the event log
    store.send_event(
        job_id,
        level="error",
        type="acp_failed",
        message=code,
        data={
            "code": code,
            "stop_reason": stop_reason,
            "stage": stage,
        },
    )

    created_at = meta.get("created")
    failed_started_at = meta.get("started_at", started_at)

    requested: dict[str, Any] = {
        "profile": provider,
        "model": model,
        "effort": effort,
        "task": task,
        "interactive": False,
        "cwd": cwd,
    }
    resolved: dict[str, Any] = {**requested, "backend": "acp"}

    envelope = build_result_envelope(
        status="failed",
        stop_reason=stop_reason,
        output=safe_output,
        created_at=created_at or "",
        started_at=failed_started_at,
        finished_at=finished_at,
        requested=requested,
        resolved=resolved,
        failure={
            "code": code,
            "retryable": retryable,
            "stage": stage,
            "next_action": next_action,
            "diagnostics": diagnostics,
        },
        technical={
            "lifecycle_events": _count_events(store, job_id),
            "native_session_id": None,
        },
    )

    store.set_result(job_id, ok=False, summary=safe_output, envelope=envelope)


def safe_acp_termination(meta: dict) -> dict:
    """Safely terminate an ACP job's child process.

    Reads the ACP process metadata stored during job creation and attempts
    to terminate the recorded child process cleanly (SIGTERM first, then
    SIGKILL after a grace period).  Idempotent — calling on an already-dead
    process is safe and returns a consistent result.

    Returns a dict with:
      - terminated: bool — whether the process was found and terminated
      - reason: str — human-readable outcome
      - pid: int | None — the process ID that was targeted
    """
    pid = meta.get("acp_pid")
    if pid is None:
        return {"terminated": False, "reason": "no_acp_pid_in_meta", "pid": None}

    import os
    import signal
    import time

    try:
        pid_int = int(pid)
    except (ValueError, TypeError):
        return {"terminated": False, "reason": f"invalid_acp_pid: {pid}", "pid": None}
    if pid_int <= 1:
        return {"terminated": False, "reason": "unsafe_acp_pid", "pid": pid_int}

    # Check if process exists
    try:
        os.kill(pid_int, 0)
    except OSError:
        return {"terminated": True, "reason": "process_already_gone", "pid": pid_int}

    # Try SIGTERM first
    try:
        os.kill(pid_int, signal.SIGTERM)
    except OSError:
        return {"terminated": True, "reason": "process_gone_during_terminate", "pid": pid_int}

    # Grace period, then SIGKILL
    time.sleep(1.0)
    try:
        os.kill(pid_int, 0)
        # Still alive — force kill
        os.kill(pid_int, signal.SIGKILL)
        return {"terminated": True, "reason": "force_killed_after_sigterm", "pid": pid_int}
    except OSError:
        return {"terminated": True, "reason": "terminated_via_sigterm", "pid": pid_int}
