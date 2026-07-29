"""ACP runtime: direct official-SDK integration via acp_client."""

from datetime import datetime, timezone
from typing import Any

from agent_crossbar.acp_client import (
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

        result: AcpResult = await run_acp_prompt(
            command,
            prompt,
            cwd,
            timeout=effective_timeout,
            autonomy=autonomy,
            model=model,
            effort=effort,
            on_process_start=_record_acp_pid,
            on_text_delta=_record_acp_text_delta,
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
