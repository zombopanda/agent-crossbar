"""ACP runtime: direct official-SDK integration via acp_client."""

import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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
from agent_crossbar.run_handles import run_handles

DEFAULT_MAX_RUNTIME_SEC: int = 1800


def build_acp_agent_command(provider: str) -> list[str]:
    """Return the CLI command for a given ACP provider."""
    if provider == "opencode":
        return ["opencode", "acp"]
    if provider == "codex":
        return ["pnpm", "dlx", "@agentclientprotocol/codex-acp@1.1.7"]
    raise ValueError(f"Unknown ACP provider: {provider!r}")


def _owned_acp_command(command: list[str]) -> list[str]:
    """Run ACP through a repository-owned process-group wrapper."""
    if os.name != "posix":
        return command
    return [sys.executable, "-m", "agent_crossbar.acp_process_wrapper", *command]


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

    # A stop may win after the durable job is created but before the
    # fire-and-forget coroutine starts. Never launch a provider for a terminal
    # job; its result must remain owned by the caller that stopped it.
    if job is None:
        return
    initial_meta = store._read_job_meta(job.path)
    if initial_meta.get("status", "running") not in {"running", ""}:
        return

    # -- creation timestamp --------------------------------------------------------
    meta = initial_meta
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
    provider_command = build_acp_agent_command(provider)
    command = _owned_acp_command(provider_command)
    effective_timeout: int = max_runtime_sec or DEFAULT_MAX_RUNTIME_SEC

    # -- persist job meta (never prompt) -------------------------------------------
    assert job is not None
    store.update_job_meta(
        job_id,
        {
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

    # Register before the final startup fence.  RunHandleRegistry preserves a
    # stop that arrived before registration as a cancelled tombstone; the
    # worker-owned startup lock never blocks the synchronous stop path.
    def _acp_stop_evidence() -> dict[str, Any]:
        current = store._read_job_meta(job.path)
        return {"acp_stop": safe_acp_termination(current)}

    handle = run_handles.register(job_id, on_cancel=_acp_stop_evidence)

    def _before_process_start() -> None:
        current = store._read_job_meta(job.path)
        if handle.cancelled or current.get("status", "running") not in {"running", ""}:
            raise AcpProtocolError(
                "ACP job was stopped before provider process startup",
                stage="prompt_delivery",
            )

    if handle.cancelled or store._read_job_meta(job.path).get("status", "running") not in {
        "running",
        "",
    }:
        run_handles.release(job_id)
        return

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

    # Persist launch intent before entering the SDK.  A controller crash
    # between provider spawn and the PID callback must remain recoverable as an
    # unknown cleanup, rather than looking like a proven no-launch job.
    store.update_job_meta(
        job_id,
        {
            "acp_launch_pending": True,
            "acp_launch_started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    spawn_observed = False
    captured_pid: int | None = None
    captured_identity: dict[str, Any] = {}

    def _record_acp_pid(pid: int) -> None:
        nonlocal captured_identity, captured_pid, spawn_observed

        # The SDK callback runs after the OS process exists.  Keep an
        # in-memory receipt before any durable operation so a metadata write
        # failure cannot turn a post-spawn error into a pre-spawn result.
        spawn_observed = True
        captured_pid = pid
        captured_identity = {}
        # Capture and persist the immutable launch identity before checking
        # cancellation.  A stop can arrive after the child exists but before
        # this callback runs; rejecting first would lose the PID/PGID and
        # falsely turn an unverified cleanup into a confirmed pre-spawn stop.
        identity = _capture_process_identity(pid)
        captured_identity = dict(identity)
        store.update_job_meta(
            job_id,
            {
                "acp_pid": pid,
                "acp_pgid": identity.get("pgid"),
                "acp_process_start": identity.get("start"),
                "acp_process_command": identity.get("command"),
                "acp_launch_pending": False,
            },
        )
        current = store._read_job_meta(job.path)
        # ``stop_job`` requests cancellation before it publishes the
        # durable terminal status.  The in-memory fence therefore has to
        # participate in this callback too; otherwise a stop that lands
        # during the awaited spawn can pass the status check and reach the
        # provider prompt.
        if handle.cancelled or current.get("status", "running") not in {"running", ""}:
            raise AcpProtocolError(
                "ACP job was stopped before provider startup completed",
                stage="prompt_delivery",
            )

    def _before_prompt() -> None:
        current = store._read_job_meta(job.path)
        if handle.cancelled or current.get("status", "running") not in {"running", ""}:
            raise AcpProtocolError(
                "ACP job was stopped before prompt delivery",
                stage="prompt_delivery",
            )

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

    try:
        result: AcpResult = await run_acp_prompt(
            command,
            prompt,
            cwd,
            timeout=effective_timeout,
            autonomy=autonomy,
            model=model,
            effort=effort,
            mode=_resolve_dev_session_mode(provider, task),
            startup_lock=handle.startup_lock,
            before_process_start=_before_process_start,
            on_process_start=_record_acp_pid,
            before_prompt=_before_prompt,
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
        try:
            store.update_job_meta(job_id, {"acp_launch_pending": False})
            meta["acp_launch_pending"] = False
        except Exception:
            # If the marker cannot be cleared, _fail retains the lease.
            pass
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
        startup_cancel_cleanup = None
        current_meta = store._read_job_meta(job.path)
        stop_observed = handle.cancelled or current_meta.get("status", "running") not in {
            "running",
            "",
        }
        if spawn_observed:
            cleanup_meta = dict(current_meta)
            if captured_pid is not None:
                cleanup_meta["acp_pid"] = captured_pid
            for key in ("acp_pgid", "acp_process_start", "acp_process_command"):
                if key in captured_identity:
                    cleanup_meta[key] = captured_identity[key]
            try:
                startup_cancel_cleanup = safe_acp_termination(cleanup_meta)
            except Exception as cleanup_exc:  # pragma: no cover - defensive cleanup
                startup_cancel_cleanup = {
                    "terminated": False,
                    "reason": "termination_error",
                    "error": type(cleanup_exc).__name__,
                    "pid": captured_pid,
                }
            meta = cleanup_meta
        elif stop_observed:
            try:
                store.update_job_meta(job_id, {"acp_launch_pending": False})
                meta["acp_launch_pending"] = False
            except Exception:
                pass
            if current_meta.get("acp_pid"):
                # The process identity is persisted before the startup fence
                # rejects a raced stop.  Reap it through the same fenced
                # helper and retain the lease whenever disappearance is not
                # positively confirmed.
                try:
                    startup_cancel_cleanup = safe_acp_termination(current_meta)
                except Exception as cleanup_exc:  # pragma: no cover - defensive cleanup
                    startup_cancel_cleanup = {
                        "terminated": False,
                        "reason": "termination_error",
                        "error": type(cleanup_exc).__name__,
                        "pid": current_meta.get("acp_pid"),
                    }
                meta = current_meta
            else:
                startup_cancel_cleanup = {
                    "terminated": True,
                    "reason": "startup_cancelled_before_spawn",
                    "pid": None,
                }
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
            cleanup=startup_cancel_cleanup,
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
            meta=current_meta,
            started_at=started_at,
            provider=provider,
            model=model,
            effort=effort,
            task=task,
            cwd=cwd,
            diagnostics={"interrupted": True, "acp_stop": termination},
            cleanup=termination,
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

    # -- fail closed: a job that produced no observable assistant output -----------
    # ACP uses a sentinel when the provider emits no session_update chunks.  An
    # empty ask/review result is just as unusable as an empty dev turn: callers
    # would otherwise receive a successful completion with no evidence that the
    # requested operation happened.  A dev turn may still have an externally
    # empty `changes` field (Agents MCP does not inventory the workspace), so the
    # output check is deliberately independent of the changes envelope.
    if _is_empty_acp_output(result.output):
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
    run_handles.release(job_id)


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
    cleanup: dict[str, Any] | None = None,
) -> None:
    """Persist a failure result.  Prompt is absent from all persisted data.

    Failure event data is kept minimal on the event (code, stop_reason,
    stage); diagnostics are only stored in the envelope for privacy.
    """
    # The caller's startup snapshot predates the durable launch-intent marker
    # and may also predate a PID identity write. Refresh it before deciding
    # whether the writer lease can be released. If the state cannot be read,
    # retain the lease conservatively: a failed read is not proof that launch
    # never happened.
    try:
        current_job = store.get_job(job_id)
        current_meta = store._read_job_meta(current_job.path) if current_job else {}
    except Exception:
        current_meta = {}
    if current_meta:
        refreshed_meta = dict(meta)
        refreshed_meta.update(current_meta)
        meta = refreshed_meta
    elif "acp_launch_pending" not in meta or not meta.get("acp_launch_pending"):
        meta = {**meta, "acp_launch_pending": True}

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

    technical: dict[str, Any] = {
        "lifecycle_events": _count_events(store, job_id),
        "native_session_id": meta.get("acp_pid"),
    }
    if cleanup is not None:
        technical["provider_cleanup"] = cleanup
        technical["cleanup_confirmed"] = bool(cleanup.get("terminated"))

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
        technical=technical,
    )

    release_writer_lease = cleanup is None or bool(cleanup.get("terminated"))
    if meta.get("acp_launch_pending"):
        # A crash/error after launch intent was durable but before the
        # identity was durably cleared is not proof that no provider exists.
        # Keep the lease until a later cleanup attempt confirms disappearance.
        release_writer_lease = bool(cleanup and cleanup.get("terminated"))
    store.set_result(
        job_id,
        ok=False,
        summary=safe_output,
        envelope=envelope,
        release_writer_lease=release_writer_lease,
    )
    run_handles.release(job_id)


def _capture_process_identity(pid: int) -> dict[str, Any]:
    """Capture start identity and process group for safe later cleanup."""
    identity: dict[str, Any] = {}
    try:
        identity["pgid"] = os.getpgid(pid)
    except OSError:
        identity["pgid"] = None
    proc_stat = f"/proc/{pid}/stat"
    try:
        fields = Path(proc_stat).read_text(encoding="utf-8").split()
        if len(fields) > 21:
            identity["start"] = fields[21]
    except OSError:
        pass
    if "start" not in identity:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                identity["start"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            identity["command"] = result.stdout.strip()[:500]
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def _process_is_zombie(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(fields) > 2 and fields[2] == "Z"
    except OSError:
        return False


def _process_group_has_live_member(pgid: int) -> bool:
    """Return whether a process group has a non-zombie member.

    ``killpg(pgid, 0)`` also succeeds for a group containing only zombie
    entries.  Use the platform process table to distinguish that case from a
    live descendant.  Any failed or ambiguous probe is treated as live so
    callers never claim cleanup without proof.
    """
    try:
        result = subprocess.run(
            ["ps", "-eo", "pgid=,stat="],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if result.returncode != 0:
        return True

    saw_row = False
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        saw_row = True
        if len(fields) < 2:
            return True
        try:
            row_pgid = int(fields[0])
        except ValueError:
            return True
        if row_pgid == pgid and not fields[1].startswith("Z"):
            return True
    return not saw_row


def _process_exit_confirmed(pid: int, pgid: int | None, timeout: float = 2.0) -> bool:
    """Return only after the recorded leader and owned group are gone."""
    deadline = time.monotonic() + timeout
    while True:
        leader_is_zombie = _process_is_zombie(pid)
        leader_alive = False
        try:
            os.kill(pid, 0)
            leader_alive = not leader_is_zombie
        except ProcessLookupError:
            leader_alive = False
        except OSError:
            # Permission or another probe failure is not proof of death.
            leader_alive = True

        group_alive = False
        if pgid is not None and pgid > 1 and pgid != os.getpgrp():
            try:
                os.killpg(pgid, 0)
                # killpg(0) succeeds while a dead leader's zombie entry still
                # belongs to the group. Only a process-table probe can tell
                # whether a live descendant remains in that group.
                group_alive = _process_group_has_live_member(pgid)
            except ProcessLookupError:
                group_alive = False
            except OSError:
                group_alive = True

        if not leader_alive and not group_alive:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


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

    import signal

    try:
        pid_int = int(pid)
    except (ValueError, TypeError):
        return {"terminated": False, "reason": f"invalid_acp_pid: {pid}", "pid": None}
    if pid_int <= 1:
        return {"terminated": False, "reason": "unsafe_acp_pid", "pid": pid_int}

    # Check process existence and prove that the PID still refers to the
    # process captured at launch. A recycled PID must never receive a signal.
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return {"terminated": True, "reason": "process_already_gone", "pid": pid_int}
    except PermissionError:
        return {"terminated": False, "reason": "process_identity_unverifiable", "pid": pid_int}
    except OSError as exc:
        return {
            "terminated": False,
            "reason": "process_probe_error",
            "error": type(exc).__name__,
            "pid": pid_int,
        }

    expected_start = meta.get("acp_process_start")
    expected_pgid = meta.get("acp_pgid")
    try:
        expected_pgid_int = int(expected_pgid)
    except (TypeError, ValueError):
        expected_pgid_int = None
    # A PID without the launch-time identity is not safely ownable.  A
    # ProcessLookupError above is still authoritative proof that it is gone,
    # but a live/unverifiable PID must never receive a signal.
    if not isinstance(expected_start, str) or not expected_start or expected_pgid_int is None:
        return {"terminated": False, "reason": "process_identity_unverifiable", "pid": pid_int}
    actual = _capture_process_identity(pid_int)
    if not actual.get("start") or actual.get("pgid") is None:
        return {"terminated": False, "reason": "process_identity_unverifiable", "pid": pid_int}
    if str(expected_start) != str(actual["start"]):
        return {"terminated": False, "reason": "pid_reused", "pid": pid_int}
    actual_pgid = actual.get("pgid")
    if expected_pgid_int != actual_pgid:
        # The repository-owned ACP wrapper calls setsid() immediately after
        # spawn.  The SDK callback can observe the pre-exec group, while the
        # same launch identity is already running in its final owned group by
        # stop time.  The verified start identity proves this is the original
        # process, so fence the current group rather than treating that normal
        # handoff as PID/group reuse.
        if actual_pgid is None:
            return {"terminated": False, "reason": "process_identity_unverifiable", "pid": pid_int}
        if actual_pgid != pid_int:
            return {"terminated": False, "reason": "process_group_reused", "pid": pid_int}
        expected_pgid_int = int(actual_pgid)

    # Only a group whose id is the recorded leader PID is owned by the
    # repository wrapper. A verified leader in a shared legacy group may be
    # signalled directly, but that foreign group must never be killed or
    # included in death confirmation.
    owned_pgid = expected_pgid_int if expected_pgid_int == pid_int else None

    # Try SIGTERM first
    try:
        if owned_pgid and owned_pgid > 1 and owned_pgid != os.getpgrp():
            os.killpg(owned_pgid, signal.SIGTERM)
        else:
            os.kill(pid_int, signal.SIGTERM)
    except ProcessLookupError:
        return {"terminated": True, "reason": "process_gone_during_terminate", "pid": pid_int}
    except OSError as exc:
        return {
            "terminated": False,
            "reason": "terminate_error",
            "error": type(exc).__name__,
            "pid": pid_int,
        }

    # Grace period, then SIGKILL
    time.sleep(1.0)
    try:
        os.kill(pid_int, 0)
        # Still alive — force kill
        if owned_pgid and owned_pgid > 1 and owned_pgid != os.getpgrp():
            os.killpg(owned_pgid, signal.SIGKILL)
        else:
            os.kill(pid_int, signal.SIGKILL)
        confirmed = _process_exit_confirmed(pid_int, owned_pgid)
        return {
            "terminated": confirmed,
            "reason": "force_killed_after_sigterm" if confirmed else "death_unconfirmed",
            "pid": pid_int,
        }
    except ProcessLookupError:
        return {"terminated": True, "reason": "terminated_via_sigterm", "pid": pid_int}
    except OSError as exc:
        try:
            waited_pid, _status = os.waitpid(pid_int, os.WNOHANG)
        except (ChildProcessError, OSError):
            waited_pid = 0
        if waited_pid == pid_int and _process_exit_confirmed(pid_int, owned_pgid):
            return {
                "terminated": True,
                "reason": "terminated_and_reaped",
                "pid": pid_int,
            }
        if (
            isinstance(exc, PermissionError)
            and _process_is_zombie(pid_int)
            and _process_exit_confirmed(pid_int, owned_pgid)
        ):
            return {
                "terminated": True,
                "reason": "terminated_zombie_pending_reap",
                "pid": pid_int,
            }
        return {
            "terminated": False,
            "reason": "death_probe_error",
            "error": type(exc).__name__,
            "pid": pid_int,
        }
