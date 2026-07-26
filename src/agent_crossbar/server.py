"""Agent Harness MCP Server — unified job harness for external agents."""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from functools import partial, wraps
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from agent_crossbar.acp_runtime import run_acp_job as _run_acp_job
from agent_crossbar.adapters.claude import LocalSubprocessRunner, start_claude_interactive_tmux
from agent_crossbar.adapters.registry import get_adapter
from agent_crossbar.agent_runner import start_agent_job
from agent_crossbar.discovery import (
    EFFORT_LEGEND,
    cached_profile_health_entry,
    live_profile_registry,
)
from agent_crossbar.env_compat import getenv
from agent_crossbar.envelope import build_result_envelope, sanitize_diagnostic_text
from agent_crossbar.jobs import JobStore, default_state_root
from agent_crossbar.profiles import list_profiles
from agent_crossbar.runner import (
    CHATGPT_PRO_DEFAULT_TIMEOUT_SEC,
    run_tmux_job,
    start_gui_job,
    start_print_job,
    start_tmux_job,
)
from agent_crossbar.telemetry import TelemetryStore
from agent_crossbar.validation import validate_start_request

mcp = FastMCP("agents")
_sync_request_state: ContextVar[tuple[str, threading.Event, asyncio.Task[Any] | None] | None] = (
    ContextVar("sync_request_state", default=None)
)


def _state_root() -> Path:
    return default_state_root()


def _job_store() -> JobStore:
    return JobStore(default_state_root())


def _client_metadata(
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Normalize optional caller-provided client metadata for telemetry."""
    data = dict(client or {})
    if client_name is not None:
        data["name"] = client_name
    else:
        env_client_name = getenv("AGENT_CROSSBAR_CLIENT_NAME")
        if env_client_name:
            data.setdefault("name", env_client_name)
    data.setdefault("name", "agent-crossbar")

    if client_version is not None:
        data["version"] = client_version
    else:
        env_client_version = getenv("AGENT_CROSSBAR_CLIENT_VERSION")
        if env_client_version:
            data.setdefault("version", env_client_version)
    data.setdefault("version", "unknown")

    if client_session_id is not None:
        data["session_id"] = client_session_id
    data.setdefault("session_id", None)
    return data


def _effective_client_session_id(
    client: dict[str, Any] | None,
    client_session_id: str | None,
) -> str | None:
    """Use the explicit session id, falling back to structured client metadata."""
    structured_session_id = (client or {}).get("session_id")
    if structured_session_id is not None:
        return str(structured_session_id)
    return client_session_id


def _cancellable_sync_tool(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Register a sync API as a cancellable MCP worker while keeping direct calls sync."""

    @wraps(fn)
    async def _tool_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        request_token = str(uuid.uuid4())
        cancelled = threading.Event()
        token = _sync_request_state.set((request_token, cancelled, asyncio.current_task()))
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        transport = bound.arguments.get("transport", "print")
        cancellable_tmux = transport == "tmux"
        worker = asyncio.create_task(asyncio.to_thread(partial(fn, *args, **kwargs)))
        try:
            return await (worker if cancellable_tmux else asyncio.shield(worker))
        except BaseException:
            cancelled.set()
            for item in _job_store().list_jobs():
                if item.get("transport") != "tmux" or item.get("status") != "running":
                    continue
                job = _job_store().get_job(item["job_id"])
                meta = _job_store()._read_job_meta(job.path) if job is not None else {}
                if job is not None and meta.get("sync_request_token") == request_token:
                    _job_store().stop_job(
                        item["job_id"],
                        reason="sync_request_cancelled",
                        client_session_id=meta.get("client_session_id"),
                    )
            if not cancellable_tmux:
                try:
                    await worker
                except Exception:
                    pass
            raise
        finally:
            _sync_request_state.reset(token)

    mcp.tool(name=fn.__name__)(_tool_wrapper)
    return fn


def _tool_error(error: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "message": message,
        "warnings": [],
        "job_created": False,
    }


def _normalize_profile_transport(profile: str, transport: str) -> str:
    """Apply profile-implied transport defaults for convenience tools."""
    if profile == "chatgpt_pro":
        return "gui"
    return transport


def _advice_timeout_sec(profile: str, timeout_sec: int | None) -> int:
    if timeout_sec is not None:
        return timeout_sec
    if profile == "chatgpt_pro":
        return CHATGPT_PRO_DEFAULT_TIMEOUT_SEC
    return 1800


def _default_dev_cwd() -> str:
    """Prefer the client's inherited shell cwd over uv's package cwd."""
    for key in ("AGENT_CROSSBAR_DEFAULT_CWD", "PWD"):
        raw = getenv(key)
        if raw and Path(raw).expanduser().exists():
            return str(Path(raw).expanduser())
    return os.getcwd()


def _apply_dev_defaults(req: dict[str, Any]) -> None:
    """Default dev jobs to the caller cwd when available."""
    if req.get("operation") != "dev":
        return
    req.setdefault("cwd", _default_dev_cwd())


def _run_logged_tool(
    tool: str,
    request_payload: dict[str, Any],
    handler: Callable[[], dict[str, Any]],
    *,
    profile: str | None = None,
    operation: str | None = None,
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Run a tool and record full request/response telemetry."""
    ts = TelemetryStore(_state_root())
    request_id = str(uuid.uuid4())
    client_meta = _client_metadata(
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )
    start = time.monotonic()

    ts.record_request(
        client=client_meta,
        tool=tool,
        request_id=request_id,
        payload=request_payload,
        profile=profile,
        operation=operation,
    )

    try:
        response_payload = handler()
    except Exception as exc:  # pragma: no cover - defensive MCP boundary
        ts.record_error(
            request_id=request_id,
            tool=tool,
            error_type=exc.__class__.__name__,
            message=str(exc),
        )
        response_payload = _tool_error("internal_error", str(exc))

    ts.record_response(
        request_id=request_id,
        ok=response_payload.get("ok", False),
        duration_ms=(time.monotonic() - start) * 1000,
        response=response_payload,
        client_name=client_meta["name"],
        tool=tool,
        job_id=response_payload.get("job_id"),
        profile=profile or response_payload.get("profile"),
        operation=operation or response_payload.get("operation"),
    )
    return response_payload


def _validate_and_create_job(
    req: dict[str, Any],
    *,
    client_session_id: str | None = None,
    client_name: str | None = None,
) -> dict[str, Any]:
    """Run validation and, if successful, create a job."""
    _apply_dev_defaults(req)
    result = validate_start_request(req, state_root=_state_root())
    if not result["ok"]:
        return result

    profile = result["profile"]
    operation = result["operation"]
    req["profile"] = profile
    req["operation"] = operation
    store = _job_store()
    job = store.create_job(
        profile=profile,
        operation=operation,
        transport=req["transport"],
        sensitivity=req["sensitivity"],
        client_session_id=client_session_id,
        client_name=client_name,
        cwd=req.get("cwd"),
    )
    job.events.write(
        level="info",
        type="job_created",
        message="Job created",
        data={
            "profile": profile,
            "operation": operation,
            "transport": req["transport"],
        },
    )
    warnings = list(result.get("warnings", []))
    if profile == "codex" and operation == "review" and req["transport"] == "print":
        warning = "context bypass risk accepted for native Codex review"
        warnings.append(warning)
        job.events.write(
            level="warn",
            type="warning",
            message=warning,
            data={
                "accepted_context_bypass_risk": True,
                "denied_path_filtering": "none",
                "manifest_accuracy": "best_effort",
            },
        )

    timeout_sec = req.get("timeout_sec")
    if req["transport"] in ("print", "auto"):
        start_print_job(store, job.job_id, req, timeout_sec=timeout_sec)
    elif req["transport"] == "gui":
        start_gui_job(store, job.job_id, req, timeout_sec=timeout_sec)
    elif req["transport"] == "tmux":
        start_tmux_job(store, job.job_id, req, timeout_sec=timeout_sec, complete_on_output=True)

    return {
        "ok": True,
        "job_id": job.job_id,
        "profile": profile,
        "operation": operation,
        "warnings": warnings,
    }


def _run_sync_tmux_request(
    req: dict[str, Any],
    result: dict[str, Any],
    *,
    timeout_sec: int,
    client_session_id: str | None = None,
    client_name: str | None = None,
) -> dict[str, Any]:
    """Create and run a synchronous tmux job, completing on interactive output."""
    run_req = dict(req)
    run_req["profile"] = result["profile"]
    run_req["operation"] = result["operation"]
    store = _job_store()
    request_state = _sync_request_state.get()

    def _request_cancelled() -> bool:
        if request_state is None:
            return False
        task = request_state[2]
        return request_state[1].is_set() or (
            task is not None and (task.cancelling() > 0 or task.cancelled())
        )

    if _request_cancelled():
        return {"ok": False, "error": "sync_request_cancelled"}
    job = store.create_job(
        profile=result["profile"],
        operation=result["operation"],
        transport=run_req["transport"],
        sensitivity=run_req["sensitivity"],
        client_session_id=client_session_id,
        client_name=client_name,
        cwd=run_req.get("cwd"),
    )
    store.update_job_meta(
        job.job_id,
        {
            "sync": True,
            "sync_request_token": request_state[0] if request_state is not None else None,
        },
    )
    if _request_cancelled():
        store.stop_job(
            job.job_id,
            reason="sync_request_cancelled",
            client_session_id=client_session_id,
        )
        return {"ok": False, "error": "sync_request_cancelled", "job_id": job.job_id}
    job.events.write(
        level="info",
        type="job_created",
        message="Job created",
        data={
            "profile": result["profile"],
            "operation": result["operation"],
            "transport": run_req["transport"],
            "sync": True,
        },
    )
    watcher_done = threading.Event()
    if request_state is not None:

        def _watch_cancellation() -> None:
            while not watcher_done.is_set():
                request_state[1].wait(0.05)
                if _request_cancelled():
                    store.stop_job(
                        job.job_id,
                        reason="sync_request_cancelled",
                        client_session_id=client_session_id,
                    )
                    return

        threading.Thread(target=_watch_cancellation, daemon=True).start()
    try:
        response = run_tmux_job(
            store,
            job.job_id,
            run_req,
            timeout_sec=timeout_sec,
            complete_on_output=True,
        )
    except BaseException:
        store.stop_job(
            job.job_id,
            reason="sync_request_cancelled",
            client_session_id=client_session_id,
        )
        raise
    finally:
        watcher_done.set()
    response["job_id"] = job.job_id
    return response


# ── MCP Tools ──────────────────────────────────────────────────────────────

_VALID_TASKS = frozenset({"ask", "review", "dev"})
_TASK_TO_OPERATION: dict[str, str] = {"ask": "advice", "review": "review", "dev": "dev"}


@mcp.tool()
def agent_start(
    profile: str,
    prompt: str,
    model: str,
    task: str = "ask",
    interactive: bool = False,
    effort: str | None = None,
    cwd: str | None = None,
    scope: dict[str, Any] | None = None,
    max_runtime_sec: int | None = None,
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Start an agent for ask/review/dev tasks with a cross-provider contract.

    Timeout semantics:
    - MCP client read timeout is external to this tool and does NOT cancel
      a durable async job — the job continues on the server.
    - Preflight/startup probe timeout is bounded internally at 30 s per probe
      (see readiness.py) and runs before any state mutation.
    - ``max_runtime_sec`` is the job wall-clock execution limit.  When
      exceeded the job result envelope reports ``max_runtime_exceeded``.
    """
    from agent_crossbar.profiles import resolve_profile

    # ── pre-validation: reject before any state mutation ──
    if task not in _VALID_TASKS:
        return _tool_error("invalid_task", f"Unknown task '{task}'")

    if max_runtime_sec is not None and max_runtime_sec <= 0:
        return _tool_error("invalid_max_runtime", "max_runtime_sec must be > 0")

    ok, resolved = resolve_profile(profile)
    if not ok:
        return _tool_error("invalid_profile", f"Unknown profile '{profile}'")

    if interactive:
        try:
            adapter = get_adapter(resolved)
        except ValueError:
            return _tool_error("invalid_profile", f"Unknown profile '{profile}'")
        if not adapter.supports_interactive:
            if resolved == "claude":
                return _tool_error(
                    "interactive_not_supported",
                    "claude --bg is the supported subscription mode; claude -p is disabled due "
                    "to ambiguous separate-credit or metered billing. Omit interactive or "
                    "choose another supported provider.",
                )
            return _tool_error(
                "interactive_not_supported",
                f"Profile '{resolved}' does not support interactive mode. "
                "Interactive send/attach is not yet implemented for this backend.",
            )

    if scope is not None and not cwd:
        return _tool_error("cwd_required", "scope requires cwd")

    # ── preflight: cached readiness probe before any state mutation ──
    from agent_crossbar.readiness import probe_profile as _readiness_probe

    try:
        readiness = _readiness_probe(resolved, use_cache=True)
        if readiness.state != "ready":
            return _tool_error(
                readiness.error_code or "not_ready",
                readiness.remediation or f"Provider '{resolved}' is not ready ({readiness.state})",
            )
    except ValueError:
        pass  # probe_profile raises for unknown profiles — already caught above

    # ── map public contract → internal request ──
    operation = _TASK_TO_OPERATION[task]

    # Fallback: if profile doesn't support the mapped operation, try alternatives.
    # Codex has no "advice" operation — use "text" for ask tasks.
    if task == "ask":
        from agent_crossbar.profiles import profile_operations as _prof_ops

        if operation not in _prof_ops(resolved):
            operation = "text"

    if resolved == "reasonix" and interactive:
        transport = "tmux"  # real interactive TUI lifecycle
    elif resolved == "chatgpt_pro":
        transport = "gui"
    elif resolved == "claude":
        transport = (
            "print"  # claude_bg adapter lifecycle; "print" is internal routing label, not claude -p
        )
    else:
        transport = "print"

    autonomy = "edit_local" if task == "dev" else "read_only"

    req: dict[str, Any] = {
        "operation": operation,
        "profile": profile,
        "transport": transport,
        "autonomy": autonomy,
        "sensitivity": "normal",
        "prompt": prompt,
    }
    if model is not None:
        req["model"] = model
    if cwd is not None:
        req["cwd"] = cwd
    if effort is not None:
        req["effort"] = effort
    timeout_sec = max_runtime_sec or 1800
    req["timeout_sec"] = timeout_sec
    _apply_dev_defaults(req)

    request_payload = dict(req)

    def _handle() -> dict[str, Any]:
        result = validate_start_request(req, state_root=_state_root())
        if not result["ok"]:
            return result

        run_req = dict(req)
        run_req["profile"] = result["profile"]
        run_req["operation"] = result["operation"]
        effective_cwd = run_req.get("cwd") or os.getcwd()

        # ── Claude: adapter-native lifecycle ──
        if result["profile"] == "claude":
            adapter = get_adapter(result["profile"])
            runner = LocalSubprocessRunner()

            # Check readiness before any state mutation
            readiness = adapter.check_readiness(runner)
            if not readiness.authenticated:
                return _tool_error(
                    readiness.error_code or "not_ready",
                    readiness.remediation or "Claude is not ready",
                )

            # Launch via adapter — use original user params, not validation defaults
            resolved_model = model  # may be None if user didn't request one
            resolved_effort = effort or "medium"
            launch_result = adapter.launch(
                runner,
                model=resolved_model,
                task=task,
                prompt=prompt,
                cwd=effective_cwd,
                effort=resolved_effort,
                interactive=interactive,
            )
            if launch_result.error:
                return _tool_error(
                    launch_result.error,
                    launch_result.message or "Launch failed",
                )

            session_id = launch_result.session_id
            if session_id is None:
                return _tool_error(
                    "session_id_missing",
                    "Claude launched but no session ID was returned",
                )

            # Create durable job
            store = _job_store()
            effective_transport = "tmux" if interactive else launch_result.backend

            job = store.create_job(
                profile=result["profile"],
                operation=result["operation"],
                transport=effective_transport,
                sensitivity=run_req["sensitivity"],
                client_session_id=_effective_client_session_id(client, client_session_id),
                client_name=_client_metadata(client, client_name)["name"],
                cwd=effective_cwd,
            )

            # ── Interactive: launch tmux claude attach session ──
            tmux_session_name: str | None = None
            if interactive:
                safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", job.job_id).strip("-")
                tmux_session_name = f"agents-{safe_id}"
                output_path = job.path / "tmux-output.log"
                tmux_result = start_claude_interactive_tmux(
                    session_id=session_id,
                    job_id=job.job_id,
                    cwd=effective_cwd,
                    tmux_session_name=tmux_session_name,
                    output_path=output_path,
                )
                if tmux_result.returncode != 0:
                    # The native background session already exists. Roll it
                    # back before exposing a failed start so a retry cannot
                    # duplicate a still-running provider turn.
                    try:
                        cancelled = adapter.cancel(runner, session_id)
                    except Exception as exc:
                        cancelled = False
                        store.send_event(
                            job.job_id,
                            level="error",
                            type="cancel_error",
                            message=f"Rollback cancel threw: {exc}",
                            data={"session_id": session_id},
                        )
                    if not cancelled:
                        store.send_event(
                            job.job_id,
                            level="warn",
                            type="cancel_warning",
                            message="Rollback cancel failed; native session may still be running",
                            data={"session_id": session_id},
                        )
                    store.set_result(
                        job.job_id,
                        ok=False,
                        summary=f"tmux session creation failed: {tmux_result.stderr[:500]}",
                    )
                    return {
                        "ok": False,
                        "error": "tmux_launch_failed",
                        "message": tmux_result.stderr[:500],
                        "job_id": job.job_id,
                    }

            store.update_job_meta(
                job.job_id,
                {
                    "backend": launch_result.backend,
                    "native_session_id": session_id,
                    "model": resolved_model,
                    "effort": resolved_effort,
                    "task": task,
                    "interactive": interactive,
                    "cwd": effective_cwd,
                    "adapter_name": adapter.name,
                    "max_runtime_sec": max_runtime_sec,
                    **(
                        {
                            "tmux_session": tmux_session_name,
                            "tmux_output_path": str(output_path),
                            "transport": "tmux",
                        }
                        if interactive
                        else {}
                    ),
                },
            )
            job.events.write(
                level="info",
                type="job_created",
                message="Job created",
                data={
                    "profile": result["profile"],
                    "operation": result["operation"],
                    "transport": run_req["transport"],
                    "backend": launch_result.backend,
                    "native_session_id": session_id,
                    **({"tmux_session": tmux_session_name} if interactive else {}),
                },
            )

            start_agent_job(
                store,
                job.job_id,
                adapter,
                session_id=session_id,
                poll_interval_sec=2.0,
                max_runtime_sec=max_runtime_sec,
            )

            return {
                "ok": True,
                "job_id": job.job_id,
                "profile": result["profile"],
                "operation": result["operation"],
                "backend": launch_result.backend,
                "warnings": list(result.get("warnings", [])),
            }

        # ── ACP path: Codex/OpenCode adapters ──
        adapter = get_adapter(result["profile"])
        _acp_profiles = frozenset({"codex", "opencode"})
        if result["profile"] in _acp_profiles and getattr(adapter, "backend", None) == "acp":
            # ── Preflight: provider-specific readiness before any state mutation ──
            # Explicit effort may fall back to print when the installed ACP
            # agent cannot prove support for the session config selector. This
            # preserves the legacy path for an older/unavailable ACP bridge.
            from agent_crossbar.acp_lifecycle import (
                check_codex_acp_readiness,
                check_opencode_acp_readiness,
            )

            preflight_runner = LocalSubprocessRunner()
            if result["profile"] == "codex":
                readiness = check_codex_acp_readiness(preflight_runner)
            else:
                readiness = check_opencode_acp_readiness(preflight_runner)

            # ── Effort routing ──
            # Some ACP agents do not advertise a session config selector for
            # effort. Keep their proven print path; agents that do advertise
            # it stay on ACP only after their version/readiness gate passes.
            acp_effort_available = bool(getattr(adapter, "supports_acp_effort", False)) and bool(
                readiness.get("ready")
            )
            if effort is not None and not acp_effort_available:
                # Route to print backend with explicit effort
                store = _job_store()
                job = store.create_job(
                    profile=result["profile"],
                    operation=result["operation"],
                    transport=run_req["transport"],
                    sensitivity=run_req["sensitivity"],
                    client_session_id=_effective_client_session_id(client, client_session_id),
                    client_name=_client_metadata(client, client_name)["name"],
                    cwd=effective_cwd,
                )
                store.update_job_meta(
                    job.job_id,
                    {
                        "backend": "print",
                        "effort_routing": "explicit_effort_print_fallback",
                        "model": model,
                        "effort": effort,
                        "task": task,
                        "interactive": interactive,
                        "cwd": effective_cwd,
                        "adapter_name": adapter.name,
                        "max_runtime_sec": max_runtime_sec,
                    },
                )
                job.events.write(
                    level="info",
                    type="job_created",
                    message="Job created (print — explicit effort fallback)",
                    data={
                        "profile": result["profile"],
                        "operation": result["operation"],
                        "backend": "print",
                        "effort_routing": "explicit_effort_print_fallback",
                        "effort": effort,
                    },
                )
                start_print_job(store, job.job_id, run_req, timeout_sec=timeout_sec)
                warnings = list(result.get("warnings", []))
                warnings.append(
                    {
                        "code": "effort_forced_print_fallback",
                        "profile": result["profile"],
                        "requested_effort": effort,
                        "message": (
                            f"Explicit effort '{effort}' forced fallback from ACP to print backend; "
                            "execution semantics may differ from ACP one-shot mode."
                        ),
                    }
                )
                return {
                    "ok": True,
                    "job_id": job.job_id,
                    "profile": result["profile"],
                    "operation": result["operation"],
                    "backend": "print",
                    "effort_routing": "explicit_effort_print_fallback",
                    "warnings": warnings,
                }

            if not readiness.get("ready"):
                return _tool_error(
                    readiness.get("error_code", "acp_not_ready"),
                    readiness.get("remediation", f"ACP preflight failed for {result['profile']}"),
                )

            # ── Create durable job ──
            store = _job_store()
            resolved_model = result["model"]
            resolved_effort = (
                result["effort"] if getattr(adapter, "supports_acp_effort", False) else None
            )

            job = store.create_job(
                profile=result["profile"],
                operation=result["operation"],
                transport="print",  # transport enum for legacy store compat
                sensitivity=run_req["sensitivity"],
                client_session_id=_effective_client_session_id(client, client_session_id),
                client_name=_client_metadata(client, client_name)["name"],
                cwd=effective_cwd,
            )
            store.update_job_meta(
                job.job_id,
                {
                    "backend": "acp",
                    "acp_transport": "sdk_stdio",
                    "model": resolved_model,
                    "effort": resolved_effort,
                    "task": task,
                    "interactive": interactive,
                    "cwd": effective_cwd,
                    "adapter_name": adapter.name,
                    "max_runtime_sec": max_runtime_sec,
                },
            )
            job.events.write(
                level="info",
                type="job_created",
                message="Job created",
                data={
                    "profile": result["profile"],
                    "operation": result["operation"],
                    "backend": "acp",
                    "acp_transport": "sdk_stdio",
                },
            )

            # Schedule ACP job (async, non-blocking)
            async def _schedule_acp():
                await _run_acp_job(
                    store,
                    job.job_id,
                    provider=result["profile"],
                    prompt=prompt,
                    cwd=effective_cwd,
                    task=task,
                    model=resolved_model,
                    effort=resolved_effort,
                    autonomy=run_req["autonomy"],
                    max_runtime_sec=max_runtime_sec,
                )

            # Fire-and-forget via asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_schedule_acp())
            except RuntimeError:
                import threading as _thr

                def _run_acp_in_thread():
                    asyncio.run(_schedule_acp())

                t = _thr.Thread(target=_run_acp_in_thread, daemon=True, name=f"acp-{job.job_id}")
                t.start()

            return {
                "ok": True,
                "job_id": job.job_id,
                "profile": result["profile"],
                "operation": result["operation"],
                "backend": "acp",
                "warnings": list(result.get("warnings", [])),
            }

        # ── legacy path: non-Claude profiles ──
        store = _job_store()
        job = store.create_job(
            profile=result["profile"],
            operation=result["operation"],
            transport=run_req["transport"],
            sensitivity=run_req["sensitivity"],
            client_session_id=_effective_client_session_id(client, client_session_id),
            client_name=_client_metadata(client, client_name)["name"],
            cwd=run_req.get("cwd"),
        )
        store.update_job_meta(
            job.job_id,
            {
                "interactive": interactive,
                "task": task,
                "max_runtime_sec": max_runtime_sec,
            },
        )
        job.events.write(
            level="info",
            type="job_created",
            message="Job created",
            data={
                "profile": result["profile"],
                "operation": result["operation"],
                "transport": run_req["transport"],
                "interactive": interactive,
                "task": task,
            },
        )
        if run_req["transport"] == "tmux":
            start_tmux_job(
                store, job.job_id, run_req, timeout_sec=timeout_sec, complete_on_output=False
            )
        elif run_req["transport"] == "gui":
            start_gui_job(store, job.job_id, run_req, timeout_sec=timeout_sec)
        else:
            start_print_job(store, job.job_id, run_req, timeout_sec=timeout_sec)

        warnings = list(result.get("warnings", []))
        return {
            "ok": True,
            "job_id": job.job_id,
            "profile": result["profile"],
            "operation": result["operation"],
            "warnings": warnings,
        }

    return _run_logged_tool(
        "agent_start",
        request_payload,
        _handle,
        profile=profile,
        operation=operation,
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


@mcp.tool()
def profiles_list(
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """List all canonical agent profiles."""
    return _run_logged_tool(
        "profiles_list",
        {},
        lambda: {
            "ok": True,
            "profiles": list_profiles(),
            "profile_details": live_profile_registry(_state_root()),
        },
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


@mcp.tool()
def profile_health(
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Return truthful readiness for every known provider profile.

    Each profile is probed (non-mutating, cached) and returns one of:
    ready, needs_auth, missing_binary, unsupported_os, misconfigured, degraded.
    Registration alone never produces "ready".
    """

    def _handle() -> dict[str, Any]:
        from agent_crossbar.readiness import probe_all_profiles

        state_root = _state_root()
        results = probe_all_profiles()
        profiles = []
        for name, r in results.items():
            entry = r.to_dict()
            try:
                entry["models"] = cached_profile_health_entry(state_root, name)
            except Exception as exc:
                entry["models"] = {
                    "name": name,
                    "discovery_available": False,
                    "models": [],
                    "error": sanitize_diagnostic_text(f"model discovery lookup crashed: {exc}"),
                }
            profiles.append(entry)
        return {"ok": True, "effort_legend": EFFORT_LEGEND, "profiles": profiles}

    return _run_logged_tool(
        "profile_health",
        {},
        _handle,
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


@mcp.tool()
def job_tail(
    job_id: str,
    since_seq: int = 0,
    output_since_bytes: int | None = None,
    max_bytes: int = 12000,
    max_events: int | None = None,
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Get incremental events for a job since a given sequence number."""
    request = {"job_id": job_id, "since_seq": since_seq, "max_bytes": max_bytes}
    if output_since_bytes is not None:
        request["output_since_bytes"] = output_since_bytes
    if max_events is not None:
        request["max_events"] = max_events

    return _run_logged_tool(
        "job_tail",
        request,
        lambda: _job_store().job_tail(
            job_id,
            since_seq=since_seq,
            max_events=max_events,
            max_bytes=max_bytes,
            output_since_bytes=output_since_bytes,
            client_session_id=_effective_client_session_id(client, client_session_id),
        ),
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


@mcp.tool()
def job_result(
    job_id: str,
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Return a final result for a job."""
    return _run_logged_tool(
        "job_result",
        {"job_id": job_id},
        lambda: _job_store().get_result(
            job_id,
            client_session_id=_effective_client_session_id(client, client_session_id),
        ),
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


@mcp.tool()
def job_send(
    job_id: str,
    text: str,
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Send redacted user input to an interactive job."""

    def _handle() -> dict[str, Any]:
        store = _job_store()
        effective_session_id = _effective_client_session_id(client, client_session_id)
        job, cross_session_note = store._get_owned_job(job_id, effective_session_id)
        if job is None:
            result = {"ok": False, "error": "job_not_found", "job_id": job_id}
            store._inject_cross_session_note(result, cross_session_note)
            return result
        return store.send_user_input(
            job_id,
            text,
            client_session_id=effective_session_id,
        )

    return _run_logged_tool(
        "job_send",
        {"job_id": job_id, "text": text},
        _handle,
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


@mcp.tool()
def job_stop(
    job_id: str,
    reason: str = "user_cancelled",
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """Stop a running job."""

    def _handle() -> dict[str, Any]:
        store = _job_store()
        effective_session_id = _effective_client_session_id(client, client_session_id)
        job, cross_session_note = store._get_owned_job(job_id, effective_session_id)
        if job is None:
            result = {"ok": False, "error": "job_not_found", "job_id": job_id}
            store._inject_cross_session_note(result, cross_session_note)
            return result
        meta = store._read_job_meta(job.path)

        # Do not terminate already-terminal jobs
        current_status = meta.get("status", "running")
        if current_status != "running":
            return {
                "ok": False,
                "error": "job_already_terminal",
                "job_id": job_id,
                "status": current_status,
            }

        # Native lifecycle: cancel via adapter before marking stopped
        backend = meta.get("backend")
        if backend in ("claude_bg", "claude_bg_pty"):
            session_id = meta.get("native_session_id")
            if session_id:
                try:
                    adapter = get_adapter(
                        meta.get("adapter_name") or meta.get("profile") or "claude"
                    )
                    runner = LocalSubprocessRunner()
                    cancelled = adapter.cancel(runner, session_id)
                    if not cancelled:
                        store.send_event(
                            job_id,
                            level="warn",
                            type="cancel_warning",
                            message="Adapter cancel returned failure; job may still be running",
                            data={"session_id": session_id},
                        )
                except Exception as exc:
                    store.send_event(
                        job_id,
                        level="error",
                        type="cancel_error",
                        message=f"Adapter cancel threw: {exc}",
                        data={"session_id": session_id},
                    )

        # ACP backend: mark stopped first (prevents background set_result race),
        # then safely terminate the recorded ACP child process.
        acp_stop_data: dict | None = None
        if backend == "acp":
            # Mark stopped BEFORE termination so background completion cannot
            # resurrect the job via set_result.
            store.stop_job(
                job_id,
                reason=reason,
                client_session_id=_effective_client_session_id(client, client_session_id),
            )
            from agent_crossbar.acp_runtime import safe_acp_termination

            meta = store._read_job_meta(job.path)  # re-read after stop_job
            acp_stop_data = safe_acp_termination(meta)
            store.send_event(
                job_id,
                level="info" if acp_stop_data.get("terminated") else "warn",
                type="acp_stop",
                message=f"ACP termination: {acp_stop_data.get('reason', 'unknown')}",
                data=acp_stop_data,
            )
            summary = f"Job stopped: {reason}"
            envelope = build_result_envelope(
                status="cancelled",
                stop_reason=reason,
                output=summary,
                summary=summary,
                created_at=meta.get("created") or meta.get("started_at"),
                started_at=meta.get("started_at"),
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
                    "backend": "acp",
                    "cwd": meta.get("cwd"),
                },
                technical={"acp_stop": acp_stop_data},
            )
            store.set_stopped_result(job_id, summary=summary, envelope=envelope)
            return {
                "ok": True,
                "job_id": job_id,
                "acp_stop": acp_stop_data,
            }

        return store.stop_job(
            job_id,
            reason=reason,
            client_session_id=effective_session_id,
        )

    return _run_logged_tool(
        "job_stop",
        {"job_id": job_id, "reason": reason},
        _handle,
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


@mcp.tool()
def job_list(
    status: str | None = None,
    profile: str | None = None,
    limit: int = 20,
    client: dict[str, Any] | None = None,
    client_name: str | None = None,
    client_version: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, Any]:
    """List all jobs in the state directory."""

    def _handle() -> dict[str, Any]:
        session_id = _effective_client_session_id(client, client_session_id)
        jobs = _job_store().list_jobs(client_session_id=session_id)
        if session_id is None:
            jobs = [job for job in jobs if "client_session_id" not in job]
        if status is not None:
            jobs = [job for job in jobs if job.get("status") == status]
        if profile is not None:
            jobs = [job for job in jobs if job.get("profile") == profile]
        return {"ok": True, "jobs": jobs[:limit]}

    request = {"status": status, "profile": profile, "limit": limit}
    return _run_logged_tool(
        "job_list",
        request,
        _handle,
        client=client,
        client_name=client_name,
        client_version=client_version,
        client_session_id=client_session_id,
    )


def main():
    mcp.run()
