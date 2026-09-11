"""Claude adapter-owned job lifecycle orchestration.

The server supplies generic durable-store and lease callbacks; provider-specific
launch, rollback, attach, and polling details stay with this adapter service.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..subprocess_runner import LocalSubprocessRunner
from .claude import start_claude_interactive_tmux


def start_claude_job(
    *,
    result: dict[str, Any],
    adapter: Any,
    interactive: bool,
    client: dict[str, Any] | None,
    client_session_id: str | None,
    client_name: str | None,
    model: str | None,
    effort: str | None,
    task: str,
    prompt: str,
    effective_cwd: str,
    max_runtime_sec: int | None,
    run_req: dict[str, Any],
    store: Any,
    attach_writer_lease: Callable[[Any, str], None],
    session_id_for: Callable[[dict[str, Any] | None, str | None], str | None],
    metadata_for: Callable[..., dict[str, Any]],
    agent_starter: Callable[..., Any],
    tool_error: Callable[[str, str], dict[str, Any]],
) -> dict[str, Any]:
    runner = LocalSubprocessRunner()

    # Defense-in-depth: server.agent_start already rejects a non-interactive
    # launch for adapters that require interactive mode before this function
    # is reached. Callers that invoke this lifecycle entry point directly
    # (bypassing the public tool) must get the same stable rejection before
    # any job/lease/provider state mutation.
    if not interactive and getattr(adapter, "requires_interactive", False):
        return tool_error(
            "interactive_required",
            f"Profile '{adapter.name}' only supports interactive launch; pass interactive=true.",
        )

    # Check readiness before any state mutation
    readiness = adapter.check_readiness(runner)
    if not readiness.authenticated:
        return tool_error(
            readiness.error_code or "not_ready",
            readiness.remediation or "Claude is not ready",
        )

    # Create the durable job before launching Claude. Provider work
    # must never become untracked between launch and storage/lease
    # attachment; any later failure can then cancel the native session.
    job = store.create_job(
        profile=result["profile"],
        operation=result["operation"],
        transport="tmux" if interactive else "print",
        sensitivity=run_req["sensitivity"],
        client_session_id=session_id_for(client, client_session_id),
        client_name=metadata_for(client, client_name)["name"],
        cwd=effective_cwd,
    )
    attach_writer_lease(store, job.job_id)

    resolved_model = model
    resolved_effort = effort or "medium"
    # Persist a launch-pending provider identity before invoking the
    # native CLI. If this controller dies after launch but before the
    # session ID is attached, reaping retains the writer lease instead
    # of claiming cleanup that was never proven.
    store.update_job_meta(
        job.job_id,
        {
            "backend": "claude_bg_pty" if interactive else "claude_bg",
            "launch_pending": True,
            "native_session_id": None,
            "model": resolved_model,
            "effort": resolved_effort,
            "task": task,
            "interactive": interactive,
            "cwd": effective_cwd,
            "adapter_name": adapter.name,
            "max_runtime_sec": max_runtime_sec,
        },
    )
    try:
        launch_result = adapter.launch(
            runner,
            model=resolved_model,
            task=task,
            prompt=prompt,
            cwd=effective_cwd,
            effort=resolved_effort,
            interactive=interactive,
        )
    except Exception as exc:
        store.set_result(
            job.job_id,
            ok=False,
            summary=f"Claude launch failed: {type(exc).__name__}",
            release_writer_lease=False,
        )
        return {
            "ok": False,
            "error": "launch_error",
            "message": f"Claude launch failed: {type(exc).__name__}",
            "job_id": job.job_id,
        }
    if launch_result.error:
        store.set_result(
            job.job_id,
            ok=False,
            summary=launch_result.message or "Launch failed",
        )
        return {
            "ok": False,
            "error": launch_result.error,
            "message": launch_result.message or "Launch failed",
            "job_id": job.job_id,
        }

    session_id = launch_result.session_id
    if session_id is None:
        store.set_result(
            job.job_id,
            ok=False,
            summary="Claude launched without a session ID",
            release_writer_lease=False,
        )
        return {
            "ok": False,
            "error": "session_id_missing",
            "message": "Claude launched but no session ID was returned",
            "job_id": job.job_id,
        }

    def persist_claude_meta(payload: dict[str, Any]) -> bool:
        try:
            store.update_job_meta(job.job_id, payload)
        except Exception as exc:
            # Storage failure after native launch must not orphan the
            # provider session or expose a retryable untracked turn.
            cancelled = False
            try:
                cancelled = bool(adapter.cancel(runner, session_id))
            except Exception as cancel_exc:
                store.send_event(
                    job.job_id,
                    level="error",
                    type="cancel_error",
                    message=f"Metadata rollback cancel threw: {type(cancel_exc).__name__}",
                    data={"session_id": session_id},
                )
            finally:
                try:
                    store.update_job_meta(
                        job.job_id,
                        {
                            "native_session_id": session_id,
                            "launch_pending": False,
                            "cleanup_pending": not cancelled,
                        },
                    )
                except Exception:
                    pass
                try:
                    store.set_result(
                        job.job_id,
                        ok=False,
                        summary=f"Claude job metadata persistence failed: {type(exc).__name__}",
                        release_writer_lease=cancelled,
                    )
                except Exception:
                    pass
            return False
        return True

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
            # Keep the native identity durable whenever rollback is not
            # confirmed.  A replacement must retain the writer lease until a
            # later reaper/operator confirms the session has stopped.
            store.update_job_meta(
                job.job_id,
                {
                    "native_session_id": session_id,
                    "launch_pending": False,
                    "cleanup_pending": not cancelled,
                },
            )
            store.set_result(
                job.job_id,
                ok=False,
                summary=f"tmux session creation failed: {tmux_result.stderr[:500]}",
                release_writer_lease=cancelled,
            )
            return {
                "ok": False,
                "error": "tmux_launch_failed",
                "message": tmux_result.stderr[:500],
                "job_id": job.job_id,
            }

    metadata_persisted = persist_claude_meta(
        {
            "backend": launch_result.backend,
            "launch_pending": False,
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
    if not metadata_persisted:
        return {
            "ok": False,
            "error": "metadata_persistence_failed",
            "message": "Claude job metadata persistence failed after launch",
            "job_id": job.job_id,
        }

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

    agent_starter(
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
