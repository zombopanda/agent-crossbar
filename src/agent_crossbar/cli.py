"""CLI entry points for Agent Crossbar.

Provides the ``doctor`` subcommand and the dispatcher that selects
between doctor mode and the MCP server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Any

from agent_crossbar import __version__

WAIT_JOB_SUCCESS = 0
WAIT_JOB_DEADLINE = 2
WAIT_JOB_TERMINAL_FAILURE = 3
WAIT_JOB_NOT_FOUND = 4
TERMINALIZE_REFUSED = 5
TERMINALIZE_GRACE_SEC = 15.0
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "stopped", "cancelled"})


def _format_text(results: dict[str, Any], profile_filter: str | None = None) -> str:
    """Format readiness results as human-readable text."""
    lines: list[str] = []
    profiles = results.get("profiles", [])

    if profile_filter:
        profiles = [p for p in profiles if p["profile"] == profile_filter]
        if not profiles:
            return f"Unknown profile: {profile_filter}\n"

    for p in profiles:
        state = p["state"]
        tier = p["support_tier"]
        icon = _state_icon(state)
        lines.append(f"{icon} {p['profile']} ({tier}): {state}")

        if p.get("error_code"):
            lines.append(f"   error: {p['error_code']}")
        if p.get("remediation"):
            lines.append(f"   action: {p['remediation']}")
        if p.get("auth_mode"):
            lines.append(f"   auth: {p['auth_mode']}")
        if p.get("billing_mode"):
            lines.append(f"   billing: {p['billing_mode']}")
        if p.get("version"):
            lines.append(f"   version: {p['version']}")
        if p.get("evidence"):
            lines.append(f"   evidence: {p['evidence']}")

    # Summary
    ready_count = sum(1 for p in profiles if p["state"] == "ready")
    total = len(profiles)
    lines.append(f"\n{ready_count}/{total} profiles ready")

    return "\n".join(lines) + "\n"


def _state_icon(state: str) -> str:
    icons = {
        "ready": "\u2705",  # ✅
        "needs_auth": "\u26a0\ufe0f",  # ⚠️
        "missing_binary": "\u274c",  # ❌
        "unsupported_os": "\u23f9\ufe0f",  # ⏹️
        "misconfigured": "\u26a0\ufe0f",  # ⚠️
        "degraded": "\u26a0\ufe0f",  # ⚠️
    }
    return icons.get(state, "?")


def doctor_cmd(json_output: bool = False, profile: str | None = None) -> None:
    """Run provider readiness checks and print results.

    Args:
        json_output: If True, print JSON to stdout.
        profile: If set, only check this profile.
    """
    from agent_crossbar.readiness import probe_all_profiles, probe_profile

    if profile:
        try:
            result = probe_profile(profile, use_cache=False)  # doctor always fresh
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        results_dict = {"profiles": [result.to_dict()]}
    else:
        results = probe_all_profiles(use_cache=False)  # doctor always fresh
        results_dict = {"profiles": [r.to_dict() for r in results.values()]}

    if json_output:
        json.dump(results_dict, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_text(results_dict, profile))


def wait_job_cmd(
    job_id: str,
    timeout_sec: float,
    poll_interval_sec: float,
    state_dir: str | None = None,
) -> int:
    """Wait for a durable job's terminal result without inferring a stall.

    Exit codes are stable for controller use: ``0`` completed successfully,
    ``2`` deadline elapsed, ``3`` terminal job failure/cancellation, and ``4``
    job not found or inaccessible.
    """
    from agent_crossbar.jobs import JobStore
    from agent_crossbar.terminal_wait import TerminalWaitTimeout, wait_for_terminal_result

    store = JobStore(state_dir)

    async def read_result() -> dict[str, Any]:
        # Explicit wildcard access is intentional for a local controller
        # recovery path; it does not alter the MCP tool surface.
        return store.get_result(job_id, client_session_id="*")

    try:
        result = asyncio.run(
            wait_for_terminal_result(
                read_result,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
            )
        )
    except TerminalWaitTimeout as exc:
        json.dump(
            {
                "ok": False,
                "error": "terminal_wait_timeout",
                "job_id": job_id,
                "timeout_sec": timeout_sec,
                "last_result": exc.last_result,
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return WAIT_JOB_DEADLINE
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    if result.get("error") == "job_not_found" or result.get("job_created") is False:
        return WAIT_JOB_NOT_FOUND
    if result.get("ok") is True and result.get("status") not in {
        "failed",
        "cancelled",
        "stopped",
    }:
        return WAIT_JOB_SUCCESS
    return WAIT_JOB_TERMINAL_FAILURE


def terminalize_job_cmd(
    job_id: str,
    reason: str,
    timeout_sec: float,
    poll_interval_sec: float,
    state_dir: str | None = None,
) -> int:
    """Explicitly stop a job, then wait for its durable terminal result.

    This command is intentionally separate from ``wait-job``: silence and an
    intermediate result never trigger cancellation. Controllers call it only
    after an explicit blocking-prompt or elapsed-deadline decision.
    """
    from agent_crossbar.jobs import JobStore
    from agent_crossbar.server import job_stop
    from agent_crossbar.terminal_wait import TerminalWaitTimeout, wait_for_terminal_result

    store = JobStore(state_dir)
    job = store.get_job(job_id)
    if job is None:
        result = {"ok": False, "error": "job_not_found", "job_id": job_id}
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return WAIT_JOB_NOT_FOUND

    status = store.job_status(job_id)
    if status in _TERMINAL_STATUSES:
        # A terminal result is already safe to collect, regardless of the
        # caller's recovery reason.  Never send a second stop request.
        pass
    elif reason == "blocking_prompt" and status != "awaiting_input":
        result = {
            "ok": False,
            "error": "terminalize_reason_not_permitted",
            "job_id": job_id,
            "status": status,
            "message": "blocking_prompt requires durable status awaiting_input",
        }
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return TERMINALIZE_REFUSED
    elif reason == "runtime_deadline":
        meta = store._read_job_meta(job.path)
        max_runtime_raw = meta.get("max_runtime_sec")
        started_raw = meta.get("started_at") or meta.get("created")
        try:
            max_runtime = float(max_runtime_raw)
            if max_runtime <= 0:
                raise ValueError("max_runtime_sec must be positive")
            started_at = datetime.fromisoformat(str(started_raw)).timestamp()
        except (TypeError, ValueError):
            result = {
                "ok": False,
                "error": "runtime_deadline_unavailable",
                "job_id": job_id,
                "status": status,
                "message": "runtime_deadline requires recorded max_runtime_sec and started_at",
            }
            json.dump(result, sys.stdout)
            sys.stdout.write("\n")
            return TERMINALIZE_REFUSED
        deadline = started_at + max_runtime + TERMINALIZE_GRACE_SEC
        remaining = deadline - datetime.now(timezone.utc).timestamp()
        if remaining > 0:
            result = {
                "ok": False,
                "error": "runtime_deadline_not_reached",
                "job_id": job_id,
                "status": status,
                "deadline": datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                "remaining_sec": round(remaining, 3),
            }
            json.dump(result, sys.stdout)
            sys.stdout.write("\n")
            return TERMINALIZE_REFUSED
    elif reason not in {"blocking_prompt", "runtime_deadline"}:
        result = {
            "ok": False,
            "error": "terminalize_reason_invalid",
            "job_id": job_id,
            "status": status,
            "message": "reason must be blocking_prompt or runtime_deadline",
        }
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return TERMINALIZE_REFUSED

    if status not in _TERMINAL_STATUSES:
        # Reuse the provider-neutral public lifecycle so ACP/Claude adapters
        # receive their cancellation path before the durable stop is collected.
        stopped = job_stop(job_id, reason=reason, client_session_id="*")
        if not stopped.get("ok") or store.job_status(job_id) not in {
            "succeeded",
            "failed",
            "stopped",
            "cancelled",
        }:
            # A provider-specific cancellation failure must not leave a
            # replacement writer racing a still-nonterminal job.  The
            # provider-neutral store stop is the bounded durable backstop.
            forced = store.stop_job(job_id, reason=reason, client_session_id="*")
            if not forced.get("ok") or store.job_status(job_id) not in {
                "succeeded",
                "failed",
                "stopped",
                "cancelled",
            }:
                json.dump(stopped if not stopped.get("ok") else forced, sys.stdout)
                sys.stdout.write("\n")
                return WAIT_JOB_TERMINAL_FAILURE

    async def read_result() -> dict[str, Any]:
        return store.get_result(job_id, client_session_id="*")

    try:
        result = asyncio.run(
            wait_for_terminal_result(
                read_result,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
            )
        )
    except TerminalWaitTimeout as exc:
        result = {
            "ok": False,
            "error": "terminal_wait_timeout",
            "job_id": job_id,
            "timeout_sec": timeout_sec,
            "last_result": exc.last_result,
        }
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return WAIT_JOB_DEADLINE
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    if result.get("error") == "job_not_found" or result.get("job_created") is False:
        return WAIT_JOB_NOT_FOUND
    if result.get("ok") is True and result.get("status") not in {
        "failed",
        "cancelled",
        "stopped",
    }:
        return WAIT_JOB_SUCCESS
    return WAIT_JOB_TERMINAL_FAILURE


def writer_lease_cmd(
    command: str,
    *,
    cwd: str | None = None,
    owner_id: str | None = None,
    owner_kind: str = "local",
    token: str | None = None,
    acknowledgement: str | None = None,
    state_dir: str | None = None,
) -> int:
    """Manage the internal durable dev-writer lease used by controllers."""
    from agent_crossbar.jobs import default_state_root
    from agent_crossbar.writer_lease import WriterLeaseStore

    store = WriterLeaseStore(state_dir or default_state_root())
    if command == "acquire":
        if not cwd or not owner_id:
            print(json.dumps({"ok": False, "error": "cwd_and_owner_required"}))
            return 1
        result = store.acquire(cwd, owner_id=owner_id, owner_kind=owner_kind)
        print(json.dumps(result.to_dict()))
        return 0 if result.ok else 1
    if command == "release":
        released = store.release(token or "")
        print(json.dumps({"ok": released, "released": released}))
        return 0 if released else 1
    if command == "heartbeat":
        refreshed = store.heartbeat(token or "")
        print(json.dumps({"ok": refreshed, "refreshed": refreshed}))
        return 0 if refreshed else 1
    if command == "reconcile":
        removed = store.reconcile()
        print(json.dumps({"ok": True, "removed": removed}))
        return 0
    if command == "recover":
        if not cwd:
            print(json.dumps({"ok": False, "error": "cwd_required"}))
            return 1
        result = store.recover(
            cwd,
            acknowledgement=acknowledgement or "",
        )
        print(json.dumps(result.to_dict()))
        return 0 if result.ok else 1
    print(json.dumps({"ok": False, "error": "invalid_writer_lease_command"}))
    return 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the lightweight CLI parser without importing the MCP server."""
    parser = argparse.ArgumentParser(
        prog="agent-crossbar",
        description="Run the Agent Crossbar MCP server or inspect provider readiness.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the installed Agent Crossbar version and exit.",
    )
    subcommands = parser.add_subparsers(dest="command")
    doctor = subcommands.add_parser("doctor", help="Check provider readiness.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.add_argument("--profile", metavar="PROFILE", help="Check one provider profile.")
    wait_job = subcommands.add_parser(
        "wait-job",
        help="Wait for a durable job result until terminal or deadline.",
    )
    wait_job.add_argument("--job-id", required=True, help="Durable job identifier.")
    wait_job.add_argument(
        "--state-dir",
        metavar="PATH",
        help=(
            "Exact durable state root used by the Agents MCP server; required when "
            "that server does not use the default state directory."
        ),
    )
    wait_job.add_argument(
        "--timeout-sec",
        type=float,
        default=1815.0,
        help="Maximum wait in seconds (default: max runtime plus grace).",
    )
    wait_job.add_argument(
        "--poll-interval-sec",
        type=float,
        default=2.0,
        help="Seconds between result polls.",
    )
    terminalize = subcommands.add_parser(
        "terminalize-job",
        help="Explicitly stop a job and wait for its terminal result.",
    )
    terminalize.add_argument("--job-id", required=True, help="Durable job identifier.")
    terminalize.add_argument(
        "--reason",
        required=True,
        help="Explicit operator reason, e.g. blocking_prompt or runtime_deadline.",
    )
    terminalize.add_argument("--state-dir", metavar="PATH")
    terminalize.add_argument("--timeout-sec", type=float, default=30.0)
    terminalize.add_argument("--poll-interval-sec", type=float, default=2.0)
    writer_lease = subcommands.add_parser(
        "writer-lease",
        help="Acquire, release, reconcile, or explicitly recover the internal dev-writer lease.",
    )
    writer_lease.add_argument(
        "writer_lease_command",
        choices=("acquire", "release", "heartbeat", "reconcile", "recover"),
    )
    writer_lease.add_argument("--state-dir", metavar="PATH")
    writer_lease.add_argument("--cwd", metavar="PATH")
    writer_lease.add_argument("--owner-id", metavar="ID")
    writer_lease.add_argument("--owner-kind", default="local", metavar="KIND")
    writer_lease.add_argument("--token", metavar="TOKEN")
    writer_lease.add_argument("--acknowledgement", metavar="TEXT")
    return parser


def main() -> None:
    """Dispatch ``agent-crossbar [doctor]`` while keeping the default MCP mode."""
    args = _build_parser().parse_args()

    if args.command == "doctor":
        doctor_cmd(json_output=args.json, profile=args.profile)
        return
    if args.command == "wait-job":
        raise SystemExit(
            wait_job_cmd(args.job_id, args.timeout_sec, args.poll_interval_sec, args.state_dir)
        )
    if args.command == "terminalize-job":
        raise SystemExit(
            terminalize_job_cmd(
                args.job_id,
                args.reason,
                args.timeout_sec,
                args.poll_interval_sec,
                args.state_dir,
            )
        )
    if args.command == "writer-lease":
        raise SystemExit(
            writer_lease_cmd(
                args.writer_lease_command,
                cwd=args.cwd,
                owner_id=args.owner_id,
                owner_kind=args.owner_kind,
                token=args.token,
                acknowledgement=args.acknowledgement,
                state_dir=args.state_dir,
            )
        )

    from agent_crossbar.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
