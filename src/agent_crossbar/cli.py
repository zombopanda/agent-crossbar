"""CLI entry points for Agent Crossbar.

Provides the ``doctor`` subcommand and the dispatcher that selects
between doctor mode and the MCP server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from agent_crossbar import __version__

WAIT_JOB_SUCCESS = 0
WAIT_JOB_DEADLINE = 2
WAIT_JOB_TERMINAL_FAILURE = 3
WAIT_JOB_NOT_FOUND = 4


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


def wait_job_cmd(job_id: str, timeout_sec: float, poll_interval_sec: float) -> int:
    """Wait for a durable job's terminal result without inferring a stall.

    Exit codes are stable for controller use: ``0`` completed successfully,
    ``2`` deadline elapsed, ``3`` terminal job failure/cancellation, and ``4``
    job not found or inaccessible.
    """
    from agent_crossbar.jobs import JobStore
    from agent_crossbar.terminal_wait import TerminalWaitTimeout, wait_for_terminal_result

    store = JobStore()

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
    return parser


def main() -> None:
    """Dispatch ``agent-crossbar [doctor]`` while keeping the default MCP mode."""
    args = _build_parser().parse_args()

    if args.command == "doctor":
        doctor_cmd(json_output=args.json, profile=args.profile)
        return
    if args.command == "wait-job":
        raise SystemExit(wait_job_cmd(args.job_id, args.timeout_sec, args.poll_interval_sec))

    from agent_crossbar.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
