"""Live provider surface gate for before-push checks.

This script intentionally calls real MCP tools and real providers. Use it for
provider/harness behavior changes, not as part of the default unit test suite.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent_crossbar.terminal_wait import TerminalWaitTimeout, wait_for_terminal_result

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAX_RUNTIME_SEC = 1800
RESULT_COMPLETION_GRACE_SEC = 15
ALL_TASKS = ("ask", "review", "dev")
BLOCKING_PROMPTS = (
    "Allow once",
    "Allow always",
    "Reject",
)
_REASONIX_FOOTER_RE = re.compile(
    r"— turns:\d+ cache:\d+(?:\.\d+)?% cost:\$\d+(?:\.\d+)? "
    r"save-vs-claude:\d+(?:\.\d+)?%"
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class GateCase:
    profile: str
    model: str
    effort: str | None
    task: str
    interactive: bool
    max_runtime_sec: int = DEFAULT_MAX_RUNTIME_SEC

    @property
    def label(self) -> str:
        effort = self.effort or "default"
        interactive = "interactive" if self.interactive else "oneshot"
        return f"{self.profile}/{self.model}/{effort}/{self.task}/{interactive}"


class _BlockingPromptDetected(RuntimeError):
    """Internal sentinel used to abort a deterministic waiter safely."""

    def __init__(self, tail: dict[str, Any]) -> None:
        super().__init__("provider requested interactive input")
        self.tail = tail


def _contains_blocking_prompt(output: str) -> bool:
    return any(prompt in output for prompt in BLOCKING_PROMPTS)


def _tool_data(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if structured:
        return dict(structured)
    return json.loads(result.content[0].text)


def _dev_prompt() -> str:
    return (
        "Create exactly two files in the current working directory: "
        "reverse_words.py and test_reverse_words.py. "
        "reverse_words.py must define reverse_words(text: str) -> str, reversing "
        "the letters in each word while preserving word order. "
        "test_reverse_words.py must contain pytest tests for normal words, "
        "multiple spaces, punctuation attached to words, empty string, and unicode. "
        "Run pytest test_reverse_words.py. Finish by reporting the test command "
        "and whether it passed."
    )


def _agent_start_args(case: GateCase, cwd: str | None = None) -> dict[str, Any]:
    """Build agent_start args from a GateCase using the current public API."""
    args: dict[str, Any] = {
        "profile": case.profile,
        "task": case.task,
        "interactive": case.interactive,
        "max_runtime_sec": case.max_runtime_sec,
        "model": case.model,
    }

    if case.task == "dev":
        args["prompt"] = _dev_prompt()
        if cwd is not None:
            args["cwd"] = cwd
    else:
        # ask or review — both use sentinel prompt
        args["prompt"] = "Reply with exactly GPT_PRO_PROVIDER_GATE_OK"

    if case.effort:
        args["effort"] = case.effort
    return args


def _prepare_workspace(cwd: Path) -> CheckResult:
    git = shutil.which("git")
    if git is None:
        return CheckResult(False, "git is not available on PATH")
    completed = subprocess.run(
        [git, "rev-parse", "--is-inside-work-tree"],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0 or completed.stdout.strip() != "true":
        output = (completed.stdout + completed.stderr)[-4000:]
        return CheckResult(False, f"workspace is not inside a git repo:\n{output}")
    (cwd / "AGENTS.md").write_text(
        "# Disposable provider gate\n\n"
        "This directory is a disposable provider surface gate. "
        "Do not create beads or OpenSpec artifacts. "
        "Do not inspect or modify parent directories. "
        "Implement only the files requested by the prompt and run only the requested test.\n",
        encoding="utf-8",
    )
    return CheckResult(True, "workspace is inside the trusted package repo")


def _workspace_tempdir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix=".agents-provider-gate-work-", dir=PACKAGE_DIR)


def _verify_reverse_words_workspace(cwd: Path) -> CheckResult:
    code_path = cwd / "reverse_words.py"
    test_path = cwd / "test_reverse_words.py"
    if not code_path.exists():
        return CheckResult(False, f"missing {code_path.name}")
    if not test_path.exists():
        return CheckResult(False, f"missing {test_path.name}")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path.name)],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr)[-4000:]
        return CheckResult(
            False, f"generated tests failed with exit {completed.returncode}:\n{output}"
        )
    return CheckResult(True, "generated tests passed")


async def _call(
    session: ClientSession, tool: str, args: dict[str, Any], timeout_sec: int = 120
) -> dict[str, Any]:
    result = await session.call_tool(
        tool, args, read_timeout_seconds=timedelta(seconds=timeout_sec)
    )
    return _tool_data(result)


async def _wait_for_result(
    session: ClientSession,
    job_id: str,
    *,
    timeout_sec: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tail: dict[str, Any] = {}

    async def _observe_tail(_result: dict[str, Any]) -> None:
        nonlocal tail
        tail = await _call(
            session, "job_tail", {"job_id": job_id, "max_bytes": 20000}, timeout_sec=60
        )
        tail_text = json.dumps(tail, ensure_ascii=False)
        if _contains_blocking_prompt(tail_text):
            raise _BlockingPromptDetected(tail)

    async def _read_result() -> dict[str, Any]:
        return await _call(session, "job_result", {"job_id": job_id}, timeout_sec=60)

    try:
        result = await wait_for_terminal_result(
            _read_result,
            timeout_sec=timeout_sec,
            poll_interval_sec=2.0,
            on_not_ready=_observe_tail,
        )
    except _BlockingPromptDetected as exc:
        tail = exc.tail
        tail_text = json.dumps(tail, ensure_ascii=False)
        return {"ok": False, "error": "blocking_prompt", "summary": tail_text[-4000:]}, tail
    except TerminalWaitTimeout as exc:
        return {"ok": False, "error": "timed_out", "summary": str(exc.last_result)}, tail

    final_tail = await _call(
        session, "job_tail", {"job_id": job_id, "max_bytes": 20000}, timeout_sec=60
    )
    final_tail_text = json.dumps(final_tail, ensure_ascii=False)
    if _contains_blocking_prompt(final_tail_text):
        return {
            "ok": False,
            "error": "blocking_prompt",
            "summary": final_tail_text[-4000:],
        }, final_tail
    return result, final_tail


async def _check_job_is_listed(
    session: ClientSession,
    job_id: str,
    case: GateCase,
) -> CheckResult | None:
    listed = await _call(
        session,
        "job_list",
        {"profile": case.profile, "limit": 100},
        timeout_sec=60,
    )
    if not listed.get("ok"):
        return CheckResult(False, f"{case.label} job_list failed: {listed}")
    jobs = listed.get("jobs")
    if not isinstance(jobs, list) or not any(
        str(job.get("job_id")) == job_id for job in jobs if isinstance(job, dict)
    ):
        return CheckResult(False, f"{case.label} job_list did not include {job_id}")
    return None


def _reasonix_sentinel_with_footer(output: str, *, profile: str) -> bool:
    if profile.casefold() not in {"reasonix", "deepseek"}:
        return False
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    return (
        len(lines) >= 2
        and lines[-2] == "GPT_PRO_PROVIDER_GATE_OK"
        and _REASONIX_FOOTER_RE.fullmatch(lines[-1]) is not None
    )


def _standalone_sentinel_received(sentinel: str, output: str) -> bool:
    """Check whether *sentinel* appears as a standalone line anywhere in output."""
    for line in output.splitlines():
        if line.strip() == sentinel:
            return True
    return False


def _sentinel_after_echo(sentinel: str, output: str) -> bool:
    """Accept *sentinel* when it appears after the echoed prompt line.

    For pipe-captured tmux output, the first occurrence of the sentinel is
    always the echoed prompt text (e.g. ``Reply with exactly SENTINEL▌``).
    A second, standalone occurrence indicates the assistant actually replied.
    """
    # Find all positions of the sentinel
    positions: list[int] = []
    start = 0
    while True:
        idx = output.find(sentinel, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(sentinel)
    if len(positions) < 2:
        # Single occurrence: accept if it's NOT an echoed prompt
        if positions:
            line_before = output[max(0, positions[0] - 60) : positions[0]]
            if "Reply with exactly" in line_before:
                return False
        return len(positions) == 1 and "Reply with exactly" not in output[: positions[0] + 60]
    # Two or more occurrences: accept (the first is likely the echo)
    return True


def _claude_sentinel_received(sentinel: str, output: str) -> bool:
    """Check whether *sentinel* appears as a distinct assistant answer.

    Claude's native TUI echoes the prompt, so a naive substring match
    would false-positive on the echoed instruction.  This checker:

    1. Accepts any line whose stripped content is exactly *sentinel*.
    2. Accepts a ``⏺``-prefixed line whose content after removing the
       ``⏺`` marker is exactly *sentinel* (case-insensitive).
    3. Rejects everything else — no buried-in-prose substring matches.
    """
    lines = output.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Standalone sentinel on its own line.
        if stripped == sentinel:
            return True
        # ⏺-prefixed line where the remainder is exactly the sentinel.
        if stripped.startswith("⏺"):
            remainder = stripped[1:].strip()
            if remainder.lower() == sentinel.lower():
                return True
        # Current Claude Code transcripts label the answer explicitly.
        if stripped.casefold().startswith("claude:"):
            remainder = stripped.split(":", 1)[1].strip()
            if remainder == sentinel:
                return True
    return False


async def _run_dev_case(session: ClientSession, case: GateCase, cwd: Path) -> CheckResult:
    """Run a dev task via agent_start, poll for completion, verify output."""
    args = _agent_start_args(case, cwd=str(cwd))

    start = await _call(session, "agent_start", args, timeout_sec=120)
    if not start.get("ok"):
        return CheckResult(False, f"{case.label} agent_start failed: {start}")

    job_id = str(start["job_id"])
    listed = await _check_job_is_listed(session, job_id, case)
    if listed is not None:
        return listed
    result, tail = await _wait_for_result(
        session,
        job_id,
        timeout_sec=case.max_runtime_sec + RESULT_COMPLETION_GRACE_SEC,
    )
    tail_text = json.dumps(tail, ensure_ascii=False)

    if _contains_blocking_prompt(tail_text):
        return CheckResult(False, f"{case.label} blocked on provider prompt:\n{tail_text[-4000:]}")
    if not result.get("ok"):
        summary = str(result.get("summary") or result.get("output") or "")[-4000:]
        return CheckResult(
            False, f"{case.label} failed: {result.get('error', 'unknown')}\n{summary}"
        )

    stream = _check_acp_stream(case, start, tail)
    if stream is not None:
        return stream

    workspace = _verify_reverse_words_workspace(cwd)
    if not workspace.ok:
        summary = str(result.get("summary") or "")[-4000:]
        return CheckResult(
            False,
            f"{case.label} workspace verification failed: {workspace.message}\nProvider summary:\n{summary}",
        )
    return CheckResult(True, f"{case.label}: {workspace.message}")


async def _run_ask_case(session: ClientSession, case: GateCase) -> CheckResult:
    """Run an ask task via agent_start, poll for completion, check sentinel."""
    if case.interactive:
        return await _run_interactive_ask_case(session, case)
    return await _run_oneshot_ask_case(session, case)


async def _run_oneshot_ask_case(session: ClientSession, case: GateCase) -> CheckResult:
    """Non-interactive ask: agent_start, wait for terminal job_result, check sentinel."""
    args = _agent_start_args(case)
    start = await _call(session, "agent_start", args, timeout_sec=120)
    if not start.get("ok"):
        return CheckResult(False, f"{case.label} agent_start failed: {start}")
    job_id = str(start["job_id"])
    listed = await _check_job_is_listed(session, job_id, case)
    if listed is not None:
        return listed
    result, tail = await _wait_for_result(
        session,
        job_id,
        timeout_sec=case.max_runtime_sec + RESULT_COMPLETION_GRACE_SEC,
    )
    if not result.get("ok"):
        return CheckResult(False, f"{case.label} failed: {result}")
    stream = _check_acp_stream(case, start, tail)
    if stream is not None:
        return stream
    output = result.get("output") or result.get("summary") or ""
    return _check_sentinel(case, output)


async def _run_interactive_ask_case(session: ClientSession, case: GateCase) -> CheckResult:
    """Interactive ask: start job, verify initial response, send second sentinel,
    verify second response, then stop and verify cleanup."""
    args = _agent_start_args(case)
    start = await _call(session, "agent_start", args, timeout_sec=120)
    if not start.get("ok"):
        return CheckResult(False, f"{case.label} agent_start failed: {start}")
    job_id = str(start["job_id"])
    listed = await _check_job_is_listed(session, job_id, case)
    if listed is not None:
        await _call(
            session, "job_stop", {"job_id": job_id, "reason": "job_list_missing"}, timeout_sec=30
        )
        return listed

    # 1. Wait for initial output (settling) and check first sentinel
    initial, _ = await _poll_tail_text(session, job_id, timeout_sec=60, settling_sec=8.0)
    initial_ok = _check_sentinel(case, initial)
    if not initial_ok.ok:
        await _call(
            session, "job_stop", {"job_id": job_id, "reason": "sentinel_missing"}, timeout_sec=30
        )
        return initial_ok

    # 2. Send a distinct second sentinel via job_send
    second_sentinel = "GPT_PROVIDER_SECOND_SENTINEL_OK"
    send = await _call(
        session,
        "job_send",
        {"job_id": job_id, "text": f"Reply with exactly {second_sentinel}"},
        timeout_sec=60,
    )
    if not send.get("ok"):
        await _call(
            session, "job_stop", {"job_id": job_id, "reason": "send_failed"}, timeout_sec=30
        )
        return CheckResult(False, f"{case.label} job_send failed: {send}")

    # 3. Poll for second sentinel (full output_tail, cleaned, standalone)
    found, _ = await _poll_until_sentinel(session, job_id, second_sentinel, timeout_sec=120)

    # 4. Clean up
    stop = await _call(
        session, "job_stop", {"job_id": job_id, "reason": "gate_complete"}, timeout_sec=30
    )
    cleanup_ok = stop.get("ok", False)
    cleanup_msg = "cleanup ok" if cleanup_ok else "cleanup failed"

    if not found:
        return CheckResult(False, f"{case.label} second sentinel not received ({cleanup_msg})")

    # 5. Verify job reached terminal state after stop
    final = await _call(session, "job_result", {"job_id": job_id}, timeout_sec=30)
    terminal_ok = final.get("ok", False)
    if not terminal_ok:
        return CheckResult(
            False, f"{case.label} job not terminal after stop: {final.get('error', 'unknown')}"
        )

    return CheckResult(True, f"{case.label}: two sentinels, {cleanup_msg}, job terminal")


CHATGPT_PROFILE = "chatgpt_pro"
CHATGPT_LONG_PROMPT = (
    "Write a detailed, multi-section architectural review of a distributed job "
    "queue, at least 1500 words, covering delivery guarantees, backpressure, "
    "observability, and failure modes."
)
CHATGPT_STOP_GRACE_SEC = 20


def _events(tail: dict[str, Any]) -> list[dict[str, Any]]:
    events = tail.get("events")
    return (
        [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    )


def _check_acp_stream(
    case: GateCase,
    start: dict[str, Any],
    tail: dict[str, Any],
) -> CheckResult | None:
    """Require ACP jobs to expose at least one real text delta in ``job_tail``."""
    if start.get("backend") != "acp":
        return None

    events = _events(tail)
    event_types = [str(event.get("type")) for event in events]
    required = ("acp_command", "log_delta", "acp_completed")
    missing = [event_type for event_type in required if event_type not in event_types]
    if missing:
        return CheckResult(
            False,
            f"{case.label} ACP stream missing lifecycle event(s): {', '.join(missing)}",
        )

    first_delta = event_types.index("log_delta")
    if not (event_types.index("acp_command") < first_delta < event_types.index("acp_completed")):
        return CheckResult(False, f"{case.label} ACP stream lifecycle order is invalid")

    text = "".join(
        str(event.get("data", {}).get("text", ""))
        for event in events
        if event.get("type") == "log_delta" and isinstance(event.get("data"), dict)
    )
    if not text.strip():
        return CheckResult(False, f"{case.label} ACP stream emitted no assistant text")
    return None


async def _run_chatgpt_readiness_case(session: ClientSession) -> CheckResult:
    """Readiness must describe the browser surface, never the desktop app."""
    data = await _call(session, "profile_health", {}, timeout_sec=120)
    profiles = data.get("profiles") or data.get("health") or data
    entry: dict[str, Any] | None = None
    if isinstance(profiles, dict):
        entry = profiles.get(CHATGPT_PROFILE)
    elif isinstance(profiles, list):
        entry = next(
            (item for item in profiles if item.get("profile") == CHATGPT_PROFILE),
            None,
        )
    if not isinstance(entry, dict):
        return CheckResult(False, f"profile_health did not report {CHATGPT_PROFILE}: {data}")

    remediation = str(entry.get("remediation") or "").casefold()
    evidence = str(entry.get("evidence") or "")
    if "desktop app" in remediation or "desktop application" in remediation:
        return CheckResult(
            False,
            f"{CHATGPT_PROFILE} readiness still requires the native desktop app: {remediation}",
        )
    state = entry.get("state")
    if state == "ready":
        if not entry.get("authenticated"):
            return CheckResult(False, "readiness reported ready without authenticated=True")
        if "browser=" not in evidence:
            return CheckResult(False, f"ready evidence does not name the browser: {evidence!r}")
        return CheckResult(True, f"{CHATGPT_PROFILE} readiness: ready ({evidence})")
    if not entry.get("error_code") or not entry.get("remediation"):
        return CheckResult(
            False, f"{CHATGPT_PROFILE} non-ready state {state} lacks actionable remediation"
        )
    return CheckResult(
        True,
        f"{CHATGPT_PROFILE} readiness: {state} ({entry.get('error_code')}) — actionable, not ready",
    )


async def _wait_for_event(
    session: ClientSession,
    job_id: str,
    event_type: str,
    *,
    timeout_sec: int,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        tail = await _call(
            session, "job_tail", {"job_id": job_id, "max_bytes": 20000}, timeout_sec=60
        )
        for event in _events(tail):
            if event.get("type") == event_type:
                return event
        if tail.get("status") not in (None, "running"):
            return None
        await anyio.sleep(2)
    return None


async def _run_chatgpt_cancellation_case(session: ClientSession, case: GateCase) -> CheckResult:
    """Maintainer-only: prove job_stop cancels a live browser generation."""
    args = _agent_start_args(case)
    args["prompt"] = CHATGPT_LONG_PROMPT
    start = await _call(session, "agent_start", args, timeout_sec=180)
    job_id = start.get("job_id")
    if not job_id:
        return CheckResult(False, f"cancellation gate could not start a job: {start}")

    submitted = await _wait_for_event(session, job_id, "prompt_submitted", timeout_sec=300)
    if submitted is None:
        tail = await _call(
            session, "job_tail", {"job_id": job_id, "max_bytes": 20000}, timeout_sec=60
        )
        seen = [event.get("type") for event in _events(tail)]
        errors = [
            str(event.get("message"))[:300]
            for event in _events(tail)
            if event.get("type") in ("error", "cancelled")
        ]
        return CheckResult(
            False,
            "cancellation gate never observed prompt_submitted "
            f"(status={tail.get('status')}, events={seen}, errors={errors})",
        )

    stopped = await _call(session, "job_stop", {"job_id": job_id}, timeout_sec=120)
    if not stopped.get("ok"):
        return CheckResult(False, f"job_stop failed: {stopped}")

    await anyio.sleep(CHATGPT_STOP_GRACE_SEC)
    tail = await _call(session, "job_tail", {"job_id": job_id, "max_bytes": 50000}, timeout_sec=60)
    if tail.get("status") != "stopped":
        return CheckResult(False, f"job is not durably stopped: status={tail.get('status')}")

    cancelled = [event for event in _events(tail) if event.get("type") == "cancelled"]
    stop_events = [event for event in _events(tail) if event.get("type") == "stopped"]
    handle_stops = [
        event
        for event in stop_events
        if isinstance(event.get("data"), dict) and "run_handle_stop" in event["data"]
    ]
    if not handle_stops:
        return CheckResult(False, "job_stop did not invoke a registered run handle")

    # For a stopped job `ok` means "a terminal answer exists", not "the provider
    # succeeded" — the late-success violation is a status that is no longer stopped.
    result = await _call(session, "job_result", {"job_id": job_id}, timeout_sec=120)
    if result.get("status") != "stopped":
        return CheckResult(
            False,
            f"a late provider result overwrote the stopped status: {str(result)[:400]}",
        )

    stop_confirmed = any(
        isinstance(event.get("data"), dict) and event["data"].get("provider_stop_confirmed")
        for event in cancelled
    )
    return CheckResult(
        True,
        f"{case.label}: stop honoured (provider_stop_confirmed={stop_confirmed}, "
        f"status=stopped, no late success)",
    )


async def _run_chatgpt_isolation_case(session: ClientSession, case: GateCase) -> CheckResult:
    """Maintainer-only: sequential turns must not cross-contaminate."""
    first = await _run_ask_case(session, case)
    if not first.ok:
        return CheckResult(False, f"isolation gate first turn failed: {first.message}")
    second = await _run_ask_case(session, case)
    if not second.ok:
        return CheckResult(False, f"isolation gate second turn failed: {second.message}")
    return CheckResult(
        True,
        f"{case.label}: two sequential turns each returned their own correlated answer",
    )


def _check_sentinel(case: GateCase, output: str) -> CheckResult:
    """Check whether *output* contains the expected gate sentinel."""
    if not isinstance(output, str) or not output.strip():
        return CheckResult(False, f"{case.label} returned no output for sentinel check")
    # Strip ANSI for tmux/interactive output
    if case.interactive:
        output = _clean_ansi(output)
    claude_profile = case.profile.casefold() in {"claude", "opus", "fable"}
    reasonix_profile = case.profile.casefold() in {"reasonix", "deepseek"}
    if reasonix_profile:
        if case.interactive:
            # tmux TUI pipes the echoed prompt — look for second occurrence
            sentinel_received = _sentinel_after_echo("GPT_PRO_PROVIDER_GATE_OK", output)
        else:
            sentinel_received = _reasonix_sentinel_with_footer(output, profile=case.profile)
    elif claude_profile:
        sentinel_received = _claude_sentinel_received("GPT_PRO_PROVIDER_GATE_OK", output)
    else:
        sentinel_received = output.strip() == "GPT_PRO_PROVIDER_GATE_OK"
    if sentinel_received:
        return CheckResult(True, f"{case.label}: live sentinel received")
    return CheckResult(False, f"{case.label} returned no exact string gate sentinel: {output!r}")


def _clean_ansi(text: str) -> str:
    """Strip ANSI escape sequences from tmux TUI output for sentinel matching."""
    from agent_crossbar.tmux_output import normalize_tmux_output

    return normalize_tmux_output(text)


def _tail_text(tail: dict[str, Any]) -> str:
    """Extract visible text from a job_tail response (handles both dict and string)."""
    ot = tail.get("output_tail")
    return ot.get("text", "") if isinstance(ot, dict) else (ot or "")


async def _poll_tail_text(
    session: ClientSession,
    job_id: str,
    *,
    timeout_sec: int,
    settling_sec: float,
    since_bytes: int | None = None,
) -> tuple[str, int]:
    """Poll job_tail until accumulated visible output stabilises.

    Returns (text, next_bytes) suitable for a subsequent incremental poll.
    """
    deadline = time.monotonic() + timeout_sec
    last_text = ""
    last_bytes = 0
    stable_since = time.monotonic()
    args: dict[str, Any] = {"job_id": job_id, "max_bytes": 50000}
    if since_bytes is not None:
        args["output_since_bytes"] = since_bytes
    while time.monotonic() < deadline:
        tail = await _call(session, "job_tail", args, timeout_sec=60)
        current = _tail_text(tail)
        ot = tail.get("output_tail")
        next_bytes = ot.get("bytes", 0) if isinstance(ot, dict) else last_bytes
        if current and current == last_text:
            if time.monotonic() - stable_since >= settling_sec:
                return current, next_bytes
        else:
            last_text = current
            last_bytes = next_bytes
            stable_since = time.monotonic()
        args.pop("output_since_bytes", None)  # full poll after first incremental
        await anyio.sleep(1)
    return last_text, last_bytes


async def _poll_until_sentinel(
    session: ClientSession,
    job_id: str,
    sentinel: str,
    *,
    timeout_sec: int,
) -> tuple[bool, str]:
    """Poll job_tail with full output_tail until *sentinel* appears as a
    standalone line in cleaned output.  Returns (found, cleaned_text)."""
    deadline = time.monotonic() + timeout_sec
    last_text = ""
    while time.monotonic() < deadline:
        tail = await _call(
            session, "job_tail", {"job_id": job_id, "max_bytes": 100000}, timeout_sec=60
        )
        current = _tail_text(tail)
        if current and current != last_text:
            last_text = current
            clean = _clean_ansi(current)
            if _sentinel_after_echo(sentinel, clean):
                return True, clean
        await anyio.sleep(1)
    return False, _clean_ansi(last_text)


def _server_params(env: dict[str, str]) -> StdioServerParameters:
    """Return StdioServerParameters for the installed ``agents-mcp`` entrypoint.

    Resolves the console script alongside the current Python interpreter
    (both live in the same venv ``bin/`` directory).  This exercises the
    real user-visible entrypoint — not ``python -m`` on a module without
    a ``__main__`` block.
    """
    bin_dir = Path(sys.executable).parent
    return StdioServerParameters(
        command=str(bin_dir / "agents-mcp"),
        args=[],
        env=env,
    )


async def _run_full_surface_preflight(
    session: ClientSession,
    cases: list[GateCase],
) -> tuple[CheckResult, dict[str, dict[str, Any]]]:
    """Exercise discovery/readiness before running the selected live cases."""
    listed = await _call(session, "profiles_list", {}, timeout_sec=120)
    if not listed.get("ok"):
        return CheckResult(False, f"profiles_list failed: {listed}"), {}
    profiles = listed.get("profiles")
    details = listed.get("profile_details")
    if not isinstance(profiles, list) or not isinstance(details, dict):
        return CheckResult(False, f"profiles_list returned malformed data: {listed}"), {}

    selected_profiles = {case.profile for case in cases}
    missing_profiles = selected_profiles - {str(profile) for profile in profiles}
    if missing_profiles:
        return CheckResult(
            False, f"profiles_list omitted selected profile(s): {sorted(missing_profiles)}"
        ), {}

    typed_details = {
        str(profile): detail for profile, detail in details.items() if isinstance(detail, dict)
    }
    for case in cases:
        models = typed_details.get(case.profile, {}).get("models")
        if not isinstance(models, list) or case.model not in models:
            return CheckResult(
                False,
                f"profiles_list did not advertise {case.profile}/{case.model}",
            ), typed_details

    health = await _call(session, "profile_health", {}, timeout_sec=120)
    if not health.get("ok"):
        return CheckResult(False, f"profile_health failed: {health}"), typed_details
    health_rows = health.get("profiles") or health.get("health")
    if isinstance(health_rows, dict):
        health_by_profile = health_rows
    elif isinstance(health_rows, list):
        health_by_profile = {
            str(row.get("profile")): row for row in health_rows if isinstance(row, dict)
        }
    else:
        return CheckResult(
            False, f"profile_health returned malformed data: {health}"
        ), typed_details
    non_ready = [
        profile
        for profile in selected_profiles
        if not isinstance(health_by_profile.get(profile), dict)
        or health_by_profile[profile].get("state") != "ready"
    ]
    if non_ready:
        return CheckResult(False, f"profile_health not ready: {sorted(non_ready)}"), typed_details
    return CheckResult(True, "profiles_list and profile_health passed"), typed_details


async def _run_cases(
    cases: list[GateCase],
    lifecycle: bool = False,
    all_supported_tools: bool = False,
) -> int:
    with tempfile.TemporaryDirectory(prefix="agents-provider-gate-state-") as state_root:
        env = os.environ.copy()
        env["AGENT_CROSSBAR_STATE_DIR"] = state_root
        env["AGENT_CROSSBAR_CLIENT_NAME"] = "provider-surface-gate"

        params = _server_params(env)
        failed = False
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                profile_details: dict[str, dict[str, Any]] = {}
                if all_supported_tools:
                    preflight, profile_details = await _run_full_surface_preflight(session, cases)
                    print(preflight.message)
                    if not preflight.ok:
                        return 1
                if lifecycle:
                    readiness = await _run_chatgpt_readiness_case(session)
                    print(readiness.message)
                    if not readiness.ok:
                        failed = True
                    for case in cases:
                        if case.profile != CHATGPT_PROFILE:
                            continue
                        for runner_fn in (
                            _run_chatgpt_cancellation_case,
                            _run_chatgpt_isolation_case,
                        ):
                            outcome = await runner_fn(session, case)
                            print(outcome.message)
                            if not outcome.ok:
                                failed = True
                    return 1 if failed else 0
                for case in cases:
                    if case.task == "dev":
                        with _workspace_tempdir() as work_dir:
                            cwd = Path(work_dir)
                            prepared = _prepare_workspace(cwd)
                            if not prepared.ok:
                                print(
                                    f"{case.label} workspace preparation failed: {prepared.message}"
                                )
                                failed = True
                                continue
                            result = await _run_dev_case(session, case, cwd)
                    else:
                        result = await _run_ask_case(session, case)
                    print(result.message)
                    if not result.ok:
                        failed = True
                if all_supported_tools:
                    interactive_cases: set[tuple[str, str, str | None]] = set()
                    for case in cases:
                        detail = profile_details.get(case.profile, {})
                        key = (case.profile, case.model, case.effort)
                        if (
                            case.interactive
                            or not detail.get("interactive")
                            or key in interactive_cases
                        ):
                            continue
                        interactive_cases.add(key)
                        result = await _run_interactive_ask_case(
                            session,
                            replace(case, task="ask", interactive=True),
                        )
                        print(result.message)
                        if not result.ok:
                            failed = True
        return 1 if failed else 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    # Strip exactly one leading '--' separator that npm/pnpm passes through
    if argv and argv[0] == "--":
        argv = argv[1:]
    parser = argparse.ArgumentParser(description="Run live provider surface gate")
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--effort", action="append", default=[])
    parser.add_argument("--task", action="append", choices=[*ALL_TASKS, "all"])
    parser.add_argument(
        "--all-supported-tools",
        action="store_true",
        help=(
            "Run the full method surface claimed by selected profiles: discovery, health, "
            "job listing, and an interactive ask for profiles that advertise it."
        ),
    )
    parser.add_argument(
        "--interactive", type=lambda s: s.lower() in ("true", "1", "yes"), default=False
    )
    parser.add_argument("--max-runtime-sec", type=int, default=DEFAULT_MAX_RUNTIME_SEC)
    parser.add_argument(
        "--chatgpt-lifecycle",
        action="store_true",
        help=(
            "Maintainer-only: run the ChatGPT Pro browser lifecycle gate "
            "(readiness, cancellation after submit, sequential isolation)."
        ),
    )
    return parser.parse_args(argv)


def _cases_from_args(args: argparse.Namespace) -> list[GateCase]:
    models = args.model
    efforts = args.effort or [None]
    requested_tasks = args.task or ["dev"]
    tasks = tuple(
        dict.fromkeys(
            task
            for requested in requested_tasks
            for task in (ALL_TASKS if requested == "all" else (requested,))
        )
    )
    return [
        GateCase(
            profile=profile,
            model=model,
            effort=effort,
            task=task,
            interactive=args.interactive,
            max_runtime_sec=args.max_runtime_sec,
        )
        for profile in args.profile
        for model in models
        for effort in efforts
        for task in tasks
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    valid_tasks = set(ALL_TASKS) | {"all"}
    unsupported = [task for task in (args.task or ["dev"]) if task not in valid_tasks]
    if unsupported:
        print(
            f"provider_surface_gate supports only task in {valid_tasks}: {unsupported}",
            file=sys.stderr,
        )
        return 2
    cases = _cases_from_args(args)
    if args.chatgpt_lifecycle:
        return anyio.run(_run_cases, cases, True, args.all_supported_tools)
    return anyio.run(_run_cases, cases, False, args.all_supported_tools)


if __name__ == "__main__":
    raise SystemExit(main())
