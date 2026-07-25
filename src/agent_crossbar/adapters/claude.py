"""Claude Code native background-session adapter.

Lifecycle is structured through ``claude --bg`` and ``claude agents --json``.
Only follow-up input requires attaching to the native session through a PTY.
This module never invokes print mode or the Agent SDK.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..profiles.claude import SUPPORT_TIER
from .base import ModelCatalog, ModelInfo, StaticAdapter, normalize_effort
from .claude_model_probe import (
    ClaudeModelProbe,
    PosixClaudeModelProbe,
    probe_to_catalog,
)

CLAUDE_NATIVE_STATES = frozenset({"working", "blocked", "done", "failed", "stopped"})
CLAUDE_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "max": "max"}
_CLAUDE_HELP_EFFORT_RE = re.compile(
    r"--effort\s+\S+\s+.*?\(([^)]+)\)",
    re.IGNORECASE | re.DOTALL,
)
_CLAUDE_VERSIONED_MODEL_ID_RE = re.compile(
    r"claude-[a-z][a-z0-9-]*-\d+(?:-\d+)*\Z",
    re.IGNORECASE,
)
_SCREEN_READER_UI_LINE_RE = re.compile(
    r"^(?:"
    r".*?\besc to interrupt\b.*|"
    r".*?\bshift\+tab to cycle\b.*|"
    r"[\d,]+\s+tokens\b.*|"
    r"effort:\s.*|"
    r"plan mode on\b.*|"
    r".*…\s+\(\s*\d.*\)|"
    r"Worked for\b.*"
    r")$",
    re.IGNORECASE,
)


def parse_claude_help_efforts(help_output: str) -> tuple[str, ...]:
    """Parse ``claude --help`` output to extract public supported effort values.

    Looks for a line like ``--effort <effort>  Reasoning effort (low, medium, high, max)``
    and returns only the public values (excluding xhigh and other non-public labels).

    Returns an empty tuple when no effort line is found.
    """
    match = _CLAUDE_HELP_EFFORT_RE.search(help_output)
    if not match:
        return ()
    raw = match.group(1)
    candidates = [v.strip() for v in raw.split(",")]
    # Only include values matching PUBLIC_EFFORTS
    from .base import PUBLIC_EFFORTS

    return tuple(v for v in candidates if v in PUBLIC_EFFORTS)


_STATE_MAP = {
    "working": "running",
    "blocked": "waiting",
    "done": "completed",
    "failed": "failed",
    "stopped": "cancelled",
}
_BACKGROUND_ID_RE = re.compile(r"backgrounded\s*·\s*([0-9a-f]{8})\b", re.IGNORECASE)
_EMPTY_MCP_CONFIG = json.dumps({"mcpServers": {}}, separators=(",", ":"))
_DIRECT_WORKSPACE_SETTINGS = json.dumps(
    {"worktree": {"bgIsolation": "none"}}, separators=(",", ":")
)


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult: ...


class LocalSubprocessRunner:
    """Production runner with an argv-only subprocess boundary."""

    def run(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=False,
        )
        return RunResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True)
class ReadinessResult:
    state: str
    authenticated: bool
    auth_mode: str | None = None
    billing_mode: str | None = None
    subscription_type: str | None = None
    error_code: str | None = None
    remediation: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class LaunchResult:
    session_id: str | None
    backend: str
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    permission_mode: str | None = None
    error: str | None = None
    message: str = ""


@dataclass(frozen=True)
class NormalizedResult:
    status: str
    stop_reason: str | None
    output: str
    session_id: str
    error: str | None = None
    error_stage: str | None = None
    waiting_for: str | None = None


def parse_claude_auth(result: RunResult) -> dict[str, Any]:
    """Parse the documented ``claude auth status --json`` response."""
    if result.returncode != 0:
        return {
            "authenticated": False,
            "auth_mode": None,
            "billing_mode": None,
            "subscription_type": None,
            "error": "auth_probe_failed",
            "remediation": "Run `claude auth login`.",
            "evidence": (result.stderr or result.stdout)[:500],
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "authenticated": False,
            "auth_mode": None,
            "billing_mode": None,
            "subscription_type": None,
            "error": "invalid_auth_status",
            "remediation": "Run `claude doctor` and retry.",
            "evidence": result.stdout[:500],
        }
    if not payload.get("loggedIn"):
        return {
            "authenticated": False,
            "auth_mode": None,
            "billing_mode": None,
            "subscription_type": None,
            "error": "not_authenticated",
            "remediation": "Run `claude auth login`.",
            "evidence": "Claude reports loggedIn=false.",
        }
    method = str(payload.get("authMethod") or "unknown")
    api_key = method in {"api_key", "console"}
    subscription_type = payload.get("subscriptionType")
    return {
        "authenticated": True,
        "auth_mode": "api_key" if api_key else "subscription",
        "billing_mode": "api" if api_key else "subscription_quota",
        "subscription_type": str(subscription_type) if subscription_type else None,
        "error": None,
        "remediation": None,
        "evidence": f"authMethod={method}; apiProvider={payload.get('apiProvider', 'unknown')}",
    }


def parse_agents_json(raw: str) -> list[dict[str, Any]]:
    """Normalize the documented array returned by ``claude agents --json``."""
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("claude agents --json must return an array")
    sessions: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        sessions.append(
            {
                "id": item.get("id"),
                "session_id": item.get("sessionId"),
                "state": item.get("state"),
                "process_status": item.get("status"),
                "waiting_for": item.get("waitingFor"),
                "cwd": item.get("cwd"),
                "started_at_ms": item.get("startedAt"),
            }
        )
    return sessions


def normalize_session_state(native_state: str) -> str:
    try:
        return _STATE_MAP[native_state]
    except KeyError as exc:
        raise ValueError(f"Unknown Claude session state: {native_state!r}") from exc


def map_claude_effort(effort: str) -> str:
    return normalize_effort(effort, CLAUDE_EFFORT_MAP)


def _screen_reader_final_response(clean_logs: str) -> str:
    """Recover the most complete final-response redraw from Claude TUI logs."""
    assistant_markers = list(re.finditer(r"(?m)^\$?claude:\s*", clean_logs))
    if not assistant_markers:
        return ""

    candidate = clean_logs[assistant_markers[-1].end() :]
    segments: list[list[str]] = []
    current: list[str] = []

    def finish_segment() -> None:
        nonlocal current
        while current and not current[-1].strip():
            current.pop()
        if current:
            segments.append(current)
        current = []

    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if line.startswith("$"):
            line = line[1:].lstrip()
        if not line:
            continue
        if _SCREEN_READER_UI_LINE_RE.match(line):
            finish_segment()
            continue
        current.append(line)
    finish_segment()

    meaningful = [
        "\n".join(lines).strip()
        for lines in segments
        if any(line.casefold() != "summary" for line in lines)
    ]
    return max(meaningful, key=len, default="")


def build_claude_launch(
    *,
    model: str | None = None,
    task: str,
    prompt: str,
    cwd: str,
    effort: str = "medium",
    interactive: bool = False,
) -> LaunchResult:
    """Build a native background launch without relying on undocumented flags.

    When *model* is ``None`` the ``--model`` flag is omitted entirely so the
    CLI can apply its own default without the harness inventing one.
    """
    try:
        native_effort = map_claude_effort(effort)
    except ValueError as exc:
        return LaunchResult(None, "claude_bg", error="invalid_effort", message=str(exc))
    if model is not None and not model.strip():
        return LaunchResult(None, "claude_bg", error="invalid_model", message="model is required")
    if model is not None and not _CLAUDE_VERSIONED_MODEL_ID_RE.fullmatch(model):
        return LaunchResult(
            None,
            "claude_bg",
            error="invalid_model",
            message="model must be a versioned Claude CLI ID discovered from profiles_list",
        )
    if task not in {"ask", "review", "dev"}:
        return LaunchResult(
            None, "claude_bg", error="invalid_task", message=f"Unknown task: {task}"
        )

    permission_mode = "auto" if task == "dev" else "dontAsk"
    backend = "claude_bg_pty" if interactive else "claude_bg"
    args = [
        "claude",
        "--bg",
        "--safe-mode",
        "--ax-screen-reader",
    ]
    if model is not None:
        args.extend(["--model", model])
    if task != "dev":
        args.extend(
            [
                "--allowedTools",
                (
                    "Read,Grep,Glob,WebFetch,WebSearch,"
                    "Bash(git diff:*),Bash(git status:*),"
                    "Bash(git show:*),Bash(git log:*)"
                ),
            ]
        )
    args.extend(
        [
            "--effort",
            native_effort,
            "--permission-mode",
            permission_mode,
            "--settings",
            _DIRECT_WORKSPACE_SETTINGS,
            "--strict-mcp-config",
            "--mcp-config",
            _EMPTY_MCP_CONFIG,
            "--disable-slash-commands",
            prompt,
        ]
    )
    return LaunchResult(
        session_id=None,
        backend=backend,
        args=args,
        cwd=cwd,
        permission_mode=permission_mode,
    )


def normalize_claude_result(entry: dict[str, Any], logs: str) -> NormalizedResult:
    from agent_crossbar.tmux_output import normalize_tmux_output

    native_state = str(entry.get("state") or "")
    try:
        status = normalize_session_state(native_state)
    except ValueError:
        status = "failed"
    short_id = str(entry.get("id") or entry.get("session_id") or "")
    clean = normalize_tmux_output(logs)
    if "[Screen Reader Mode:" in clean:
        clean = _screen_reader_final_response(clean)
        if native_state == "done" and not clean:
            message = "Claude completed without a recoverable final response."
            return NormalizedResult(
                status="failed",
                stop_reason="result_output_unavailable",
                output=message,
                session_id=short_id,
                error=message,
                error_stage="finalization",
                waiting_for=entry.get("waiting_for"),
            )
    return NormalizedResult(
        status=status,
        stop_reason=native_state or "unknown_state",
        output=clean,
        session_id=short_id,
        error=clean if status == "failed" else None,
        error_stage="execution" if status == "failed" else None,
        waiting_for=entry.get("waiting_for"),
    )



def start_claude_interactive_tmux(
    *,
    session_id: str,
    job_id: str,
    cwd: str,
    tmux_session_name: str,
    output_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Start a harness-owned tmux session running claude attach <session_id>.

    The tmux session is created detached (-d); output is captured via
    pipe-pane into *output_path*.  Callers are responsible for
    verifying the result and recording the session name in job meta.
    """
    args: list[str] = [
        "tmux", "new-session", "-d", "-s", tmux_session_name,
        "-c", cwd,
        "claude", "attach", session_id,
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return result

    pipe_result = subprocess.run(
        ["tmux", "pipe-pane", "-t", tmux_session_name, "-o",
         f"cat >> {shlex.quote(str(output_path))}"],
        capture_output=True, text=True, timeout=10,
    )
    if pipe_result.returncode == 0:
        return result

    # An interactive job without a readable transcript is unusable: job_tail
    # cannot report the model's output. Do not leave an orphaned attach pane.
    subprocess.run(
        ["tmux", "kill-session", "-t", tmux_session_name],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return subprocess.CompletedProcess(
        args=args,
        returncode=pipe_result.returncode,
        stdout=pipe_result.stdout,
        stderr=pipe_result.stderr,
    )


class ClaudeAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="claude",
            support_tier=SUPPORT_TIER,
            backend="claude_bg",
            supports_interactive=True,
            effort_map=CLAUDE_EFFORT_MAP,
        )

    def discover_models(
        self,
        probe: ClaudeModelProbe | None = None,
        help_output: str | None = None,
    ) -> ModelCatalog:
        """Discover models via the interactive ``/model`` picker PTY probe.

        When *probe* is None (production), a ``PosixClaudeModelProbe`` is
        used.  Tests inject a fake probe for deterministic output.

        Also runs ``claude --help`` (or uses *help_output* when provided)
        to extract the CLI's advertised effort values for catalog exposure.

        On any probe failure or unrecognized picker format, returns an
        honest empty catalog with ``error`` set — never fakes a model list.
        """
        if probe is None:
            probe = PosixClaudeModelProbe()

        result = probe.probe()
        data = probe_to_catalog(result)

        # Augment native_efforts with help-based discovery
        native_efforts = tuple(data.get("native_efforts", []))
        if help_output is not None:
            help_efforts = parse_claude_help_efforts(help_output)
        else:
            try:
                help_result = subprocess.run(
                    ["claude", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                help_efforts = parse_claude_help_efforts(
                    help_result.stdout if help_result.returncode == 0 else ""
                )
            except Exception:
                help_efforts = ()
        if help_efforts:
            native_efforts = tuple(sorted(set(native_efforts) | set(help_efforts)))

        # Build model_info, merging help-discovered efforts into each model's
        # supported_efforts when the picker didn't surface per-model efforts.
        raw_model_info = data.get("model_info", [])
        merged_model_info: list[ModelInfo] = []
        for mi in raw_model_info:
            picker_efforts = set(mi.get("supported_efforts", []))
            if help_efforts:
                merged_efforts = tuple(sorted(picker_efforts | set(help_efforts)))
            else:
                merged_efforts = tuple(picker_efforts)
            merged_model_info.append(
                ModelInfo(
                    id=mi["id"],
                    supported_efforts=merged_efforts,
                    default_effort=mi.get("default_effort"),
                )
            )

        return ModelCatalog(
            models=tuple(data.get("models", [])),
            default_model=data.get("default_model"),
            native_efforts=native_efforts,
            source=data.get("source", "claude interactive /model picker"),
            error=data.get("error"),
            model_info=tuple(merged_model_info),
        )

    def check_readiness(self, runner: SubprocessRunner) -> ReadinessResult:
        try:
            raw = runner.run(["claude", "auth", "status", "--json"], timeout=15)
        except FileNotFoundError:
            return ReadinessResult(
                state="missing_binary",
                authenticated=False,
                error_code="claude_binary_missing",
                remediation="Install Claude Code and retry.",
            )
        parsed = parse_claude_auth(raw)
        if not parsed["authenticated"]:
            return ReadinessResult(
                state="needs_auth",
                authenticated=False,
                error_code=parsed["error"],
                remediation=parsed["remediation"],
                evidence=parsed["evidence"],
            )
        return ReadinessResult(
            state="ready",
            authenticated=True,
            auth_mode=parsed["auth_mode"],
            billing_mode=parsed["billing_mode"],
            subscription_type=parsed["subscription_type"],
            evidence=parsed["evidence"],
        )

    def launch(
        self,
        runner: SubprocessRunner,
        *,
        model: str | None = None,
        task: str,
        prompt: str,
        cwd: str,
        effort: str = "medium",
        interactive: bool = False,
    ) -> LaunchResult:
        plan = build_claude_launch(
            model=model,
            task=task,
            prompt=prompt,
            cwd=cwd,
            effort=effort,
            interactive=interactive,
        )
        if plan.error:
            return plan
        result = runner.run(plan.args, timeout=30, cwd=plan.cwd)
        if result.returncode != 0:
            return LaunchResult(
                None,
                plan.backend,
                args=plan.args,
                cwd=plan.cwd,
                permission_mode=plan.permission_mode,
                error="launch_failed",
                message=(result.stderr or result.stdout)[:500],
            )
        match = _BACKGROUND_ID_RE.search(result.stdout)
        if not match:
            return LaunchResult(
                None,
                plan.backend,
                args=plan.args,
                cwd=plan.cwd,
                permission_mode=plan.permission_mode,
                error="session_id_missing",
                message=result.stdout[:500],
            )
        return LaunchResult(
            match.group(1),
            plan.backend,
            args=plan.args,
            cwd=plan.cwd,
            permission_mode=plan.permission_mode,
        )

    def status(self, runner: SubprocessRunner, session_id: str) -> dict[str, Any]:
        result = runner.run(["claude", "agents", "--json", "--all"], timeout=15)
        if result.returncode != 0:
            return {"state": "failed", "error": (result.stderr or result.stdout)[:500]}
        for session in parse_agents_json(result.stdout):
            if session["id"] == session_id or session["session_id"] == session_id:
                return session
        return {"state": "failed", "error": "session_not_found"}

    def cancel(self, runner: SubprocessRunner, session_id: str) -> bool:
        return runner.run(["claude", "stop", session_id], timeout=15).returncode == 0

    def get_logs(self, runner: SubprocessRunner, session_id: str) -> str:
        result = runner.run(["claude", "logs", session_id], timeout=15)
        return result.stdout if result.returncode == 0 else ""

    def normalize_result(self, entry: dict[str, Any], logs: str) -> NormalizedResult:
        """Normalize a Claude agents entry + logs into a lifecycle result."""
        return normalize_claude_result(entry, logs)


adapter = ClaudeAdapter()
