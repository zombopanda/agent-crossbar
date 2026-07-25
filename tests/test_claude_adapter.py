"""Contract tests for the Claude native background-session adapter."""

from __future__ import annotations

import json
import re

import pytest

from agent_crossbar.adapters.claude import (
    ClaudeAdapter,
    RunResult,
    build_claude_launch,
    normalize_claude_result,
    normalize_session_state,
    parse_agents_json,
    parse_claude_auth,
)


class FakeRunner:
    def __init__(self, results: list[RunResult] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict[str, object]] = []

    def run(self, args, *, timeout=None, cwd=None, env=None):
        self.calls.append({"args": list(args), "timeout": timeout, "cwd": cwd, "env": env})
        if not self.results:
            raise AssertionError(f"Unexpected subprocess call: {args}")
        return self.results.pop(0)


def _auth(**overrides):
    payload = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "team",
    }
    payload.update(overrides)
    return RunResult(0, json.dumps(payload), "")


def _agents(*entries):
    return RunResult(0, json.dumps(list(entries)), "")


def test_auth_parser_uses_native_auth_status_fields() -> None:
    result = parse_claude_auth(_auth())

    assert result["authenticated"] is True
    assert result["auth_mode"] == "subscription"
    assert result["subscription_type"] == "team"
    assert "email" not in result


def test_auth_parser_reports_explicit_api_key_as_api_billing() -> None:
    result = parse_claude_auth(
        _auth(authMethod="api_key", apiProvider="firstParty", subscriptionType=None)
    )

    assert result["auth_mode"] == "api_key"
    assert result["billing_mode"] == "api"


def test_auth_parser_rejects_logged_out_status() -> None:
    result = parse_claude_auth(_auth(loggedIn=False, authMethod="none"))

    assert result["authenticated"] is False
    assert result["error"] == "not_authenticated"
    assert result["remediation"] == "Run `claude auth login`."


def test_agents_parser_matches_real_cli_array_and_fields() -> None:
    raw = _agents(
        {
            "id": "e2accc98",
            "sessionId": "e2accc98-9fd6-4813-a6e6-3cb0d134de46",
            "state": "done",
            "status": "idle",
            "waitingFor": None,
            "cwd": "/repo",
            "startedAt": 1784739414928,
        }
    ).stdout

    parsed = parse_agents_json(raw)

    assert parsed == [
        {
            "id": "e2accc98",
            "session_id": "e2accc98-9fd6-4813-a6e6-3cb0d134de46",
            "state": "done",
            "process_status": "idle",
            "waiting_for": None,
            "cwd": "/repo",
            "started_at_ms": 1784739414928,
        }
    ]


@pytest.mark.parametrize(
    ("native", "normalized"),
    [
        ("working", "running"),
        ("blocked", "waiting"),
        ("done", "completed"),
        ("failed", "failed"),
        ("stopped", "cancelled"),
    ],
)
def test_native_states_are_normalized(native: str, normalized: str) -> None:
    assert normalize_session_state(native) == normalized


def test_launch_uses_subprocess_cwd_and_never_invents_cwd_flag() -> None:
    plan = build_claude_launch(
        model="opus",
        task="review",
        prompt="review this",
        cwd="/repo",
        effort="high",
        interactive=False,
    )

    assert plan.error is None
    assert "--cwd" not in plan.args
    assert plan.cwd == "/repo"
    assert plan.args[:2] == ["claude", "--bg"]
    assert plan.args[plan.args.index("--model") + 1] == "claude-opus-5"
    assert "-p" not in plan.args and "--print" not in plan.args


def test_launch_without_model_uses_cli_default_without_inventing_opus() -> None:
    plan = build_claude_launch(
        model=None, task="ask", prompt="answer", cwd="/repo", effort="medium"
    )

    assert plan.error is None
    assert "--model" not in plan.args
    assert "opus" not in plan.args


def test_review_launch_is_noninteractive_read_only_and_uses_empty_mcp_surface() -> None:
    plan = build_claude_launch(model="opus", task="review", prompt="review", cwd="/repo")

    assert plan.permission_mode == "dontAsk"
    assert plan.args[plan.args.index("--permission-mode") + 1] == "dontAsk"
    allowed = plan.args[plan.args.index("--allowedTools") + 1]
    assert "Read" in allowed
    assert "Bash(git diff:*)" in allowed
    assert "Edit" not in allowed
    assert "--strict-mcp-config" in plan.args
    assert "--safe-mode" in plan.args
    assert "--bare" not in plan.args
    assert "--ax-screen-reader" in plan.args


def test_dev_launch_never_bypasses_permissions_and_disables_bg_worktree_isolation() -> None:
    plan = build_claude_launch(model="sonnet", task="dev", prompt="implement", cwd="/repo")

    assert "bypassPermissions" not in plan.args
    assert "--dangerously-skip-permissions" not in plan.args
    assert plan.permission_mode == "auto"
    settings = json.loads(plan.args[plan.args.index("--settings") + 1])
    assert settings["worktree"]["bgIsolation"] == "none"


def test_launch_extracts_real_short_id_and_preserves_native_metadata() -> None:
    runner = FakeRunner(
        [RunResult(0, "backgrounded · e2accc98 · relay-test\n  claude attach e2accc98\n", "")]
    )
    adapter = ClaudeAdapter()

    result = adapter.launch(
        runner,
        model="opus",
        task="ask",
        prompt="hello",
        cwd="/repo",
        effort="low",
        interactive=True,
    )

    assert result.session_id == "e2accc98"
    assert result.backend == "claude_bg_pty"
    assert runner.calls[0]["cwd"] == "/repo"


def test_status_uses_short_or_full_session_id_and_native_state() -> None:
    runner = FakeRunner(
        [
            _agents(
                {
                    "id": "e2accc98",
                    "sessionId": "e2accc98-full",
                    "state": "blocked",
                    "status": "waiting",
                    "waitingFor": "permission prompt",
                    "cwd": "/repo",
                    "startedAt": 1,
                }
            )
        ]
    )

    status = ClaudeAdapter().status(runner, "e2accc98-full")

    assert status["state"] == "blocked"
    assert status["waiting_for"] == "permission prompt"


def test_readiness_uses_claude_auth_status_not_agents_listing() -> None:
    runner = FakeRunner([_auth()])

    readiness = ClaudeAdapter().check_readiness(runner)

    assert readiness.state == "ready"
    assert runner.calls[0]["args"] == ["claude", "auth", "status", "--json"]
    assert readiness.auth_mode == "subscription"


def test_model_discovery_never_labels_a_static_fallback_as_live() -> None:
    """Claude discover_models with probe error returns honest empty — never a static fallback."""
    from agent_crossbar.adapters.claude_model_probe import ProbeResult

    class FakeProbe:
        def probe(self) -> ProbeResult:
            return ProbeResult(output=None, error="interactive model probe unavailable")

    catalog = ClaudeAdapter().discover_models(probe=FakeProbe(), help_output="")

    assert catalog.models == ()
    assert catalog.default_model is None
    assert catalog.source != "static fallback"
    assert catalog.error is not None


def test_cancel_and_logs_use_native_commands() -> None:
    runner = FakeRunner([RunResult(0, "stopped", ""), RunResult(0, "final output", "")])
    adapter = ClaudeAdapter()

    assert adapter.cancel(runner, "e2accc98") is True
    assert adapter.get_logs(runner, "e2accc98") == "final output"
    assert runner.calls[0]["args"] == ["claude", "stop", "e2accc98"]
    assert runner.calls[1]["args"] == ["claude", "logs", "e2accc98"]


def test_result_normalization_uses_state_and_waiting_for() -> None:
    result = normalize_claude_result(
        {
            "id": "e2accc98",
            "session_id": "e2accc98-full",
            "state": "blocked",
            "waiting_for": "permission prompt",
        },
        "needs approval",
    )

    assert result.status == "waiting"
    assert result.session_id == "e2accc98"
    assert result.waiting_for == "permission prompt"


# ── Task 3.6: Claude effort in launch argv + catalog exposure (TDD) ──


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_claude_launch_includes_effort_in_argv(effort):
    """Claude launch argv includes --effort with the requested value."""
    from agent_crossbar.adapters.claude import build_claude_launch

    result = build_claude_launch(
        model="opus",
        task="ask",
        prompt="hello",
        cwd="/tmp",
        effort=effort,
    )
    assert result.error is None
    assert "--effort" in result.args
    idx = result.args.index("--effort")
    assert result.args[idx + 1] == effort


def test_claude_launch_rejects_unknown_effort():
    """Claude launch rejects an effort not in PUBLIC_EFFORTS."""
    from agent_crossbar.adapters.claude import build_claude_launch

    result = build_claude_launch(
        model="opus",
        task="ask",
        prompt="hello",
        cwd="/tmp",
        effort="xhigh",
    )
    assert result.error == "invalid_effort"


# ── ANSI / control character normalization ────────────────────────────────

CLAUDE_ANSI_OUTPUT = (
    "\x1b]777;notify;Claude Code;Claude is waiting for your input\x07"
    "\x1b[?25l\x1b[2K\r⏺ \x1b[1mGPT_PRO_PROVIDER_GATE_OK\x1b[0m\r\n"
    "\x1b[?25h"
    "\x1b]0;claude\x07"
    "\x1b[?2004h\r\n"
    "\x1b[?2004l"
)


def test_normalize_claude_result_strips_ansi_osc_and_control_chars():
    """Claude TUI output must be cleaned of raw ANSI/OSC/control sequences."""
    from agent_crossbar.adapters.claude import normalize_claude_result

    result = normalize_claude_result(
        {"id": "abc123", "state": "done"},
        CLAUDE_ANSI_OUTPUT,
    )

    assert result.status == "completed"
    # Must contain the visible sentinel text
    assert "GPT_PRO_PROVIDER_GATE_OK" in result.output
    # Must NOT contain raw escape sequences
    assert "\x1b" not in result.output
    assert "\x07" not in result.output
    # Must NOT contain control chars (below 0x20, except newline/tab)
    assert "\r" not in result.output
    assert "\x00" not in result.output


def test_normalize_claude_screen_reader_logs_returns_final_assistant_message():
    logs = """[Screen Reader Mode: on via flag]
Claude Code v2.1.211
$you: Reply with exactly GPT_PRO_PROVIDER_GATE_OK
$claude: G
Schlepping…   ( 1s )
$claude: GPT_PRO_PROVIDER_GATE_OK
Schlepping…   ( 2s · 6 tokens )
plan mode on (shift+tab to cycle)
36413 tokens
effort: medium · /effort
$
"""
    result = normalize_claude_result(
        {"id": "abc123", "state": "done"},
        logs,
    )

    assert result.output == "GPT_PRO_PROVIDER_GATE_OK"


def test_normalize_claude_screen_reader_logs_prefers_complete_final_redraw():
    logs = """[Screen Reader Mode: on via flag]
$claude: Summary
I've completed a read-only advers
don't ask on (shift+tab to cycle)  ·  esc to interrupt
86436 tokens
$I've completed a read-only adversarial review of the README.
don't ask on (shift+tab to cycle)  ·  esc to interrupt
86436 tokens
$
1. Invalid model ID with ANSI escape code
  - The opus model ID contains terminal formatting.
2. Broken anchor link
  - The README link does not match a heading.
3. Misleading interactive support
  - The table should name both supported modes.
4. Vague language
  - Replace internal gate jargon with user-facing wording.
All findings are verified against the source code.
Unravelling…   ( 2m 47s · 13.8k tokens )
don't ask on (shift+tab to cycle)  ·  esc to interrupt
86436 tokens
$Worked for 2m 49s
"""
    result = normalize_claude_result(
        {"id": "abc123", "state": "done"},
        logs,
    )

    assert result.status == "completed"
    assert result.output.startswith("1. Invalid model ID")
    assert "4. Vague language" in result.output
    assert "don't ask on" not in result.output
    assert "86436 tokens" not in result.output


def test_normalize_claude_done_with_only_tui_chrome_is_not_a_false_success():
    result = normalize_claude_result(
        {"id": "abc123", "state": "done"},
        """[Screen Reader Mode: on via flag]
$claude: Summary
don't ask on (shift+tab to cycle) · esc to interrupt
0 tokens
$Worked for 2s
""",
    )

    assert result.status == "failed"
    assert result.stop_reason == "result_output_unavailable"
    assert result.error == "Claude completed without a recoverable final response."


def test_claude_static_model_ids_do_not_contain_terminal_formatting():
    from agent_crossbar.profiles.claude import CLAUDE_MODEL_IDS

    for model_id in CLAUDE_MODEL_IDS.values():
        assert "\x1b" not in model_id
        assert not re.search(r"\[[0-9;]*m", model_id)


def test_normalize_claude_result_clean_error_field():
    """Error field must also be stripped of ANSI when status is failed."""
    from agent_crossbar.adapters.claude import normalize_claude_result

    raw = "error: \x1b[31mfatal\x1b[0m\n"
    result = normalize_claude_result(
        {"id": "abc123", "state": "failed"},
        raw,
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "\x1b" not in result.error
    assert "fatal" in result.error
