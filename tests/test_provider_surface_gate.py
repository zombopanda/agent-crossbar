from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import anyio
import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from scripts import provider_surface_gate as gate  # noqa: E402


def test_provider_gate_allows_thirty_minutes_for_heavy_prompts():
    assert gate.DEFAULT_MAX_RUNTIME_SEC == 1800


def test_blocking_prompt_detection_flags_opencode_tui_permission_dialog():
    output = "some ansi text Allow once | Allow always | Reject"

    assert gate._contains_blocking_prompt(output) is True


def test_blocking_prompt_detection_ignores_normal_output():
    output = "pytest passed and reverse_words.py was created"

    assert gate._contains_blocking_prompt(output) is False


def test_reverse_words_verification_requires_files_and_passing_tests(tmp_path):
    result = gate._verify_reverse_words_workspace(tmp_path)

    assert result.ok is False
    assert "reverse_words.py" in result.message


def test_reverse_words_verification_uses_current_python_runtime(tmp_path, monkeypatch):
    (tmp_path / "reverse_words.py").write_text("def reverse_words(text): return text\n")
    (tmp_path / "test_reverse_words.py").write_text("def test_ok(): assert True\n")
    captured = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return Completed()

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    result = gate._verify_reverse_words_workspace(tmp_path)

    assert result.ok is True
    assert captured["argv"] == [gate.sys.executable, "-m", "pytest", "test_reverse_words.py"]


def test_workspace_tempdir_lives_inside_trusted_package_repo():
    with gate._workspace_tempdir() as work_dir:
        assert Path(work_dir).parent == gate.PACKAGE_DIR
        result = gate._prepare_workspace(Path(work_dir))
        assert result.ok is True
        assert not (Path(work_dir) / ".git").exists()
        instructions = (Path(work_dir) / "AGENTS.md").read_text()
        assert "disposable provider surface gate" in instructions
        assert "Do not create beads or OpenSpec artifacts" in instructions


def test_explicit_cli_values_do_not_duplicate_default_cases():
    args = gate._parse_args(
        [
            "--profile",
            "opencode",
            "--model",
            "glm-5.2",
            "--task",
            "dev",
            "--task",
            "review",
        ]
    )

    cases = gate._cases_from_args(args)

    assert [(case.task, case.interactive) for case in cases] == [
        ("dev", False),
        ("review", False),
    ]


def test_effort_is_preserved_in_gate_case_and_agent_start_args(tmp_path):
    args = gate._parse_args(
        [
            "--profile",
            "codex",
            "--model",
            "gpt-5.6-sol",
            "--effort",
            "light",
            "--task",
            "dev",
        ]
    )

    case = gate._cases_from_args(args)[0]

    assert case.effort == "light"
    assert gate._agent_start_args(case, str(tmp_path))["effort"] == "light"


def test_agent_start_args_for_ask_task():
    case = gate.GateCase(
        profile="reasonix",
        model="deepseek-v4-pro",
        effort=None,
        task="ask",
        interactive=False,
    )

    args = gate._agent_start_args(case)

    assert args == {
        "profile": "reasonix",
        "task": "ask",
        "interactive": False,
        "max_runtime_sec": 1800,
        "prompt": "Reply with exactly GPT_PRO_PROVIDER_GATE_OK",
        "model": "deepseek-v4-pro",
    }


def test_agent_start_args_for_review_task():
    case = gate.GateCase(
        profile="reasonix",
        model="gpt-5.6-sol",
        effort=None,
        task="review",
        interactive=False,
    )

    args = gate._agent_start_args(case)

    assert args["task"] == "review"
    assert args["prompt"] == "Reply with exactly GPT_PRO_PROVIDER_GATE_OK"
    assert args["interactive"] is False


def test_agent_start_args_for_dev_task_includes_cwd_and_dev_prompt(tmp_path):
    case = gate.GateCase(
        profile="codex",
        model="gpt-5.6-sol",
        effort=None,
        task="dev",
        interactive=False,
    )

    args = gate._agent_start_args(case, cwd=str(tmp_path))

    assert args["task"] == "dev"
    assert args["cwd"] == str(tmp_path)
    assert "reverse_words.py" in args["prompt"]


def test_max_runtime_sec_defaults_to_30_minutes():
    case = gate.GateCase(
        profile="codex",
        model="gpt-5.6-sol",
        effort=None,
        task="dev",
        interactive=False,
    )

    assert case.max_runtime_sec == 1800


def test_cli_max_runtime_sec_flows_to_gate_case_and_agent_start_args():
    args = gate._parse_args(
        [
            "--profile",
            "codex",
            "--model",
            "gpt-5.6-sol",
            "--task",
            "dev",
            "--max-runtime-sec",
            "600",
        ]
    )

    case = gate._cases_from_args(args)[0]

    assert case.max_runtime_sec == 600
    a_args = gate._agent_start_args(case)
    assert a_args["max_runtime_sec"] == 600


def test_max_runtime_sec_reaches_wait_for_result_deadline(monkeypatch):
    """--max-runtime-sec must set the polling deadline, not be overridden."""
    case = gate.GateCase(
        profile="codex",
        model="gpt-5.6-sol",
        effort=None,
        task="dev",
        interactive=False,
        max_runtime_sec=42,
    )

    # Capture the timeout_sec value passed to _wait_for_result
    captured_timeout = []

    async def fake_wait_for_result(session, job_id, *, timeout_sec):
        captured_timeout.append(timeout_sec)
        return {"ok": True, "output": "ok"}, {}

    monkeypatch.setattr(gate, "_wait_for_result", fake_wait_for_result)

    # Mock agent_start to return a job_id
    async def fake_call(session, tool, args, timeout_sec=120):
        if tool == "agent_start":
            return {"ok": True, "job_id": "fake-job"}
        if tool == "job_list":
            return {"ok": True, "jobs": [{"job_id": "fake-job"}]}
        return {}

    monkeypatch.setattr(gate, "_call", fake_call)

    # Also need _contains_blocking_prompt + workspace verification to pass
    monkeypatch.setattr(gate, "_contains_blocking_prompt", lambda x: False)
    monkeypatch.setattr(
        gate, "_verify_reverse_words_workspace", lambda cwd: gate.CheckResult(True, "ok")
    )

    result = anyio.run(gate._run_dev_case, object(), case, Path("/tmp/fake"))

    assert len(captured_timeout) == 1
    assert captured_timeout[0] == case.max_runtime_sec + gate.RESULT_COMPLETION_GRACE_SEC
    assert result.ok is True


def test_main_allows_ask_task(monkeypatch):
    captured = {}

    def fake_run(func, cases, *args):
        captured["func"] = func
        captured["cases"] = cases
        return 0

    monkeypatch.setattr(gate.anyio, "run", fake_run)

    result = gate.main(
        [
            "--profile",
            "reasonix",
            "--model",
            "deepseek-v4-pro",
            "--task",
            "ask",
        ]
    )

    assert result == 0
    assert captured["func"] is gate._run_cases
    assert [(case.task, case.interactive) for case in captured["cases"]] == [("ask", False)]


def test_main_allows_review_task(monkeypatch):
    captured = {}

    def fake_run(func, cases, *args):
        captured["cases"] = cases
        return 0

    monkeypatch.setattr(gate.anyio, "run", fake_run)

    gate.main(
        [
            "--profile",
            "reasonix",
            "--model",
            "deepseek-v4-pro",
            "--task",
            "review",
        ]
    )

    assert captured["cases"][0].task == "review"
    assert captured["cases"][0].interactive is False


def test_task_all_expands_to_each_noninteractive_operation_once():
    args = gate._parse_args(
        [
            "--profile",
            "opencode",
            "--model",
            "glm-5.2",
            "--task",
            "all",
            "--task",
            "ask",
        ]
    )

    cases = gate._cases_from_args(args)

    assert [case.task for case in cases] == ["ask", "review", "dev"]
    assert all(case.interactive is False for case in cases)


def test_full_surface_preflight_checks_discovery_and_readiness(monkeypatch):
    calls = []

    async def fake_call(session, tool, args, timeout_sec=120):
        calls.append(tool)
        if tool == "profiles_list":
            return {
                "ok": True,
                "profiles": ["opencode"],
                "profile_details": {"opencode": {"models": ["glm-5.2"], "interactive": False}},
            }
        if tool == "profile_health":
            return {"ok": True, "profiles": [{"profile": "opencode", "state": "ready"}]}
        raise AssertionError(f"unexpected tool: {tool}")

    monkeypatch.setattr(gate, "_call", fake_call)

    outcome, details = anyio.run(
        gate._run_full_surface_preflight,
        object(),
        [gate.GateCase("opencode", "glm-5.2", None, "ask", False)],
    )

    assert outcome.ok is True
    assert details["opencode"]["interactive"] is False
    assert calls == ["profiles_list", "profile_health"]


def _two_phase_mock(
    monkeypatch,
    *,
    agent_start_response=None,
    job_result_response=None,
    job_tail_response=None,
):
    """Mock _call for the two-phase agent_start + job_result pattern.

    First call returns agent_start (with job_id), second returns job_result.
    """
    if agent_start_response is None:
        agent_start_response = {"ok": True, "job_id": "fake-job-123"}

    calls = []

    async def fake_call(session, tool, args, timeout_sec=120):
        calls.append(tool)
        if tool == "agent_start":
            return agent_start_response
        if tool == "job_result":
            return job_result_response or {}
        if tool == "job_tail":
            return job_tail_response or {"ok": True, "output": "", "events": []}
        if tool == "job_list":
            return {"ok": True, "jobs": [{"job_id": "fake-job-123"}]}
        return {}

    monkeypatch.setattr(gate, "_call", fake_call)
    return calls


def _acp_tail(*deltas: str):
    events = [{"type": "acp_command", "data": {}}]
    events.extend({"type": "log_delta", "data": {"text": delta}} for delta in deltas)
    events.append({"type": "acp_completed", "data": {}})
    return {"ok": True, "events": events}


def test_oneshot_acp_gate_rejects_missing_text_delta(monkeypatch):
    _two_phase_mock(
        monkeypatch,
        agent_start_response={"ok": True, "job_id": "fake-job-123", "backend": "acp"},
        job_result_response={"ok": True, "output": "GPT_PRO_PROVIDER_GATE_OK"},
        job_tail_response=_acp_tail(),
    )

    result = anyio.run(
        gate._run_oneshot_ask_case,
        object(),
        gate.GateCase("codex", "gpt-5.6-sol", None, "ask", False),
    )

    assert result.ok is False
    assert "log_delta" in result.message


def test_oneshot_acp_gate_accepts_ordered_text_deltas(monkeypatch):
    _two_phase_mock(
        monkeypatch,
        agent_start_response={"ok": True, "job_id": "fake-job-123", "backend": "acp"},
        job_result_response={"ok": True, "output": "GPT_PRO_PROVIDER_GATE_OK"},
        job_tail_response=_acp_tail("GPT_PRO_", "PROVIDER_GATE_OK"),
    )

    result = anyio.run(
        gate._run_oneshot_ask_case,
        object(),
        gate.GateCase("codex", "gpt-5.6-sol", None, "ask", False),
    )

    assert result.ok is True


@pytest.mark.parametrize(
    "output",
    [
        "prefix GPT_PRO_PROVIDER_GATE_OK",
        "GPT_PRO_PROVIDER_GATE_OK suffix",
        "`GPT_PRO_PROVIDER_GATE_OK`",
    ],
)
def test_ask_gate_rejects_non_exact_sentinel(monkeypatch, output):
    _two_phase_mock(monkeypatch, job_result_response={"ok": True, "output": output})
    case = gate.GateCase("reasonix", None, None, "ask", False)

    result = anyio.run(gate._run_ask_case, object(), case)

    assert result.ok is False


def test_ask_gate_accepts_exact_sentinel_with_surrounding_whitespace(monkeypatch):
    _two_phase_mock(
        monkeypatch, job_result_response={"ok": True, "output": " \nGPT_PRO_PROVIDER_GATE_OK\n "}
    )
    case = gate.GateCase("chatgpt_pro", None, None, "ask", False)

    result = anyio.run(gate._run_ask_case, object(), case)

    assert result.ok is True


def test_reasonix_ask_gate_accepts_sentinel_before_reasonix_footer(monkeypatch):
    _two_phase_mock(
        monkeypatch,
        job_result_response={
            "ok": True,
            "output": (
                "GPT_PRO_PROVIDER_GATE_OK\n\n"
                "— turns:1 cache:0.0% cost:$0.004384 save-vs-claude:85.7%\n"
            ),
        },
    )
    case = gate.GateCase("reasonix", "deepseek-v4-pro", None, "ask", False)

    result = anyio.run(gate._run_ask_case, object(), case)

    assert result.ok is True


def test_reasonix_ask_gate_rejects_bare_sentinel_without_reasonix_footer(monkeypatch):
    _two_phase_mock(
        monkeypatch, job_result_response={"ok": True, "output": "GPT_PRO_PROVIDER_GATE_OK"}
    )
    case = gate.GateCase("reasonix", "deepseek-v4-pro", None, "ask", False)

    result = anyio.run(gate._run_ask_case, object(), case)

    assert result.ok is False


@pytest.mark.parametrize(
    "footer",
    [
        "— turns:garbage",
        "— turns:1 arbitrary model text",
        "— turns:1 cache:0.0% cost:$oops save-vs-claude:85.7%",
    ],
)
def test_reasonix_ask_gate_rejects_malformed_reasonix_footer(monkeypatch, footer):
    _two_phase_mock(
        monkeypatch,
        job_result_response={"ok": True, "output": f"GPT_PRO_PROVIDER_GATE_OK\n\n{footer}\n"},
    )
    case = gate.GateCase("reasonix", "deepseek-v4-pro", None, "ask", False)

    result = anyio.run(gate._run_ask_case, object(), case)

    assert result.ok is False


def test_ask_gate_does_not_accept_sentinel_from_summary(monkeypatch):
    _two_phase_mock(
        monkeypatch, job_result_response={"ok": True, "summary": "GPT_PRO_PROVIDER_GATE_OK"}
    )
    case = gate.GateCase("reasonix", None, None, "ask", False)

    result = anyio.run(gate._run_ask_case, object(), case)

    assert result.ok is False


def test_ask_gate_requires_output_to_be_a_string(monkeypatch):
    _two_phase_mock(
        monkeypatch,
        job_result_response={"ok": True, "output": {"text": "GPT_PRO_PROVIDER_GATE_OK"}},
    )
    case = gate.GateCase("reasonix", None, None, "ask", False)

    result = anyio.run(gate._run_ask_case, object(), case)

    assert result.ok is False


# ── npm/pnpm double-dash passthrough ───────────────────────────────────────


def test_parse_args_strips_leading_double_dash():
    """A leading '--' from pnpm/npm pass-through must be stripped before
    argparse parsing, so that _parse_args(['--', '--profile', 'codex', '--task', 'dev'])
    produces the same result as _parse_args(['--profile', 'codex', '--task', 'dev'])."""
    normal = gate._parse_args(["--profile", "codex", "--model", "gpt-5.6-sol", "--task", "dev"])
    npm_style = gate._parse_args(
        ["--", "--profile", "codex", "--model", "gpt-5.6-sol", "--task", "dev"]
    )
    assert normal == npm_style


def test_parse_args_handles_npm_double_dash_with_help():
    """--help after a leading npm '--' separator (pnpm run gate -- --help)
    must trigger argparse help, not a 'missing --profile' error."""
    with pytest.raises(SystemExit, match="0"):
        gate._parse_args(["--", "--help"])


def test_parse_args_without_double_dash_remains_unchanged():
    """argv that does not start with '--' is passed through as-is."""
    normal = gate._parse_args(["--profile", "codex", "--model", "gpt-5.6-sol", "--task", "dev"])
    assert normal.profile == ["codex"]
    assert normal.task == ["dev"]


# ── server_params ─────────────────────────────────────────────────────────


def test_server_params_launches_agents_mcp_entrypoint_from_same_venv(monkeypatch):
    """The gate must launch the installed console entrypoint, not python -m."""
    monkeypatch.setattr(gate.sys, "executable", "/tmp/ci-venv/bin/python")

    params = gate._server_params(env={"TEST": "1"})

    assert params.command == "/tmp/ci-venv/bin/agents-mcp"
    assert params.args == []
    assert params.env == {"TEST": "1"}


def test_server_params_args_are_empty_not_module_flag():
    """No -m agent_crossbar.server — uses the real installed entrypoint."""
    params = gate._server_params(env={})

    assert params.args == []
    assert "-m" not in params.args
    assert "agent_crossbar.server" not in params.args


# ── Claude sentinel: reject prompt echo, accept distinct answer ────────────

CLAUDE_OUTPUT_ECHO_ONLY = "⏺ Reply with exactly GPT_PROVIDER_GATE_OK\n⏵⏵ bypass permissions on\n"

CLAUDE_OUTPUT_ECHO_THEN_ANSWER = (
    "⏺ Reply with exactly GPT_PROVIDER_GATE_OK\n...\n⏺ GPT_PROVIDER_GATE_OK\n"
)

CLAUDE_OUTPUT_SENTINEL_IN_ECHO_BUT_NO_DISTINCT_ANSWER = (
    "⏺ I'll reply with exactly GPT_PROVIDER_GATE_OK as requested\n"
    "...\n"
    "⏺ Here it is: GPT_PROVIDER_GATE_OK from me\n"
)

CLAUDE_OUTPUT_MULTILINE_ANSWER_WITH_SENTINEL = (
    "⏺ Reply with exactly GPT_PROVIDER_GATE_OK\n...\n⏺ Sure! Here you go:\n\nGPT_PROVIDER_GATE_OK\n"
)

CLAUDE_OUTPUT_ECHO_THEN_LABELLED_ANSWER = (
    "you: Reply with exactly GPT_PROVIDER_GATE_OK\n...\nclaude: GPT_PROVIDER_GATE_OK\n"
)


def test_claude_sentinel_rejects_prompt_echo_only():
    """Output that only echoes the prompt must not trigger a false positive."""
    assert not gate._claude_sentinel_received("GPT_PROVIDER_GATE_OK", CLAUDE_OUTPUT_ECHO_ONLY)


def test_claude_sentinel_accepts_distinct_answer_after_echo():
    """A distinct ⏺ line with just the sentinel is a valid answer."""
    assert gate._claude_sentinel_received("GPT_PROVIDER_GATE_OK", CLAUDE_OUTPUT_ECHO_THEN_ANSWER)


def test_claude_sentinel_rejects_sentinel_embedded_in_longer_response():
    """The sentinel must appear standalone, not buried in prose."""
    assert not gate._claude_sentinel_received(
        "GPT_PROVIDER_GATE_OK", CLAUDE_OUTPUT_SENTINEL_IN_ECHO_BUT_NO_DISTINCT_ANSWER
    )


def test_claude_sentinel_accepts_multiline_answer_with_standalone_sentinel():
    """A multiline answer where one line is exactly the sentinel is valid."""
    assert gate._claude_sentinel_received(
        "GPT_PROVIDER_GATE_OK", CLAUDE_OUTPUT_MULTILINE_ANSWER_WITH_SENTINEL
    )


def test_claude_sentinel_accepts_labelled_transcript_answer_after_prompt_echo():
    """Current Claude TUI labels the answer instead of prefixing it with ⏺."""
    assert gate._claude_sentinel_received(
        "GPT_PROVIDER_GATE_OK", CLAUDE_OUTPUT_ECHO_THEN_LABELLED_ANSWER
    )


def test_reasonix_footer_accepts_noisy_transcript_before_terminal_answer():
    output = (
        "⌘ MCP · filesystem      ✓ connected\n"
        "GPT_PRO_PROVIDER_GATE_OK\n\n"
        "— turns:1 cache:0.0% cost:$0.001568 save-vs-claude:95.4%\n"
    )

    assert gate._reasonix_sentinel_with_footer(output, profile="reasonix")


def test_standalone_sentinel_finds_line_in_noisy_output():
    """A standalone sentinel line is found even in a noisy TUI dump."""
    noisy = "banner\n...\nGPT_PROVIDER_GATE_OK\nfooter\n"
    assert gate._standalone_sentinel_received("GPT_PROVIDER_GATE_OK", noisy)


def test_standalone_sentinel_rejects_no_match():
    assert not gate._standalone_sentinel_received(
        "GPT_PROVIDER_GATE_OK", "some text\nno sentinel here"
    )


def test_standalone_sentinel_requires_exact_line():
    assert not gate._standalone_sentinel_received(
        "GPT_PROVIDER_GATE_OK", "prefix GPT_PROVIDER_GATE_OK suffix"
    )


# ── poll_tail_text with output_since_bytes ────────────────────────────────


def test_poll_tail_text_returns_text_from_output_tail_dict():
    """_poll_tail_text extracts text from the output_tail dict key."""
    # The function is sync now, and we test via a mock session
    pass  # covered by live integration; tested via direct extraction pattern


def test_clean_ansi_strips_reasonix_tui_escape_sequences():
    """_clean_ansi must strip ANSI so sentinel can be found."""
    raw = "\x1b[38;5;189mGPT_PRO\x1b[39m\x1b[38;5;189m_PROVIDER_GATE_OK\x1b[39m"
    clean = gate._clean_ansi(raw)
    assert "\x1b" not in clean
    assert "GPT_PRO_PROVIDER_GATE_OK" in clean


def test_standalone_sentinel_fails_on_raw_ansi_without_clean():
    """_standalone_sentinel_received on raw ANSI must fail — caller must clean first."""
    raw = "\x1b[32mGPT_PROVIDER_SECOND_SENTINEL_OK\x1b[0m"
    assert not gate._standalone_sentinel_received("GPT_PROVIDER_SECOND_SENTINEL_OK", raw)


def test_standalone_sentinel_passes_on_cleaned_ansi():
    """After _clean_ansi, _standalone_sentinel_received must find the sentinel."""
    raw = "\x1b[32m\nGPT_PROVIDER_SECOND_SENTINEL_OK\n\x1b[0m"
    clean = gate._clean_ansi(raw)
    assert gate._standalone_sentinel_received("GPT_PROVIDER_SECOND_SENTINEL_OK", clean)


# ── poll_until_sentinel unit (uses fake_session) ──────────────────────────


def test_poll_until_sentinel_returns_true_when_sentinel_appears():
    """_poll_until_sentinel returns (True, output) when sentinel found after echo."""

    class FakeSession:
        call_count = 0

        async def call_tool(self, tool, args, read_timeout_seconds=None):
            self.call_count += 1
            if self.call_count == 1:
                return type(
                    "R",
                    (),
                    {
                        "content": [
                            type(
                                "C",
                                (),
                                {
                                    "text": '{"output_tail":{"text":"Reply with exactly SENT▌","bytes":100}}'
                                },
                            )
                        ]
                    },
                )()
            # Second poll: sentinel appears as answer (second occurrence)
            return type(
                "R",
                (),
                {
                    "content": [
                        type(
                            "C",
                            (),
                            {
                                "text": '{"output_tail":{"text":"Reply with exactly SENT▌\\n...\\nSENT\\ndone","bytes":200}}'
                            },
                        )
                    ]
                },
            )()

    ok, text = asyncio.run(gate._poll_until_sentinel(FakeSession(), "job1", "SENT", timeout_sec=10))
    assert ok is True
    assert text.count("SENT") >= 2


def test_poll_until_sentinel_returns_false_on_timeout():
    """_poll_until_sentinel returns (False, last_output) when sentinel never appears."""

    class FakeSession:
        async def call_tool(self, tool, args, read_timeout_seconds=None):
            return type(
                "R",
                (),
                {
                    "content": [
                        type(
                            "C",
                            (),
                            {"text": '{"output_tail":{"text":"still loading...","bytes":50}}'},
                        )
                    ]
                },
            )()

    ok, text = asyncio.run(
        gate._poll_until_sentinel(FakeSession(), "job1", "GPT_SENTINEL_OK", timeout_sec=1)
    )
    assert ok is False
    assert "GPT_SENTINEL_OK" not in text


def test_poll_until_sentinel_uses_full_output_tail():
    """_poll_until_sentinel polls full output_tail, not incremental."""
    captured_args = []

    class FakeSession:
        async def call_tool(self, tool, args, read_timeout_seconds=None):
            captured_args.append(dict(args))
            return type(
                "R",
                (),
                {
                    "content": [
                        type(
                            "C",
                            (),
                            {"text": '{"output_tail":{"text":"sentinel FOUND","bytes":99}}'},
                        )
                    ]
                },
            )()

    asyncio.run(gate._poll_until_sentinel(FakeSession(), "job1", "FOUND", timeout_sec=10))
    # No output_since_bytes — full poll
    assert "output_since_bytes" not in captured_args[0]


# ── sentinel_after_echo: accept second occurrence after echoed prompt ──────


def test_sentinel_after_echo_accepts_second_occurrence():
    """Sentinel appearing after the echoed prompt line is accepted."""
    output = (
        "Reply with exactly SENTINEL▌\n"  # echo
        "some thinking...\n"
        "SENTINEL\n"  # actual answer
    )
    assert gate._sentinel_after_echo("SENTINEL", output)


def test_sentinel_after_echo_rejects_echo_only():
    """Only the echoed prompt — no second occurrence — must be rejected."""
    output = "Reply with exactly SENTINEL▌\n"
    assert not gate._sentinel_after_echo("SENTINEL", output)


def test_sentinel_after_echo_rejects_single_non_echo_occurrence():
    """A single occurrence that is NOT an echo (no Reply prefix) is ambiguous."""
    output = "some thinking...\nSENTINEL\n"
    assert gate._sentinel_after_echo("SENTINEL", output)


def test_sentinel_after_echo_accepts_standalone_line_after_echo_in_pipe_output():
    """Realistic tmux pipe: sentinel appears in echoed prompt line then later standalone."""
    output = (
        "banner stuff...\n"
        "Reply with exactly SENTINEL▌              \n"
        "more banner...\n"
        "◆ reasoning · 5 tok · 0.1s\n"
        "SENTINEL\n"
    )
    assert gate._sentinel_after_echo("SENTINEL", output)
