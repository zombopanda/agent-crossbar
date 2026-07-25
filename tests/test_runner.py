from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_crossbar import runner as runner_module
from agent_crossbar.jobs import JobStore
from agent_crossbar.runner import (
    CommandCandidate,
    CuaDriverClient,
    _active_chatgpt_model_is_pro,
    _extract_marked_response,
    _tmux_shell_command,
    run_gui_request,
    run_print_job,
    run_print_request,
)


def test_runner_default_timeout_is_thirty_minutes():
    assert runner_module._DEFAULT_TIMEOUT_SEC == 1800


def test_codex_runtime_candidate_passes_validated_model_and_effort():
    candidate = runner_module._candidates(
        {
            "profile": "codex",
            "operation": "dev",
            "transport": "print",
            "model": "gpt-5.6-terra",
            "effort": "xhigh",
            "cwd": "/repo",
        },
        "implement",
    )[0]

    assert candidate.argv == [
        "codex",
        "exec",
        "--ephemeral",
        "--model",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="xhigh"',
        "-C",
        "/repo",
        "implement",
    ]


def test_codex_tmux_candidate_passes_validated_model_and_effort(tmp_path):
    candidates, error = runner_module._tmux_candidates(
        {
            "profile": "codex",
            "operation": "dev",
            "transport": "tmux",
            "model": "gpt-5.6-terra",
            "effort": "xhigh",
            "cwd": "/repo",
            "prompt": "do it",
        },
        tmp_path,
    )

    assert error is None
    assert candidates[0].argv[:6] == [
        "codex",
        "--model",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="xhigh"',
        "--no-alt-screen",
    ]


def _completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_cua_driver_client_uses_configured_call_timeout(monkeypatch):
    calls: list[dict] = []

    def fake_run(_args, **kwargs):
        calls.append(kwargs)
        return _completed('{"ok": true}')

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    result = CuaDriverClient(bin_path="/tmp/cua-driver", call_timeout_sec=75).call("list_apps", {})

    assert result == {"ok": True}
    assert calls[0]["timeout"] == 75


def test_cua_driver_client_supports_a_shorter_per_call_timeout(monkeypatch):
    calls: list[dict] = []

    def fake_run(_args, **kwargs):
        calls.append(kwargs)
        return _completed('{"ok": true}')

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    result = CuaDriverClient(bin_path="/tmp/cua-driver", call_timeout_sec=180).call(
        "page", {"action": "get_text"}, timeout_sec=7
    )

    assert result == {"ok": True}
    assert calls[0]["timeout"] == 7


def test_cua_driver_client_honors_a_longer_explicit_per_call_timeout(monkeypatch):
    calls: list[dict] = []

    def fake_run(_args, **kwargs):
        calls.append(kwargs)
        return _completed('{"ok": true}')

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    result = CuaDriverClient(bin_path="/tmp/cua-driver", call_timeout_sec=60).call(
        "page", {"action": "get_text"}, timeout_sec=240
    )

    assert result == {"ok": True}
    assert calls[0]["timeout"] == 240


def test_reasonix_print_runner_uses_run_with_config_and_model():
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _completed("DEEPSEEK_OK\n")

    result = run_print_request(
        {
            "profile": "reasonix",
            "operation": "review",
            "transport": "print",
            "prompt": "review this",
            "model": "deepseek-v4-flash",
        },
        run=fake_run,
    )

    assert result["ok"] is True
    assert result["selected_candidate"] == "reasonix deepseek-v4-flash"
    assert calls == [
        [
            "reasonix",
            "run",
            "-m",
            "deepseek-v4-flash",
            "--effort",
            "high",
            "review this",
        ]
    ]


def test_reasonix_dev_print_runner_uses_high_effort_run_with_config():
    calls: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        envs.append(kwargs["env"])
        return _completed("READY\n")

    result = run_print_request(
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "print",
            "prompt": "Create no files. Reply READY.",
            "model": "deepseek-v4-flash",
        },
        run=fake_run,
    )

    assert result["ok"] is True
    assert result["selected_candidate"] == "reasonix run deepseek-v4-flash --mcp shell"
    assert calls[0][:7] == [
        "reasonix",
        "run",
        "-m",
        "deepseek-v4-flash",
        "--effort",
        "high",
        "--mcp",
    ]
    assert calls[0][7].startswith("agent_crossbar_shell=")
    assert "run_shell_command" in calls[0][-1]
    assert "Create no files. Reply READY." in calls[0][-1]
    assert envs[0]["AGENT_CROSSBAR_SHELL_CWD"] == str(Path.cwd())


def test_opencode_dev_print_runner_uses_opencode_go_model_candidate():
    calls: list[list[str]] = []
    cwd = "/tmp/opencode-workspace"

    def fake_run(args, **kwargs):
        calls.append(args)
        return _completed("READY\n")

    result = run_print_request(
        {
            "profile": "opencode",
            "operation": "dev",
            "transport": "print",
            "prompt": "Create no files. Reply READY.",
            "model": "kimi-k2.7-code",
            "cwd": cwd,
        },
        run=fake_run,
    )

    assert result["ok"] is True
    assert result["selected_candidate"] == "opencode run opencode-go/kimi-k2.7-code"
    assert calls == [
        [
            "opencode",
            "run",
            "-m",
            "opencode-go/kimi-k2.7-code",
            "--dangerously-skip-permissions",
            "--dir",
            cwd,
            "Create no files. Reply READY.",
        ]
    ]


def test_codex_native_review_runs_in_repo_cwd():
    seen: dict[str, str | None] = {}

    def fake_run(args, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        return _completed("CODEX_OK\n")

    result = run_print_request(
        {
            "profile": "codex",
            "operation": "review",
            "transport": "print",
            "prompt": "review this",
            "cwd": "/repo/codex-native-context",
            "model": "gpt-5.6-sol",
        },
        run=fake_run,
    )

    assert result["ok"] is True
    assert seen["cwd"] == "/repo/codex-native-context"


def test_tmux_prompt_buffers_are_named_per_job_session():
    first = runner_module._tmux_prompt_buffer_name("agents-job-one")
    second = runner_module._tmux_prompt_buffer_name("agents-job-two")

    assert first == "agents-job-one-prompt"
    assert second == "agents-job-two-prompt"
    assert first != second


def test_opencode_review_print_passes_prompt_on_stdin_and_uses_caller_timeout():
    seen: dict[str, object] = {}
    long_prompt = "Review this.\n" + ("context\n" * 1000)

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["stdin"] = kwargs.get("stdin")
        seen["input"] = kwargs.get("input")
        seen["timeout"] = kwargs.get("timeout")
        return _completed("GLM_REVIEW_OK\n")

    result = run_print_request(
        {
            "profile": "opencode",
            "operation": "review",
            "transport": "print",
            "prompt": long_prompt,
            "model": "glm-5.2",
            "timeout_sec": 300,
        },
        run=fake_run,
        timeout_sec=600,
    )

    assert result["ok"] is True
    assert seen["args"] == ["opencode", "run", "-m", "opencode-go/glm-5.2"]
    assert seen["stdin"] is None
    assert seen["input"] == long_prompt
    assert seen["timeout"] == 600


def test_run_print_job_records_events_and_result(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix",
        operation="review",
        transport="print",
        sensitivity="normal",
    )

    def fake_run(args, **kwargs):
        return _completed("DEEPSEEK_OK\n")

    result = run_print_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "review",
            "transport": "print",
            "prompt": "review this",
            "model": "deepseek-v4-flash",
        },
        run=fake_run,
    )

    assert result["ok"] is True
    stored = store.get_result(job.job_id)
    assert stored["ok"] is True
    assert stored["summary"] == "DEEPSEEK_OK\n"
    tail = store.job_tail(job.job_id)
    assert tail["status"] == "succeeded"
    event_types = [event["type"] for event in tail["events"]]
    assert "provider_started" in event_types
    assert "stdout" in event_types
    assert "result" in event_types


def test_run_gui_job_persists_prompt_and_generation_progress(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="chatgpt_pro",
        operation="advice",
        transport="gui",
    )

    def fake_run_gui_request(req, **kwargs):
        progress = kwargs["progress"]
        progress(
            "prompt_submitted",
            "ChatGPT prompt submitted",
            {"browser": "Helium"},
        )
        progress(
            "generation_in_progress",
            "ChatGPT is still generating",
            {"browser": "Helium", "elapsed_sec": 30},
        )
        return {
            "ok": True,
            "output": "DONE",
            "selected_candidate": "ChatGPT Pro web via Helium",
        }

    monkeypatch.setattr(runner_module, "run_gui_request", fake_run_gui_request)

    runner_module.run_gui_job(
        store,
        job.job_id,
        {"profile": "chatgpt_pro", "operation": "advice"},
        timeout_sec=1200,
    )

    events = job.events.read_since(0)
    assert [event["type"] for event in events] == [
        "provider_started",
        "prompt_submitted",
        "generation_in_progress",
        "stdout",
        "result",
    ]
    assert events[0]["data"]["timeout_sec"] == 1200
    assert events[2]["data"] == {"browser": "Helium", "elapsed_sec": 30}


def test_run_tmux_job_starts_session_and_records_metadata(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix",
        operation="dev",
        transport="tmux",
        sensitivity="normal",
    )
    calls: list[list[str]] = []
    sleep_calls: list[float] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["tmux", "capture-pane"]:
            return _completed("› ask anything\n")
        return _completed("")

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "tmux",
            "autonomy": "edit_local",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "Smoke only. Reply READY.",
            "model": "deepseek-v4-flash",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        sleep=fake_sleep,
        monitor=False,
    )

    assert result["ok"] is True
    assert any(call[:2] == ["tmux", "new-session"] for call in calls)
    assert any(call[:2] == ["tmux", "capture-pane"] for call in calls)
    assert any(call[:2] == ["tmux", "send-keys"] for call in calls)
    assert sleep_calls == [2.0, 0.5]
    assert [
        "tmux",
        "send-keys",
        "-t",
        result["tmux_target"],
        "-l",
        "Smoke only. Reply READY.",
    ] in calls
    assert ["tmux", "send-keys", "-t", result["tmux_target"], "C-m"] in calls
    tail = store.job_tail(job.job_id)
    event_types = [event["type"] for event in tail["events"]]
    assert "provider_started" in event_types
    assert "tmux_started" in event_types
    meta = json.loads((job.path / "meta.json").read_text())
    assert meta["tmux_session"].startswith("agents-")
    assert meta["tmux_transcript_path"].endswith("transcript.jsonl")
    assert meta["prompt_sent"] is True
    assert meta["prompt_ready"] is True
    assert meta["candidate_argv"][:2] == ["reasonix", "code"]
    assert all("Smoke only. Reply READY." not in arg for arg in meta["candidate_argv"])


def test_run_tmux_job_resolves_claude_binary_and_opus_model_for_claude_profile(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CLAUDE_BIN", "/tmp/current-claude")
    store = JobStore(tmp_path)

    job = store.create_job(
        profile="claude",
        operation="dev",
        transport="tmux",
        sensitivity="normal",
    )

    def fake_run(args, **kwargs):
        return _completed("")

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "claude",
            "operation": "dev",
            "transport": "tmux",
            "model": "opus",
            "prompt": "Reply READY.",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        monitor=False,
    )

    assert result["ok"] is True
    assert result["selected_candidate"] == "claude claude-opus-5 interactive"
    assert result["candidate_argv"][:3] == [
        "/tmp/current-claude",
        "--model",
        "claude-opus-5",
    ]
    assert "-p" not in result["candidate_argv"]
    assert "--print" not in result["candidate_argv"]
    assert "Reply READY." in result["candidate_argv"]
    assert result["prompt_sent"] is False


def test_run_tmux_job_resolves_opencode_binary_and_model_for_opencode_profile(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENCODE_BIN", "/tmp/current-opencode")
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode",
        operation="dev",
        transport="tmux",
        sensitivity="normal",
    )

    def fake_run(args, **kwargs):
        return _completed("")

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "opencode",
            "operation": "dev",
            "transport": "tmux",
            "model": "glm-5.2",
            "prompt": "Reply READY.",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        monitor=False,
    )

    assert result["ok"] is True
    assert result["selected_candidate"] == "opencode run opencode-go/glm-5.2 tmux"
    assert result["candidate_argv"][:4] == [
        "/tmp/current-opencode",
        "run",
        "-m",
        "opencode-go/glm-5.2",
    ]
    assert "--interactive" not in result["candidate_argv"]
    assert "opencode-go/glm-5.2" in result["candidate_argv"]
    assert "--dangerously-skip-permissions" in result["candidate_argv"]
    assert "--dir" in result["candidate_argv"]
    assert str(tmp_path) in result["candidate_argv"]
    assert "Reply READY." in result["candidate_argv"]
    assert result["prompt_sent"] is False
    assert result["interactive"] is False


def test_run_tmux_job_reports_bootstrap_diagnostics_when_session_disappears(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix",
        operation="dev",
        transport="tmux",
        sensitivity="normal",
    )
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["tmux", "new-session"]:
            return _completed("")
        if args[:2] == ["tmux", "list-panes"]:
            return _completed("", returncode=1, stderr="can't find window: 0")
        if args[:2] == ["tmux", "pipe-pane"]:
            return _completed("", returncode=1, stderr="can't find pane")
        if args[:2] == ["tmux", "send-keys"]:
            return _completed("", returncode=1, stderr="can't find pane")
        if args[:3] == ["tmux", "has-session", "-t"]:
            return _completed("", returncode=1, stderr="can't find session")
        raise AssertionError(f"unexpected call: {args}")

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "tmux",
            "autonomy": "edit_local",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "Smoke only. Reply READY.",
            "model": "deepseek-v4-flash",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        sleep=lambda _seconds: None,
        poll_interval_sec=0,
    )

    assert result["ok"] is False
    assert result["error"] == "missing_exit_status"
    assert "before prompt delivery" in result["message"]

    error_events = [
        event for event in store.job_tail(job.job_id)["events"] if event["type"] == "error"
    ]
    diagnostic = error_events[-1]["data"]
    assert diagnostic["candidate_argv"][:5] == [
        "reasonix",
        "code",
        "-m",
        "deepseek-v4-flash",
        "--effort",
    ]
    assert all("Smoke only. Reply READY." not in arg for arg in diagnostic["candidate_argv"])
    assert diagnostic["pipe_pane_ok"] is False
    assert diagnostic["prompt_sent"] is False
    assert diagnostic["tmux_output_path"].endswith("tmux-output.log")
    assert diagnostic["tmux_exit_status_path"].endswith("tmux-exit-status.txt")


def test_run_tmux_job_waits_for_tmux_target_before_prompt_delivery(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix",
        operation="dev",
        transport="tmux",
        sensitivity="normal",
    )
    calls: list[list[str]] = []
    readiness_checks = 0

    def fake_run(args, **kwargs):
        nonlocal readiness_checks
        calls.append(args)
        if args[:2] == ["tmux", "new-session"]:
            return _completed("")
        if args[:2] == ["tmux", "list-panes"]:
            readiness_checks += 1
            return _completed(
                "%1\n" if readiness_checks > 1 else "",
                returncode=0 if readiness_checks > 1 else 1,
                stderr="" if readiness_checks > 1 else "can't find window: 0",
            )
        if args[:2] == ["tmux", "capture-pane"]:
            return _completed("› ask anything\n")
        if args[:2] in (["tmux", "pipe-pane"], ["tmux", "send-keys"]):
            return _completed("")
        raise AssertionError(f"unexpected call: {args}")

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "tmux",
            "autonomy": "edit_local",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "Smoke only. Reply READY.",
            "model": "deepseek-v4-flash",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        sleep=lambda _seconds: None,
        monitor=False,
    )

    list_panes_index = next(i for i, call in enumerate(calls) if call[:2] == ["tmux", "list-panes"])
    pipe_pane_index = next(i for i, call in enumerate(calls) if call[:2] == ["tmux", "pipe-pane"])
    assert list_panes_index < pipe_pane_index
    assert result["ok"] is True
    assert result["tmux_target_ready"] is True
    assert result["pipe_pane_ok"] is True
    assert result["prompt_sent"] is True
    assert all("Smoke only. Reply READY." not in arg for arg in result["candidate_argv"])


def test_run_tmux_job_waits_for_reasonix_prompt_ready_before_sending_prompt(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix",
        operation="dev",
        transport="tmux",
        sensitivity="normal",
    )
    calls: list[list[str]] = []
    sleep_calls: list[float] = []
    capture_checks = 0

    def fake_run(args, **kwargs):
        nonlocal capture_checks
        calls.append(args)
        if args[:2] == ["tmux", "new-session"]:
            return _completed("")
        if args[:2] == ["tmux", "list-panes"]:
            return _completed("%1\n")
        if args[:2] == ["tmux", "pipe-pane"]:
            return _completed("")
        if args[:2] == ["tmux", "capture-pane"]:
            capture_checks += 1
            return _completed("Loading\n" if capture_checks == 1 else "› ask anything\n")
        if args[:2] == ["tmux", "send-keys"]:
            return _completed("")
        raise AssertionError(f"unexpected call: {args}")

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        calls.append(["sleep", str(seconds)])

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "tmux",
            "autonomy": "edit_local",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "Smoke only. Reply READY.",
            "model": "deepseek-v4-flash",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        sleep=fake_sleep,
        monitor=False,
    )

    capture_indexes = [i for i, call in enumerate(calls) if call[:2] == ["tmux", "capture-pane"]]
    send_indexes = [i for i, call in enumerate(calls) if call[:2] == ["tmux", "send-keys"]]
    assert capture_indexes
    assert send_indexes
    assert max(capture_indexes) < min(send_indexes)
    literal_index = calls.index(
        [
            "tmux",
            "send-keys",
            "-t",
            result["tmux_target"],
            "-l",
            "Smoke only. Reply READY.",
        ]
    )
    enter_index = calls.index(["tmux", "send-keys", "-t", result["tmux_target"], "C-m"])
    assert any(
        literal_index < index < enter_index and call == ["sleep", "0.5"]
        for index, call in enumerate(calls)
    )
    assert capture_checks == 2
    assert result["prompt_ready"] is True
    assert result["prompt_ready_detail"]["matched_pattern"] == "ask anything"
    assert result["prompt_sent"] is True
    assert sleep_calls == [2.0, 0.5, 0.5]


def test_run_tmux_job_targets_session_current_pane_not_window_zero(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="reasonix",
        operation="dev",
        transport="tmux",
        sensitivity="normal",
    )
    tmux_targets: list[str] = []

    def fake_run(args, **kwargs):
        if args[:2] == ["tmux", "new-session"]:
            return _completed("")
        if args[:2] == ["tmux", "capture-pane"]:
            tmux_targets.append(args[4])
            return _completed("› ask anything\n")
        if args[:2] in (
            ["tmux", "list-panes"],
            ["tmux", "pipe-pane"],
            ["tmux", "send-keys"],
        ):
            tmux_targets.append(args[3])
            return _completed("")
        raise AssertionError(f"unexpected call: {args}")

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "tmux",
            "autonomy": "edit_local",
            "external_context": "allowed",
            "sensitivity": "normal",
            "prompt": "Smoke only. Reply READY.",
            "model": "deepseek-v4-flash",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        monitor=False,
    )

    assert result["tmux_target"] == result["tmux_session"]
    assert all(":0.0" not in target for target in tmux_targets)


def test_tmux_shell_command_wraps_status_script_for_tmux_parser(tmp_path):
    command = _tmux_shell_command(
        CommandCandidate(name="demo", argv=["echo", "hi"]),
        tmp_path / "tmux-exit-status.txt",
    )

    parts = shlex.split(command)
    assert parts[:3] == ["exec", "/bin/sh", "-lc"]
    script = parts[3]
    assert script.startswith("echo hi; __agents_status=$?;")
    assert "printf '%s\\n' \"$__agents_status\"" in script
    assert script.endswith("exit $__agents_status")


def test_tmux_shell_command_redirects_noninteractive_candidate_output(tmp_path):
    command = _tmux_shell_command(
        CommandCandidate(name="demo", argv=["qwen", "-p", "hi"], send_prompt=False),
        tmp_path / "tmux-exit-status.txt",
        tmp_path / "tmux-output.log",
    )

    script = shlex.split(command)[3]
    assert "> " in script
    assert "tmux-output.log" in script
    assert "2>&1 < /dev/null" in script


def test_tmux_shell_command_keeps_interactive_candidate_attached_to_pane(tmp_path):
    command = _tmux_shell_command(
        CommandCandidate(
            name="qwen",
            argv=["qwen", "--prompt-interactive", "hi"],
            send_prompt=False,
            redirect_output=False,
        ),
        tmp_path / "tmux-exit-status.txt",
        tmp_path / "tmux-output.log",
    )

    script = shlex.split(command)[3]
    assert "--prompt-interactive hi" in script
    assert "2>&1 < /dev/null" not in script


def test_tmux_shell_command_can_run_candidate_with_clean_env(tmp_path):
    command = _tmux_shell_command(
        CommandCandidate(
            name="qwen",
            argv=["/tmp/qwen", "-p", "hi"],
            send_prompt=False,
            clean_env=True,
        ),
        tmp_path / "tmux-exit-status.txt",
        tmp_path / "tmux-output.log",
        env={
            "CODEX_CI": "1",
            "HOME": "/tmp/home",
            "PATH": "/tmp/node",
            "TERM": "tmux-256color",
            "USER": "bo",
        },
    )

    script = shlex.split(command)[3]
    assert script.startswith("env -i ")
    assert "HOME=/tmp/home" in script
    assert "PATH=/tmp/node" in script
    assert "TERM=dumb" in script
    assert "CODEX_CI" not in script


@pytest.mark.parametrize("provider_exit_code", [0, 7])
def test_tmux_shell_command_writes_status_and_exits_real_pane(tmp_path, provider_exit_code):
    if shutil.which("tmux") is None:
        pytest.skip("tmux is not installed")

    session = f"agents-wrapper-{provider_exit_code}-{tmp_path.name}"[:80]
    status_path = tmp_path / f"status-{provider_exit_code}.txt"
    command = _tmux_shell_command(
        CommandCandidate(
            name="exit",
            argv=["/bin/sh", "-c", f"exit {provider_exit_code}"],
        ),
        status_path,
    )
    try:
        started = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, command],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert started.returncode == 0, started.stderr
        for _ in range(50):
            alive = subprocess.run(
                ["tmux", "has-session", "-t", session],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if alive.returncode != 0:
                break
            time.sleep(0.02)
        assert alive.returncode != 0
        assert status_path.read_text(encoding="utf-8").strip() == str(provider_exit_code)
    finally:
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            capture_output=True,
            text=True,
            timeout=10,
        )


def test_job_stop_kills_recorded_tmux_session(tmp_path, monkeypatch):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    meta_path = job.path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["tmux_session"] = "agents-test-session"
    meta_path.write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _completed("")

    monkeypatch.setattr("agent_crossbar.jobs.subprocess.run", fake_run)

    response = store.stop_job(job.job_id, reason="user_cancelled")

    assert response["ok"] is True
    assert ["tmux", "has-session", "-t", "agents-test-session"] in calls
    assert ["tmux", "kill-session", "-t", "agents-test-session"] in calls
    tail = store.job_tail(job.job_id)
    stopped = [event for event in tail["events"] if event["type"] == "stopped"][-1]
    assert stopped["data"]["tmux_session"] == "agents-test-session"
    assert stopped["data"]["tmux_stop"] == "killed"


def test_tmux_monitor_preserves_stopped_status(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    has_session_calls = 0

    def fake_run(args, **kwargs):
        nonlocal has_session_calls
        if args[:3] == ["tmux", "has-session", "-t"]:
            has_session_calls += 1
            return _completed("", returncode=0 if has_session_calls == 1 else 1)
        return _completed("")

    def fake_sleep(_seconds):
        meta_path = job.path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["status"] = "stopped"
        meta_path.write_text(json.dumps(meta, separators=(",", ":")) + "\n")
        job.events.write(
            level="info",
            type="stopped",
            message="Job stopped: test_stop",
            data={"reason": "test_stop"},
        )

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "tmux",
            "prompt": "Smoke only. Reply READY.",
            "model": "deepseek-v4-flash",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        sleep=fake_sleep,
        poll_interval_sec=0,
    )

    assert result["ok"] is False
    assert result["error"] == "job_stopped"
    assert store.job_tail(job.job_id)["status"] == "stopped"
    stopped_result = store.get_result(job.job_id)
    assert stopped_result["ok"] is True
    assert stopped_result["status"] == "stopped"
    assert "stop_reason" in stopped_result


def test_run_tmux_job_complete_on_output_returns_before_interactive_session_exits(
    tmp_path,
):
    store = JobStore(tmp_path)
    job = store.create_job(profile="codex", operation="dev", transport="tmux")
    calls: list[list[str]] = []
    sleep_calls = 0

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["tmux", "new-session"]:
            return _completed("")
        if args[:2] == ["tmux", "list-panes"]:
            return _completed("%1\n")
        if args[:2] == ["tmux", "pipe-pane"]:
            return _completed("")
        if args[:2] == ["tmux", "capture-pane"]:
            return _completed("› ask anything\n")
        if args[:2] == ["tmux", "send-keys"]:
            return _completed("")
        if args[:3] == ["tmux", "has-session", "-t"]:
            return _completed("", returncode=0)
        if args[:2] == ["tmux", "kill-session"]:
            return _completed("")
        raise AssertionError(f"unexpected call: {args}")

    def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        output_path = job.path / "tmux-output.log"
        if sleep_calls == 1:
            output = "• UserPromptSubmit hook (completed)\n• TRANSIENT\n"
        elif sleep_calls == 2:
            output = "• UserPromptSubmit hook (completed)\n• TRANSIENT\n• Working (1s • esc to interrupt)\n"
        else:
            output = "• UserPromptSubmit hook (completed)\n• FINAL\n• Running Stop hook: test\n"
        output_path.write_text(output, encoding="utf-8")

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "codex",
            "operation": "dev",
            "transport": "tmux",
            "prompt": "Reply with the single word OK and nothing else.",
            "model": "gpt-5.6-sol",
            "effort": "medium",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        sleep=fake_sleep,
        poll_interval_sec=0,
        timeout_sec=180,
        complete_on_output=True,
    )

    assert result["ok"] is True
    assert result["completion_reason"] == "interactive_output_complete"
    assert "• FINAL" in result["output"]
    assert result["tmux_cleanup"] == "killed"
    assert sleep_calls >= 2
    assert ["tmux", "kill-session", "-t", result["tmux_session"]] in calls
    stored = store.get_result(job.job_id)
    assert stored["ok"] is True
    assert "FINAL" in stored["summary"]


def test_run_tmux_job_fails_fast_when_reasonix_resumes_session(tmp_path):
    store = JobStore(tmp_path)
    job = store.create_job(profile="reasonix", operation="dev", transport="tmux")
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["tmux", "new-session"]:
            return _completed("")
        if args[:2] == ["tmux", "list-panes"]:
            return _completed("%1\n")
        if args[:2] == ["tmux", "pipe-pane"]:
            return _completed("")
        if args[:2] == ["tmux", "capture-pane"]:
            return _completed("› ask anything\n")
        if args[:2] == ["tmux", "send-keys"]:
            return _completed("")
        if args[:3] == ["tmux", "has-session", "-t"]:
            return _completed("", returncode=0)
        if args[:2] == ["tmux", "kill-session"]:
            return _completed("")
        raise AssertionError(f"unexpected call: {args}")

    def fake_sleep(_seconds):
        output_path = job.path / "tmux-output.log"
        output_path.write_text(
            '✓▸resumed session "code-claude" with 10 prior messages · /new to start fresh\n'
            "◇ you · just now\n"
            "↳ Reply OK.\n"
            "‹reply v4-flash\n"
            "OK\n"
            "› ask anything\n",
            encoding="utf-8",
        )

    result = runner_module.run_tmux_job(
        store,
        job.job_id,
        {
            "profile": "reasonix",
            "operation": "dev",
            "transport": "tmux",
            "prompt": "Reply OK.",
            "model": "deepseek-v4-flash",
            "cwd": str(tmp_path),
        },
        run=fake_run,
        sleep=fake_sleep,
        poll_interval_sec=0,
        timeout_sec=180,
        complete_on_output=True,
    )

    assert result["ok"] is False
    assert result["error"] == "session_resumed"
    assert result["tmux_cleanup"] == "killed"
    assert ["tmux", "kill-session", "-t", result["tmux_session"]] in calls
    stored = store.get_result(job.job_id)
    assert stored["ok"] is False
    assert stored["summary"].startswith("Reasonix resumed an existing session")


def legacy_chatgpt_native_runner_uses_native_app_and_extracts_nonce(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    calls: list[tuple[str, dict]] = []
    typed: dict[str, str] = {}

    class FakeCua:
        def call(self, tool: str, payload: dict):
            calls.append((tool, payload))
            if tool == "list_apps":
                return {
                    "apps": [
                        {
                            "bundle_id": "com.openai.chat",
                            "name": "ChatGPT",
                            "pid": 123,
                            "running": True,
                            "active": False,
                        }
                    ]
                }
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "on_current_space": True,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                if typed:
                    prompt = typed["text"]
                    begin = next(
                        line
                        for line in prompt.splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    end = begin.replace("BEGIN_", "END_")
                    return {
                        "bundle_id": "com.openai.chat",
                        "tree_markdown": f"AXStaticText ({begin}\nGPT_PRO_OK\n{end})",
                    }
                return {
                    "bundle_id": "com.openai.chat",
                    "tree_markdown": (
                        '- AXApplication "ChatGPT"\n'
                        "  - [112] AXButton (New chat)\n"
                        '  - [106] AXButton = "5.5 Pro" (Options)\n'
                        "  - [101] AXTextArea\n"
                        '  - [109] AXButton (Send) help="Send message"\n'
                    ),
                }
            if tool == "click":
                return {"ok": True}
            if tool == "type_text":
                typed["text"] = payload["text"]
                return {"ok": True}
            if tool == "press_key":
                return {"ok": True}
            raise AssertionError(f"unexpected CUA tool {tool}")

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "transport": "gui",
            "prompt": "reply with the sentinel",
            "timeout_sec": 5,
        },
        cua=FakeCua(),
        sleep=lambda _: None,
    )

    assert result["ok"] is True
    assert result["output"] == "GPT_PRO_OK"
    assert result["selected_candidate"] == "ChatGPT native app via cua-driver"
    assert "reply with the sentinel" in typed["text"]
    assert "BEGIN_AGENTS_MCP_RESPONSE_" in typed["text"]
    assert "END_AGENTS_MCP_RESPONSE_" in typed["text"]
    assert ("click", {"pid": 123, "window_id": 456, "element_index": 112}) in calls


def test_chatgpt_browser_runner_falls_back_in_fixed_order(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    seen: list[str] = []

    def fake_candidate(candidate, req, cua, sleep, deadline, nonce):
        seen.append(candidate.key)
        if candidate.key == "safari":
            return {"ok": True, "output": "BROWSER_OK", "browser": candidate.name}
        return {
            "ok": False,
            "error": "browser_unavailable",
            "message": f"{candidate.name} unavailable",
        }

    monkeypatch.setattr(runner_module, "_run_chatgpt_browser_candidate", fake_candidate)

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "prompt": "advise",
            "timeout_sec": 5,
        },
        cua=object(),
        sleep=lambda _: None,
    )

    assert seen == ["helium", "chrome", "safari"]
    assert result["ok"] is True
    assert result["output"] == "BROWSER_OK"
    assert result["selected_candidate"] == "ChatGPT Pro web via Safari"
    assert [attempt["browser"] for attempt in result["attempts"]] == [
        "Helium",
        "Chrome",
        "Safari",
    ]


def test_chatgpt_browser_candidates_exclude_unsupported_zen():
    assert [candidate.key for candidate in runner_module._CHATGPT_BROWSER_CANDIDATES] == [
        "helium",
        "chrome",
        "safari",
    ]


def test_chatgpt_browser_runner_reports_all_fallback_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    def fake_candidate(candidate, req, cua, sleep, deadline, nonce):
        return {
            "ok": False,
            "error": "authentication_required",
            "message": f"Sign in to {candidate.name}",
            "diagnostics": {
                "stage": "prompt_delivery",
                "failure": "composer_stayed_empty",
            },
        }

    monkeypatch.setattr(runner_module, "_run_chatgpt_browser_candidate", fake_candidate)

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "prompt": "advise",
            "timeout_sec": 5,
        },
        cua=object(),
        sleep=lambda _: None,
    )

    assert result["ok"] is False
    assert result["error"] == "browser_fallback_exhausted"
    assert [attempt["browser"] for attempt in result["attempts"]] == [
        "Helium",
        "Chrome",
        "Safari",
    ]
    assert all(
        attempt["diagnostics"]
        == {
            "stage": "prompt_delivery",
            "failure": "composer_stayed_empty",
        }
        for attempt in result["attempts"]
    )


def test_chatgpt_browser_runner_preserves_shared_screen_time_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    def fake_candidate(candidate, req, cua, sleep, deadline, nonce):
        return {
            "ok": False,
            "error": "browser_time_limit",
            "message": f"macOS Screen Time blocked {candidate.name}",
        }

    monkeypatch.setattr(runner_module, "_run_chatgpt_browser_candidate", fake_candidate)

    result = run_gui_request(
        {"profile": "chatgpt_pro", "operation": "advice", "prompt": "advise"},
        cua=object(),
        sleep=lambda _: None,
        timeout_sec=5,
    )

    assert result["ok"] is False
    assert result["error"] == "browser_time_limit"
    assert result["message"] == "macOS Screen Time blocked every ChatGPT browser"
    assert len(result["attempts"]) == 3


def test_chatgpt_browser_runner_does_not_aggregate_partial_screen_time_failure(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    calls = 0

    def fake_candidate(candidate, req, cua, sleep, deadline, nonce):
        nonlocal calls
        calls += 1
        # Exhaust the shared deadline after the first attempted browser.
        monkeypatch.setattr(runner_module.time, "monotonic", lambda: 10.0)
        return {
            "ok": False,
            "error": "browser_time_limit",
            "message": f"macOS Screen Time blocked {candidate.name}",
        }

    monkeypatch.setattr(runner_module, "_run_chatgpt_browser_candidate", fake_candidate)
    monotonic_values = iter([0.0, 0.0])
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: next(monotonic_values))

    result = run_gui_request(
        {"profile": "chatgpt_pro", "operation": "advice", "prompt": "advise"},
        cua=object(),
        sleep=lambda _: None,
        timeout_sec=5,
    )

    assert calls == 1
    assert result["error"] == "browser_fallback_exhausted"


def test_chatgpt_browser_runner_preserves_post_submit_timeout_state(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    def fake_candidate(candidate, req, cua, sleep, deadline, nonce):
        return {
            "ok": False,
            "error": "generation_timed_out",
            "message": "ChatGPT prompt was submitted and is still generating",
            "diagnostics": {
                "stage": "generation_in_progress",
                "prompt_submitted": True,
            },
        }

    monkeypatch.setattr(runner_module, "_run_chatgpt_browser_candidate", fake_candidate)

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "prompt": "analyze deeply",
        },
        cua=object(),
        sleep=lambda _: None,
        timeout_sec=1200,
    )

    assert result["ok"] is False
    assert result["error"] == "generation_timed_out"
    assert result["message"] == "ChatGPT prompt was submitted and is still generating"
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["diagnostics"]["prompt_submitted"] is True


def test_chatgpt_web_candidate_requires_pro_before_typing(monkeypatch):
    typed: list[str] = []
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": "net.imput.helium", "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "page":
                return {"text": "Good to see you\nAsk ChatGPT\nAuto"}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWindow "ChatGPT"\n'
                        'AXWebArea "ChatGPT"\n'
                        '[1] AXTextArea = "Ask ChatGPT"\n'
                        '[2] AXPopUpButton "Auto"'
                    )
                }
            if tool == "click":
                return {"ok": True}
            if tool == "type_text":
                typed.append(payload["text"])
                return {"ok": True}
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        runner_module._CHATGPT_BROWSER_CANDIDATES[0],
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is False
    assert result["error"] == "model_not_pro"
    assert result["diagnostics"]["stage"] == "model_detection"
    assert result["diagnostics"]["model_detection"] == {
        "composer_found": True,
        "picker_found": True,
        "pro_detected": False,
    }
    assert typed == []


def test_chatgpt_web_candidate_reports_macos_screen_time_limit(monkeypatch):
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": "net.imput.helium", "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "",
                            "layer": 0,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        "AXWindow\n"
                        'AXStaticText = "Time Limit"\n'
                        'AXStaticText = "You’ve reached your limit on Helium."\n'
                        '[1] AXButton "Ask for More Time"'
                    )
                }
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        runner_module._CHATGPT_BROWSER_CANDIDATES[0],
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is False
    assert result["error"] == "browser_time_limit"
    assert result["message"] == "macOS Screen Time blocked Helium"
    assert result["diagnostics"]["stage"] == "browser_time_limit"
    assert result["diagnostics"]["screen_time_message"] == ("You’ve reached your limit on Helium.")


def test_chatgpt_web_candidate_falls_through_when_signed_out(monkeypatch):
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": "net.imput.helium", "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "page":
                return {"text": "Log in\nSign up"}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWindow "ChatGPT"\n'
                        '- [5] AXTextField = "chatgpt.com" (Address and search bar)'
                    )
                }
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        runner_module._CHATGPT_BROWSER_CANDIDATES[0],
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is False
    assert result["error"] == "authentication_required"


def test_chatgpt_browser_app_ignores_headless_chrome_and_uses_gui_pid(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    opened: list[object] = []

    class FakeCua:
        def call(self, tool, payload):
            assert tool == "list_apps"
            return {
                "apps": [
                    {"bundle_id": candidate.bundle_id, "pid": 111, "running": True},
                ]
            }

    monkeypatch.setattr(
        runner_module, "_process_is_headless", lambda pid: pid == 111, raising=False
    )
    monkeypatch.setattr(runner_module, "_find_gui_browser_pid", lambda item: 222, raising=False)
    monkeypatch.setattr(
        runner_module.subprocess, "run", lambda *args, **kwargs: opened.append(args)
    )

    app = runner_module._chatgpt_browser_app(FakeCua(), candidate, lambda _: None)

    assert app["pid"] == 222
    assert app["bundle_id"] == candidate.bundle_id
    assert opened == []


def test_chatgpt_chrome_launch_requests_a_new_gui_instance(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    discovered = iter([None, 222])
    opened: list[list[str]] = []

    class FakeCua:
        def call(self, tool, payload):
            assert tool == "list_apps"
            return {"apps": []}

    monkeypatch.setattr(
        runner_module,
        "_find_gui_browser_pid",
        lambda item: next(discovered),
        raising=False,
    )

    def fake_run(args, **kwargs):
        opened.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)

    app = runner_module._chatgpt_browser_app(FakeCua(), candidate, lambda _: None)

    assert app["pid"] == 222
    assert opened == [
        [
            "open",
            "-na",
            "Google Chrome",
            "--args",
            "--new-window",
            "https://chatgpt.com/",
        ]
    ]


def test_chatgpt_chrome_window_open_does_not_route_through_headless_bundle(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    list_calls = 0
    opened: list[list[str]] = []

    class FakeCua:
        def call(self, tool, payload):
            nonlocal list_calls
            if tool == "list_windows":
                list_calls += 1
                windows = [{"pid": 222, "window_id": 1, "title": "Signal Room", "layer": 0}]
                if list_calls >= 7:
                    windows.insert(
                        0,
                        {
                            "pid": 222,
                            "window_id": 2,
                            "title": "ChatGPT",
                            "layer": 0,
                        },
                    )
                return {"windows": windows}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"'
                        if payload["window_id"] == 2
                        else 'AXWebArea "Signal Room"'
                    )
                }
            raise AssertionError(tool)

    def fake_run(args, **kwargs):
        opened.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 222, lambda _: None)

    assert window["window_id"] == 2
    assert opened == [
        [
            "open",
            "-na",
            "Google Chrome",
            "--args",
            "--new-window",
            "https://chatgpt.com/",
        ]
    ]


def test_chatgpt_browser_window_does_not_accept_stale_chatgpt_title(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )
    opened = False

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                windows = [
                    {
                        "pid": 123,
                        "window_id": 1,
                        "title": "ChatGPT",
                        "layer": 0,
                    }
                ]
                if opened:
                    windows.insert(
                        0,
                        {
                            "pid": 123,
                            "window_id": 2,
                            "title": "ChatGPT",
                            "layer": 0,
                        },
                    )
                return {"windows": windows}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n[7] AXTextArea = "Ask ChatGPT"'
                        if payload["window_id"] == 2
                        else 'AXWindow "ChatGPT"'
                    )
                }
            raise AssertionError(tool)

    def fake_run(args, **kwargs):
        nonlocal opened
        opened = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert opened is True
    assert window["window_id"] == 2


def test_chatgpt_browser_window_rejects_unrelated_visible_textarea(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 1,
                            "title": "Notes",
                            "layer": 0,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                return {"tree_markdown": '[7] AXTextArea = "Write a note"'}
            raise AssertionError(tool)

    monkeypatch.setattr(runner_module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    with pytest.raises(RuntimeError, match="did not expose a ChatGPT window"):
        runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)


def test_chatgpt_browser_window_rediscovers_new_chrome_pid_after_launch(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    opened = False
    app_lists_after_open = 0

    class FakeCua:
        def call(self, tool, payload):
            nonlocal app_lists_after_open
            if tool == "list_apps":
                if opened:
                    app_lists_after_open += 1
                apps = [{"bundle_id": candidate.bundle_id, "pid": 111, "running": True}]
                # The new process may not be visible in CUA's first app snapshot.
                if app_lists_after_open >= 2:
                    apps.append({"bundle_id": candidate.bundle_id, "pid": 222, "running": True})
                return {"apps": apps}
            if tool == "list_windows":
                return {
                    "windows": (
                        [{"pid": 222, "window_id": 9, "title": "ChatGPT", "layer": 0}]
                        if payload["pid"] == 222
                        else []
                    )
                }
            if tool == "get_window_state":
                return {"tree_markdown": 'AXWebArea "ChatGPT"'}
            raise AssertionError(tool)

    def fake_run(*args, **kwargs):
        nonlocal opened
        opened = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 111, lambda _: None)

    assert window["pid"] == 222
    assert window["window_id"] == 9


def test_chatgpt_browser_window_prefers_new_pid_over_other_preexisting_chrome(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    opened = False

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                apps = [
                    {"bundle_id": candidate.bundle_id, "pid": 111, "running": True},
                    {"bundle_id": candidate.bundle_id, "pid": 333, "running": True},
                ]
                if opened:
                    apps.append({"bundle_id": candidate.bundle_id, "pid": 222, "running": True})
                return {"apps": apps}
            if tool == "list_windows":
                windows = {
                    111: [],
                    222: [{"pid": 222, "window_id": 22, "title": "ChatGPT", "layer": 0}],
                    333: [{"pid": 333, "window_id": 33, "title": "ChatGPT", "layer": 0}],
                }
                return {"windows": windows[payload["pid"]]}
            if tool == "get_window_state":
                return {"tree_markdown": 'AXWebArea "ChatGPT"'}
            raise AssertionError(tool)

    def fake_run(*args, **kwargs):
        nonlocal opened
        opened = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 111, lambda _: None)

    assert window["pid"] == 222
    assert window["window_id"] == 22


def test_chatgpt_browser_window_probes_exact_chatgpt_title_before_site_popup(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    probed: list[int] = []

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 1,
                            "title": "chatgpt.com",
                            "layer": 0,
                            "is_on_screen": True,
                        },
                        {
                            "pid": 123,
                            "window_id": 2,
                            "title": "ChatGPT",
                            "layer": 0,
                            "is_on_screen": True,
                        },
                    ]
                }
            if tool == "get_window_state":
                probed.append(payload["window_id"])
                if payload["window_id"] == 1:
                    return {"tree_markdown": 'AXWindow "site info"'}
                return {"tree_markdown": 'AXWebArea "ChatGPT"'}
            raise AssertionError(tool)

    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2
    assert probed == [2]


def test_chatgpt_browser_window_prefers_frontmost_exact_chatgpt_window(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    probed: list[int] = []

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {
                    "windows": [
                        {"pid": 123, "window_id": 1, "title": "ChatGPT", "layer": 0, "z_index": 2},
                        {"pid": 123, "window_id": 2, "title": "ChatGPT", "layer": 0, "z_index": 9},
                    ]
                }
            if tool == "get_window_state":
                probed.append(payload["window_id"])
                return {"tree_markdown": 'AXWebArea "ChatGPT"'}
            raise AssertionError(tool)

    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2
    assert probed == [2]


def test_chatgpt_browser_window_prefers_live_onscreen_conversation_over_offscreen_exact_title(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 1,
                            "title": "ChatGPT",
                            "layer": 0,
                            "is_on_screen": False,
                        },
                        {
                            "pid": 123,
                            "window_id": 2,
                            "title": "GPT_PRO_PROVIDER_GATE_OK",
                            "layer": 0,
                            "is_on_screen": True,
                        },
                    ]
                }
            if tool == "get_window_state":
                if payload["window_id"] == 2:
                    return {
                        "tree_markdown": '[7] AXTextField = "chatgpt.com/c/1" (Address and search bar)\nAXWebArea "GPT_PRO_PROVIDER_GATE_OK"'
                    }
                return {"tree_markdown": 'AXWebArea "ChatGPT"'}
            raise AssertionError(tool)

    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2


def test_activate_chatgpt_browser_raises_exact_ax_window_number(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    calls = []
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )

    runner_module._activate_chatgpt_browser(candidate, 123, 456)

    args, kwargs = calls[0]
    assert args[-2:] == ["123", "456"]
    assert "whose unix id is targetPid" in args[2]
    assert 'value of attribute "AXWindowNumber" of targetWindow' in args[2]
    assert 'perform action "AXRaise" of targetWindow' in args[2]
    assert kwargs["timeout"] == 10


def test_chatgpt_web_model_detection_uses_composer_adjacent_picker_only():
    tree = (
        "[10] AXPopUpButton (Bohdan Lytvynenko Pro, open profile menu)\n"
        '[20] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n'
        '[21] AXPopUpButton "Medium"'
    )

    assert runner_module._chatgpt_web_active_model_is_pro(tree) is False
    assert runner_module._find_chatgpt_web_model_picker(tree) == 21


def test_chatgpt_web_composer_ignores_preceding_search_and_dialog_textareas():
    tree = (
        '[7] AXTextArea = "Search conversations" (Search)\n'
        '[8] AXTextArea = "Tell us more" (Feedback dialog)\n'
        '[20] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n'
        '[21] AXPopUpButton "Pro"'
    )

    assert runner_module._find_first_text_area(tree) == 20


@pytest.mark.parametrize(
    "label",
    [
        'AXPopUpButton "Pro account"',
        'AXPopUpButton "Pro mode"',
        "AXPopUpButton (Bohdan Pro, open profile menu)",
    ],
)
def test_chatgpt_web_active_model_rejects_nonexact_pro_control_labels(label):
    tree = f'[20] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n[21] {label}'

    assert runner_module._chatgpt_web_active_model_is_pro(tree) is False


def test_chatgpt_web_pro_option_requires_exact_complete_control_label():
    tree = (
        '[22] AXMenuItem "Pro account"\n'
        '[23] AXMenuItem "Pro mode"\n'
        "[24] AXMenuItem (Bohdan Pro, open profile menu)\n"
        '[25] AXMenuItem "Pro"'
    )

    assert runner_module._find_chatgpt_web_pro_option(tree) == 25


def test_chatgpt_web_high_effort_picker_does_not_impersonate_pro_mode():
    tree = (
        '[20] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n'
        '[21] AXPopUpButton "High"\n'
        '[22] AXMenu "High"\n'
        '[23] AXMenuItem "Extra High"\n'
        '[24] AXMenuItem "Pro"'
    )

    assert runner_module._chatgpt_web_active_model_is_pro(tree) is False
    assert runner_module._find_chatgpt_web_pro_option(tree) == 24


def test_chatgpt_candidate_opens_adjacent_picker_and_picks_exact_pro(monkeypatch):
    candidate = runner_module._CHATGPT_BROWSER_CANDIDATES[0]
    phase = {"value": "high"}
    actions: list[tuple[int, str | None]] = []
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )

    def snapshot(*_args):
        picker = "Pro" if phase["value"] == "pro" else "High"
        menu = '\n[3] AXMenu "High"\n[4] AXMenuItem "Pro"' if phase["value"] == "menu" else ""
        return (
            'AXWebArea "ChatGPT"\n'
            '[1] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n'
            f'[2] AXPopUpButton "{picker}"{menu}'
        )

    monkeypatch.setattr(runner_module, "_chatgpt_browser_snapshot", snapshot)
    monkeypatch.setattr(runner_module, "_chatgpt_page_text", lambda *_args, **_kwargs: "signed in")
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", lambda *args, **kwargs: False)

    class FakeCua:
        def call(self, tool, payload):
            if tool == "get_window_state":
                return {"tree_markdown": snapshot()}
            assert tool == "click"
            actions.append((payload["element_index"], payload.get("action")))
            if payload["element_index"] == 2:
                phase["value"] = "menu"
            elif payload["element_index"] == 4:
                phase["value"] = "pro"
            return {"ok": True}

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["error"] == "prompt_insertion_failed"
    assert actions == [(2, "press"), (4, "press")]


def test_chatgpt_browser_snapshot_waits_until_composer_is_visible(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )
    snapshots = iter(
        [
            {"tree_markdown": 'AXWindow "ChatGPT"'},
            {"tree_markdown": 'AXWebArea "ChatGPT"'},
            {
                "tree_markdown": (
                    'AXWebArea "ChatGPT"\n[1] AXTextArea = "Ask ChatGPT"\n[2] AXButton "Pro"'
                )
            },
        ]
    )

    class FakeCua:
        def call(self, tool, payload):
            assert tool == "get_window_state"
            return next(snapshots)

    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_: None)

    tree = runner_module._chatgpt_browser_snapshot(FakeCua(), candidate, 123, 456, lambda _: None)

    assert '[1] AXTextArea = "Ask ChatGPT"' in tree
    assert runner_module._chatgpt_web_active_model_is_pro(tree) is True


def test_chatgpt_browser_window_accepts_newly_opened_challenge_window(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    snapshots = iter(
        [
            [{"pid": 123, "window_id": 1, "title": "Signal Room", "layer": 0}],
            [
                {"pid": 123, "window_id": 2, "title": "Just a moment...", "layer": 0},
                {"pid": 123, "window_id": 1, "title": "Signal Room", "layer": 0},
            ],
        ]
    )

    class FakeCua:
        def call(self, tool, payload):
            assert tool == "list_windows"
            return {"windows": next(snapshots)}

    monkeypatch.setattr(runner_module.subprocess, "run", lambda *args, **kwargs: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2
    assert window["title"] == "Just a moment..."


def test_chatgpt_browser_window_accepts_conversation_title_when_url_is_chatgpt(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    opened: list[object] = []

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 222,
                            "window_id": 456,
                            "title": "Порівняння результатів",
                            "layer": 0,
                        },
                    ]
                }
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        '- [5] AXTextField = "chatgpt.com" (Address and search bar)\n'
                        '- [14] AXWebArea "ChatGPT"'
                    )
                }
            raise AssertionError(tool)

    monkeypatch.setattr(
        runner_module.subprocess, "run", lambda *args, **kwargs: opened.append(args)
    )
    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 222, lambda _: None)

    assert window["window_id"] == 456
    assert opened == []


def test_chatgpt_browser_window_skips_chatgpt_title_without_a_live_chatgpt_surface(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {
                    "windows": [
                        {"pid": 123, "window_id": 1, "title": "ChatGPT", "layer": 0},
                        {"pid": 123, "window_id": 2, "title": "", "layer": 0},
                    ]
                }
            if tool == "get_window_state":
                return {
                    "tree_markdown": 'AXWindow "Settings"'
                    if payload["window_id"] == 1
                    else 'AXWebArea "ChatGPT"\n[7] AXTextArea = "Ask ChatGPT"'
                }
            raise AssertionError(tool)

    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2


def test_chatgpt_browser_window_waits_for_visible_chrome_chatgpt_surface_before_launching(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    snapshots = iter(
        [
            [
                {
                    "pid": 123,
                    "window_id": 1,
                    "title": "chatgpt.com",
                    "layer": 0,
                    "is_on_screen": False,
                }
            ],
            [
                {
                    "pid": 123,
                    "window_id": 1,
                    "title": "chatgpt.com",
                    "layer": 0,
                    "is_on_screen": False,
                },
                {
                    "pid": 123,
                    "window_id": 2,
                    "title": "Untitled",
                    "layer": 0,
                    "is_on_screen": True,
                },
            ],
        ]
    )
    opened: list[object] = []

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {"windows": next(snapshots)}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWindow "chatgpt.com"'
                        if payload["window_id"] == 1
                        else '- [5] AXTextField = "chatgpt.com" (Address and search bar)'
                    )
                }
            raise AssertionError(tool)

    monkeypatch.setattr(
        runner_module.subprocess, "run", lambda *args, **kwargs: opened.append(args)
    )
    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_args: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2
    assert opened == []


def test_chatgpt_browser_window_prefers_live_chatgpt_webarea_over_blank_address_match(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 1,
                            "title": "Untitled",
                            "layer": 0,
                            "is_on_screen": True,
                        },
                        {
                            "pid": 123,
                            "window_id": 2,
                            "title": "ChatGPT",
                            "layer": 0,
                            "is_on_screen": True,
                        },
                    ]
                }
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        '- [5] AXTextField = "chatgpt.com" (Address and search bar)'
                        if payload["window_id"] == 1
                        else '- [5] AXTextField = "chatgpt.com" (Address and search bar)\n- [14] AXWebArea "ChatGPT"'
                    )
                }
            raise AssertionError(tool)

    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_args: None)

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2


def test_chatgpt_browser_window_accepts_parenthesized_chatgpt_webarea(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_windows":
                return {"windows": [{"pid": 123, "window_id": 2, "title": "", "layer": 0}]}
            if tool == "get_window_state":
                return {"tree_markdown": "AXWebArea (ChatGPT)"}
            raise AssertionError(tool)

    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )

    assert (
        runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)[
            "window_id"
        ]
        == 2
    )


def test_chatgpt_browser_window_reuses_existing_challenge_without_opening_another_tab(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    opened: list[object] = []

    class FakeCua:
        def call(self, tool, payload):
            assert tool == "list_windows"
            return {
                "windows": [
                    {
                        "pid": 123,
                        "window_id": 2,
                        "title": "Just a moment...",
                        "layer": 0,
                    },
                    {"pid": 123, "window_id": 1, "title": "Signal Room", "layer": 0},
                ]
            }

    monkeypatch.setattr(
        runner_module.subprocess, "run", lambda *args, **kwargs: opened.append(args)
    )
    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )

    window = runner_module._chatgpt_browser_window(FakeCua(), candidate, 123, lambda _: None)

    assert window["window_id"] == 2
    assert opened == []


def test_chatgpt_web_candidate_reports_browser_challenge_before_page_interaction(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "chrome"
    )
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 2,
                            "title": "Just a moment...",
                            "layer": 0,
                        }
                    ]
                }
            raise AssertionError(f"challenge must stop before {tool}")

    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advisor prompt"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is False
    assert result["error"] == "browser_challenge_required"


def test_chatgpt_web_pro_detection_accepts_button_role():
    assert (
        runner_module._chatgpt_web_active_model_is_pro(
            '[1431] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n[1432] AXButton "Pro"'
        )
        is True
    )


def test_chatgpt_web_composer_recognizes_safari_empty_role_only_representation():
    tree = '[182] AXTextArea "Chat with ChatGPT" (Chat with ChatGPT) [actions=[press,showmenu]]'

    assert runner_module._chatgpt_web_composer_value(tree, 182) == ""
    assert runner_module._chatgpt_web_composer_is_empty(tree, 182) is True


def test_chatgpt_web_composer_parses_bulleted_ax_value_representation():
    tree = '- [221] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT) [actions=[press,showmenu]]'

    assert runner_module._chatgpt_web_composer_value(tree, 221) == "Ask ChatGPT"
    assert runner_module._chatgpt_web_composer_is_empty(tree, 221) is True


def test_chatgpt_web_composer_parses_safari_titled_ax_value_representation():
    tree = '[184] AXTextArea "Chat with ChatGPT" = "complete prompt" (Chat with ChatGPT)'

    assert runner_module._chatgpt_web_composer_value(tree, 184) == "complete prompt"
    assert runner_module._chatgpt_web_composer_is_empty(tree, 184) is False


def test_chatgpt_composer_match_accepts_editor_canonicalized_blank_lines_only():
    expected = "first\n\nsecond\n"
    actual = "first\nsecond"

    assert runner_module._chatgpt_web_composer_matches_prompt(actual, expected) is True
    assert runner_module._chatgpt_web_composer_matches_prompt("first\nchanged", expected) is False


def test_chatgpt_prompt_delivery_waits_for_a_fresh_ax_snapshot_after_paste(monkeypatch):
    clipboard = {"text": "user clipboard"}
    snapshots = iter(
        [
            '[1] AXTextArea = "Ask ChatGPT"',
            '[1] AXTextArea = "Ask ChatGPT"',
            '[1] AXTextArea = "complete prompt"',
        ]
    )
    pauses: list[float] = []
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                return {"ok": True}
            if tool == "hotkey":
                assert payload == {
                    "pid": 123,
                    "window_id": 456,
                    "keys": ["command", "v"],
                    "delivery_mode": "foreground",
                }
                return {"ok": True}
            if tool == "get_window_state":
                return {"tree_markdown": next(snapshots)}
            raise AssertionError(tool)

    delivered = runner_module._chatgpt_deliver_prompt(
        FakeCua(), 123, 456, 1, "complete prompt", sleep=pauses.append
    )

    assert delivered is True
    assert pauses == [0.2, 0.2]
    assert clipboard["text"] == "user clipboard"


def test_chatgpt_prompt_delivery_can_preserve_existing_composer_focus(monkeypatch):
    clipboard = {"text": "user clipboard"}
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                raise AssertionError("Safari must not lose its existing web composer focus")
            if tool == "hotkey":
                assert payload == {
                    "pid": 123,
                    "window_id": 456,
                    "keys": ["command", "v"],
                    "delivery_mode": "foreground",
                }
                return {"ok": True}
            if tool == "get_window_state":
                return {"tree_markdown": '[1] AXTextArea = "complete prompt"'}
            raise AssertionError(tool)

    assert (
        runner_module._chatgpt_deliver_prompt(
            FakeCua(), 123, 456, 1, "complete prompt", preserve_existing_focus=True
        )
        is True
    )
    assert clipboard["text"] == "user clipboard"


def test_safari_candidate_does_not_probe_page_before_clipboard_delivery(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "safari"
    )
    clipboard = {"text": "user clipboard"}
    state = {"composer": "Ask ChatGPT", "dom_composer": "", "submitted": False}
    activated: list[tuple[str, int]] = []
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )
    monkeypatch.setattr(
        runner_module,
        "_activate_chatgpt_browser",
        lambda browser, pid: activated.append((browser.key, pid)),
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {"windows": [{"pid": 123, "window_id": 456, "title": "ChatGPT", "layer": 0}]}
            if tool == "get_window_state":
                send = "\n[3] AXButton (Send prompt)" if state["composer"] != "Ask ChatGPT" else ""
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n'
                        f'''[1] AXTextArea = "{state["composer"]}" (Chat with ChatGPT)\n[2] AXButton "Pro"{send}'''
                    )
                }
            if tool == "hotkey":
                state["composer"] = clipboard["text"]
                return {"ok": True}
            if tool == "click":
                assert payload["element_index"] == 3
                state["submitted"] = True
                return {"ok": True}
            if tool == "page":
                assert state["composer"] != "Ask ChatGPT", (
                    "Safari page probe stole composer focus before paste"
                )
                begin = next(
                    line
                    for line in state["composer"].splitlines()
                    if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                )
                return {"text": f"{begin}\\nSAFARI_OK\\n{begin.replace('BEGIN_', 'END_')}"}
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is True
    assert result["output"] == "SAFARI_OK"
    assert state["submitted"] is True
    assert activated == []


def test_safari_candidate_falls_back_to_page_insert_only_after_clipboard_delivery_fails(
    monkeypatch,
):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "safari"
    )
    clipboard = {"text": "user clipboard"}
    state = {"composer": "Ask ChatGPT", "dom_composer": "", "submitted": False}
    hotkeys: list[dict[str, object]] = []
    page_actions: list[str] = []
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )
    monkeypatch.setattr(runner_module, "_activate_chatgpt_browser", lambda *_args: None)

    def safari_javascript(javascript: str) -> str | None:
        if javascript.startswith("document.querySelector"):
            return state["dom_composer"]
        assert "document.execCommand" in javascript
        assert ".focus(" not in javascript
        match = re.search(
            r'document\.execCommand\("insertText", false, ("(?:[^"\\]|\\.)*")\)',
            javascript,
        )
        assert match is not None
        state["dom_composer"] = json.loads(match.group(1))
        state["composer"] = state["dom_composer"]
        return "true"

    monkeypatch.setattr(
        runner_module, "_safari_execute_javascript", safari_javascript, raising=False
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {"windows": [{"pid": 123, "window_id": 456, "title": "ChatGPT", "layer": 0}]}
            if tool == "get_window_state":
                send = "\n[3] AXButton (Send prompt)" if state["composer"] != "Ask ChatGPT" else ""
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n'
                        f'''[1] AXTextArea = "{state["composer"]}"
[2] AXButton "Pro"{send}'''
                    ),
                    "elements": [
                        {
                            "element_index": 1,
                            "frame": {"x": 10, "y": 20, "w": 100, "h": 40},
                        }
                    ],
                }
            if tool == "hotkey":
                hotkeys.append(payload)
                return {"ok": True}
            if tool == "click":
                assert payload["element_index"] == 3
                state["submitted"] = True
                return {"ok": True}
            if tool == "page":
                page_actions.append(payload["action"])
                if payload["action"] == "click_element":
                    assert page_actions == ["click_element"]
                    assert (
                        payload["selector"]
                        == '[contenteditable="true"][aria-label="Chat with ChatGPT"]'
                    )
                    return {"screen_x": 10, "screen_y": 20}
                begin = next(
                    line
                    for line in state["composer"].splitlines()
                    if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                )
                return {"text": f"{begin}\nSAFARI_FALLBACK_OK\n{begin.replace('BEGIN_', 'END_')}"}
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is True
    assert result["output"] == "SAFARI_FALLBACK_OK"
    assert len(hotkeys) == 1
    assert page_actions == ["click_element", "get_text"]
    assert state["submitted"] is True


def test_chatgpt_prompt_delivery_uses_pixel_focus_fallback_before_clipboard_paste(
    monkeypatch,
):
    clipboard = {"text": "user clipboard"}
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                return {"ok": False, "error": "AXPress failed"}
            if tool == "get_window_state":
                return {
                    "tree_markdown": '[1] AXTextArea = "complete prompt"',
                    "elements": [
                        {
                            "element_index": 1,
                            "frame": {"x": 10, "y": 20, "w": 100, "h": 40},
                        }
                    ],
                }
            if tool == "hotkey":
                assert payload == {
                    "pid": 123,
                    "window_id": 456,
                    "keys": ["command", "v"],
                    "x": 60,
                    "y": 40,
                    "delivery_mode": "foreground",
                }
                return {"ok": True}
            raise AssertionError(tool)

    assert runner_module._chatgpt_deliver_prompt(FakeCua(), 123, 456, 1, "complete prompt") is True
    assert clipboard["text"] == "user clipboard"


def test_chatgpt_prompt_delivery_retries_with_pixel_focus_when_ax_click_is_false_positive(
    monkeypatch,
):
    clipboard = {"text": "user clipboard"}
    writes: list[dict[str, object]] = []
    state = {"typed": False}
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                return {}
            if tool == "get_window_state":
                composer = "complete prompt" if state["typed"] else "Ask ChatGPT"
                return {
                    "tree_markdown": f'[1] AXTextArea = "{composer}"',
                    "elements": [
                        {
                            "element_index": 1,
                            "frame": {"x": 10, "y": 20, "w": 100, "h": 40},
                        }
                    ],
                }
            if tool == "hotkey":
                writes.append(payload)
                return {"ok": True}
            if tool == "type_text":
                writes.append(payload)
                state["typed"] = True
                return {"ok": True}
            raise AssertionError(tool)

    assert (
        runner_module._chatgpt_deliver_prompt(
            FakeCua(), 123, 456, 1, "complete prompt", sleep=lambda _: None
        )
        is True
    )
    assert writes == [
        {
            "pid": 123,
            "window_id": 456,
            "keys": ["command", "v"],
            "delivery_mode": "foreground",
        },
        {
            "pid": 123,
            "window_id": 456,
            "keys": ["command", "a"],
            "x": 60,
            "y": 40,
            "delivery_mode": "foreground",
        },
        {
            "pid": 123,
            "window_id": 456,
            "keys": ["backspace"],
            "delivery_mode": "foreground",
        },
        {
            "pid": 123,
            "window_id": 456,
            "x": 60,
            "y": 40,
            "text": "complete prompt",
            "delay_ms": 0,
            "delivery_mode": "foreground",
        },
    ]


def test_chatgpt_prompt_delivery_pixel_retry_replaces_possible_stale_success(
    monkeypatch,
):
    clipboard = {"text": "user clipboard"}
    state = {"actual": "", "snapshots": 0}
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                return {}
            if tool == "hotkey":
                if payload["keys"] == ["backspace"]:
                    state["actual"] = ""
                elif payload["keys"] == ["command", "v"]:
                    state["actual"] += clipboard["text"]
                return {"ok": True}
            if tool == "type_text":
                state["actual"] += payload["text"]
                return {"ok": True}
            if tool == "get_window_state":
                state["snapshots"] += 1
                # Simulate five stale empty AX reads after the first successful paste.
                value = "Ask ChatGPT" if state["snapshots"] <= 5 else state["actual"]
                return {
                    "tree_markdown": f'[1] AXTextArea = "{value}"',
                    "elements": [
                        {
                            "element_index": 1,
                            "frame": {"x": 10, "y": 20, "w": 100, "h": 40},
                        }
                    ],
                }
            raise AssertionError(tool)

    assert runner_module._chatgpt_deliver_prompt(
        FakeCua(), 123, 456, 1, "complete prompt", sleep=lambda _: None
    )
    assert state["actual"] == "complete prompt"


def test_chatgpt_prompt_delivery_converts_ax_screen_frame_to_window_pixels(
    monkeypatch,
):
    clipboard = {"text": "user clipboard"}
    hotkeys: list[dict[str, object]] = []
    state = {"typed": False}
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                return {}
            if tool == "hotkey":
                hotkeys.append(payload)
                return {"ok": True}
            if tool == "type_text":
                state["typed"] = True
                return {"ok": True}
            if tool == "get_window_state":
                composer = "complete prompt" if state["typed"] else "Ask ChatGPT"
                return {
                    "tree_markdown": f'[1] AXTextArea = "{composer}"',
                    "elements": [
                        {
                            "element_index": 0,
                            "role": "AXWindow",
                            "frame": {"x": 40, "y": 30, "w": 500, "h": 400},
                        },
                        {
                            "element_index": 1,
                            "role": "AXTextArea",
                            "frame": {"x": 50, "y": 70, "w": 100, "h": 40},
                        },
                    ],
                }
            raise AssertionError(tool)

    assert runner_module._chatgpt_deliver_prompt(
        FakeCua(), 123, 456, 1, "complete prompt", sleep=lambda _: None
    )
    assert hotkeys[1] == {
        "pid": 123,
        "window_id": 456,
        "keys": ["command", "a"],
        "x": 60,
        "y": 60,
        "delivery_mode": "foreground",
    }


def test_chatgpt_prompt_delivery_types_multiline_prompt_with_soft_line_breaks(
    monkeypatch,
):
    clipboard = {"text": "user clipboard"}
    calls: list[tuple[str, dict[str, object]]] = []
    state = {"value": "Ask ChatGPT"}
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                return {}
            if tool == "hotkey":
                calls.append((tool, payload))
                if payload["keys"] == ["backspace"]:
                    state["value"] = ""
                elif payload["keys"] == ["shift", "enter"]:
                    state["value"] += "\n"
                return {"ok": True}
            if tool == "type_text":
                calls.append((tool, payload))
                state["value"] += payload["text"]
                return {"ok": True}
            if tool == "get_window_state":
                return {
                    "tree_markdown": f"[1] AXTextArea = {json.dumps(state['value'])}",
                    "elements": [
                        {
                            "element_index": 1,
                            "frame": {"x": 10, "y": 20, "w": 100, "h": 40},
                        }
                    ],
                }
            raise AssertionError(tool)

    assert runner_module._chatgpt_deliver_prompt(
        FakeCua(), 123, 456, 1, "line one\n\nline two", sleep=lambda _: None
    )
    assert [(tool, payload.get("text"), payload.get("keys")) for tool, payload in calls[-4:]] == [
        ("type_text", "line one", None),
        ("hotkey", None, ["shift", "enter"]),
        ("hotkey", None, ["shift", "enter"]),
        ("type_text", "line two", None),
    ]


def test_chatgpt_candidate_ignores_progress_callback_failure_after_submit(monkeypatch):
    candidate = runner_module._CHATGPT_BROWSER_CANDIDATES[0]
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )
    tree = '[1] AXTextArea = "Ask ChatGPT"\n[2] AXButton "Pro"\n[3] AXButton (Send prompt)'
    monkeypatch.setattr(runner_module, "_chatgpt_browser_snapshot", lambda *_: tree)
    monkeypatch.setattr(runner_module, "_chatgpt_page_text", lambda *_: "signed in")
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", lambda *args, **kwargs: True)

    class FakeCua:
        def call(self, tool, payload):
            if tool == "click":
                return {"ok": True}
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() - 1,
        "nonce",
        progress=lambda *_: (_ for _ in ()).throw(RuntimeError("log sink failed")),
    )

    assert result["error"] == "generation_timed_out"
    assert result["diagnostics"]["prompt_submitted"] is True
    assert result["diagnostics"]["progress_reporting"][0]["exception_type"] == "RuntimeError"


def test_chatgpt_browser_runner_never_falls_back_after_post_submit_page_exception(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    seen: list[str] = []

    def fake_candidate(candidate, req, cua, sleep, deadline, nonce):
        seen.append(candidate.key)
        return {
            "ok": False,
            "error": "RuntimeError",
            "message": "page get_text failed after submit",
            "diagnostics": {
                "stage": "generation_in_progress",
                "prompt_submitted": True,
                "failure": "exception",
            },
        }

    monkeypatch.setattr(runner_module, "_run_chatgpt_browser_candidate", fake_candidate)

    result = run_gui_request(
        {"profile": "chatgpt_pro", "operation": "advice", "prompt": "deep analysis"},
        cua=object(),
        sleep=lambda _: None,
        timeout_sec=60,
    )

    assert seen == ["helium"]
    assert result["error"] == "generation_status_unavailable"
    assert result["attempts"][0]["diagnostics"]["prompt_submitted"] is True


def test_chatgpt_generation_page_reads_are_bounded_by_heartbeat_and_deadline(
    monkeypatch,
):
    candidate = runner_module._CHATGPT_BROWSER_CANDIDATES[0]
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )
    tree = '[1] AXTextArea = "Ask ChatGPT"\n[2] AXButton "Pro"\n[3] AXButton (Send prompt)'
    monkeypatch.setattr(runner_module, "_chatgpt_browser_snapshot", lambda *_: tree)
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", lambda *args, **kwargs: True)
    clock = {"now": 0.0}
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock["now"])
    read_timeouts: list[float] = []
    progress_events: list[str] = []

    class FakeCua:
        def call(self, tool, payload, timeout_sec=None):
            if tool == "page":
                return {"text": "signed in"}
            if tool == "click":
                return {"ok": True}
            raise AssertionError(tool)

        def call_with_timeout(self, tool, payload, timeout_sec):
            assert tool == "page"
            read_timeouts.append(timeout_sec)
            clock["now"] += timeout_sec
            return {"text": "still generating"}

    def fake_sleep(seconds):
        clock["now"] += seconds

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advise"},
        FakeCua(),
        fake_sleep,
        35.0,
        "nonce",
        progress=lambda event, *_: progress_events.append(event),
    )

    assert result["error"] == "generation_timed_out"
    assert read_timeouts
    assert max(read_timeouts) <= 25
    assert sum(read_timeouts) <= 35
    assert "generation_in_progress" in progress_events


def test_chatgpt_generation_retries_timed_out_page_reads_then_returns_response(
    monkeypatch,
):
    candidate = runner_module._CHATGPT_BROWSER_CANDIDATES[0]
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_: {"pid": 123})
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda *_: {"pid": 123, "window_id": 456, "title": "ChatGPT"},
    )
    tree = '[1] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n[2] AXButton "Pro"\n[3] AXButton (Send prompt)'
    monkeypatch.setattr(runner_module, "_chatgpt_browser_snapshot", lambda *_: tree)
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", lambda *args, **kwargs: True)
    clock = {"now": 0.0}
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock["now"])
    reads = {"count": 0}
    progress_events: list[str] = []

    class FakeCua:
        def call(self, tool, payload):
            if tool == "page":
                return {"text": "signed in"}
            if tool == "click":
                return {"ok": True}
            raise AssertionError(tool)

        def call_with_timeout(self, tool, payload, timeout_sec):
            reads["count"] += 1
            clock["now"] += timeout_sec
            if reads["count"] <= 2:
                raise subprocess.TimeoutExpired("page", timeout_sec)
            return {
                "text": ("BEGIN_AGENTS_MCP_RESPONSE_nonce\nDONE\nEND_AGENTS_MCP_RESPONSE_nonce")
            }

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advise"},
        FakeCua(),
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
        90.0,
        "nonce",
        progress=lambda event, *_: progress_events.append(event),
    )

    assert result["ok"] is True
    assert result["output"] == "DONE"
    assert reads["count"] == 3
    assert "generation_in_progress" in progress_events


def test_chatgpt_generation_read_timeouts_reach_deadline_without_browser_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    candidate = runner_module._CHATGPT_BROWSER_CANDIDATES[0]
    monkeypatch.setattr(
        runner_module,
        "_CHATGPT_BROWSER_CANDIDATES",
        (candidate, runner_module._CHATGPT_BROWSER_CANDIDATES[1]),
    )
    monkeypatch.setattr(runner_module, "_chatgpt_browser_app", lambda *_: {"pid": 123})
    seen_windows: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_chatgpt_browser_window",
        lambda _cua, browser, *_: (
            seen_windows.append(browser.key) or {"pid": 123, "window_id": 456, "title": "ChatGPT"}
        ),
    )
    tree = '[1] AXTextArea = "Ask ChatGPT" (Chat with ChatGPT)\n[2] AXButton "Pro"\n[3] AXButton (Send prompt)'
    monkeypatch.setattr(runner_module, "_chatgpt_browser_snapshot", lambda *_: tree)
    monkeypatch.setattr(runner_module, "_chatgpt_deliver_prompt", lambda *args, **kwargs: True)
    clock = {"now": 0.0}
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock["now"])

    class FakeCua:
        def call(self, tool, payload):
            if tool == "page":
                return {"text": "signed in"}
            if tool == "click":
                return {"ok": True}
            raise AssertionError(tool)

        def call_with_timeout(self, tool, payload, timeout_sec):
            clock["now"] += timeout_sec
            raise subprocess.TimeoutExpired("page", timeout_sec)

    result = run_gui_request(
        {"profile": "chatgpt_pro", "operation": "advice", "prompt": "deep"},
        cua=FakeCua(),
        sleep=lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
        timeout_sec=35,
    )

    assert result["error"] == "generation_timed_out"
    assert seen_windows == [candidate.key]


def test_chatgpt_web_send_button_ignores_unrelated_send_actions():
    tree = '[7] AXButton "Send Chrome feedback"\n[8] AXButton (Send prompt)'

    assert runner_module._find_chatgpt_web_send_button(tree) == 8


def test_chatgpt_web_candidate_refreshes_stale_tree_before_pasting(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    state_calls = 0
    clipboard = {"text": "user clipboard"}
    state = {"composer": "Ask ChatGPT"}
    activated: list[tuple[str, int]] = []
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            nonlocal state_calls
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {"windows": [{"pid": 123, "window_id": 456, "title": "ChatGPT", "layer": 0}]}
            if tool == "page":
                if state["composer"] != "Ask ChatGPT":
                    begin = next(
                        line
                        for line in state["composer"].splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    return {"text": f"{begin}\nZEN_OK\n{begin.replace('BEGIN_', 'END_')}"}
                return {"text": "Ask ChatGPT\nPro"}
            if tool == "get_window_state":
                state_calls += 1
                if state_calls == 1:
                    return {"tree_markdown": '[2] AXButton "Pro"'}
                send = "\n[3] AXButton (Send prompt)" if state["composer"] != "Ask ChatGPT" else ""
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n'
                        f'[1] AXTextArea = "{state["composer"]}"\n[2] AXButton "Pro"{send}'
                    )
                }
            if tool == "click":
                return {"ok": True}
            if tool == "type_text":
                raise AssertionError("browser prompt delivery must use clipboard paste")
            if tool == "hotkey":
                assert payload["keys"] == ["command", "v"]
                state["composer"] = clipboard["text"]
                return {"ok": True}
            if tool == "press_key":
                raise AssertionError("browser submit must click Send prompt")
            raise AssertionError(tool)

    monkeypatch.setattr(
        runner_module,
        "_activate_chatgpt_browser",
        lambda candidate, pid, window_id=None: activated.append((candidate.key, pid)),
        raising=False,
    )

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advise"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is True
    assert result["output"] == "ZEN_OK"
    assert state_calls >= 2
    assert activated
    assert "BEGIN_AGENTS_MCP_RESPONSE_" in state["composer"]
    assert clipboard["text"] == "user clipboard"


@pytest.mark.parametrize(
    "candidate",
    runner_module._CHATGPT_BROWSER_CANDIDATES,
    ids=lambda candidate: candidate.key,
)
def test_chatgpt_web_candidate_pastes_and_verifies_full_prompt_before_single_submit(
    candidate,
    monkeypatch,
):
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    clipboard = {"text": "user clipboard"}
    state = {"composer": "Ask ChatGPT", "submitted": False}
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module, "_write_text_clipboard", lambda text: clipboard.update(text=text)
    )

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                send = "\n[3] AXButton (Send prompt)" if state["composer"] != "Ask ChatGPT" else ""
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n'
                        f'[1] AXTextArea = "{state["composer"]}"\n[2] AXPopUpButton "Pro"{send}'
                    )
                }
            if tool == "page":
                full_text = state["composer"]
                if "BEGIN_AGENTS_MCP_RESPONSE_" in full_text:
                    begin = next(
                        line
                        for line in full_text.splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    end = begin.replace("BEGIN_", "END_")
                    return {
                        "text": f"{begin}\nWEB_PRO_OK\n{end}"
                        if state["submitted"]
                        else "Ask ChatGPT\nPro"
                    }
                return {"text": "Good to see you\nAsk ChatGPT\nPro"}
            if tool == "click":
                events.append(("click", str(payload["element_index"])))
                if payload["element_index"] == 3:
                    state["submitted"] = True
                return {"ok": True}
            if tool == "type_text":
                raise AssertionError("browser prompt delivery must use clipboard paste")
            if tool == "hotkey":
                assert payload["keys"] == ["command", "v"]
                state["composer"] = clipboard["text"]
                events.append(("hotkey", "command+v"))
                return {"ok": True}
            if tool == "press_key":
                raise AssertionError(
                    "browser submit must click the fresh Send button, not press Return"
                )
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "first line\nsecond line"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is True
    assert result["output"] == "WEB_PRO_OK"
    assert events.count(("hotkey", "command+v")) == 1
    assert "first line\nsecond line" in state["composer"]
    assert clipboard["text"] == "user clipboard"
    submit_indexes = [index for index, event in enumerate(events) if event == ("click", "3")]
    assert submit_indexes == [len(events) - 1]


def test_chatgpt_web_candidate_does_not_submit_or_clobber_new_clipboard_after_paste_mismatch(
    monkeypatch,
):
    candidate = runner_module._CHATGPT_BROWSER_CANDIDATES[0]
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )
    clipboard = {"text": "user clipboard"}
    writes: list[str] = []
    monkeypatch.setattr(runner_module, "_read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        runner_module,
        "_write_text_clipboard",
        lambda text: (writes.append(text), clipboard.update(text=text)),
    )
    sent = False

    class FakeCua:
        def call(self, tool, payload):
            nonlocal sent
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {"windows": [{"pid": 123, "window_id": 456, "title": "ChatGPT", "layer": 0}]}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n'
                        '[1] AXTextArea = "Ask ChatGPT"\n[2] AXPopUpButton "Pro"'
                    )
                }
            if tool == "page":
                return {"text": "Ask ChatGPT\nPro"}
            if tool == "click":
                if payload["element_index"] == 3:
                    sent = True
                return {"ok": True}
            if tool == "hotkey":
                assert payload["keys"] == ["command", "v"]
                clipboard["text"] = "new user clipboard"
                return {"ok": True}
            if tool == "type_text":
                raise AssertionError("browser prompt delivery must use clipboard paste")
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "full prompt"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is False
    assert result["error"] == "prompt_insertion_failed"
    assert sent is False
    assert clipboard["text"] == "new user clipboard"
    assert writes[0] != "user clipboard"
    assert "user clipboard" not in writes[1:]


def test_chatgpt_web_candidate_refuses_to_overwrite_existing_draft(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    mutations: list[str] = []

    class FakeCua:
        def call(self, tool, payload):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {"windows": [{"pid": 123, "window_id": 456, "title": "ChatGPT", "layer": 0}]}
            if tool == "page":
                return {"text": "Ask ChatGPT\nPro"}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n'
                        '[1] AXTextArea = "my unfinished draft" (Chat with ChatGPT)\n'
                        '[2] AXButton "Pro"'
                    )
                }
            if tool in {"click", "type_text", "hotkey", "press_key"}:
                mutations.append(tool)
                return {"ok": True}
            raise AssertionError(tool)

    monkeypatch.setattr(
        runner_module, "_activate_chatgpt_browser", lambda candidate, pid, window_id=None: None
    )

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "advisor prompt"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is False
    assert result["error"] == "composer_not_empty"
    assert mutations == []


def test_chatgpt_web_candidate_does_not_submit_when_clipboard_paste_fails(monkeypatch):
    candidate = next(
        item for item in runner_module._CHATGPT_BROWSER_CANDIDATES if item.key == "helium"
    )
    monkeypatch.setattr(runner_module, "_process_is_headless", lambda _pid: False)
    submitted = False

    class FakeCua:
        def call(self, tool, payload):
            nonlocal submitted
            if tool == "list_apps":
                return {"apps": [{"bundle_id": candidate.bundle_id, "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {"windows": [{"pid": 123, "window_id": 456, "title": "ChatGPT", "layer": 0}]}
            if tool == "get_window_state":
                return {
                    "tree_markdown": (
                        'AXWebArea "ChatGPT"\n'
                        '[1] AXTextArea = "Ask ChatGPT"\n[2] AXPopUpButton "Pro"'
                    )
                }
            if tool == "page":
                return {"text": "Ask ChatGPT\nPro"}
            if tool == "click":
                return {"ok": True}
            if tool == "hotkey":
                return {"ok": False, "error": "paste_failed"}
            if tool == "press_key":
                submitted = True
                return {"ok": True}
            raise AssertionError(tool)

    result = runner_module._run_chatgpt_browser_candidate(
        candidate,
        {"prompt": "first line\nsecond line"},
        FakeCua(),
        lambda _: None,
        time.monotonic() + 1,
        "nonce",
    )

    assert result["ok"] is False
    assert result["error"] == "prompt_insertion_failed"
    assert submitted is False


def legacy_chatgpt_native_runner_accepts_chatgpt_classic_launch_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    typed: dict[str, str] = {}

    class FakeCua:
        def call(self, tool: str, payload: dict):
            if tool == "list_apps":
                return {
                    "apps": [
                        {
                            "bundle_id": "com.openai.chat",
                            "launch_path": "/Applications/ChatGPT Classic.app",
                            "name": "ChatGPT",
                            "pid": 0,
                            "running": False,
                        }
                    ]
                }
            if tool == "launch_app":
                return {"bundle_id": "?", "name": "?", "pid": 123}
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "on_current_space": True,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                if typed:
                    prompt = typed["text"]
                    begin = next(
                        line
                        for line in prompt.splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    end = begin.replace("BEGIN_", "END_")
                    return {
                        "bundle_id": "com.openai.chat",
                        "tree_markdown": f"AXStaticText ({begin}\nCLASSIC_PRO_OK\n{end})",
                    }
                return {
                    "bundle_id": "com.openai.chat",
                    "tree_markdown": (
                        '- AXApplication "ChatGPT"\n'
                        "  - [112] AXButton (New chat)\n"
                        '  - [106] AXButton = "5.5 Pro" (Options)\n'
                        "  - [101] AXTextArea\n"
                    ),
                }
            if tool == "click":
                return {"ok": True}
            if tool == "type_text":
                typed["text"] = payload["text"]
                return {"ok": True}
            if tool == "press_key":
                return {"ok": True}
            raise AssertionError(f"unexpected CUA tool {tool}")

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "transport": "gui",
            "prompt": "reply with the sentinel",
            "timeout_sec": 5,
        },
        cua=FakeCua(),
        sleep=lambda _: None,
    )

    assert result["ok"] is True
    assert result["output"] == "CLASSIC_PRO_OK"


def test_chatgpt_app_rejects_reliably_different_launched_bundle():
    class FakeCua:
        def call(self, tool: str, payload: dict):
            if tool == "list_apps":
                return {"apps": []}
            if tool == "launch_app":
                return {"bundle_id": "com.example.other", "pid": 123}
            raise AssertionError(f"unexpected CUA tool {tool}")

    with pytest.raises(RuntimeError, match="Native ChatGPT app did not launch"):
        runner_module._chatgpt_app(FakeCua())


def test_chatgpt_snapshot_rejects_unknown_bundle_without_chatgpt_marker():
    class FakeCua:
        def call(self, tool: str, payload: dict):
            assert tool == "get_window_state"
            return {"bundle_id": None, "tree_markdown": '- AXApplication "Other App"'}

    with pytest.raises(RuntimeError, match="CUA target is not the native ChatGPT app"):
        runner_module._chatgpt_snapshot(FakeCua(), 123, 456)


def legacy_chatgpt_native_runner_accepts_visible_window_when_space_flag_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    typed: dict[str, str] = {}

    class FakeCua:
        def call(self, tool: str, payload: dict):
            if tool == "list_apps":
                return {
                    "apps": [
                        {
                            "bundle_id": "com.openai.chat",
                            "name": "ChatGPT",
                            "pid": 123,
                            "running": True,
                            "active": True,
                        }
                    ]
                }
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "on_current_space": None,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                if typed:
                    prompt = typed["text"]
                    begin = next(
                        line
                        for line in prompt.splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    end = begin.replace("BEGIN_", "END_")
                    return {
                        "bundle_id": "com.openai.chat",
                        "tree_markdown": f"AXStaticText ({begin}\nGPT_PRO_OK\n{end})",
                    }
                return {
                    "bundle_id": "com.openai.chat",
                    "tree_markdown": (
                        '- AXApplication "ChatGPT"\n'
                        "  - [112] AXButton (New chat)\n"
                        '  - [106] AXButton = "5.5 Pro" (Options)\n'
                        "  - [101] AXTextArea\n"
                    ),
                }
            if tool == "click":
                return {"ok": True}
            if tool == "type_text":
                typed["text"] = payload["text"]
                return {"ok": True}
            if tool == "press_key":
                return {"ok": True}
            raise AssertionError(f"unexpected CUA tool {tool}")

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "transport": "gui",
            "prompt": "reply with the sentinel",
            "timeout_sec": 5,
        },
        cua=FakeCua(),
        sleep=lambda _: None,
    )

    assert result["ok"] is True
    assert result["output"] == "GPT_PRO_OK"


def legacy_chatgpt_native_runner_accepts_missing_bundle_id_when_tree_is_chatgpt(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    typed: dict[str, str] = {}

    class FakeCua:
        def call(self, tool: str, payload: dict):
            if tool == "list_apps":
                return {
                    "apps": [
                        {
                            "bundle_id": "com.openai.chat",
                            "name": "ChatGPT",
                            "pid": 123,
                            "running": True,
                            "active": True,
                        }
                    ]
                }
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "on_current_space": True,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                if typed:
                    prompt = typed["text"]
                    begin = next(
                        line
                        for line in prompt.splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    end = begin.replace("BEGIN_", "END_")
                    return {
                        "bundle_id": None,
                        "tree_markdown": f'AXWindow "ChatGPT"\nAXStaticText ({begin}\nGPT_PRO_OK\n{end})',
                    }
                return {
                    "bundle_id": None,
                    "tree_markdown": (
                        '- AXWindow "ChatGPT"\n'
                        "  - [112] AXButton (New chat)\n"
                        '  - [106] AXButton = "5.5 Pro" (Options)\n'
                        "  - [101] AXTextArea\n"
                    ),
                }
            if tool == "click":
                return {"ok": True}
            if tool == "type_text":
                typed["text"] = payload["text"]
                return {"ok": True}
            if tool == "press_key":
                return {"ok": True}
            raise AssertionError(f"unexpected CUA tool {tool}")

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "transport": "gui",
            "prompt": "reply with the sentinel",
            "timeout_sec": 5,
        },
        cua=FakeCua(),
        sleep=lambda _: None,
    )

    assert result["ok"] is True
    assert result["output"] == "GPT_PRO_OK"


def legacy_chatgpt_native_runner_switches_model_to_pro_before_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    events: list[tuple[str, int | None]] = []
    typed: dict[str, str] = {}
    state = {"picker_open": False, "pro_selected": False}

    class FakeCua:
        def call(self, tool: str, payload: dict):
            events.append((tool, payload.get("element_index")))
            if tool == "list_apps":
                return {
                    "apps": [
                        {
                            "bundle_id": "com.openai.chat",
                            "name": "ChatGPT",
                            "pid": 123,
                            "running": True,
                        }
                    ]
                }
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "on_current_space": True,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                if typed:
                    prompt = typed["text"]
                    begin = next(
                        line
                        for line in prompt.splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    end = begin.replace("BEGIN_", "END_")
                    return {
                        "bundle_id": "com.openai.chat",
                        "tree_markdown": f"AXStaticText ({begin}\nGPT_PRO_OK\n{end})",
                    }
                if state["picker_open"]:
                    return {
                        "bundle_id": "com.openai.chat",
                        "tree_markdown": (
                            '- AXApplication "ChatGPT"\n'
                            '  - [106] AXButton = "Auto" (Options) help="Pick a model or GPT" DISABLED\n'
                            "  - [201] AXButton (Pro, Research-grade intelligence)\n"
                        ),
                    }
                model = "5.5 Pro" if state["pro_selected"] else "Auto"
                return {
                    "bundle_id": "com.openai.chat",
                    "tree_markdown": (
                        '- AXApplication "ChatGPT"\n'
                        "  - [112] AXButton (New chat)\n"
                        f'  - [106] AXButton = "{model}" (Options) help="Pick a model or GPT"\n'
                        "  - [101] AXTextArea\n"
                    ),
                }
            if tool == "click":
                if payload["element_index"] == 106:
                    state["picker_open"] = True
                if payload["element_index"] == 201:
                    state["picker_open"] = False
                    state["pro_selected"] = True
                return {"ok": True}
            if tool == "type_text":
                assert state["pro_selected"] is True
                typed["text"] = payload["text"]
                return {"ok": True}
            if tool == "press_key":
                return {"ok": True}
            raise AssertionError(f"unexpected CUA tool {tool}")

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "transport": "gui",
            "prompt": "reply with the sentinel",
            "timeout_sec": 5,
        },
        cua=FakeCua(),
        sleep=lambda _: None,
    )

    assert result["ok"] is True
    assert ("click", 106) in events
    assert ("click", 201) in events
    assert events.index(("click", 201)) < events.index(("type_text", 101))


def legacy_chatgpt_native_runner_uses_copy_menu_when_ax_text_is_blank(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    # Enough ticks for the window polling loop (8s deadline, 0.5s sleep intervals)
    # plus the main response polling deadline.
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0])
    monkeypatch.setattr("agent_crossbar.runner.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("agent_crossbar.runner.time.sleep", lambda _: None)
    clipboard = {"text": "original clipboard"}
    typed: dict[str, str] = {}
    state = {"menu_open": False}

    monkeypatch.setattr("agent_crossbar.runner._read_text_clipboard", lambda: clipboard["text"])
    monkeypatch.setattr(
        "agent_crossbar.runner._write_text_clipboard",
        lambda text: clipboard.update(text=text),
    )

    class FakeCua:
        def call(self, tool: str, payload: dict):
            if tool == "list_apps":
                return {"apps": [{"bundle_id": "com.openai.chat", "pid": 123, "running": True}]}
            if tool == "list_windows":
                return {
                    "windows": [
                        {
                            "pid": 123,
                            "window_id": 456,
                            "title": "ChatGPT",
                            "layer": 0,
                            "on_current_space": True,
                            "is_on_screen": True,
                        }
                    ]
                }
            if tool == "get_window_state":
                if state["menu_open"]:
                    return {
                        "bundle_id": "com.openai.chat",
                        "tree_markdown": (
                            '- AXApplication "ChatGPT"\n  - [300] AXMenuItem (Copy)\n'
                        ),
                    }
                if typed:
                    return {
                        "bundle_id": "com.openai.chat",
                        "tree_markdown": (
                            '- AXApplication "ChatGPT"\n'
                            "  - [222] AXButton (Thought for 1s) actions=[AXShowMenu, Copy]\n"
                            '  - [106] AXButton = "5.5 Pro" (Options) help="Pick a model or GPT"\n'
                            "  - [101] AXTextArea\n"
                            '  - [109] AXButton (Send) help="Send message" DISABLED\n'
                        ),
                    }
                return {
                    "bundle_id": "com.openai.chat",
                    "tree_markdown": (
                        '- AXApplication "ChatGPT"\n'
                        "  - [112] AXButton (New chat)\n"
                        '  - [106] AXButton = "5.5 Pro" (Options) help="Pick a model or GPT"\n'
                        "  - [101] AXTextArea\n"
                    ),
                }
            if tool == "click":
                if payload["element_index"] == 222 and payload.get("action") == "show_menu":
                    state["menu_open"] = True
                if payload["element_index"] == 300:
                    prompt = typed["text"]
                    begin = next(
                        line
                        for line in prompt.splitlines()
                        if line.startswith("BEGIN_AGENTS_MCP_RESPONSE_")
                    )
                    end = begin.replace("BEGIN_", "END_")
                    clipboard["text"] = f"{begin}\nCOPIED_GPT_PRO_OK\n{end}"
                return {"ok": True}
            if tool == "type_text":
                typed["text"] = payload["text"]
                return {"ok": True}
            if tool == "press_key":
                return {"ok": True}
            raise AssertionError(f"unexpected CUA tool {tool}")

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "transport": "gui",
            "prompt": "reply with the sentinel",
            "timeout_sec": 1,
        },
        cua=FakeCua(),
        sleep=lambda _: None,
    )

    assert result["ok"] is True
    assert result["output"] == "COPIED_GPT_PRO_OK"
    assert clipboard["text"] == "original clipboard"


def test_chatgpt_response_extraction_preserves_trailing_parenthesis():
    tree = "AXStaticText (BEGIN_MARKER\nThe answer is (yes)\nEND_MARKER)"

    output = _extract_marked_response(tree, "BEGIN_MARKER", "END_MARKER")

    assert output == "The answer is (yes)"


def test_chatgpt_response_extraction_ignores_user_template_placeholder():
    begin = "BEGIN_AGENTS_MCP_RESPONSE_nonce"
    end = "END_AGENTS_MCP_RESPONSE_nonce"
    tree = (
        f"AXStaticText (User request included template:\n{begin}\\n<your answer>\\n{end})\n"
        f"AXStaticText (Assistant final:\n{begin}\\nGPT_PRO_OK\\n{end})"
    )

    output = _extract_marked_response(tree, begin, end)

    assert output == "GPT_PRO_OK"


def test_chatgpt_response_extraction_rejects_placeholder_only_output():
    begin = "BEGIN_AGENTS_MCP_RESPONSE_nonce"
    end = "END_AGENTS_MCP_RESPONSE_nonce"
    tree = f"AXStaticText ({begin}\\n<your answer>\\n{end})"

    output = _extract_marked_response(tree, begin, end)

    assert output is None


def test_chatgpt_active_model_check_requires_selected_model_label():
    tree = '[39] AXButton = "Auto" (Options) help="Pick a model or GPT; Pro available"'

    assert _active_chatgpt_model_is_pro(tree) is False
    assert (
        _active_chatgpt_model_is_pro(
            '[39] AXButton = "5.5 Pro" (Options) help="Pick a model or GPT"'
        )
        is True
    )


def test_chatgpt_pro_gui_runner_lock_blocks_second_job(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / "chatgpt_pro.lock").write_text('{"created_at": 9999999999}', encoding="utf-8")

    class FailingCua:
        def call(self, tool: str, payload: dict):  # pragma: no cover - should not be called
            raise AssertionError("CUA should not be touched when lock is busy")

    result = run_gui_request(
        {
            "profile": "chatgpt_pro",
            "operation": "advice",
            "transport": "gui",
            "prompt": "x",
        },
        cua=FailingCua(),
        sleep=lambda _: None,
    )

    assert result["ok"] is False
    assert result["error"] == "busy"
    assert (lock_dir / "chatgpt_pro.lock").exists()


def test_cua_driver_client_accepts_plain_text_action_output(monkeypatch):
    def fake_run(args, **kwargs):
        return _completed("clicked\n")

    monkeypatch.setattr("agent_crossbar.runner.subprocess.run", fake_run)

    result = CuaDriverClient(bin_path="cua-driver").call("click", {"pid": 123})

    assert result == {"raw_output": "clicked"}
