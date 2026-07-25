"""Regression audit for the public MCP tool surface."""

from __future__ import annotations

import inspect

from agent_crossbar import server
from agent_crossbar.profiles import profile_registry

CLIENT_ARGS = {"client", "client_name", "client_version", "client_session_id"}

EXPECTED_TOOL_ARGS = {
    "agent_start": {
        "profile",
        "prompt",
        "task",
        "interactive",
        "model",
        "effort",
        "cwd",
        "scope",
        "max_runtime_sec",
        *CLIENT_ARGS,
    },
    "profiles_list": CLIENT_ARGS,
    "profile_health": CLIENT_ARGS,
    "job_tail": {
        "job_id",
        "since_seq",
        "output_since_bytes",
        "max_bytes",
        "max_events",
        *CLIENT_ARGS,
    },
    "job_result": {"job_id", *CLIENT_ARGS},
    "job_send": {"job_id", "text", *CLIENT_ARGS},
    "job_stop": {"job_id", "reason", *CLIENT_ARGS},
    "job_list": {"status", "profile", "limit", *CLIENT_ARGS},
}


def test_public_tool_signatures_match_audited_argument_surface():
    for tool_name, expected in EXPECTED_TOOL_ARGS.items():
        actual = set(inspect.signature(getattr(server, tool_name)).parameters)
        assert actual == expected, tool_name


def test_agent_start_preserves_cwd_and_effort(tmp_path, monkeypatch):
    """agent_start forwards cwd/effort to print and ACP backends."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    captured: dict[str, dict] = {}
    captured_acp: list[dict] = []

    def fake_start_print_job(store, job_id, req, **kwargs):
        captured[req.get("operation", "unknown")] = dict(req)

    monkeypatch.setattr(server, "start_print_job", fake_start_print_job)

    import threading

    acp_started = threading.Event()

    async def fake_run_acp_job(store, job_id, **kwargs):
        captured_acp.append(kwargs)
        store.set_result(job_id, ok=True, summary="ACP_OK\n")
        acp_started.set()

    monkeypatch.setattr(server, "_run_acp_job", fake_run_acp_job)

    # Mock the readiness probe for all profiles
    import agent_crossbar.readiness as rmod

    def fake_probe(profile, _runner=None, use_cache=True):
        import time

        from agent_crossbar.readiness import ReadinessResult

        return ReadinessResult(
            profile=profile,
            state="ready",
            support_tier="supported",
            authenticated=True,
            probe_version=1,
            timestamp=time.time(),
        )

    monkeypatch.setattr(rmod, "probe_profile", fake_probe)

    # ask task with cwd and effort — maps to advice operation, reasonix supports it
    server.agent_start(
        profile="reasonix",
        prompt="explain this",
        model="deepseek-v4-flash",
        task="ask",
        cwd="/repo",
        effort="low",
    )
    assert captured["advice"]["cwd"] == "/repo", f"cwd not preserved: {captured}"
    assert captured["advice"]["effort"] == "low", f"effort not preserved: {captured}"

    # review task with cwd and effort
    server.agent_start(
        profile="reasonix",
        prompt="review that",
        model="deepseek-v4-flash",
        task="review",
        cwd="/other",
        effort="medium",
    )
    assert captured["review"]["cwd"] == "/other"
    assert captured["review"]["effort"] == "medium"

    # dev task with explicit effort on ACP-backed Codex
    server.agent_start(
        profile="codex",
        prompt="implement it",
        model="gpt-5.6-sol",
        task="dev",
        cwd="/repo",
        effort="high",
    )
    assert acp_started.wait(2), "ACP job did not start"
    assert captured_acp == [
        {
            "provider": "codex",
            "prompt": "implement it",
            "cwd": "/repo",
            "task": "dev",
            "model": "gpt-5.6-sol",
            "effort": "high",
            "autonomy": "edit_local",
            "max_runtime_sec": None,
        }
    ]


def test_every_advertised_transport_has_server_routing():
    """Every transport implied by profiles has a corresponding server runner."""
    routed = {
        ("print", "auto"): "start_print_job",
        ("gui",): "start_gui_job",
        ("tmux",): "start_tmux_job",
    }
    implemented_transports = {transport for transports in routed for transport in transports}
    # Derive advertised transports from flat profile schema:
    # interactive=True → tmux available; all profiles → print; chatgpt_pro → gui
    advertised: set[str] = set()
    for name, entry in profile_registry().items():
        advertised.add("print")
        advertised.add("auto")
        if entry.get("interactive"):
            advertised.add("tmux")
        if name == "chatgpt_pro":
            advertised.add("gui")

    assert advertised <= implemented_transports
    for runner_name in set(routed.values()):
        assert hasattr(server, runner_name), runner_name
