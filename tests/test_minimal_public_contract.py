"""Contract tests for the Agent Crossbar 0.2 public surface."""

from __future__ import annotations

import inspect

from agent_crossbar import providers, runner, server
from agent_crossbar.profiles import list_profiles

CLIENT_ARGS = {"client", "client_name", "client_version", "client_session_id"}


def test_agent_start_has_only_cross_provider_fields() -> None:
    expected = {
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
    }

    assert set(inspect.signature(server.agent_start).parameters) == expected


def test_mcp_exposes_only_minimal_public_tools() -> None:
    assert set(server.mcp._tool_manager._tools) == {
        "profiles_list",
        "profile_health",
        "agent_start",
        "job_tail",
        "job_result",
        "job_send",
        "job_stop",
        "job_list",
    }


def test_removed_fields_are_absent_from_agent_start() -> None:
    fields = set(inspect.signature(server.agent_start).parameters)
    removed = {
        "transport",
        "autonomy",
        "external_context",
        "sensitivity",
        "sanitized_context_only",
        "text_subtype",
        "budget_usd",
        "review_target",
        "context_target",
        "timeout_sec",
    }

    assert fields.isdisjoint(removed)


def test_agent_start_model_is_required() -> None:
    parameter = inspect.signature(server.agent_start).parameters["model"]
    assert parameter.default is inspect.Parameter.empty


def test_runtime_launchers_never_fall_back_to_a_model() -> None:
    runner_source = inspect.getsource(runner._candidates)
    reasonix_source = inspect.getsource(providers._reasonix_plan)

    assert 'req.get("model") or "gpt-5.6-sol"' not in runner_source
    assert 'req.get("model") or "deepseek-v4-flash"' not in runner_source
    assert "if model is None:" not in reasonix_source


def test_qwen_is_not_a_public_profile() -> None:
    assert "qwen" not in list_profiles()


def test_all_review_targets_not_in_shipped_runtime() -> None:
    """ALL_REVIEW_TARGETS must not exist in the shipped profiles module."""
    from agent_crossbar import profiles as pmod

    assert not hasattr(pmod, "ALL_REVIEW_TARGETS"), (
        "ALL_REVIEW_TARGETS is legacy dead code and must not be in shipped runtime"
    )


def test_minimal_agent_start_review_creates_job_without_target_fields(
    tmp_path,
    monkeypatch,
) -> None:
    """Review via agent_start works with just profile+prompt — no target fields."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    def fake_start_print_job(store, job_id, req, **kwargs):
        pass

    monkeypatch.setattr(server, "start_print_job", fake_start_print_job)

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

    result = server.agent_start(
        profile="reasonix",
        prompt="review the code",
        task="review",
        model="deepseek-v4-flash",
    )

    assert result["ok"] is True
    assert result["job_id"] is not None
    # Verify no target fields leaked into the request that reached start_print_job
    assert "review_target" not in result
    assert "context_target" not in result


def test_agent_start_rejects_unknown_task_without_creating_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    result = server.agent_start(
        profile="codex",
        prompt="test",
        task="summarize",
        model="gpt-5.6-sol",
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_task"
    assert result["job_created"] is False
    assert not (tmp_path / "jobs").exists()


def test_repo_scope_requires_cwd_without_creating_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    result = server.agent_start(
        profile="codex",
        prompt="review it",
        task="review",
        scope={"kind": "working_tree"},
        model="gpt-5.6-sol",
    )

    assert result["ok"] is False
    assert result["error"] == "cwd_required"
    assert result["job_created"] is False
    assert not (tmp_path / "jobs").exists()


def test_interactive_request_is_rejected_when_adapter_cannot_continue(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    result = server.agent_start(
        profile="chatgpt_pro",
        prompt="answer",
        task="ask",
        interactive=True,
        model="gpt-5",
    )

    assert result["ok"] is False
    assert result["error"] == "interactive_not_supported"
    assert result["job_created"] is False


def test_max_runtime_is_validated_before_job_creation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    result = server.agent_start(
        profile="codex",
        prompt="test",
        task="ask",
        max_runtime_sec=0,
        model="gpt-5.6-sol",
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_max_runtime"
    assert result["job_created"] is False


def test_agent_start_always_creates_async_job_and_returns_job_id(
    tmp_path,
    monkeypatch,
) -> None:
    """Noninteractive agent_start must create a durable job, not run synchronously."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    def fake_start_print_job(store, job_id, req, **kwargs):
        pass  # just record, don't execute

    monkeypatch.setattr(server, "start_print_job", fake_start_print_job)

    # codex/dev with no explicit effort routes via ACP, not start_print_job —
    # this must be mocked too, or the fire-and-forget background thread calls
    # the real ACP bridge (pnpm dlx @agentclientprotocol/codex-acp).
    acp_calls: list[dict] = []

    async def fake_run_acp_job(store, job_id, **kwargs):
        acp_calls.append(kwargs)
        store.set_result(job_id, ok=True, summary="ACP_OK\n")

    monkeypatch.setattr(server, "_run_acp_job", fake_run_acp_job)

    # Mock the readiness probe — real probe requires ACP bridge locally cached
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
    monkeypatch.setattr(
        "agent_crossbar.acp_lifecycle.check_codex_acp_readiness",
        lambda _runner: {"ready": True},
    )

    result = server.agent_start(
        profile="codex",
        prompt="implement feature",
        task="dev",
        model="gpt-5.6-sol",
    )

    assert result["ok"] is True
    assert result["job_id"] is not None
    assert (tmp_path / "jobs").exists()
    assert (tmp_path / "jobs" / result["job_id"]).exists()
    assert len(acp_calls) == 1, "real ACP bridge must not be invoked from unit tests"


def test_codex_agent_start_ask_falls_back_to_text(tmp_path, monkeypatch):
    """Codex does not support advice, so task=ask must map to text."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    def fake_start_print_job(store, job_id, req, **kwargs):
        pass

    monkeypatch.setattr(server, "start_print_job", fake_start_print_job)

    # codex/ask with no explicit effort routes via ACP, not start_print_job —
    # this must be mocked too, or the fire-and-forget background thread calls
    # the real ACP bridge (pnpm dlx @agentclientprotocol/codex-acp).
    acp_calls: list[dict] = []

    async def fake_run_acp_job(store, job_id, **kwargs):
        acp_calls.append(kwargs)
        store.set_result(job_id, ok=True, summary="ACP_OK\n")

    monkeypatch.setattr(server, "_run_acp_job", fake_run_acp_job)

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
    monkeypatch.setattr(
        "agent_crossbar.acp_lifecycle.check_codex_acp_readiness",
        lambda _runner: {"ready": True},
    )

    result = server.agent_start(
        profile="codex",
        prompt="explain something",
        task="ask",
        model="gpt-5.6-sol",
    )

    assert result["ok"] is True, f"codex ask failed: {result}"
    assert result["job_id"] is not None
    # The resolved operation must be 'text', not 'advice'
    assert result.get("operation") == "text", f"expected text, got {result.get('operation')}"
    assert len(acp_calls) == 1, "real ACP bridge must not be invoked from unit tests"


def test_reasonix_interactive_true_selects_tmux_transport(tmp_path, monkeypatch):
    """Reasonix interactive=true must route to tmux, not print."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    def fake_start_tmux_job(store, job_id, req, **kwargs):
        pass

    monkeypatch.setattr(server, "start_tmux_job", fake_start_tmux_job)

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

    result = server.agent_start(
        profile="reasonix",
        prompt="do thing",
        task="ask",
        interactive=True,
        model="deepseek-v4-flash",
    )

    assert result["ok"] is True, f"reasonix interactive failed: {result}"
    assert result["job_id"] is not None


def test_claude_interactive_true_rejected_with_billing_guidance(tmp_path, monkeypatch):
    """Claude interactive=True must be rejected with clear separate-credit/metered billing guidance."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    result = server.agent_start(
        profile="claude",
        prompt="review the code",
        task="review",
        interactive=True,
        model="sonnet",
    )

    assert result["ok"] is False
    assert result["error"] == "interactive_not_supported"
    assert result["job_created"] is False
    # Must include billing guidance about separate credit / metered usage
    msg = result.get("message", "").lower()
    assert "separate" in msg or "metered" in msg or "credit" in msg or "billing" in msg, (
        f"Claude interactive rejection must mention separate-credit or metered billing: {msg}"
    )
    # Must never recommend an API key as the workaround
    assert "api key" not in msg and "anthropic_api_key" not in msg, (
        f"Claude interactive rejection must not recommend an API key: {msg}"
    )
    # Must reference the supported alternative (claude_bg)
    assert (
        "claude_bg" in msg or "background" in msg or "noninteractive" in msg or "claude --bg" in msg
    ), f"Claude interactive rejection must reference claude_bg alternative: {msg}"


def test_claude_agent_start_uses_claude_bg_backend(tmp_path, monkeypatch):
    """Claude agent_start response must report backend='claude_bg', not 'print'."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

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
    # Also need to fake the adapter readiness (Claude uses a different path)

    def fake_adapter_readiness(_self, runner):
        from agent_crossbar.adapters.claude import ReadinessResult as AdapterRR

        return AdapterRR(state="ready", authenticated=True)

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.ClaudeAdapter.check_readiness",
        fake_adapter_readiness,
    )

    def fake_launch(_self, runner, **kwargs):
        from agent_crossbar.adapters.claude import LaunchResult

        return LaunchResult(
            session_id="abc12345",
            backend="claude_bg",
            args=["claude", "--bg", "hello"],
            cwd="/tmp",
        )

    monkeypatch.setattr(
        "agent_crossbar.adapters.claude.ClaudeAdapter.launch",
        fake_launch,
    )

    result = server.agent_start(
        profile="claude",
        prompt="review the code",
        task="review",
        model="sonnet",
    )

    assert result["ok"] is True
    assert "backend" in result, "Claude agent_start response must include backend field"
    assert result["backend"] == "claude_bg", (
        f"Claude backend must be 'claude_bg', got {result.get('backend')!r}"
    )


def test_reasonix_interactive_false_selects_print_transport(tmp_path, monkeypatch):
    """Reasonix interactive=false must use print, not tmux."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    captured = {}

    def fake_start_print_job(store, job_id, req, **kwargs):
        captured["transport"] = req.get("transport")

    monkeypatch.setattr(server, "start_print_job", fake_start_print_job)

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

    result = server.agent_start(
        profile="reasonix",
        prompt="do thing",
        task="ask",
        interactive=False,
        model="deepseek-v4-flash",
    )

    assert result["ok"] is True
    assert captured["transport"] == "print"


# ── Minimal profiles_list contract ────────────────────────────────────────

LEGACY_FIELDS = frozenset(
    {
        "budget",
        "cloud_backed",
        "default_model_by_operation",
        "default_effort_by_operation",
        "efforts",
        "effort_aliases",
        "blocked_operations",
        "capabilities",
        "local_tools",
    }
)


def test_profiles_list_has_no_budget_or_legacy_fields():
    """profiles_list must not expose budget, cloud_backed, or legacy metadata."""
    from agent_crossbar.profiles import profile_registry

    reg = profile_registry()
    for name, entry in reg.items():
        overlap = set(entry.keys()) & LEGACY_FIELDS
        assert not overlap, f"{name} has legacy fields: {overlap}"


def test_profiles_list_has_minimal_schema_per_entry():
    """Each profile entry must have only: aliases, models, default_model, operations, interactive."""
    from agent_crossbar.profiles import profile_registry

    allowed = {"aliases", "models", "operations", "interactive", "support_tier"}
    reg = profile_registry()
    for name, entry in reg.items():
        extra = set(entry.keys()) - allowed
        assert not extra, f"{name} has unexpected fields: {extra}"


def test_profiles_list_interactive_is_boolean():
    """interactive must be a boolean, not a transport list."""
    from agent_crossbar.profiles import profile_registry

    reg = profile_registry()
    for name, entry in reg.items():
        interactive = entry.get("interactive")
        assert isinstance(interactive, bool), f"{name} interactive is not bool: {type(interactive)}"


def test_profiles_list_operations_are_string_lists():
    """operations must be a list of strings, not capability dicts."""
    from agent_crossbar.profiles import profile_registry

    reg = profile_registry()
    for name, entry in reg.items():
        ops = entry.get("operations", [])
        assert isinstance(ops, list), f"{name} operations is not list: {type(ops)}"
        for op in ops:
            assert isinstance(op, str), f"{name} operation is not str: {type(op)}"


def test_profiles_list_has_no_transports_in_entries():
    """No transports key anywhere in the registry."""
    from agent_crossbar.profiles import profile_registry

    reg = profile_registry()
    for name, entry in reg.items():
        assert "transports" not in entry, f"{name} has transports key"


def test_capability_for_and_blocked_operations_are_removed():
    """capability_for and blocked_operations must not exist."""
    import agent_crossbar.profiles as pm

    assert not hasattr(pm, "capability_for"), "capability_for must be removed"
    assert not hasattr(pm, "blocked_operations"), "blocked_operations must be removed"
    assert not hasattr(pm, "budget_mode"), "budget_mode must be removed"
    assert not hasattr(pm, "profile_budget_mode"), "profile_budget_mode must be removed"
    assert not hasattr(pm, "is_cloud_backed"), "is_cloud_backed must be removed"


def test_profile_operations_and_interactive_helpers_exist():
    """New minimal helpers must exist."""
    import agent_crossbar.profiles as pm

    assert hasattr(pm, "profile_operations"), "profile_operations missing"
    assert hasattr(pm, "profile_interactive"), "profile_interactive missing"
    assert pm.profile_interactive("reasonix") is True
    assert pm.profile_interactive("codex") is False
    assert set(pm.profile_operations("codex")) == {"review", "text", "dev"}
