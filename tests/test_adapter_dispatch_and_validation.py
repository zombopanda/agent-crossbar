"""Regression tests for provider-neutral architecture extraction.

Covers:
- LocalSubprocessRunner/RunResult living in a provider-neutral module and
  still being importable from agent_crossbar.adapters.claude for back-compat.
- Claude lifecycle dispatch being adapter-owned (ClaudeAdapter.start_lifecycle)
  so the server has no direct claude_lifecycle import on the normal path.
- validation.py driving model/effort resolution through adapter methods with
  no provider-name branches.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import agent_crossbar.server as server
from agent_crossbar.adapters.base import ModelCatalog, ModelInfo
from agent_crossbar.adapters.claude import ClaudeAdapter
from agent_crossbar.adapters.claude import LocalSubprocessRunner as ClaudeLocalSubprocessRunner
from agent_crossbar.adapters.claude import RunResult as ClaudeRunResult
from agent_crossbar.adapters.codex import adapter as codex_adapter
from agent_crossbar.adapters.opencode import adapter as opencode_adapter
from agent_crossbar.adapters.reasonix import adapter as reasonix_adapter
from agent_crossbar.adapters.registry import get_adapter
from agent_crossbar.subprocess_runner import LocalSubprocessRunner, RunResult
from agent_crossbar.validation import validate_start_request

# ---------------------------------------------------------------------------
# Task 1: LocalSubprocessRunner/RunResult live in a provider-neutral module
# ---------------------------------------------------------------------------


def test_local_subprocess_runner_reexported_identically_from_claude_adapter():
    """adapters.claude must re-export the same objects, not copies."""
    assert ClaudeLocalSubprocessRunner is LocalSubprocessRunner
    assert ClaudeRunResult is RunResult


def test_local_subprocess_runner_module_is_provider_neutral():
    """The runner's home module must not be Claude-specific."""
    assert LocalSubprocessRunner.__module__ == "agent_crossbar.subprocess_runner"
    assert RunResult.__module__ == "agent_crossbar.subprocess_runner"


def test_server_and_agent_runner_import_neutral_module_not_claude_adapter():
    import agent_crossbar.agent_runner as agent_runner_module

    assert server.LocalSubprocessRunner is LocalSubprocessRunner
    assert agent_runner_module.LocalSubprocessRunner is LocalSubprocessRunner
    server_source = inspect.getsource(server)
    assert "from agent_crossbar.adapters.claude import LocalSubprocessRunner" not in server_source


# ---------------------------------------------------------------------------
# Task 2: Claude lifecycle dispatch is adapter-owned
# ---------------------------------------------------------------------------


def test_server_has_no_direct_claude_lifecycle_import_on_normal_path():
    """The server must dispatch through adapter.start_lifecycle, never importing
    agent_crossbar.adapters.claude_lifecycle directly."""
    server_source = inspect.getsource(server)
    assert "claude_lifecycle" not in server_source


def test_claude_adapter_exposes_start_lifecycle():
    adapter = get_adapter("claude")
    assert callable(getattr(adapter, "start_lifecycle", None))


def test_readiness_dispatches_through_registered_adapter_probe(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import agent_crossbar.readiness as readiness
    from agent_crossbar.adapters import registry

    calls = []

    def fake_probe(runner):
        calls.append(runner)
        return readiness.ProbeResult(state="ready", authenticated=True, evidence="fixture")

    fake = SimpleNamespace(support_tier="supported", readiness_probe=fake_probe)
    monkeypatch.setitem(registry.ADAPTERS, "fixture", fake)
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    result = readiness.probe_profile("fixture", _runner=object(), use_cache=False)

    assert result.state == "ready"
    assert len(calls) == 1


def test_claude_adapter_start_lifecycle_delegates_to_claude_lifecycle_module():
    """ClaudeAdapter.start_lifecycle must forward to claude_lifecycle.start_claude_job
    with itself bound as the adapter, and forward through all other kwargs."""
    adapter = ClaudeAdapter()
    sentinel = {"ok": True, "job_id": "sentinel-job"}
    captured: dict[str, object] = {}

    def _fake_start_claude_job(**kwargs):
        captured.update(kwargs)
        return sentinel

    with patch(
        "agent_crossbar.adapters.claude_lifecycle.start_claude_job",
        _fake_start_claude_job,
    ):
        result = adapter.start_lifecycle(
            result={"profile": "claude", "operation": "advice"},
            interactive=False,
            client=None,
            client_session_id=None,
            client_name="test",
            model="claude-sonnet-5",
            effort=None,
            task="ask",
            prompt="hi",
            effective_cwd="/tmp",
            max_runtime_sec=None,
            run_req={},
            store=object(),
            attach_writer_lease=lambda *a, **k: None,
            session_id_for=lambda *a, **k: None,
            metadata_for=lambda *a, **k: {"name": "test"},
            agent_starter=lambda *a, **k: None,
            tool_error=lambda *a, **k: {"ok": False},
        )

    assert result is sentinel
    assert captured["adapter"] is adapter
    assert captured["model"] == "claude-sonnet-5"
    assert captured["task"] == "ask"


def test_agent_start_reports_lifecycle_dispatch_unsupported_when_adapter_lacks_start_lifecycle(
    monkeypatch,
):
    """A hypothetical lifecycle-owning adapter without start_lifecycle must fail
    with an explicit error rather than the server importing a provider module."""
    from types import SimpleNamespace

    from agent_crossbar.adapters import registry as adapters_registry

    fake_adapter = SimpleNamespace(
        name="claude",
        check_readiness=lambda runner: None,
        launch=lambda *a, **k: None,
        supports_interactive=False,
    )

    monkeypatch.setattr(adapters_registry, "get_adapter", lambda name: fake_adapter)
    monkeypatch.setattr(server, "get_adapter", lambda name: fake_adapter)
    monkeypatch.setattr(
        "agent_crossbar.readiness.probe_profile",
        lambda *a, **k: SimpleNamespace(state="ready", error_code=None, remediation=None),
    )

    result = server.agent_start(
        profile="claude",
        prompt="hi",
        model="claude-sonnet-5",
        task="ask",
    )
    assert result.get("ok") is False
    assert result.get("error") == "lifecycle_dispatch_unsupported"


# ---------------------------------------------------------------------------
# Task 3: validation.py drives model/effort resolution through adapters
# ---------------------------------------------------------------------------


def _base_req(**overrides):
    req = {
        "operation": "review",
        "profile": "codex",
        "transport": "print",
        "autonomy": "read_only",
        "sensitivity": "normal",
        "prompt": "x",
        "model": "gpt-5.6-sol",
    }
    req.update(overrides)
    return req


def test_validation_has_no_provider_name_branches():
    """Guard against reintroducing per-provider `if resolved == "<name>"` branches."""
    import agent_crossbar.validation as validation_module

    source = inspect.getsource(validation_module)
    for profile in ("codex", "opencode", "reasonix", "claude"):
        assert f'"{profile}"' not in source, (
            f"validation.py must not branch on the literal profile name '{profile}'"
        )


def test_validation_dispatches_model_resolution_through_adapter(tmp_path, monkeypatch):
    """validate_start_request must call adapter.resolve_model_id, not reimplement
    per-provider matching itself."""
    catalog = ModelCatalog(
        models=("gpt-5.6-sol", "gpt-5.6-terra"),
        default_model="gpt-5.6-sol",
        native_efforts=("low", "medium", "high", "max"),
        source="test",
    )
    calls = []
    original = codex_adapter.resolve_model_id

    def _spy(requested, catalog_arg):
        calls.append(requested)
        return original(requested, catalog_arg)

    monkeypatch.setattr(codex_adapter, "resolve_model_id", _spy)

    import agent_crossbar.discovery as discovery

    with patch.object(discovery, "discover_profile_models", return_value=catalog):
        result = validate_start_request(_base_req(), state_root=tmp_path)

    assert result["ok"] is True
    assert calls == ["gpt-5.6-sol"]


def test_validation_dispatches_effort_resolution_through_adapter(tmp_path, monkeypatch):
    catalog = ModelCatalog(
        models=("gpt-5.6-sol",),
        default_model="gpt-5.6-sol",
        native_efforts=("low", "medium", "high", "max"),
        source="test",
    )
    calls = []
    original = codex_adapter.validate_effort

    def _spy(effort, catalog_arg, model_id):
        calls.append(effort)
        return original(effort, catalog_arg, model_id)

    monkeypatch.setattr(codex_adapter, "validate_effort", _spy)

    import agent_crossbar.discovery as discovery

    with patch.object(discovery, "discover_profile_models", return_value=catalog):
        result = validate_start_request(_base_req(effort="light"), state_root=tmp_path)

    assert result["ok"] is True
    assert result["effort"] == "low"
    assert calls == ["light"]


def test_validation_skips_adapter_catalog_dispatch_for_static_allowlist_profiles(
    tmp_path, monkeypatch
):
    """chatgpt_pro has no live model discovery — its adapter's discovery-only
    hooks must never be invoked."""
    from agent_crossbar.adapters.chatgpt_pro import adapter as chatgpt_pro_adapter

    called = []
    monkeypatch.setattr(
        chatgpt_pro_adapter,
        "resolve_model_id",
        lambda *a, **k: called.append(True) or (None, ("invalid_model", "boom")),
    )

    req = _base_req(profile="chatgpt_pro", transport="gui", operation="advice", model="gpt-5")
    result = validate_start_request(req, state_root=tmp_path)

    assert result["ok"] is True
    assert called == []


def test_opencode_adapter_ignores_effort_when_not_provided():
    catalog = ModelCatalog(
        models=("opencode-go/glm-5.2",),
        default_model="opencode-go/glm-5.2",
        native_efforts=("low", "medium"),
        source="test",
        model_info=(ModelInfo(id="opencode-go/glm-5.2", supported_efforts=("low", "medium")),),
    )
    normalized, resolved, error = opencode_adapter.validate_effort(
        None, catalog, "opencode-go/glm-5.2"
    )
    assert (normalized, resolved, error) == (None, None, None)


def test_reasonix_adapter_never_validates_effort():
    """Reasonix has no native effort/model-capability relationship — the base
    no-op validate_effort must apply even when an effort value is supplied."""
    catalog = ModelCatalog(
        models=("deepseek-flash/deepseek-v4-flash",),
        default_model="deepseek-flash/deepseek-v4-flash",
        native_efforts=(),
        source="test",
    )
    normalized, resolved, error = reasonix_adapter.validate_effort(
        "high", catalog, "deepseek-flash/deepseek-v4-flash"
    )
    assert (normalized, resolved, error) == (None, None, None)


def test_reasonix_and_opencode_share_fuzzy_suffix_resolution():
    catalog = ModelCatalog(
        models=("deepseek-flash/deepseek-v4-flash", "deepseek-pro/deepseek-v4-pro"),
        default_model="deepseek-flash/deepseek-v4-flash",
        native_efforts=(),
        source="test",
    )
    model_id, error = reasonix_adapter.resolve_model_id("deepseek-v4-flash", catalog)
    assert error is None
    assert model_id == "deepseek-flash/deepseek-v4-flash"


def test_claude_adapter_requires_exact_model_match_no_fuzzy():
    catalog = ModelCatalog(
        models=("claude-opus-5",),
        default_model="claude-opus-5",
        native_efforts=("low", "medium", "high"),
        source="test",
    )
    claude_adapter = get_adapter("claude")
    model_id, error = claude_adapter.resolve_model_id("opus-5", catalog)
    assert model_id is None
    assert error is not None
    assert error[0] == "invalid_model"
