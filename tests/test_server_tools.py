"""FastMCP tool surface and job runner tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_crossbar.server import (
    profile_health,
    profiles_list,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _today_dir(tmp_path: Path) -> Path:
    telemetry_root = tmp_path / "telemetry"
    days = list(telemetry_root.iterdir())
    assert len(days) == 1
    return days[0]


def _logged_tools(tmp_path: Path, file_name: str) -> list[str]:
    return [entry["tool"] for entry in _read_jsonl(_today_dir(tmp_path) / file_name)]


@pytest.fixture(autouse=True)
def _no_real_provider_execution(monkeypatch):
    """Server tool tests should not call real external CLIs."""

    def fake_start_print_job(store, job_id, req, **kwargs):
        return store.set_result(job_id, True, summary="PROVIDER_OK\n")

    monkeypatch.setattr(
        "agent_crossbar.server.start_print_job",
        fake_start_print_job,
    )


@pytest.fixture(autouse=True)
def _no_live_model_discovery(monkeypatch):
    """profiles_list now attempts bounded live discovery on a cache miss —
    default to a fast, deterministic failure so tests never spawn the real
    codex/opencode/claude CLIs that may be installed on the test machine.
    Tests that want to exercise the discovery path override this mock.
    """
    import agent_crossbar.discovery as _disc

    def _boom(state_root, profile, *, refresh=False):
        raise RuntimeError(f"no live {profile} probe in unit tests")

    monkeypatch.setattr(_disc, "discover_profile_models", _boom)


# ── profiles_list ──────────────────────────────────────────────────────────


def test_profiles_list_returns_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    result = profiles_list(client_name="codex", client_session_id="s1")
    assert result["ok"] is True
    assert "reasonix" in result["profiles"]
    assert "codex" in result["profiles"]
    assert "claude" in result["profiles"]
    assert "opus" not in result["profiles"]
    assert "fable" not in result["profiles"]
    assert "qwen" not in result["profiles"]
    assert len(result["profiles"]) == 5
    assert result["profile_details"]["claude"]["aliases"] == ["opus", "fable"]
    assert result["profile_details"]["codex"]["models"] == ["gpt-5.6-sol", "gpt-5.6-terra"]
    assert "default_model" not in result["profile_details"]["codex"]
    assert result["profile_details"]["codex"]["operations"] == ["review", "text", "dev"]
    assert result["profile_details"]["codex"]["interactive"] is False
    request = _read_jsonl(_today_dir(tmp_path) / "requests.jsonl")[0]
    assert request["tool"] == "profiles_list"
    assert request["client"]["name"] == "codex"
    assert request["client"]["session_id"] == "s1"


def test_profiles_list_uses_env_client_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_CROSSBAR_CLIENT_NAME", "claude")
    monkeypatch.setenv("AGENT_CROSSBAR_CLIENT_VERSION", "2.1.98")

    result = profiles_list()

    assert result["ok"] is True
    request = _read_jsonl(_today_dir(tmp_path) / "requests.jsonl")[0]
    assert request["client"]["name"] == "claude"
    assert request["client"]["version"] == "2.1.98"


# ── profile_health ─────────────────────────────────────────────────────────


def test_profile_health_returns_known_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    result = profile_health(client_name="codex")
    assert result["ok"] is True
    assert "profiles" in result
    # profile_health now returns real readiness, not just "registered"
    assert len(result["profiles"]) == 5
    profile_names = {p["profile"] for p in result["profiles"]}
    assert profile_names == {"reasonix", "codex", "claude", "opencode", "chatgpt_pro"}
    for p in result["profiles"]:
        assert "state" in p
        assert p["state"] in {
            "ready",
            "needs_auth",
            "missing_binary",
            "unsupported_os",
            "misconfigured",
            "degraded",
        }
        assert "support_tier" in p
        assert "authenticated" in p
        assert "probe_version" in p
        # Model discovery metadata lives here, not in profiles_list's entries
        assert "models" in p
        assert "discovery_available" in p["models"]
    assert "profile_health" in _logged_tools(tmp_path, "requests.jsonl")


def test_profile_health_models_never_spawns_discovery_subprocess(tmp_path, monkeypatch):
    """profile_health must stay cache-only for model discovery — never a live fetch."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    import agent_crossbar.discovery as _disc

    def _boom(*args, **kwargs):
        raise AssertionError("profile_health must not trigger live model discovery")

    monkeypatch.setattr(_disc, "discover_profile_models", _boom)

    result = profile_health()
    assert result["ok"] is True


def test_profile_health_sanitizes_model_lookup_crash(tmp_path, monkeypatch):
    """A crash in the model-discovery lookup must never leak a raw secret or
    an unbounded traceback into profile_health's error field."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    secret = "sk-ant-super-secret-token-should-never-appear"

    def _boom(state_root, profile):
        raise RuntimeError(f"lookup failed, token={secret} " + ("x" * 4000))

    monkeypatch.setattr("agent_crossbar.server.cached_profile_health_entry", _boom)

    result = profile_health()
    assert result["ok"] is True
    for entry in result["profiles"]:
        error = entry["models"]["error"]
        assert secret not in error
        assert len(error.encode("utf-8")) <= 2048


def test_profiles_list_uses_cached_models_when_available(tmp_path, monkeypatch):
    """profiles_list reflects a live-discovered model catalog for codex."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    import agent_crossbar.discovery as _disc
    from agent_crossbar.adapters.base import ModelCatalog

    def _fake_discover(state_root, profile, *, refresh=False):
        if profile == "codex":
            return ModelCatalog(
                models=("gpt-live-a", "gpt-live-b"),
                default_model="gpt-live-a",
                native_efforts=(),
                source="codex app-server model/list",
            )
        raise RuntimeError(f"no live {profile} probe in unit tests")

    monkeypatch.setattr(_disc, "discover_profile_models", _fake_discover)

    result = profiles_list()
    assert result["ok"] is True
    assert result["profile_details"]["codex"]["models"] == ["gpt-live-a", "gpt-live-b"]
    assert "default_model" not in result["profile_details"]["codex"]
    # Schema stays minimal
    allowed = {"aliases", "models", "operations", "interactive", "support_tier"}
    for entry in result["profile_details"].values():
        assert set(entry.keys()) == allowed


def test_profiles_list_falls_back_to_static_without_cache(tmp_path, monkeypatch):
    """With no cache and a failed live probe, profiles_list returns the
    deterministic static models — never empty."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    result = profiles_list()
    assert result["ok"] is True
    assert result["profile_details"]["codex"]["models"] == ["gpt-5.6-sol", "gpt-5.6-terra"]
    assert result["profile_details"]["codex"]["models"] != []


def test_profiles_list_invokes_live_discovery_on_clean_cache(tmp_path, monkeypatch):
    """A completely clean cache must still attempt real discover_profile_models —
    profiles_list never silently skips discovery just because there's no cache yet."""
    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))

    import agent_crossbar.discovery as _disc
    from agent_crossbar.adapters.base import ModelCatalog

    calls: list[str] = []

    def _fake_discover(state_root, profile, *, refresh=False):
        calls.append(profile)
        if profile == "codex":
            return ModelCatalog(
                models=("gpt-live-1", "gpt-live-2"),
                default_model="gpt-live-1",
                native_efforts=(),
                source="codex app-server model/list",
            )
        raise RuntimeError(f"no live {profile} probe in unit tests")

    monkeypatch.setattr(_disc, "discover_profile_models", _fake_discover)

    result = profiles_list()

    assert "codex" in calls  # discovery was actually attempted, not skipped
    assert result["profile_details"]["codex"]["models"] == ["gpt-live-1", "gpt-live-2"]
    assert "default_model" not in result["profile_details"]["codex"]


def test_job_stop_acp_persists_terminal_result(tmp_path, monkeypatch):
    from agent_crossbar.jobs import JobStore
    from agent_crossbar.server import job_stop

    monkeypatch.setenv("AGENT_CROSSBAR_STATE_DIR", str(tmp_path))
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode",
        operation="dev",
        transport="print",
        cwd=str(tmp_path),
    )
    meta = store._read_job_meta(job.path)
    store.update_job_meta(
        job.job_id,
        {
            **meta,
            "status": "running",
            "backend": "acp",
            "model": "opencode-go/deepseek-v4-flash",
            "task": "dev",
        },
    )

    stopped = job_stop(job.job_id, reason="test_stop")
    assert stopped["ok"] is True

    result = store.get_result(job.job_id)
    assert result["status"] == "cancelled"
    assert result["stop_reason"] == "test_stop"
    assert result["technical"]["acp_stop"]["reason"] == "no_acp_pid_in_meta"
