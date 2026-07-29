"""Shared test fixtures for agent-crossbar tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_browser_windows(monkeypatch):
    """Prevent _chatgpt_open_fresh_window from opening real browser windows.

    Every test gets this by default.  Tests that explicitly mock the function
    override this fixture because monkeypatch.setattr replaces the previous
    value for the same scope.
    """
    from agent_crossbar import runner as runner_module

    monkeypatch.setattr(runner_module, "_chatgpt_open_fresh_window", lambda *_: None)


@pytest.fixture(autouse=True)
def _no_live_model_discovery_in_unit_tests(request, monkeypatch):
    """Keep ordinary unit tests independent of locally installed provider CLIs.

    Discovery-specific tests own their cache and subprocess behavior directly.
    Every other test receives representative discovered catalogs, so server and
    validation tests exercise the same dynamic-model path as production without
    depending on credentials or a provider binary being present in CI.
    """
    if request.path.name == "test_discovery.py":
        return

    import agent_crossbar.discovery as discovery
    from agent_crossbar.adapters.base import ModelCatalog

    catalogs = {
        "codex": ModelCatalog(
            models=("gpt-5.6-sol", "gpt-5.6-terra"),
            default_model="gpt-5.6-sol",
            native_efforts=("low", "medium", "high", "max"),
            source="test fixture",
        ),
        "claude": ModelCatalog(
            models=("claude-sonnet-5", "claude-opus-5"),
            default_model="claude-sonnet-5",
            native_efforts=("low", "medium", "high"),
            source="test fixture",
        ),
        "opencode": ModelCatalog(
            models=("opencode-go/qwen3.6-plus", "opencode-go/glm-5.2"),
            default_model="opencode-go/qwen3.6-plus",
            native_efforts=("low", "medium", "high"),
            source="test fixture",
        ),
    }
    original_discover = discovery.discover_profile_models

    def discover(state_root, profile, *, refresh=False):
        if profile in catalogs:
            return catalogs[profile]
        return original_discover(state_root, profile, refresh=refresh)

    monkeypatch.setattr(discovery, "discover_profile_models", discover)
