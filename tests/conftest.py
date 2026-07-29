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
