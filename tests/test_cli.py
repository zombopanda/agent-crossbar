"""CLI help and version regression tests."""

from __future__ import annotations

import sys

import pytest

from agent_crossbar import __version__
from agent_crossbar.cli import main


def test_help_exits_without_starting_the_mcp_server(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-crossbar", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Run the Agent Crossbar MCP server" in output
    assert "doctor" in output


def test_version_prints_installed_package_version(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-crossbar", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"agent-crossbar {__version__}\n"
