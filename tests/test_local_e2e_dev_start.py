"""Maintainer-only live MCP smoke for the current eight-tool contract."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENT_CROSSBAR_RUN_LOCAL_E2E") != "1",
    reason="set AGENT_CROSSBAR_RUN_LOCAL_E2E=1 for maintainer provider e2e",
)


def _model() -> str:
    model = os.environ.get("AGENT_CROSSBAR_E2E_MODEL")
    if not model:
        pytest.skip("set AGENT_CROSSBAR_E2E_MODEL to an explicitly discovered model")
    return model


async def _run_live_gate(tmp_path: Path) -> dict:
    uv = shutil.which("uv")
    provider = shutil.which("opencode")
    if not uv or not provider:
        pytest.skip("uv and opencode are required for the opt-in live gate")
    state_root = tmp_path / "state"
    env = os.environ.copy()
    env["AGENT_CROSSBAR_STATE_DIR"] = str(state_root)
    env["AGENT_CROSSBAR_CLIENT_NAME"] = "local-e2e"
    package_dir = Path(__file__).parents[1]
    params = StdioServerParameters(
        command=uv,
        args=["run", "--directory", str(package_dir), "agents-mcp"],
        env=env,
        cwd=str(package_dir),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            health = await session.call_tool("profile_health", {})
            health_data = dict(health.structuredContent or {})
            if not health_data.get("ok"):
                pytest.skip(f"OpenCode readiness unavailable: {health_data}")
            opencode_health = next(
                (
                    entry
                    for entry in health_data.get("profiles", [])
                    if entry.get("profile", entry.get("name")) == "opencode"
                ),
                {},
            )
            if opencode_health.get("state") not in {"ready", None}:
                pytest.skip(f"OpenCode readiness unavailable: {opencode_health}")
            started = await session.call_tool(
                "agent_start",
                {
                    "profile": "opencode",
                    "task": "dev",
                    "interactive": False,
                    "model": _model(),
                    "prompt": "Create no files. Reply exactly AGENT_CROSSBAR_LIVE_DEV_OK.",
                    "cwd": str(tmp_path),
                    "max_runtime_sec": 120,
                },
                read_timeout_seconds=timedelta(seconds=30),
            )
            data = dict(started.structuredContent or {})
            if not data.get("ok"):
                pytest.skip(f"OpenCode live start unavailable: {data}")
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                terminal = await session.call_tool("job_result", {"job_id": data["job_id"]})
                terminal_data = dict(terminal.structuredContent or {})
                if terminal_data.get("error") == "result_not_ready":
                    await asyncio.sleep(2)
                    continue
                return terminal_data
            pytest.fail("OpenCode live dev job did not reach a terminal result")


def test_local_mcp_agent_start_dev_reaches_terminal_result(tmp_path: Path):
    result = asyncio.run(_run_live_gate(tmp_path))
    assert result.get("ok") is True, result
    assert "AGENT_CROSSBAR_LIVE_DEV_OK" in str(result.get("summary") or result.get("output")), (
        result
    )
