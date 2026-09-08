"""Tests for acp_runtime — build_acp_agent_command and run_acp_job.

TDD RED step — run_acp_job does not exist yet, imports will fail.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch

import pytest
from acp.schema import (
    AgentMessageChunk,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    TextContentBlock,
)

# ── production target ────────────────────────────────────────────────
from agent_crossbar.acp_client import (
    AcpError,
    AcpLaunchError,
    AcpProtocolError,
    AcpProviderUnavailableError,
    AcpResult,
    AcpTimeoutError,
)
from agent_crossbar.acp_runtime import build_acp_agent_command, run_acp_job
from agent_crossbar.envelope import FAILURE_STAGES
from agent_crossbar.jobs import JobStore
from agent_crossbar.models import Autonomy
from agent_crossbar.run_handles import run_handles

# ── helpers ──────────────────────────────────────────────────────────

SECRET = "ssh-ed25519 AAA... bogus key"


def test_run_acp_job_requires_explicit_model():
    model = inspect.signature(run_acp_job).parameters["model"]
    assert model.default is inspect.Parameter.empty


def _create_job_store(tmp_path) -> tuple[JobStore, str]:
    """Return (store, job_id) with a fresh job already stored."""
    store = JobStore(tmp_path)
    job = store.create_job(
        profile="opencode",
        operation="dev",
        transport="print",
        sensitivity="normal",
        cwd=str(tmp_path),
    )
    return store, job.job_id


def _read_store_events(store: JobStore, job_id: str) -> list[dict]:
    events_file = store.get_job(job_id).path / "events.jsonl"
    if not events_file.exists():
        return []
    return [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]


def _read_store_meta(store: JobStore, job_id: str) -> dict:
    meta_file = store.get_job(job_id).path / "meta.json"
    return json.loads(meta_file.read_text())


class _ChunkingAcpConnection:
    """Protocol-shaped fake that emits real AgentMessageChunk notifications."""

    def __init__(self, texts: list[str], model: str) -> None:
        self.texts = texts
        self.model = model
        self.mode = "orchestrator"
        self.client = None

    async def initialize(self, protocol_version, **_kwargs):
        return SimpleNamespace(protocol_version=protocol_version)

    async def new_session(self, **_kwargs):
        return SimpleNamespace(
            session_id="protocol-fake-1",
            config_options=[self._model_config(self.model), self._mode_config(self.mode)],
        )

    async def set_config_option(self, value, **_kwargs):
        if value in {"orchestrator", "build", "plan"}:
            self.mode = value
        else:
            self.model = value
        return SimpleNamespace(
            config_options=[self._model_config(self.model), self._mode_config(self.mode)]
        )

    async def prompt(self, session_id, **_kwargs):
        assert self.client is not None
        for text in self.texts:
            await self.client.session_update(
                session_id,
                AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=TextContentBlock(type="text", text=text),
                ),
            )
        return SimpleNamespace(stop_reason="end_turn")

    @staticmethod
    def _model_config(model: str) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            type="select",
            id="model",
            name="Model",
            description=None,
            category="model",
            current_value=model,
            options=[SessionConfigSelectOption(value=model, name=model, description=None)],
        )

    @staticmethod
    def _mode_config(mode: str) -> SessionConfigOptionSelect:
        return SessionConfigOptionSelect(
            type="select",
            id="mode",
            name="Mode",
            description=None,
            category="mode",
            current_value=mode,
            options=[
                SessionConfigSelectOption(value=value, name=value, description=None)
                for value in ("orchestrator", "build", "plan")
            ],
        )


def _chunking_acp_spawn(connection: _ChunkingAcpConnection):
    @asynccontextmanager
    async def _spawn(client, *_args, **_kwargs):
        connection.client = client
        yield connection, SimpleNamespace(pid=17)

    return _spawn


# ── command builder ──────────────────────────────────────────────────


class TestBuildAcpAgentCommand:
    def test_opencode(self):
        assert build_acp_agent_command("opencode") == ["opencode", "acp"]

    def test_codex(self):
        assert build_acp_agent_command("codex") == [
            "pnpm",
            "dlx",
            "@agentclientprotocol/codex-acp@1.1.7",
        ]

    def test_unknown_provider_raises_valueerror(self):
        with pytest.raises(ValueError):
            build_acp_agent_command("claude-code")


# ── run_acp_job ──────────────────────────────────────────────────────


class TestRunAcpJobSuccess:
    def test_happy_path(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        acp_result = AcpResult(
            output="done",
            stop_reason="end_turn",
            session_id="native-1",
        )

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ) as mock_run:
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        # ── assert run_acp_prompt called correctly (model forwarded) ──
        # provider="opencode" + task="dev" resolves the adapter-owned "build"
        # session mode (see adapters/opencode.py OpencodeAdapter.dev_acp_mode).
        mock_run.assert_awaited_once_with(
            [sys.executable, "-m", "agent_crossbar.acp_process_wrapper", "opencode", "acp"],
            SECRET,
            str(tmp_path),
            timeout=12,
            autonomy=Autonomy.EDIT_LOCAL,
            model="glm",
            effort=None,
            mode="build",
            startup_lock=ANY,
            before_process_start=ANY,
            on_process_start=ANY,
            before_prompt=ANY,
            on_text_delta=ANY,
            on_execution_heartbeat=ANY,
        )

        # ── assert store result ──
        stored = store.get_result(job_id)
        assert stored["ok"] is True
        assert stored["status"] == "completed"
        assert stored["summary"] == "done"
        assert stored["output"] == "done"
        assert stored["stop_reason"] == "end_turn"
        assert stored["resolved"]["backend"] == "acp"
        assert stored["technical"]["native_session_id"] == "native-1"

        # ── secret absent from events and meta ──
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text

    def test_execution_heartbeat_is_persisted_without_native_working_claim(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        async def fake_run(*_args, **kwargs):
            kwargs["on_execution_heartbeat"](
                {
                    "process_alive": True,
                    "prompt_active": True,
                    "elapsed_sec": 31,
                    "last_protocol_activity_sec": 31,
                }
            )
            return AcpResult(output="done", stop_reason="end_turn", session_id="native-1")

        with patch("agent_crossbar.acp_runtime.run_acp_prompt", new=fake_run):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="safe",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    max_runtime_sec=30,
                )
            )

        heartbeat = [
            event
            for event in _read_store_events(store, job_id)
            if event["type"] == "execution_heartbeat"
        ]
        assert len(heartbeat) == 1
        assert heartbeat[0]["data"]["process_alive"] is True
        assert "native_state" not in heartbeat[0]["data"]


class TestRunAcpJobDevEmptyResultFailsClosed:
    """A "dev" task returning whitespace/empty output must fail closed with
    a stable ``acp_empty_result`` diagnostic — never report `completed`."""

    @pytest.mark.parametrize("blank_output", ["\n\n", "", "   ", "\t \n"])
    def test_whitespace_only_output_fails_closed(self, tmp_path, blank_output):
        store, job_id = _create_job_store(tmp_path)
        acp_result = AcpResult(output=blank_output, stop_reason="end_turn", session_id="native-1")

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["ok"] is False
        assert stored["failure"]["code"] == "acp_empty_result"
        assert stored["failure"]["stage"] in FAILURE_STAGES
        assert stored["failure"]["stage"] == "execution"
        assert stored["failure"]["retryable"] is True
        assert "OpenCode" not in stored["output"]

    def test_no_output_sentinel_fails_closed(self, tmp_path):
        """Zero session_update chunks at all (the "(no output)" sentinel) is
        just as much a no-op as whitespace text."""
        from agent_crossbar.acp_client import NO_OUTPUT_SENTINEL

        store, job_id = _create_job_store(tmp_path)
        acp_result = AcpResult(
            output=NO_OUTPUT_SENTINEL, stop_reason="end_turn", session_id="native-1"
        )

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["code"] == "acp_empty_result"

    def test_concise_nonempty_dev_output_still_succeeds(self, tmp_path):
        """A legitimately concise dev reply must NOT be treated as empty —
        `changes` being externally empty (Agents MCP doesn't inventory the
        workspace) is expected and is not itself a failure signal."""
        store, job_id = _create_job_store(tmp_path)
        acp_result = AcpResult(output="Done.", stop_reason="end_turn", session_id="native-1")

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "completed"
        assert stored["ok"] is True
        assert stored["output"] == "Done."
        assert stored["envelope"]["changes"] == []

    def test_stop_before_worker_start_never_invokes_provider(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        stopped = store.stop_job(job_id, reason="user_cancelled")
        assert stopped["ok"] is True

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=AssertionError("provider must not start")),
        ) as prompt:
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )
        prompt.assert_not_awaited()

    def test_stop_after_final_fence_blocks_provider_start(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        async def stop_before_spawn(*_args, before_process_start, **_kwargs):
            run_handles.cancel(job_id)
            before_process_start()

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=stop_before_spawn,
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["code"] == "acp_protocol_error"
        assert stored["technical"]["provider_cleanup"]["reason"] == (
            "startup_cancelled_before_spawn"
        )

    def test_cancel_before_process_identity_fence_blocks_prompt(self, tmp_path):
        """The in-memory stop fence wins before durable status is published."""
        store, job_id = _create_job_store(tmp_path)
        prompt_started = False

        async def stop_during_spawn(*_args, on_process_start, **_kwargs):
            nonlocal prompt_started
            run_handles.cancel(job_id)
            # stop_job requests the handle before it writes status=stopped. The
            # production callback must persist identity, then reject this
            # before any provider prompt.
            on_process_start(7001)
            prompt_started = True

        with (
            patch(
                "agent_crossbar.acp_runtime.run_acp_prompt",
                new=stop_during_spawn,
            ),
            patch(
                "agent_crossbar.acp_runtime._capture_process_identity",
                return_value={"pgid": 7001, "start": "launch-1", "command": "fake"},
            ),
            patch(
                "agent_crossbar.acp_runtime.safe_acp_termination",
                return_value={"terminated": False, "reason": "death_unconfirmed", "pid": 7001},
            ) as terminate,
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert prompt_started is False
        assert stored["status"] == "failed"
        assert stored["failure"]["code"] == "acp_protocol_error"
        assert _read_store_meta(store, job_id)["acp_pid"] == 7001
        assert stored["technical"]["provider_cleanup"] == {
            "terminated": False,
            "reason": "death_unconfirmed",
            "pid": 7001,
        }
        assert terminate.call_count == 2
        assert terminate.call_args_list[-1].args[0]["acp_pid"] == 7001

    @pytest.mark.parametrize("cancel", [False, True])
    def test_pid_record_failure_keeps_post_spawn_cleanup_receipt(self, tmp_path, cancel):
        """A metadata failure after spawn never becomes a pre-spawn success."""
        store, job_id = _create_job_store(tmp_path)
        original_update = store.update_job_meta
        prompt_started = False

        def fail_pid_record(update_job_id, updates):
            if "acp_pid" in updates:
                raise OSError("metadata unavailable")
            return original_update(update_job_id, updates)

        async def fail_after_spawn(*_args, on_process_start, **_kwargs):
            nonlocal prompt_started
            if cancel:
                run_handles.cancel(job_id)
            try:
                on_process_start(7001)
            except OSError as exc:
                raise AcpProtocolError("could not persist process identity") from exc
            prompt_started = True

        with (
            patch("agent_crossbar.acp_runtime.run_acp_prompt", new=fail_after_spawn),
            patch(
                "agent_crossbar.acp_runtime._capture_process_identity",
                return_value={"pgid": 7001, "start": "launch-1", "command": "fake"},
            ),
            patch(
                "agent_crossbar.acp_runtime.safe_acp_termination",
                return_value={"terminated": False, "reason": "death_unconfirmed", "pid": 7001},
            ) as terminate,
            patch.object(store, "update_job_meta", side_effect=fail_pid_record),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert prompt_started is False
        assert stored["status"] == "failed"
        assert stored["failure"]["code"] == "acp_protocol_error"
        assert stored["technical"]["provider_cleanup"] == {
            "terminated": False,
            "reason": "death_unconfirmed",
            "pid": 7001,
        }
        assert stored["technical"]["cleanup_confirmed"] is False
        assert stored["technical"]["native_session_id"] == 7001
        assert terminate.call_args_list[-1].args[0]["acp_pid"] == 7001

    def test_failure_after_launch_marker_refresh_retains_lease_before_pid(self, tmp_path):
        """Early provider errors refresh durable launch intent before release."""
        from agent_crossbar.writer_lease import WriterLeaseStore

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        state = tmp_path / "state"
        leases = WriterLeaseStore(state)
        pending = leases.acquire(workspace, owner_id="pending", owner_kind="pending_dev")
        assert pending.ok and pending.token
        store = JobStore(state)
        job = store.create_job("opencode", "dev", transport="print", cwd=str(workspace))
        assert leases.attach(pending.token, job_id=job.job_id)
        store.update_job_meta(job.job_id, {"writer_lease_token": pending.token})

        async def provider_error(*_args, **_kwargs):
            raise AcpProviderUnavailableError(
                "provider_unavailable", "provider failed before PID callback"
            )

        with patch("agent_crossbar.acp_runtime.run_acp_prompt", new=provider_error):
            asyncio.run(
                run_acp_job(
                    store,
                    job.job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(workspace),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        result = store.get_result(job.job_id)
        assert result["status"] == "failed"
        assert _read_store_meta(store, job.job_id)["acp_launch_pending"] is True
        assert leases.acquire(workspace, owner_id="replacement").error == "writer_busy"

    def test_cancel_before_prompt_fence_blocks_provider_prompt(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        prompt_started = False

        async def stop_before_prompt(*_args, before_prompt, **_kwargs):
            nonlocal prompt_started
            run_handles.cancel(job_id)
            before_prompt()
            prompt_started = True

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=stop_before_prompt,
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert prompt_started is False
        assert stored["status"] == "failed"
        assert stored["failure"]["code"] == "acp_protocol_error"
        assert stored["technical"]["provider_cleanup"]["reason"] == (
            "startup_cancelled_before_spawn"
        )

    @pytest.mark.parametrize("task", ["ask", "review"])
    def test_non_dev_whitespace_output_fails_closed(self, tmp_path, task):
        """Ask/review must not claim success without provider output either."""
        store = JobStore(tmp_path)
        job = store.create_job(
            profile="opencode",
            operation=task,
            transport="print",
            sensitivity="normal",
            cwd=str(tmp_path),
        )
        job_id = job.job_id
        acp_result = AcpResult(output="\n\n", stop_reason="end_turn", session_id="native-1")

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task=task,
                    model="glm",
                    autonomy=Autonomy.READ_ONLY,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["ok"] is False
        assert stored["failure"]["code"] == "acp_empty_result"

    @pytest.mark.parametrize("task", ["ask", "review"])
    def test_non_dev_no_output_sentinel_fails_closed(self, tmp_path, task):
        from agent_crossbar.acp_client import NO_OUTPUT_SENTINEL

        store, job_id = _create_job_store(tmp_path)
        acp_result = AcpResult(
            output=NO_OUTPUT_SENTINEL, stop_reason="end_turn", session_id="native-1"
        )
        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task=task,
                    model="glm",
                    autonomy=Autonomy.READ_ONLY,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["ok"] is False
        assert stored["failure"]["code"] == "acp_empty_result"


class TestRunAcpJobTimeout:
    def test_timeout_sets_failure_status(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=AcpTimeoutError("ACP prompt timed out after 1.0s")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["stop_reason"] == "timeout"
        assert stored["failure"]["code"] == "acp_timeout"
        assert stored["failure"]["retryable"] is True
        assert stored["failure"]["next_action"] == "check_provider_limits_or_retry_with_free_model"
        assert stored["failure"]["diagnostics"]["max_runtime_sec"] == 12
        assert "quota" in stored["output"].lower()
        assert "free model" in stored["output"].lower()

        # ── secret absent everywhere ──
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text
        if "error" in stored:
            assert SECRET not in json.dumps(stored["error"])
        if "failure" in stored:
            assert SECRET not in json.dumps(stored["failure"])


class TestRunAcpJobPromptDeliveryTimeout:
    def test_timeout_before_prompt_delivery_is_not_generic_timeout(self, tmp_path):
        """A prompt-delivery-stage timeout must not collapse into the
        generic acp_timeout/execution classification — it is diagnosable
        as a distinct stage with its own code and next action.
        """
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(
                side_effect=AcpTimeoutError(
                    "ACP prompt timed out after 12.0s", stage="prompt_delivery"
                )
            ),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["stage"] == "prompt_delivery"
        assert stored["failure"]["code"] != "acp_timeout"
        assert stored["failure"]["retryable"] is True
        assert stored["failure"]["next_action"]

        # ── secret absent everywhere ──
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text


class TestRunAcpJobLaunchError:
    def test_missing_provider_binary(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=AcpLaunchError("Provider binary not found: opencode")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["stage"] == "launch"
        assert stored["failure"]["code"] == "acp_launch_error"
        assert (
            "install" in stored["failure"]["next_action"].lower()
            or "check" in stored["failure"]["next_action"].lower()
        )

        # prompt absent
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text


class TestRunAcpJobProtocolError:
    def test_handshake_failure(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(
                side_effect=AcpProtocolError("handshake failed", stage="prompt_delivery")
            ),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["code"] == "acp_protocol_error"
        assert "handshake" in stored["failure"]["diagnostics"]["error"].lower()
        # The exception's own stage must be forwarded, not collapsed into a
        # hardcoded (and out-of-taxonomy) "protocol" bucket.
        assert stored["failure"]["stage"] == "prompt_delivery"
        assert stored["failure"]["stage"] in FAILURE_STAGES

        # prompt absent
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text

    def test_protocol_failure_after_prompt_dispatch_keeps_execution_stage(self, tmp_path):
        """A protocol error raised after the prompt was already sent to the
        agent is a later-provider failure — it must classify as
        execution, not collapse to the same bucket as a pre-dispatch
        handshake failure.
        """
        store, job_id = _create_job_store(tmp_path)

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=AcpProtocolError("stream corrupted", stage="execution")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["failure"]["stage"] == "execution"
        assert stored["failure"]["stage"] in FAILURE_STAGES


class TestRunAcpJobInvalidAutonomyStage:
    def test_invalid_autonomy_is_preflight_not_protocol(self, tmp_path):
        """Invalid autonomy is rejected before any provider interaction —
        it belongs in the 'preflight' bucket, not the removed 'protocol'
        bucket (which was never one of the six allowed stages)."""
        store, job_id = _create_job_store(tmp_path)

        asyncio.run(
            run_acp_job(
                store,
                job_id,
                provider="opencode",
                prompt=SECRET,
                cwd=str(tmp_path),
                task="dev",
                model="glm",
                effort=None,
                autonomy="not-a-real-autonomy-value",
                max_runtime_sec=12,
            )
        )

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["failure"]["stage"] == "preflight"
        assert stored["failure"]["stage"] in FAILURE_STAGES


class TestSafeErrorRedaction:
    """_safe_error must redact secrets beyond just the raw prompt text —
    a provider exception can embed live credentials from its own
    environment or stderr, not only the prompt we sent it."""

    def test_bearer_token_in_exception_message_is_redacted(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        leaked = "sk-live-should-not-leak-1234567890"

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(
                side_effect=Exception(f"upstream call failed: Authorization: Bearer {leaked}")
            ),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert leaked not in stored["summary"]
        assert leaked not in stored["output"]
        assert leaked not in json.dumps(stored["failure"])

    def test_key_value_secret_in_exception_message_is_redacted(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        leaked = "abcDEF1234567890"

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=Exception(f"config error: OPENAI_API_KEY={leaked}")),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt="hello",
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert leaked not in stored["summary"]
        assert leaked not in stored["output"]
        assert leaked not in json.dumps(stored["failure"])


class TestRunAcpJobEvents:
    def test_protocol_shaped_acp_chunks_reach_job_tail(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        connection = _ChunkingAcpConnection(["streamed ", "text"], model="glm")

        with patch(
            "agent_crossbar.acp_client.spawn_agent_process",
            _chunking_acp_spawn(connection),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        events = _read_store_events(store, job_id)
        assert [event["data"]["text"] for event in events if event["type"] == "log_delta"] == [
            "streamed ",
            "text",
        ]
        assert store.get_result(job_id)["output"] == "streamed text"

    def test_success_events_stream_each_acp_text_delta(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        async def _streaming_acp_prompt(*_args, **kwargs):
            on_text_delta = kwargs["on_text_delta"]
            on_text_delta("first ")
            on_text_delta("second")
            return AcpResult(
                output="first second",
                stop_reason="end_turn",
                session_id="native-1",
            )

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(side_effect=_streaming_acp_prompt),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        events = _read_store_events(store, job_id)
        acp_events = [
            event
            for event in events
            if event["type"] in {"acp_command", "log_delta", "acp_completed"}
        ]
        assert [event["type"] for event in acp_events] == [
            "acp_command",
            "log_delta",
            "log_delta",
            "acp_completed",
        ]
        assert [event["data"]["text"] for event in acp_events if event["type"] == "log_delta"] == [
            "first ",
            "second",
        ]

    def test_success_events_contain_acp_types(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)

        acp_result = AcpResult(
            output="done",
            stop_reason="end_turn",
            session_id="native-1",
        )

        with patch(
            "agent_crossbar.acp_runtime.run_acp_prompt",
            new=AsyncMock(return_value=acp_result),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        events = _read_store_events(store, job_id)
        event_types = {e["type"] for e in events}
        assert "acp_command" in event_types
        assert "acp_completed" in event_types

        # no "acpx" strings anywhere in events
        for event in events:
            json_str = json.dumps(event)
            assert "acpx" not in json_str


# ── Exhaustive stage-taxonomy regression (review fix #1) ──────────────
#
# Every failure branch of run_acp_job must emit one of the six allowed
# envelope failure stages. This is more than a per-branch spot check: it
# drives EVERY branch through the real run_acp_job and asserts each
# resulting stage is in FAILURE_STAGES — and explicitly rejects the old
# out-of-taxonomy "protocol" bucket. build_result_envelope also enforces
# this at runtime (raises ValueError on an unknown stage), so this test
# additionally proves no branch tries to publish a value outside the six.


def _run_job_and_get_failure_stage(tmp_path, *, autonomy, side_effect) -> str:
    store, job_id = _create_job_store(tmp_path)
    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(side_effect=side_effect),
    ):
        asyncio.run(
            run_acp_job(
                store,
                job_id,
                provider="opencode",
                prompt=SECRET,
                cwd=str(tmp_path),
                task="dev",
                model="glm",
                effort=None,
                autonomy=autonomy,
                max_runtime_sec=12,
            )
        )
    stored = store.get_result(job_id)
    assert stored["status"] == "failed"
    return stored["failure"]["stage"]


class TestExhaustiveFailureStageTaxonomy:
    @pytest.mark.parametrize(
        ("label", "autonomy", "side_effect"),
        [
            ("invalid_autonomy", "not-a-real-value", None),
            ("timeout_execution_default", Autonomy.EDIT_LOCAL, AcpTimeoutError("timed out")),
            (
                "timeout_prompt_delivery",
                Autonomy.EDIT_LOCAL,
                AcpTimeoutError("timed out", stage="prompt_delivery"),
            ),
            (
                "launch_error",
                Autonomy.EDIT_LOCAL,
                AcpLaunchError("binary not found"),
            ),
            (
                "protocol_error_prompt_delivery",
                Autonomy.EDIT_LOCAL,
                AcpProtocolError("handshake failed", stage="prompt_delivery"),
            ),
            (
                "protocol_error_execution",
                Autonomy.EDIT_LOCAL,
                AcpProtocolError("stream corrupted", stage="execution"),
            ),
            ("generic_acp_error", Autonomy.EDIT_LOCAL, AcpError("unexpected acp failure")),
            ("generic_exception", Autonomy.EDIT_LOCAL, RuntimeError("boom")),
        ],
    )
    def test_every_failure_branch_emits_an_allowed_stage(
        self, tmp_path, label, autonomy, side_effect
    ):
        stage = _run_job_and_get_failure_stage(tmp_path, autonomy=autonomy, side_effect=side_effect)
        assert stage in FAILURE_STAGES, (
            f"{label}: stage {stage!r} is outside the six allowed failure stages "
            f"{sorted(FAILURE_STAGES)}"
        )
        # The specific bug reported in review: the old code emitted this
        # literal value, which was never one of the six allowed stages.
        assert stage != "protocol", f"{label}: regressed to the removed 'protocol' stage"


# ── safe_acp_termination tests ──────────────────────────────────────────


def test_provider_limit_failure_persists_terminal_actionable_envelope(tmp_path):
    store, job_id = _create_job_store(tmp_path)
    with patch(
        "agent_crossbar.acp_runtime.run_acp_prompt",
        new=AsyncMock(
            side_effect=AcpProviderUnavailableError(
                "provider_limit_exhausted",
                "The selected provider has exhausted its quota or rate limit",
            )
        ),
    ):
        asyncio.run(
            run_acp_job(
                store,
                job_id,
                provider="opencode",
                prompt="safe",
                cwd=str(tmp_path),
                model="opencode-go/deepseek-v4-flash",
                max_runtime_sec=60,
            )
        )

    result = store.get_result(job_id)
    assert result is not None
    assert result["ok"] is False
    assert result["envelope"]["status"] == "failed"
    assert result["envelope"]["failure"]["code"] == "provider_limit_exhausted"
    assert result["envelope"]["failure"]["retryable"] is True


class TestSafeAcpTermination:
    def test_no_pid_in_meta(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        result = safe_acp_termination({})
        assert result["terminated"] is False
        assert result["reason"] == "no_acp_pid_in_meta"
        assert result["pid"] is None

    def test_invalid_pid_type(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        result = safe_acp_termination({"acp_pid": "not-a-number"})
        assert result["terminated"] is False
        assert "invalid_acp_pid" in result["reason"]
        assert result["pid"] is None

    def test_nonexistent_pid(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        result = safe_acp_termination({"acp_pid": 99999999})
        assert result["terminated"] is True
        assert result["reason"] == "process_already_gone"

    def test_idempotent_double_call(self):
        from agent_crossbar.acp_runtime import safe_acp_termination

        meta = {"acp_pid": 99999999}
        r1 = safe_acp_termination(meta)
        r2 = safe_acp_termination(meta)
        assert r1 == r2
        assert r1["terminated"] is True

    def test_pid_reuse_is_not_signalled(self, monkeypatch):
        import agent_crossbar.acp_runtime as runtime

        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(runtime.os, "kill", lambda pid, sig: calls.append((pid, sig)))
        monkeypatch.setattr(
            runtime,
            "_capture_process_identity",
            lambda pid: {"start": "new-start", "pgid": pid},
        )
        result = runtime.safe_acp_termination(
            {"acp_pid": 123, "acp_process_start": "old-start", "acp_pgid": 123}
        )
        assert result["terminated"] is False
        assert result["reason"] == "pid_reused"
        assert calls == [(123, 0)]

    def test_live_pid_without_launch_identity_is_not_signalled(self, monkeypatch):
        import agent_crossbar.acp_runtime as runtime

        calls: list[tuple[int, int]] = []
        pid = 4321
        monkeypatch.setattr(runtime.os, "kill", lambda target, sig: calls.append((target, sig)))
        monkeypatch.setattr(runtime.os, "getpid", lambda: pid)

        result = runtime.safe_acp_termination({"acp_pid": pid})

        assert result == {
            "terminated": False,
            "reason": "process_identity_unverifiable",
            "pid": pid,
        }
        assert calls == [(pid, 0)]

    def test_owned_identity_cannot_adopt_foreign_process_group(self, monkeypatch):
        import agent_crossbar.acp_runtime as runtime

        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(runtime.os, "kill", lambda pid, sig: calls.append((pid, sig)))
        monkeypatch.setattr(
            runtime,
            "_capture_process_identity",
            lambda pid: {"start": "same-start", "pgid": pid + 1},
        )
        result = runtime.safe_acp_termination(
            {"acp_pid": 123, "acp_process_start": "same-start", "acp_pgid": 123}
        )
        assert result["terminated"] is False
        assert result["reason"] == "process_group_reused"
        assert calls == [(123, 0)]

    def test_shared_group_uses_leader_signal_without_killpg(self, monkeypatch):
        import signal

        import agent_crossbar.acp_runtime as runtime

        calls: list[tuple[str, int, int]] = []
        monkeypatch.setattr(
            runtime.os,
            "kill",
            lambda pid, sig: calls.append(("kill", pid, sig)),
        )
        monkeypatch.setattr(
            runtime.os,
            "killpg",
            lambda pgid, sig: calls.append(("killpg", pgid, sig)),
        )
        monkeypatch.setattr(
            runtime, "_capture_process_identity", lambda _pid: {"start": "same", "pgid": 456}
        )
        monkeypatch.setattr(runtime, "_process_exit_confirmed", lambda *_args: True)
        monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)

        result = runtime.safe_acp_termination(
            {"acp_pid": 123, "acp_process_start": "same", "acp_pgid": 456}
        )

        assert result["terminated"] is True
        assert not any(kind == "killpg" for kind, _pid, _sig in calls)
        assert ("kill", 123, signal.SIGTERM) in calls

    def test_owned_process_group_kills_provider_and_grandchild(self):
        import agent_crossbar.acp_runtime as runtime

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agent_crossbar.acp_process_wrapper",
                sys.executable,
                "-c",
                "import subprocess,time; subprocess.Popen(['sleep','30']); time.sleep(30)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(0.1)
            identity = runtime._capture_process_identity(proc.pid)
            result = runtime.safe_acp_termination(
                {
                    "acp_pid": proc.pid,
                    "acp_pgid": identity.get("pgid"),
                    "acp_process_start": identity.get("start"),
                }
            )
            assert result["terminated"] is True
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_zombie_leader_with_live_group_member_is_not_confirmed(self, monkeypatch):
        """A zombie leader must not hide a live descendant from cleanup."""
        import signal

        import agent_crossbar.acp_runtime as runtime

        group_signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            runtime,
            "_capture_process_identity",
            lambda _pid: {"start": "start", "pgid": 123},
        )
        monkeypatch.setattr(runtime, "_process_is_zombie", lambda _pid: True)
        monkeypatch.setattr(runtime, "_process_group_has_live_member", lambda _pgid: True)
        monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
        clock = iter((0.0, 3.0))
        monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(
            runtime.os,
            "kill",
            lambda _target, _sig: None,
        )
        monkeypatch.setattr(
            runtime.os,
            "killpg",
            lambda pgid, sig: group_signals.append((pgid, sig)),
        )

        result = runtime.safe_acp_termination(
            {"acp_pid": 123, "acp_process_start": "start", "acp_pgid": 123}
        )

        assert result["terminated"] is False
        assert result["reason"] == "death_unconfirmed"
        assert (123, signal.SIGTERM) in group_signals
        assert (123, signal.SIGKILL) in group_signals

    def test_process_group_probe_fails_closed_and_ignores_zombies(self, monkeypatch):
        import agent_crossbar.acp_runtime as runtime

        class _Result:
            returncode = 0
            stdout = "123 S\n123 Z+\n"

        monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: _Result())
        assert runtime._process_group_has_live_member(123) is True

        class _ZombieOnlyResult:
            returncode = 0
            stdout = "123 Z\n123 Z+\n"

        monkeypatch.setattr(
            runtime.subprocess,
            "run",
            lambda *args, **kwargs: _ZombieOnlyResult(),
        )
        assert runtime._process_group_has_live_member(123) is False

        class _EmptyResult:
            returncode = 0
            stdout = ""

        monkeypatch.setattr(
            runtime.subprocess,
            "run",
            lambda *args, **kwargs: _EmptyResult(),
        )
        assert runtime._process_group_has_live_member(123) is True

        class _FailedResult:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(
            runtime.subprocess,
            "run",
            lambda *args, **kwargs: _FailedResult(),
        )
        assert runtime._process_group_has_live_member(123) is True

    def test_reaped_leader_still_checks_owned_group(self, monkeypatch):
        """A missing leader does not prove zombie descendants are gone."""
        import agent_crossbar.acp_runtime as runtime

        probes: list[int] = []

        def missing_leader(_pid, _sig):
            raise ProcessLookupError

        monkeypatch.setattr(runtime.os, "kill", missing_leader)
        monkeypatch.setattr(runtime.os, "killpg", lambda _pgid, _sig: None)
        monkeypatch.setattr(runtime.os, "getpgrp", lambda: 999)
        monkeypatch.setattr(
            runtime,
            "_process_group_has_live_member",
            lambda pgid: probes.append(pgid) or False,
        )

        assert runtime._process_exit_confirmed(123, 123, timeout=0) is True
        assert probes == [123]

    def test_importable_from_acp_runtime(self):
        """safe_acp_termination must be importable — no ImportError."""
        from agent_crossbar.acp_runtime import safe_acp_termination  # noqa: F811

        assert callable(safe_acp_termination)


class TestRunAcpJobCallerInterruption:
    """A cancelled ACP delegation must persist a durable terminal result before
    the cancellation propagates, so caller interruption never orphans a job."""

    def test_cancellation_persists_terminal_result_then_propagates(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        store.update_job_meta(job_id, {"acp_pid": 42})

        async def interrupted(*_args, **_kwargs):
            raise asyncio.CancelledError()

        with (
            patch("agent_crossbar.acp_runtime.run_acp_prompt", new=interrupted),
            patch(
                "agent_crossbar.acp_runtime.safe_acp_termination",
                return_value={"terminated": True, "reason": "process_already_gone", "pid": 42},
            ) as terminate,
        ):
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(
                    run_acp_job(
                        store,
                        job_id,
                        provider="opencode",
                        prompt=SECRET,
                        cwd=str(tmp_path),
                        task="dev",
                        model="glm",
                        effort=None,
                        autonomy=Autonomy.EDIT_LOCAL,
                        max_runtime_sec=12,
                    )
                )

        terminate.assert_called_once()
        assert terminate.call_args.args[0]["acp_pid"] == 42

        stored = store.get_result(job_id)
        assert stored["status"] == "failed"
        assert stored["ok"] is False
        assert stored["stop_reason"] == "cancelled"
        assert stored["failure"]["code"] == "acp_interrupted"
        assert stored["failure"]["stage"] in FAILURE_STAGES

        # No partial/prompt content is persisted.
        events_text = (store.get_job(job_id).path / "events.jsonl").read_text()
        meta_text = (store.get_job(job_id).path / "meta.json").read_text()
        assert SECRET not in events_text
        assert SECRET not in meta_text

    def test_cancellation_of_already_stopped_job_keeps_exactly_one_result(self, tmp_path):
        """If job_stop already terminalized the job, the cancellation handler
        must not overwrite it — exactly one durable terminal result survives."""
        store, job_id = _create_job_store(tmp_path)
        store.update_job_meta(job_id, {"status": "stopped", "stop_reason": "user_cancelled"})
        store.set_stopped_result(
            job_id,
            summary="Job stopped: user_cancelled",
            envelope={
                "status": "cancelled",
                "stop_reason": "user_cancelled",
                "summary": "Job stopped: user_cancelled",
            },
        )

        async def interrupted(*_args, **_kwargs):
            raise asyncio.CancelledError()

        with patch("agent_crossbar.acp_runtime.run_acp_prompt", new=interrupted):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    effort=None,
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["status"] == "cancelled"
        assert stored["stop_reason"] == "user_cancelled"

    def test_cancellation_with_unconfirmed_cleanup_retains_writer_lease(self, tmp_path):
        store, job_id = _create_job_store(tmp_path)
        released: list[str] = []
        store._release_writer_lease = lambda job_id, meta=None: released.append(job_id) or True

        async def interrupted(*_args, **_kwargs):
            raise asyncio.CancelledError()

        cleanup = {
            "terminated": False,
            "reason": "process_identity_unverifiable",
            "pid": 123,
        }
        with (
            patch("agent_crossbar.acp_runtime.run_acp_prompt", new=interrupted),
            patch("agent_crossbar.acp_runtime.safe_acp_termination", return_value=cleanup),
            pytest.raises(asyncio.CancelledError),
        ):
            asyncio.run(
                run_acp_job(
                    store,
                    job_id,
                    provider="opencode",
                    prompt=SECRET,
                    cwd=str(tmp_path),
                    task="dev",
                    model="glm",
                    autonomy=Autonomy.EDIT_LOCAL,
                    max_runtime_sec=12,
                )
            )

        stored = store.get_result(job_id)
        assert stored["technical"]["provider_cleanup"] == cleanup
        assert stored["technical"]["cleanup_confirmed"] is False
        assert released == []
        meta = store._read_job_meta(store.get_job(job_id).path)
        assert meta["cleanup_pending"] is True
        result_events = [e for e in _read_store_events(store, job_id) if e["type"] == "result"]
        assert len(result_events) == 1
