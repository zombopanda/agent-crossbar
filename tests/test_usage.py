from __future__ import annotations

import json
from pathlib import Path

from agent_crossbar.usage import (
    extract_acp_usage,
    extract_claude_usage,
    resolve_usage_for_meta,
    unavailable_usage,
)

FIXTURES = Path(__file__).parent / "fixtures" / "usage_jobs"


def test_claude_job_replay_sums_unique_request_ids_and_nested_subagent() -> None:
    fixture = FIXTURES / "1789457044771-job"
    meta = json.loads((fixture / "meta.json").read_text())

    usage = extract_claude_usage(
        cwd=meta["cwd"],
        full_session_id=meta["native_full_session_id"],
        projects_root=fixture,
    )

    assert usage["available"] is True
    assert usage["status"] == "complete"
    assert usage["source"] == "claude_code_transcript"
    assert usage["input_tokens"] == 76
    assert usage["cache_creation_input_tokens"] == 89452
    assert usage["cache_read_input_tokens"] == 3252655
    assert usage["output_tokens"] == 41949
    assert usage["reasoning_tokens"] is None
    assert usage["total_tokens"] == 3384132

    assert len(usage["subagents"]) == 1
    nested = usage["subagents"][0]
    assert nested["agent_id"] == "a0e481025c4c25ec7"
    assert nested["agent_type"] == "Explore"
    assert nested["input_tokens"] == 2
    assert nested["cache_creation_input_tokens"] == 41285
    assert nested["cache_read_input_tokens"] == 312130
    assert nested["output_tokens"] == 9410
    assert nested["total_tokens"] == 362827


def test_acp_native_usage_preserves_provider_fields() -> None:
    usage = extract_acp_usage(
        {
            "inputTokens": 120,
            "cachedWriteTokens": 30,
            "cachedReadTokens": 450,
            "outputTokens": 80,
            "thoughtTokens": 20,
            "totalTokens": 700,
        }
    )
    assert usage["available"] is True
    assert usage["status"] == "complete"
    assert usage["source"] == usage["provenance"] == "acp_native"
    assert usage["reason"] is None
    assert usage["input_tokens"] == 120
    assert usage["cache_creation_input_tokens"] == usage["cache_write_tokens"] == 30
    assert usage["cache_read_input_tokens"] == 450
    assert usage["output_tokens"] == 80
    assert usage["reasoning_tokens"] == 20
    assert usage["total_tokens"] == 700
    assert usage["subagents"] == []


def test_opencode_historical_timeout_without_native_usage_is_explicitly_unavailable() -> None:
    fixture = FIXTURES / "1789455240484-job"
    meta = json.loads((fixture / "meta.json").read_text())
    historical = json.loads((fixture / "result.json").read_text())

    usage = resolve_usage_for_meta(meta)

    assert historical["envelope"]["usage"] == {"available": False}
    assert usage["available"] is False
    assert usage["status"] == "unavailable"
    assert usage["reason"] == "acp_session_terminated_before_response"
    # The fixture is a replay copy; extraction must never rewrite it.
    assert json.loads((fixture / "result.json").read_text()) == historical


def test_acp_missing_optional_fields_is_explicitly_partial() -> None:
    usage = extract_acp_usage({"input_tokens": 10, "output_tokens": 5})
    assert usage["available"] is True
    assert usage["status"] == "partial"
    assert usage["total_tokens"] is None
    assert "cache_read_input_tokens" in usage["reason"]
    assert "reasoning_tokens" in usage["reason"]


def test_acp_invalid_native_counts_are_not_coerced_to_usage() -> None:
    usage = extract_acp_usage({"inputTokens": "120", "outputTokens": 5})
    assert usage["available"] is True
    assert usage["status"] == "partial"
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] == 5
    assert usage["total_tokens"] is None
    assert "input_tokens" in usage["reason"]


def test_unavailable_usage_is_structured_and_never_looks_like_zero() -> None:
    usage = unavailable_usage("fixture_missing")
    assert usage["available"] is False
    assert usage["status"] == "unavailable"
    assert usage["reason"] == "fixture_missing"
    assert usage["source"] is None
    assert usage["total_tokens"] is None


def test_lazy_tmux_finalization_persists_usage_in_result_and_public_result(
    tmp_path: Path, monkeypatch
) -> None:
    from agent_crossbar.jobs import JobStore

    fixture = FIXTURES / "1789457044771-job"
    fixture_meta = json.loads((fixture / "meta.json").read_text())
    fake_usage = extract_claude_usage(
        cwd=fixture_meta["cwd"],
        full_session_id=fixture_meta["native_full_session_id"],
        projects_root=fixture,
    )
    monkeypatch.setattr("agent_crossbar.jobs.resolve_usage_for_meta", lambda _meta: fake_usage)

    store = JobStore(tmp_path)
    job = store.create_job(
        profile="claude",
        operation="dev",
        transport="tmux",
        cwd=fixture_meta["cwd"],
    )
    store.update_job_meta(
        job.job_id,
        {
            "model": fixture_meta["model"],
            "effort": fixture_meta["effort"],
            "task": fixture_meta["task"],
            "backend": fixture_meta["backend"],
            "native_full_session_id": fixture_meta["native_full_session_id"],
            "tmux_output_path": str(job.path / "tmux-output.log"),
            "tmux_exit_status_path": str(job.path / "tmux-exit-status.txt"),
        },
    )
    (job.path / "tmux-output.log").write_text("done\n", encoding="utf-8")
    (job.path / "tmux-exit-status.txt").write_text("0\n", encoding="utf-8")

    result = store.get_result(job.job_id)
    persisted = json.loads((job.path / "result.json").read_text())

    assert result["usage"]["total_tokens"] == 3384132
    assert persisted["envelope"]["usage"]["total_tokens"] == 3384132
    assert persisted["envelope"]["usage"]["subagents"][0]["model"] == "claude-sonnet-5"
