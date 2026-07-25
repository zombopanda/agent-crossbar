"""Telemetry audit logs and zstd retention tests."""

import json

from agent_crossbar.telemetry import TelemetryStore


def test_telemetry_writes_daily_request_response_and_usage(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_request(
        client={"name": "codex"},
        tool="review_start",
        request_id="r1",
        payload={"x": 1},
    )
    telemetry.record_response(request_id="r1", ok=True, duration_ms=12, response={"ok": True})
    today = telemetry.today_dir()
    assert oct(today.stat().st_mode & 0o777) == "0o700"
    assert (today / "requests.jsonl").exists()
    assert (today / "responses.jsonl").exists()
    assert (today / "client_usage.jsonl").exists()
    assert oct((today / "requests.jsonl").stat().st_mode & 0o777) == "0o600"


def test_retention_compresses_logs_older_than_14_days(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    old_dir = telemetry.day_dir("2026-05-01")
    (old_dir / "requests.jsonl").write_text('{"x":1}\n')
    calls = []
    telemetry.compress_old_logs(now_date="2026-05-25", run=lambda args: calls.append(args))
    assert any(args[:2] == ["zstd", "-10"] for args in calls)
    # injected runner: tmp written, renamed to .zst, original removed
    assert (old_dir / "requests.jsonl.zst").exists()
    assert oct((old_dir / "requests.jsonl.zst").stat().st_mode & 0o777) == "0o600"
    assert not (old_dir / "requests.jsonl").exists()


def test_error_log_is_written(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_error(
        request_id="e1",
        tool="review_start",
        error_type="ValidationError",
        message="bad profile",
    )
    today = telemetry.today_dir()
    error_path = today / "errors.jsonl"
    assert error_path.exists()
    line = error_path.read_text().strip()
    entry = json.loads(line)
    assert entry["request_id"] == "e1"
    assert entry["error_type"] == "ValidationError"
    assert "ts" in entry


def test_client_usage_record(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_client_usage(
        client_name="codex",
        tool="review_start",
        profile="codex",
        operation="review",
        job_id="j1",
        status="ok",
    )
    today = telemetry.today_dir()
    usage_path = today / "client_usage.jsonl"
    assert usage_path.exists()
    line = usage_path.read_text().strip()
    entry = json.loads(line)
    assert entry["client_name"] == "codex"
    assert entry["tool"] == "review_start"
    assert entry["profile"] == "codex"
    assert entry["operation"] == "review"
    assert entry["job_id"] == "j1"
    assert entry["status"] == "ok"
    assert "ts" in entry


def test_file_permissions_on_all_logs(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_request(
        client={"name": "qwen"}, tool="text_start", request_id="r2", payload={}
    )
    telemetry.record_response(request_id="r2", ok=True, duration_ms=5, response={})
    telemetry.record_error(
        request_id="r2", tool="text_start", error_type="RuntimeError", message="boom"
    )
    telemetry.record_client_usage(
        client_name="qwen",
        tool="text_start",
        profile="qwen",
        operation="text",
        job_id="j2",
        status="ok",
    )
    today = telemetry.today_dir()
    for fname in ("requests.jsonl", "responses.jsonl", "errors.jsonl", "client_usage.jsonl"):
        fpath = today / fname
        assert fpath.exists(), f"{fname} should exist"
        assert oct(fpath.stat().st_mode & 0o777) == "0o600", f"{fname} should be 0600"


def test_day_dir_creates_with_correct_permissions(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    day = telemetry.day_dir("2026-06-15")
    assert oct(tmp_path.stat().st_mode & 0o777) == "0o700"
    assert oct((tmp_path / "telemetry").stat().st_mode & 0o777) == "0o700"
    assert oct(day.stat().st_mode & 0o777) == "0o700"


def test_compress_skips_current_day(tmp_path):
    """compress_old_logs must not compress today's directory."""
    telemetry = TelemetryStore(tmp_path)
    today_str = telemetry.today_date()
    today_d = telemetry.day_dir(today_str)
    (today_d / "requests.jsonl").write_text('{"x":1}\n')
    calls = []
    telemetry.compress_old_logs(now_date=today_str, run=lambda args: calls.append(args))
    # Should not have compressed anything — it's today
    assert not any(today_str in str(a) for a in calls)


def test_compress_does_not_touch_recent_dirs(tmp_path):
    """Directories within the 14-day retention window must not be compressed."""
    telemetry = TelemetryStore(tmp_path)
    # 10 days ago — within retention
    recent_dir = telemetry.day_dir("2026-05-15")
    (recent_dir / "requests.jsonl").write_text('{"x":1}\n')
    calls = []
    telemetry.compress_old_logs(now_date="2026-05-25", run=lambda args: calls.append(args))
    assert not any("2026-05-15" in str(a) for a in calls)


def test_compress_injected_runner_uses_zstd_10_and_removes_original(tmp_path):
    """Injected runner creates tmp zst; assert args include zstd -10 and original removed after rename."""
    telemetry = TelemetryStore(tmp_path)
    old_dir = telemetry.day_dir("2026-05-01")
    log_file = old_dir / "requests.jsonl"
    log_file.write_text('{"req": "test"}\n')

    calls = []
    telemetry.compress_old_logs(now_date="2026-05-25", run=lambda args: calls.append(args))

    assert len(calls) == 1
    args = calls[0]
    assert args[0] == "zstd"
    assert args[1] == "-10"
    # tmp.zst should be in -o arg
    assert "-o" in args
    tmp_idx = args.index("-o")
    tmp_path_arg = args[tmp_idx + 1]
    assert tmp_path_arg.endswith(".tmp.zst")

    # After injected flow: .zst exists, original removed
    zst_file = old_dir / "requests.jsonl.zst"
    assert zst_file.exists(), "Compressed file should exist after rename"
    assert not log_file.exists(), "Original should be removed after compression"


def test_record_request_writes_valid_jsonl(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_request(
        client={"name": "reasonix", "version": "1.0"},
        tool="advice_start",
        request_id="req-abc",
        payload={"operation": "advice", "profile": "reasonix"},
        profile="reasonix",
        operation="advice",
        job_id="j-abc",
    )
    today = telemetry.today_dir()
    req_path = today / "requests.jsonl"
    lines = req_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["request_id"] == "req-abc"
    assert entry["tool"] == "advice_start"
    assert entry["client"]["name"] == "reasonix"
    assert "request" in entry
    assert entry["request"] == {"operation": "advice", "profile": "reasonix"}
    assert "ts" in entry
    assert entry["profile"] == "reasonix"
    assert entry["operation"] == "advice"
    assert entry["job_id"] == "j-abc"


def test_record_response_writes_valid_jsonl(tmp_path):
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_response(
        request_id="req-abc", ok=True, duration_ms=42, response={"job_id": "j1"}, job_id="j1"
    )
    today = telemetry.today_dir()
    resp_path = today / "responses.jsonl"
    lines = resp_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["request_id"] == "req-abc"
    assert entry["ok"] is True
    assert entry["duration_ms"] == 42
    assert "ts" in entry
    assert entry["response"] == {"job_id": "j1"}


def test_no_blank_tool_in_client_usage_after_request_and_response(tmp_path):
    """record_request + record_response must not produce blank-tool client_usage entries."""
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_request(
        client={"name": "codex"}, tool="review_start", request_id="r1", payload={}
    )
    telemetry.record_response(request_id="r1", ok=True, duration_ms=5, response={})
    today = telemetry.today_dir()
    usage_path = today / "client_usage.jsonl"
    assert usage_path.exists()
    entries = [json.loads(line) for line in usage_path.read_text().strip().splitlines()]
    for entry in entries:
        assert entry.get("tool", "") != "", f"blank tool in entry: {entry}"


def test_record_response_writes_client_usage_when_explicit(tmp_path):
    """record_response writes client_usage only when client_name and tool are provided."""
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_response(
        request_id="r1",
        ok=True,
        duration_ms=7,
        response={},
        client_name="codex",
        tool="review_start",
        job_id="j1",
    )
    today = telemetry.today_dir()
    usage_path = today / "client_usage.jsonl"
    assert usage_path.exists()
    lines = usage_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["client_name"] == "codex"
    assert entry["tool"] == "review_start"
    assert entry["status"] == "ok"
    assert entry["job_id"] == "j1"


def test_record_response_without_client_name_does_not_write_client_usage(tmp_path):
    """record_response with tool but no client_name must not write client_usage."""
    telemetry = TelemetryStore(tmp_path)
    telemetry.record_response(
        request_id="r1",
        ok=True,
        duration_ms=3,
        response={},
        tool="review_start",
    )
    today = telemetry.today_dir()
    usage_path = today / "client_usage.jsonl"
    assert not usage_path.exists()
