from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

from agent_crossbar.admission import admit_request


def _request() -> dict[str, str]:
    return {
        "profile": "claude",
        "model": "claude-sonnet-5",
        "operation": "dev",
        "task": "dev",
        "cwd": "/repo",
    }


def test_admission_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_CROSSBAR_ADMISSION_MODE", raising=False)
    monkeypatch.delenv("AGENT_CROSSBAR_ADMISSION_COMMAND", raising=False)
    result = admit_request(_request())
    assert result.allowed is True


def test_admission_requires_exact_allow_response(monkeypatch):
    command = [
        sys.executable,
        "-c",
        "import json,sys; value=json.load(sys.stdin); print(json.dumps({'version':1,'decision':'allow' if value['model']=='claude-sonnet-5' else 'deny','reason':'fixture'}))",
    ]
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_MODE", "strict")
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_COMMAND", json.dumps(command))
    assert admit_request(_request()).allowed is True
    denied = admit_request({**_request(), "model": "claude-opus-5"})
    assert denied.allowed is False
    assert denied.error == "admission_denied"


def test_admission_malformed_and_nonzero_fail_closed(monkeypatch):
    command = [sys.executable, "-c", "print('not json')"]
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_MODE", "strict")
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_COMMAND", json.dumps(command))
    result = admit_request(_request())
    assert result.allowed is False
    assert result.error == "admission_malformed"


def test_admission_timeout_kills_owned_process_group(monkeypatch):
    script = textwrap.dedent(
        """
        import time
        time.sleep(10)
        """
    )
    command = [sys.executable, "-c", script]
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_MODE", "strict")
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_COMMAND", json.dumps(command))
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_TIMEOUT_SEC", "0.05")
    result = admit_request(_request())
    assert result.allowed is False
    assert result.error == "admission_timeout"


def test_invalid_admission_timeout_fails_closed(monkeypatch):
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_MODE", "strict")
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_COMMAND", json.dumps([sys.executable]))
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_TIMEOUT_SEC", "not-a-number")
    result = admit_request(_request())
    assert result.allowed is False
    assert result.error == "admission_config"


def test_admission_rejects_substitution_fields(monkeypatch):
    command = [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'version':1,'decision':'allow','reason':'ok','model':'rewrite'}))",
    ]
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_MODE", "strict")
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_COMMAND", json.dumps(command))
    result = admit_request(_request())
    assert result.allowed is False
    assert result.error == "admission_malformed"


def test_admission_nonzero_fails_closed(monkeypatch):
    command = [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"]
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_MODE", "strict")
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_COMMAND", json.dumps(command))
    result = admit_request(_request())
    assert result.allowed is False
    assert result.error == "admission_failed"


def test_admission_timeout_reaps_child_process_group(monkeypatch, tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(10)"
    )
    command = [sys.executable, "-c", script]
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_MODE", "strict")
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_COMMAND", json.dumps(command))
    monkeypatch.setenv("AGENT_CROSSBAR_ADMISSION_TIMEOUT_SEC", "0.1")
    result = admit_request(_request())
    assert result.error == "admission_timeout"
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not pid_file.exists():
        time.sleep(0.01)
    if pid_file.exists():
        child_pid = int(pid_file.read_text())
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("admission child process survived timeout cleanup")
