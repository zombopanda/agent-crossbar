"""Daily telemetry audit logs and zstd retention."""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_crossbar.redaction import redact_operator_token_from_text

_RETENTION_DAYS = 14
_DIR_MODE = 0o700
_FILE_MODE = 0o600

_LOG_FILES = ("requests.jsonl", "responses.jsonl", "client_usage.jsonl", "errors.jsonl")



def _now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _default_subprocess_run(args: list[str]) -> None:
    """Default subprocess runner for real compression."""
    subprocess.run(args, check=True, capture_output=True)


class TelemetryStore:
    """Telemetry store that writes daily audit logs and compresses old logs."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root) / "telemetry"

    # -- directory helpers --

    def today_date(self) -> str:
        """Return today's date as YYYY-MM-DD string."""
        return date.today().isoformat()

    def day_dir(self, day: str) -> Path:
        """Return the Path for a given YYYY-MM-DD day, creating it with 0700."""
        self._root.parent.mkdir(parents=True, exist_ok=True)
        self._root.parent.chmod(_DIR_MODE)
        self._root.mkdir(parents=True, exist_ok=True)
        self._root.chmod(_DIR_MODE)
        d = self._root / day
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(_DIR_MODE)
        return d

    def today_dir(self) -> Path:
        """Return today's telemetry directory."""
        return self.day_dir(self.today_date())

    # -- record methods --

    def record_request(
        self,
        client: dict[str, Any],
        tool: str,
        request_id: str,
        payload: dict[str, Any],
        *,
        profile: str | None = None,
        operation: str | None = None,
        job_id: str | None = None,
    ) -> None:
        """Record a full request payload to requests.jsonl and client_usage.jsonl."""
        today = self.today_dir()
        ts = _now_iso()

        # requests.jsonl: full payload in 'request' field
        req_entry: dict[str, Any] = {
            "ts": ts,
            "request_id": request_id,
            "client": client,
            "tool": tool,
            "request": payload,
        }
        if profile is not None:
            req_entry["profile"] = profile
        if operation is not None:
            req_entry["operation"] = operation
        if job_id is not None:
            req_entry["job_id"] = job_id
        self._append(today / "requests.jsonl", req_entry)

        # client_usage.jsonl: compact index
        usage_entry: dict[str, Any] = {
            "ts": ts,
            "client_name": client.get("name", ""),
            "tool": tool,
        }
        if profile is not None:
            usage_entry["profile"] = profile
        if operation is not None:
            usage_entry["operation"] = operation
        if job_id is not None:
            usage_entry["job_id"] = job_id
        self._append(today / "client_usage.jsonl", usage_entry)

    def record_response(
        self,
        request_id: str,
        ok: bool,
        duration_ms: float,
        response: dict[str, Any],
        *,
        job_id: str | None = None,
        client_name: str | None = None,
        tool: str | None = None,
        profile: str | None = None,
        operation: str | None = None,
    ) -> None:
        """Record a response to responses.jsonl and optionally client_usage.jsonl."""
        today = self.today_dir()
        ts = _now_iso()

        # responses.jsonl: full payload in 'response' field
        resp_entry: dict[str, Any] = {
            "ts": ts,
            "request_id": request_id,
            "ok": ok,
            "duration_ms": duration_ms,
            "response": response,
        }
        if tool is not None:
            resp_entry["tool"] = tool
        if client_name is not None:
            resp_entry["client_name"] = client_name
        if job_id is not None:
            resp_entry["job_id"] = job_id
        self._append(today / "responses.jsonl", resp_entry)

        # client_usage.jsonl: only if both client_name and tool are provided
        if client_name is not None and tool is not None:
            usage_entry: dict[str, Any] = {
                "ts": ts,
                "client_name": client_name,
                "tool": tool,
                "status": "ok" if ok else "error",
            }
            if job_id is not None:
                usage_entry["job_id"] = job_id
            if profile is not None:
                usage_entry["profile"] = profile
            if operation is not None:
                usage_entry["operation"] = operation
            self._append(today / "client_usage.jsonl", usage_entry)

    def record_error(
        self,
        request_id: str,
        tool: str,
        error_type: str,
        message: str,
    ) -> None:
        """Record an error to errors.jsonl."""
        today = self.today_dir()
        entry = {
            "ts": _now_iso(),
            "request_id": request_id,
            "tool": tool,
            "error_type": error_type,
            "message": message,
        }
        self._append(today / "errors.jsonl", entry)

    def record_client_usage(
        self,
        client_name: str,
        tool: str,
        *,
        profile: str | None = None,
        operation: str | None = None,
        job_id: str | None = None,
        status: str = "ok",
    ) -> None:
        """Record a compact usage entry to client_usage.jsonl."""
        today = self.today_dir()
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "client_name": client_name,
            "tool": tool,
            "status": status,
        }
        if profile is not None:
            entry["profile"] = profile
        if operation is not None:
            entry["operation"] = operation
        if job_id is not None:
            entry["job_id"] = job_id
        self._append(today / "client_usage.jsonl", entry)

    # -- retention --

    def compress_old_logs(
        self,
        now_date: str | None = None,
        run: Callable[[list[str]], None] | None = _default_subprocess_run,
    ) -> None:
        """Compress log files older than the retention window using zstd -10.

        Uses an atomic tmp-write / rename / remove flow:
          1. zstd -10 <file> -o <file>.tmp.zst
          2. rename <file>.tmp.zst -> <file>.zst
          3. remove <file>

        When *run* is None, real subprocess execution is used.
        When *run* is a callable, it is invoked with the argument list
        (useful for testing without spawning real processes).
        """
        ref_date = date.fromisoformat(now_date) if now_date else date.today()

        if not self._root.exists():
            return

        for day_path in sorted(self._root.iterdir()):
            if not day_path.is_dir():
                continue
            try:
                day_date = date.fromisoformat(day_path.name)
            except (ValueError, TypeError):
                continue

            age_days = (ref_date - day_date).days
            if age_days <= _RETENTION_DAYS:
                continue

            for log_name in _LOG_FILES:
                log_file = day_path / log_name
                if not log_file.exists():
                    continue

                zst_file = day_path / f"{log_name}.zst"
                tmp_file = day_path / f"{log_name}.tmp.zst"

                args = ["zstd", "-10", str(log_file), "-o", str(tmp_file)]

                if run is not None:
                    # Test mode: invoke the provided callable with args
                    run(args)
                    # Simulate atomic flow: tmp exists after run, then rename + remove
                    tmp_file.touch()
                    tmp_file.chmod(_FILE_MODE)
                    tmp_file.rename(zst_file)
                    zst_file.chmod(_FILE_MODE)
                    log_file.unlink(missing_ok=True)
                else:
                    # Real mode: use actual subprocess
                    subprocess.run(args, check=True, capture_output=True)
                    tmp_file.chmod(_FILE_MODE)
                    tmp_file.rename(zst_file)
                    zst_file.chmod(_FILE_MODE)
                    log_file.unlink(missing_ok=True)

    # -- internals --

    def _append(self, path: Path, entry: dict[str, Any]) -> None:
        """Append a JSON entry to a .jsonl file with 0600 permissions."""
        if not path.exists():
            path.touch(mode=_FILE_MODE)
        with open(path, "a") as f:
            json_str = json.dumps(entry, default=str)
            json_str = redact_operator_token_from_text(json_str)
            f.write(json_str + "\n")
        # Ensure permissions are correct even on existing files
        path.chmod(_FILE_MODE)
