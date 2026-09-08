"""Provider-neutral local subprocess runner.

Argv-only, no shell — used by adapter lifecycle code (launch, readiness,
status, cancel) across providers, not just Claude.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult: ...


class LocalSubprocessRunner:
    """Production runner with an argv-only subprocess boundary."""

    def run(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            check=False,
        )
        return RunResult(completed.returncode, completed.stdout, completed.stderr)
