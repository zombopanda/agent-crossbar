"""ACP server readiness checks for Codex and OpenCode providers.

These are CLIENT-SIDE preflight checks — they probe the local binaries
before attempting to spawn an ACP agent process.  Returns structured
results with actionable remediation so callers can surface clear errors.

- Codex: no native ``codex acp`` subcommand. Uses the standalone
  ``@agentclientprotocol/codex-acp`` package (>= 1.1.7), accessed via
  ``pnpm dlx @agentclientprotocol/codex-acp@1.1.7 --version``.
- OpenCode: uses the native ``opencode acp`` subcommand directly.
  Readiness probes ``opencode --version`` followed by a nonblocking
  ``opencode acp --help`` to verify the subcommand is available.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

_CODEX_ACP_MIN_VERSION = (1, 1, 7)
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class SubprocessRunner(Protocol):
    def run(self, args, *, timeout=None, cwd=None, env=None) -> Any: ...


_ACP_READINESS_PROBES = {
    "codex": "check_codex_acp_readiness",
    "opencode": "check_opencode_acp_readiness",
}


def check_acp_readiness(profile: str, runner: SubprocessRunner) -> dict:
    """Dispatch an ACP readiness probe through the adapter-owned registry."""
    try:
        probe_name = _ACP_READINESS_PROBES[profile]
        probe = globals()[probe_name]
    except KeyError:
        return {
            "ready": False,
            "state": "unsupported",
            "error_code": "acp_profile_unsupported",
            "remediation": f"No ACP readiness probe is registered for '{profile}'.",
            "version": None,
        }
    return probe(runner)


def _parse_version(output: str):
    match = _VERSION_RE.search(output)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_codex_acp_readiness(runner) -> dict:
    # Step 1: pnpm --version
    try:
        pnpm_result = runner.run(["pnpm", "--version"], timeout=10)
    except FileNotFoundError:
        return {
            "ready": False,
            "state": "missing_binary",
            "error_code": "pnpm_missing",
            "remediation": (
                "pnpm is not installed or not on PATH. "
                "Install pnpm (https://pnpm.io/installation). "
                "The harness uses `pnpm dlx @agentclientprotocol/codex-acp` "
                "(>= 1.1.7)."
            ),
            "version": None,
        }
    except Exception as exc:
        return {
            "ready": False,
            "state": "probe_failed",
            "error_code": "pnpm_probe_failed",
            "remediation": f"Failed to probe pnpm: {exc}",
            "version": None,
        }
    if pnpm_result.returncode != 0:
        return {
            "ready": False,
            "state": "missing_binary",
            "error_code": "pnpm_broken",
            "remediation": "pnpm is present but broken. Reinstall pnpm.",
            "version": None,
        }

    # Step 2: codex --version
    try:
        codex_result = runner.run(["codex", "--version"], timeout=10)
    except FileNotFoundError:
        return {
            "ready": False,
            "state": "missing_binary",
            "error_code": "codex_missing",
            "remediation": (
                "codex CLI is not installed or not on PATH. "
                "Install the Codex CLI to use the codex provider."
            ),
            "version": None,
        }
    except Exception as exc:
        return {
            "ready": False,
            "state": "probe_failed",
            "error_code": "codex_probe_failed",
            "remediation": f"Failed to probe codex: {exc}",
            "version": None,
        }
    if codex_result.returncode != 0:
        return {
            "ready": False,
            "state": "missing_binary",
            "error_code": "codex_broken",
            "remediation": "codex CLI is present but broken. Reinstall the Codex CLI.",
            "version": None,
        }

    # Step 3: pnpm dlx @agentclientprotocol/codex-acp@1.1.7 --version
    try:
        result = runner.run(
            ["pnpm", "dlx", "@agentclientprotocol/codex-acp@1.1.7", "--version"],
            timeout=30,
        )
    except FileNotFoundError:
        return {
            "ready": False,
            "state": "missing_binary",
            "error_code": "pnpm_missing",
            "remediation": (
                "pnpm is not installed. Install pnpm and run: "
                "pnpm dlx @agentclientprotocol/codex-acp@1.1.7 --version"
            ),
            "version": None,
        }
    except Exception as exc:
        return {
            "ready": False,
            "state": "probe_failed",
            "error_code": "codex_acp_probe_failed",
            "remediation": (
                f"Failed to probe @agentclientprotocol/codex-acp: {exc}. "
                "Install with: pnpm dlx @agentclientprotocol/codex-acp@1.1.7"
            ),
            "version": None,
        }

    if result.returncode != 0:
        return {
            "ready": False,
            "state": "probe_failed",
            "error_code": "codex_acp_version_failed",
            "remediation": (
                f"@agentclientprotocol/codex-acp --version exited with "
                f"code {result.returncode}. "
                f"stderr: {(result.stderr or '')[:200]}. "
                "Install with: pnpm dlx @agentclientprotocol/codex-acp@1.1.7"
            ),
            "version": None,
        }

    version = _parse_version(result.stdout)
    if version is None:
        return {
            "ready": True,
            "state": "ready",
            "error_code": None,
            "remediation": None,
            "version": None,
        }

    if version < _CODEX_ACP_MIN_VERSION:
        current = ".".join(str(v) for v in version)
        required = ".".join(str(v) for v in _CODEX_ACP_MIN_VERSION)
        return {
            "ready": False,
            "state": "version_too_low",
            "error_code": "codex_acp_version_too_low",
            "remediation": (
                f"@agentclientprotocol/codex-acp version {current} is too old. "
                f"Upgrade to >= {required}. "
                f"Run: pnpm dlx @agentclientprotocol/codex-acp@{required}"
            ),
            "version": current,
        }

    return {
        "ready": True,
        "state": "ready",
        "error_code": None,
        "remediation": None,
        "version": ".".join(str(v) for v in version),
    }


def check_opencode_acp_readiness(runner) -> dict:
    # Step 1: opencode --version
    try:
        opencode_result = runner.run(["opencode", "--version"], timeout=10)
    except FileNotFoundError:
        return {
            "ready": False,
            "state": "missing_binary",
            "error_code": "opencode_missing",
            "remediation": (
                "opencode CLI is not installed or not on PATH. "
                "Install the OpenCode CLI to use the opencode provider."
            ),
        }
    except Exception as exc:
        return {
            "ready": False,
            "state": "probe_failed",
            "error_code": "opencode_probe_failed",
            "remediation": f"Failed to probe opencode: {exc}",
        }
    if opencode_result.returncode != 0:
        return {
            "ready": False,
            "state": "missing_binary",
            "error_code": "opencode_broken",
            "remediation": "opencode CLI is present but broken. Reinstall the OpenCode CLI.",
        }

    # Step 2: opencode acp --help (nonblocking probe)
    try:
        acp_result = runner.run(["opencode", "acp", "--help"], timeout=30)
    except Exception as exc:
        stderr_snippet = str(exc)[:200]
        return {
            "ready": False,
            "state": "probe_failed",
            "error_code": "opencode_acp_probe_failed",
            "remediation": (
                f"Failed to probe `opencode acp --help`: {stderr_snippet}. "
                "Ensure the opencode CLI is installed and the `acp` subcommand is available."
            ),
        }

    if acp_result.returncode != 0:
        stderr_snippet = (acp_result.stderr or "")[:200]
        return {
            "ready": False,
            "state": "unavailable",
            "error_code": "opencode_acp_unavailable",
            "remediation": (
                f"`opencode acp --help` exited with code {acp_result.returncode}. "
                f"stderr: {stderr_snippet}. "
                "Ensure your opencode version supports the native `acp` subcommand. "
                "Upgrade opencode if needed."
            ),
        }

    return {
        "ready": True,
        "state": "ready",
        "error_code": None,
        "remediation": None,
    }
