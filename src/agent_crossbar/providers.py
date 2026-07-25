"""Provider launch plans for Reasonix, Codex, Claude, and OpenCode."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_crossbar.profiles import (
    CODEX_DEFAULT_EFFORT,
    OPENCODE_MODELS,
    OPENCODE_PROVIDER_ID,
)


@dataclass
class LaunchCandidate:
    """A single candidate command/transport for launching an agent."""

    name: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    send_prompt: bool = True
    redirect_output: bool = True
    prompt_delay_sec: float = 0.0
    prompt_ready_patterns: tuple[str, ...] = ()
    prompt_submit_delay_sec: float = 0.0
    prompt_text: str | None = None


@dataclass
class LaunchPlan:
    """A launch plan containing candidates and metadata."""

    candidates: list[LaunchCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    message: str = ""


# Reasonix model allowlist.
REASONIX_ALLOWED_MODELS: frozenset[str] = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})


def reasonix_shell_mcp_spec() -> str:
    """Return the local YOLO shell MCP spec for Reasonix dev runs."""
    return f"agent_crossbar_shell={sys.executable} -m agent_crossbar.shell_server"


def reasonix_shell_env(cwd: str | None) -> dict[str, str]:
    """Root Reasonix shell tools in the requested dev cwd."""
    return {"AGENT_CROSSBAR_SHELL_CWD": str(cwd or os.getcwd())}


def reasonix_dev_prompt(prompt: str) -> str:
    """Tell Reasonix dev runs that the harness-provided shell tool is available."""
    return (
        "You have YOLO shell access through the agent_crossbar_shell MCP server. "
        "Use its run_shell_command tool for local shell commands and return the "
        "actual command output when the user asks for it.\n\n"
        f"{prompt}"
    )


def build_launch_plan(
    *,
    profile: str,
    operation: str,
    transport: str,
    prompt: str,
    model: str | None = None,
    **kwargs: Any,
) -> LaunchPlan:
    """Build a launch plan for the given profile/operation/transport.

    Returns a ``LaunchPlan`` with candidates and metadata, or an error string.
    No subprocess execution is performed here — this only produces the plan.
    """
    if profile == "reasonix":
        return _reasonix_plan(
            operation=operation,
            transport=transport,
            prompt=prompt,
            model=model,
            cwd=kwargs.get("cwd"),
            job_dir=kwargs.get("job_dir"),
        )
    if profile == "codex":
        return _codex_plan(
            operation=operation,
            transport=transport,
            prompt=prompt,
            model=model,
            effort=kwargs.get("effort"),
            cwd=kwargs.get("cwd"),
        )
    if profile == "claude":
        return _claude_plan(
            profile=profile,
            operation=operation,
            transport=transport,
            prompt=prompt,
            model=model,
        )
    if profile == "opencode":
        return _opencode_plan(
            operation=operation,
            transport=transport,
            prompt=prompt,
            model=model,
            cwd=kwargs.get("cwd"),
        )

    return LaunchPlan(error="invalid_profile", message=f"Profile '{profile}' has no launch plan")


def _reasonix_plan(
    *,
    operation: str,
    transport: str,
    prompt: str,
    model: str | None,
    cwd: str | None = None,
    job_dir: str | None = None,
) -> LaunchPlan:
    """Build a Reasonix launch plan for an explicitly selected model."""
    if not model:
        return LaunchPlan(
            error="missing_model",
            message="Model is required for Reasonix",
        )
    if model not in REASONIX_ALLOWED_MODELS:
        return LaunchPlan(
            error="invalid_model",
            message=f"Model '{model}' not allowed for Reasonix. Allowed: {sorted(REASONIX_ALLOWED_MODELS)}",
        )
    if operation == "dev":
        prompt = reasonix_dev_prompt(prompt)
        shell_args = ["--mcp", reasonix_shell_mcp_spec()]
        shell_env = reasonix_shell_env(cwd)
    else:
        shell_args = []
        shell_env = {}
    if operation == "dev" and transport == "tmux":
        transcript = f"{job_dir}/transcript.jsonl" if job_dir else "transcript.jsonl"
        code_dir = cwd or os.getcwd()
        candidates = [
            LaunchCandidate(
                name=f"reasonix code {model}",
                args=[
                    "reasonix",
                    "code",
                    "-m",
                    model,
                    "--effort",
                    "high",
                    "--new",
                    "--transcript",
                    transcript,
                    code_dir,
                ],
                send_prompt=True,
                redirect_output=False,
                prompt_delay_sec=2.0,
                prompt_ready_patterns=("ask anything", "type a message to start your session"),
                prompt_submit_delay_sec=0.5,
            )
        ]
        return LaunchPlan(
            candidates=candidates,
            metadata={
                "profile": "reasonix",
                "operation": operation,
                "model": model,
                "transport": transport,
            },
        )
    if operation == "dev":
        return LaunchPlan(
            candidates=[
                LaunchCandidate(
                    name=f"reasonix run {model} --mcp shell",
                    args=[
                        "reasonix",
                        "run",
                        "-m",
                        model,
                        "--effort",
                        "high",
                        *shell_args,
                        prompt,
                    ],
                    env=shell_env,
                    send_prompt=False,
                )
            ],
            metadata={
                "profile": "reasonix",
                "operation": operation,
                "model": model,
                "transport": transport,
            },
        )
    # Non-dev: for tmux transport, enter a clean interactive chat session
    # identified by a unique --session name derived from job_dir.
    if transport == "tmux":
        import re as _re

        transcript = f"{job_dir}/transcript.jsonl" if job_dir else None
        session_name = (
            f"agent-crossbar-gate-{_re.sub(r'[^A-Za-z0-9_-]+', '-', Path(job_dir).name).strip('-')}"
            if job_dir
            else "agent-crossbar-gate-default"
        )
        chat_args = [
            "reasonix",
            "chat",
            "-m",
            model,
            "--session",
            session_name,
            "--new",
            "--no-dashboard",
        ]
        if transcript is not None:
            chat_args.extend(["--transcript", transcript])
        candidates = [
            LaunchCandidate(
                name=f"reasonix chat {model}",
                args=chat_args,
                send_prompt=True,
                redirect_output=False,
                prompt_delay_sec=3.0,
                prompt_ready_patterns=("ask anything", "type a message to start your session"),
                prompt_submit_delay_sec=0.5,
                prompt_text=prompt,
            ),
        ]
    else:
        candidates = [
            LaunchCandidate(
                name=f"reasonix {model}",
                args=["--model", model, "--prompt", prompt],
            ),
        ]
    return LaunchPlan(
        candidates=candidates,
        metadata={
            "profile": "reasonix",
            "operation": operation,
            "model": model,
            "transport": transport,
        },
    )


def _codex_plan(
    *,
    operation: str,
    transport: str,
    prompt: str,
    model: str | None,
    effort: str | None,
    cwd: str | None = None,
) -> LaunchPlan:
    """Codex native review with accepted_context_bypass_risk=true only for review/print."""
    selected_model = model or ""
    selected_effort = effort or CODEX_DEFAULT_EFFORT
    selection_args = [
        "--model",
        selected_model,
        "-c",
        f'model_reasoning_effort="{selected_effort}"',
    ]
    if operation == "dev" and transport == "tmux":
        argv = [
            "codex",
            *selection_args,
            "--no-alt-screen",
            "--ask-for-approval",
            "never",
            "--sandbox",
            "workspace-write",
        ]
        if cwd:
            argv += ["-C", cwd]
        argv.append(prompt)
        candidates = [
            LaunchCandidate(
                name="codex interactive tmux",
                args=argv,
                send_prompt=False,
                redirect_output=False,
            ),
        ]
    else:
        candidates = [
            LaunchCandidate(
                name="codex exec",
                args=["codex", "exec", "--ephemeral", *selection_args, prompt],
            ),
        ]
    is_review_print = operation == "review" and transport == "print"
    metadata: dict[str, Any] = {
        "profile": "codex",
        "operation": operation,
        "model": selected_model,
        "effort": selected_effort,
        "accepted_context_bypass_risk": is_review_print,
    }
    if is_review_print:
        metadata.update(
            {
                "context_gathering": "provider",
                "denied_path_filtering": "none",
                "manifest_accuracy": "best_effort",
                "warnings": ["context bypass risk accepted for native review"],
            }
        )
    return LaunchPlan(candidates=candidates, metadata=metadata)


def _claude_plan(
    *,
    profile: str,
    operation: str,
    transport: str,
    prompt: str,
    model: str | None,
) -> LaunchPlan:
    selected_model = model or ""
    if not re.fullmatch(r"claude-[a-z][a-z0-9-]*-\d+(?:-\d+)*", selected_model):
        return LaunchPlan(
            error="invalid_model",
            message=(
                f"Model '{selected_model}' is not a versioned Claude CLI model ID. "
                "Discover models through profiles_list and pass that exact ID."
            ),
        )
    model_id = selected_model
    if transport != "tmux":
        return LaunchPlan(
            error="unsupported_transport",
            message=f"Profile '{profile}' does not support transport '{transport}' for operation '{operation}'",
        )
    args = ["claude", "--model", model_id]
    if operation == "dev":
        args += ["--permission-mode", "bypassPermissions", prompt]
        send_prompt = False
        prompt_text = None
        prompt_ready_patterns: tuple[str, ...] = ()
    elif operation == "review":
        args += [
            "--permission-mode",
            "plan",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--disable-slash-commands",
        ]
        send_prompt = True
        prompt_text = prompt
        prompt_ready_patterns = ("❯",)
    else:
        args.append(prompt)
        send_prompt = False
        prompt_text = None
        prompt_ready_patterns = ()
    return LaunchPlan(
        candidates=[
            LaunchCandidate(
                name=f"claude {model_id} interactive",
                args=args,
                send_prompt=send_prompt,
                redirect_output=False,
                prompt_text=prompt_text,
                prompt_ready_patterns=prompt_ready_patterns,
            )
        ],
        metadata={
            "profile": profile,
            "operation": operation,
            "model": selected_model,
            "model_id": model_id,
            "transport": transport,
        },
    )


def _opencode_model_id(model: str) -> str:
    if "/" in model:
        return model
    if model.startswith(f"{OPENCODE_PROVIDER_ID}/"):
        model = model.removeprefix(f"{OPENCODE_PROVIDER_ID}/")
    return f"{OPENCODE_PROVIDER_ID}/{model}"


def _opencode_plan(
    *,
    operation: str,
    transport: str,
    prompt: str,
    model: str | None,
    cwd: str | None = None,
) -> LaunchPlan:
    """OpenCode uses opencode-go models through the native opencode CLI."""
    selected_model = model or ""
    if selected_model.startswith(f"{OPENCODE_PROVIDER_ID}/"):
        selected_model = selected_model.removeprefix(f"{OPENCODE_PROVIDER_ID}/")
    if selected_model not in OPENCODE_MODELS:
        return LaunchPlan(
            error="invalid_model",
            message=f"Model '{model or selected_model}' not allowed for OpenCode. Allowed: {OPENCODE_MODELS}",
        )

    model_id = _opencode_model_id(selected_model)
    if operation == "dev" and transport == "tmux":
        args = [
            "opencode",
            "run",
            "-m",
            model_id,
            "--dangerously-skip-permissions",
        ]
        if cwd:
            args += ["--dir", cwd]
        args.append(prompt)
        candidates = [
            LaunchCandidate(
                name=f"opencode run {model_id} tmux",
                args=args,
                send_prompt=False,
                redirect_output=True,
            )
        ]
    else:
        args = ["opencode", "run", "-m", model_id]
        if operation == "dev":
            args.append("--dangerously-skip-permissions")
            if cwd:
                args += ["--dir", cwd]
        args.append(prompt)
        candidates = [
            LaunchCandidate(
                name=f"opencode run {model_id}",
                args=args,
                send_prompt=False,
            )
        ]

    return LaunchPlan(
        candidates=candidates,
        metadata={
            "profile": "opencode",
            "operation": operation,
            "model": selected_model,
            "model_id": model_id,
            "transport": transport,
        },
    )
