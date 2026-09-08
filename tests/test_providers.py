"""Provider launch plan tests for the agent harness MCP."""

from agent_crossbar.profiles import profile_registry
from agent_crossbar.providers import build_launch_plan

INTERACTIVE_TMUX_DEV_PROFILES = ["reasonix", "codex", "claude"]
TMUX_DEV_PROFILES = ["reasonix", "codex", "claude", "opencode"]
PRINT_MODE_FLAGS = {"-p", "--print", "--prompt"}


def _tmux_dev_plan(profile: str):
    return build_launch_plan(
        profile=profile,
        operation="dev",
        transport="tmux",
        prompt="Smoke only. Reply READY.",
        model="deepseek-flash/deepseek-v4-flash" if profile == "reasonix" else None,
        cwd="/repo",
        job_dir="/state/jobs/12345678-job",
    )


def _assert_interactive_tmux_candidate(profile: str, args: list[str]) -> None:
    assert not (PRINT_MODE_FLAGS & set(args)), profile
    assert args[:2] != ["codex", "exec"], profile
    assert args[:2] != ["reasonix", "run"], profile
    tmux_profiles = [
        profile
        for profile, entry in profile_registry().items()
        for capability in entry["capabilities"]
        if capability["operation"] == "dev" and "tmux" in capability["transports"]
    ]

    assert tmux_profiles == TMUX_DEV_PROFILES
    for profile in tmux_profiles:
        plan = _tmux_dev_plan(profile)
        assert plan.error is None, profile
        assert plan.candidates, profile
        _assert_interactive_tmux_candidate(profile, plan.candidates[0].args)
        if profile in INTERACTIVE_TMUX_DEV_PROFILES:
            assert plan.candidates[0].redirect_output is False, profile
        else:
            assert plan.candidates[0].redirect_output is True, profile


def test_reasonix_rejects_invalid_model():
    result = build_launch_plan(
        profile="reasonix", operation="review", transport="print", prompt="x", model="bad model"
    )
    assert result.error == "invalid_model"


def test_opencode_print_plan_requires_model_and_rejects_invalid_model():
    plan = build_launch_plan(
        profile="opencode", operation="review", transport="print", prompt="x", model=None
    )

    assert plan.error == "invalid_model"
    assert plan.candidates == []

    invalid = build_launch_plan(
        profile="opencode",
        operation="review",
        transport="print",
        prompt="x",
        model="not-a-model",
    )
    assert invalid.error == "invalid_model"


def test_opencode_dev_print_plan_roots_workspace_with_dir():
    plan = build_launch_plan(
        profile="opencode",
        operation="dev",
        transport="print",
        prompt="x",
        model="glm-5.2",
        cwd="/repo",
    )

    assert plan.error is None
    assert plan.candidates[0].args == [
        "opencode",
        "run",
        "-m",
        "opencode-go/glm-5.2",
        "--dangerously-skip-permissions",
        "--dir",
        "/repo",
        "x",
    ]


def test_codex_text_transport_no_bypass_risk():
    plan = build_launch_plan(profile="codex", operation="text", transport="print", prompt="x")
    assert plan.metadata.get("accepted_context_bypass_risk") is False
    assert "context_gathering" not in plan.metadata


def test_reasonix_tmux_non_dev_includes_isolation_flags():
    """Interactive tmux launch for ask/review must use --session <unique> --new --no-dashboard."""
    plan = build_launch_plan(
        profile="reasonix",
        operation="advice",
        transport="tmux",
        prompt="hello",
        model="deepseek-flash/deepseek-v4-flash",
        job_dir="/tmp/jobs/gate-abc123",
    )
    assert not plan.error, plan.message
    assert len(plan.candidates) == 1
    args = plan.candidates[0].args
    assert "--new" in args
    assert "--no-dashboard" in args
    # --session must contain a sanitised unique name derived from job_dir
    session_idx = args.index("--session")
    session_name = args[session_idx + 1]
    assert "gate-abc123" in session_name, (
        f"session name {session_name!r} should derive from job_dir"
    )
