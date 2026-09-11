import time

import pytest

from agent_crossbar.tmux_output import (
    interactive_tmux_output_complete,
    interactive_tmux_output_complete_since,
    interactive_tmux_output_summary,
    reconstruct_tmux_output,
)


@pytest.mark.parametrize("profile", ["reasonix", "deepseek"])
def test_reasonix_busy_prompt_does_not_fall_through_to_opencode(profile):
    output = """
waiting for model…
reasoning…
thinking…
› type to steer current task — commands disabled while busy
waiting for model…
esc to stop
"""

    assert interactive_tmux_output_complete(output, profile=profile) is False


@pytest.mark.parametrize("profile", ["reasonix", "deepseek"])
def test_reasonix_reply_followed_by_ready_prompt_is_complete(profile):
    output = """
‹ reply
The requested analysis is complete.
Ask anything
"""

    assert interactive_tmux_output_complete(output, profile=profile) is True


def test_unknown_named_profile_fails_closed_instead_of_cross_detecting():
    output = """
› opencode-looking prompt
Apparent answer
"""

    assert interactive_tmux_output_complete(output, profile="reasonixx") is False
    assert interactive_tmux_output_complete(output) is True


def test_unknown_named_profile_summary_does_not_use_cross_provider_marker():
    output = "› opencode-looking prompt\nApparent answer\n" + ("x" * 100) + "TAIL"

    summary = interactive_tmux_output_summary(
        output,
        profile="reasonixx",
        max_chars=20,
    )

    assert summary.endswith("TAIL")


def test_reasonix_summary_uses_only_reasonix_completion_marker():
    output = """
› stale opencode-looking prompt
stale provider output
‹ reply
The actual Reasonix answer.
Ask anything
"""

    summary = interactive_tmux_output_summary(output, profile="reasonix", max_chars=45)

    assert "actual Reasonix answer" in summary


def test_claude_tool_cycle_with_busy_prompt_is_not_complete():
    output = """
⏺ Bash (git diff --stat)
  ⎿ diff --git a/demo.py b/demo.py
❯
esc to interrupt ◐ medium · /effort
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_tool_cycle_followed_by_final_answer_is_complete():
    output = """
⏺ Bash (git diff --stat)
  ⎿ diff --git a/demo.py b/demo.py
esc to interrupt ◐ medium · /effort
⏺ PASS
❯
\x1b]777;notify;Claude Code;Claude is waiting for your input\x07
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


@pytest.mark.parametrize(
    "spinner",
    [
        "Ebbing… (35s · ↓ 1.5k tokens · thought for 9s)",
        "Crystallizing… (57s · almost done thinking)",
        "Simmering… (22s · ↓ 900 tokens)",
        "Churning… (1m 2s · thinking some more)",
        "Pondering… (18s · thought for 4s)",
    ],
)
def test_claude_visible_prompt_during_active_thinking_is_not_complete(spinner):
    output = f"""
⏸ plan mode on · esctointerrupt · ← 1 agent
80427 tokens
        ⏺(/home/demo/src/example-project/runner.py)
  ⎿  Read 240 lines
✶ {spinner}
❯
80427 tokens
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_elapsed_only_spinner_is_not_complete():
    output = """
⏺ Earlier text copied into the prompt context.
❯
✽ Ruminating… (0s)
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_markers_copied_inside_current_prompt_are_not_complete():
    output = """
❯ Review this bundled context:
  ⏺ PASS copied from an earlier transcript
  ❯ copied idle prompt
❯
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_notification_text_copied_in_prompt_is_not_idle_signal():
    output = """
❯ Review this test fixture:
  ⏺ Final verdict: PASS.
  ❯
  Claude is waiting for your input
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_answer_with_indented_prompt_example_completes_on_idle_notification():
    output = """
❯ Review the tmux completion code
⏺ Here is the shell prompt example:
    ❯ run the command
❯
\x1b]777;notify;Claude Code;Claude is waiting for your input\x07
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_prompt_redraw_after_idle_notification_stays_complete():
    output = """
❯ Review the tmux completion code
⏺ PASS
\x1b]777;notify;Claude Code;Claude is waiting for your input\x07
❯
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_plan_ready_approval_prompt_is_terminal_for_read_only_work():
    output = """
Review findings and verification steps.
Claude has written up a plan and is ready to execute. Would you like to proceed?
❯ 1. Yes, and use auto mode
2. Yes, manually approve edits
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_plan_ready_text_copied_in_prompt_is_not_terminal():
    output = """
❯ Review this source line:
  r"Claude has written up a plan and is ready to execute"
❯
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_plan_ready_phrase_without_approval_ui_is_not_terminal():
    output = """Claude has written up a plan and is ready to execute.

This sentence is reviewed source text, not the live approval prompt.
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_final_verdict_with_idle_prompt_is_complete():
    output = """
⏺ Read
  ⎿  Read 240 lines
⏺ Final verdict: PASS. No actionable findings.
❯
80427 tokens
\x1b]777;notify;Claude Code;Claude is waiting for your input\x07
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_live_idle_footer_without_osc_notification_is_complete():
    output = """
⏺ Bash(python3 -m pytest test_reverse_words.py -v)
  ⎿  Pytest: 5 passed

⏺ Готово.

  - Результат: 5 passed.

✻ Sautéed for 30s
❯ add a test for numbers mixed with letters
Opus 4.8 │ 63k/1M (6%)
⏵⏵ bypass permissions on (shift+tab to cycle) · ← 1 agent
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_append_log_finalization_marker_overrides_stale_busy_frame():
    output = """
⏺ Готово. Результат: 5 passed.
· Thinking… (25s · ↓ 1.2k tokens)
running stop hooks…
✻ Churned for 25s
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_final_answer_discussing_spinner_tokens_is_complete():
    output = """
⏺ Final verdict: avoid text like Ebbing… (900 tokens) in diagnostics.
❯
80427 tokens
\x1b]777;notify;Claude Code;Claude is waiting for your input\x07
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_idle_notification_completes_after_finalization_spinner():
    output = """
⏺ Final verdict: PASS. No actionable findings.
✶ Metamorphosing… (3m 52s · ↓ 17.8k tokens)
❯
]777;notify;Claude Code;Claude is waiting for your input
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_stale_idle_notification_does_not_complete_new_active_turn():
    output = """
⏺ Earlier answer.
❯
]777;notify;Claude Code;Claude is waiting for your input
⏺ Read
  ⎿ Read 20 lines
✶ Crystallizing… (25s · ↓ 900 tokens)
❯
esc to interrupt
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_claude_screen_reader_final_after_reply_boundary_is_complete():
    """The screen-reader ``claude:`` redraw plus lifecycle footer is final."""
    prefix = "previous turn transcript\n"
    appended = """\n⏺ User answered Claude's questions:
· Яка риса характеру у собаки? → Грайливий
Calculating…
auto mode on (shift+tab to cycle) · esc to interrupt
35892 tokens
effort: medium · /effort
$claude: F
Calculating…
auto mode on (shift+tab to cycle) · esc to interrupt
35892 tokens
effort: medium · /effort
$claude: FINAL: Барні
Calculating…
auto mode on (shift+tab to cycle) · esc to interrupt
36502 tokens
effort: medium · /effort
$Cogitated for 7s
auto mode on (shift+tab to cycle)
"""

    assert (
        interactive_tmux_output_complete_since(
            prefix + appended,
            baseline_bytes=len(prefix.encode()),
            profile="claude",
        )
        is True
    )


def test_claude_screen_reader_stale_redraw_without_final_footer_is_not_complete():
    prefix = "previous turn transcript\n"
    stale_redraw = """\n$claude: FINAL: old answer
Calculating…
auto mode on (shift+tab to cycle) · esc to interrupt
36502 tokens
effort: medium · /effort
"""

    assert (
        interactive_tmux_output_complete_since(
            prefix + stale_redraw,
            baseline_bytes=len(prefix.encode()),
            profile="claude",
        )
        is False
    )


def test_claude_screen_reader_generic_duration_footer_is_complete():
    output = """
$claude: LIVE_CLAUDE_TWO_AWAITS_OK
Crunched for 10s
"""

    assert interactive_tmux_output_complete(output, profile="claude") is True


def test_claude_screen_reader_generic_footer_requires_capitalized_past_tense_line():
    output = """
$claude: LIVE_CLAUDE_TWO_AWAITS_OK
crunched for 10s
"""

    assert interactive_tmux_output_complete(output, profile="claude") is False


def test_reconstruct_cursor_renderer_handles_newline_and_cursor_right():
    raw = "first line\npartial\n\x1b[1A\x1b[8Gcontinuation\n"

    rendered = reconstruct_tmux_output(raw)

    assert rendered.splitlines()[:2] == ["first line", "partialcontinuation"]


def test_reconstruct_cursor_renderer_preserves_text_before_line_erase():
    raw = "final answer\n\x1b[1A\x1b[2Kreplacement\n"

    rendered = reconstruct_tmux_output(raw)

    assert "final answer" in rendered
    assert "replacement" in rendered


def test_claude_exact_cursor_rewrite_after_reply_is_complete_and_full():
    prefix = "x" * 2595 + "\n"
    appended = (
        "\x1b(B\x0f\x1b[?1000h"
        "\x1b[Gclaude: L\r\r\n"
        "Contemplating…\r\r\n"
        "auto mode on (shift+tab to cycle)  ·  esc to interrupt\r\r\n"
        "37349 tokens\r\r\n"
        "effort: high · /effort\r\r\n"
        "$\x1b[2G\x1b[5A\x1b[10GIVE_CLAUDE_TWO_AWAITS_OK"
        "\x1b[2G\x1b[5B\x1b[2K\x1b[1A\x1b[2K"
        "\x1b[G37509 tokens\r\r\n"
        "\x1b[GChurned for 6s\r\r\n"
    )
    raw = prefix + appended

    assert len(prefix.encode("utf-8")) == 2596
    assert (
        interactive_tmux_output_complete_since(
            raw,
            baseline_bytes=2596,
            profile="claude",
        )
        is True
    )
    assert "claude: LIVE_CLAUDE_TWO_AWAITS_OK" in reconstruct_tmux_output(raw)


@pytest.mark.parametrize("movement", ["B", "C", "H"])
def test_cursor_renderer_rejects_pathological_motion_without_allocating(movement):
    raw = f"$claude: FINAL\n\x1b[999999999{movement}\nChurned for 1s\n"
    started = time.perf_counter()

    assert reconstruct_tmux_output(raw) == ""
    assert interactive_tmux_output_complete(raw, profile="claude") is False
    assert time.perf_counter() - started < 0.5


def test_codex_intermediate_answer_while_working_is_not_complete():
    output = """
› Create files and run tests.
• UserPromptSubmit hook (failed)
• Використаю TDD: спершу додам тести.
• Working (16s • esc to interrupt)
› Summarize recent commits
"""

    assert interactive_tmux_output_complete(output, profile="codex") is False


def test_codex_tool_result_bullet_is_not_a_final_answer():
    output = """
• UserPromptSubmit hook (completed)
• Ran rtk cat SKILL.md
  └ skill contents
"""

    assert interactive_tmux_output_complete(output, profile="codex") is False


def test_codex_tool_hook_bullet_is_not_a_final_answer():
    output = """
• UserPromptSubmit hook (completed)
• Ran rtk cat SKILL.md
• PostToolUse hook (completed)
  hook context: guidance
"""

    assert interactive_tmux_output_complete(output, profile="codex") is False


def test_codex_file_diff_without_stop_hook_is_not_complete():
    output = """
• UserPromptSubmit hook (completed)
• Added test_reverse_words.py (+21 -0)
     1 +def test_ok(): pass
"""

    assert interactive_tmux_output_complete(output, profile="codex") is False


def test_codex_answer_followed_by_stop_hook_is_complete():
    output = """
• UserPromptSubmit hook (completed)
• Tests passed.
• Running Stop hook: mem0 fact extraction
"""

    assert interactive_tmux_output_complete(output, profile="codex") is True


def test_codex_answer_followed_by_worked_for_boundary_is_complete():
    output = """
• UserPromptSubmit hook (completed)
• Created both files. Tests passed.
─ Worked for 1m 15s ─
› Explain this codebase
"""

    assert interactive_tmux_output_complete(output, profile="codex") is True
