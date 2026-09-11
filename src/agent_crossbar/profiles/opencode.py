"""OpenCode profile constants and entry."""

from __future__ import annotations

# Single source of truth for this provider's support tier — the adapter
# module re-exports this constant rather than hardcoding its own literal.
SUPPORT_TIER = "supported"

OPENCODE_PROVIDER_ID = "opencode-go"
OPENCODE_MODELS = [
    "opencode/deepseek-v4-flash-free",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "mimo-v2.5",
    "mimo-v2.5-pro",
    "minimax-m2.7",
    "minimax-m3",
]


def build_entry() -> dict:
    return {
        "aliases": [],
        "models": OPENCODE_MODELS,
        "operations": ["review", "text", "advice", "dev"],
        "interactive": False,
        "support_tier": SUPPORT_TIER,
    }


def build_matrix_entry() -> dict:
    entry = build_entry()
    return {
        "support_tier": entry["support_tier"],
        "os": ["darwin", "linux"],
        "operations": entry["operations"],
        "backend": "acp",
        "interaction_modes": ["noninteractive"],
        "effort_support": True,
        "billing_mode": "free_defaults",
        # OpenCode has no free-text continuation transport, but its ACP
        # permission callback is held and resolved through structured
        # ``job_send`` decisions while the ACP connection remains alive.
        "owner_permission_decisions": True,
        "job_send_supported": False,
    }
