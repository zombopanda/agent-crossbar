"""ChatGPT Pro GUI adapter."""

from __future__ import annotations

from ..profiles.chatgpt_pro import SUPPORT_TIER
from .base import StaticAdapter


def _readiness_probe(runner=None):
    from ..readiness import check_chatgpt_pro_readiness

    return check_chatgpt_pro_readiness(runner)


class ChatgptProAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="chatgpt_pro",
            support_tier=SUPPORT_TIER,
            backend="gui",
            supports_interactive=False,
            effort_map={},
            default_transport="gui",
            readiness_probe=_readiness_probe,
        )


adapter = ChatgptProAdapter()
