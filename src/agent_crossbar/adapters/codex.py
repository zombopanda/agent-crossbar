"""Codex adapter and Codex App Server model discovery."""

from __future__ import annotations

from typing import Any

from ..discovery_runner import CodexSession, PopenCodexSession, discover_codex_models
from ..profiles.codex import SUPPORT_TIER
from .base import ModelCatalog, ModelInfo, StaticAdapter

# ── Session factory (injectable for tests) ───────────────────────────────────


_CODEX_SESSION_FACTORY: type[CodexSession] | None = None


def _create_codex_session() -> CodexSession:
    """Create a Codex session — production uses Popen, tests inject fake."""
    if _CODEX_SESSION_FACTORY is not None:
        return _CODEX_SESSION_FACTORY()
    return PopenCodexSession(["codex", "app-server"])


# ── Response parsing ─────────────────────────────────────────────────────────


def parse_model_list_response(response: dict[str, Any]) -> ModelCatalog:
    """Parse a Codex App Server model/list response into a ModelCatalog.

    Extracts per-model supported efforts and default effort.
    """
    visible = [item for item in response.get("data", []) if not item.get("hidden", False)]
    models = tuple(str(item.get("model") or item["id"]) for item in visible)
    default = next(
        (str(item.get("model") or item["id"]) for item in visible if item.get("isDefault")),
        models[0] if models else None,
    )
    efforts: list[str] = []
    model_info: list[ModelInfo] = []
    for item in visible:
        model_id = str(item.get("model") or item["id"])
        supported: list[str] = []
        for option in item.get("supportedReasoningEfforts", []):
            value = option.get("reasoningEffort")
            if isinstance(value, str):
                if value not in efforts:
                    efforts.append(value)
                if value not in supported:
                    supported.append(value)
        default_effort = item.get("defaultReasoningEffort")
        model_info.append(
            ModelInfo(
                id=model_id,
                supported_efforts=tuple(supported),
                default_effort=str(default_effort) if default_effort else None,
            )
        )
    return ModelCatalog(
        models=models,
        default_model=default,
        native_efforts=tuple(efforts),
        source="codex app-server model/list",
        model_info=tuple(model_info),
    )


def _max_effort_for_model(catalog: ModelCatalog, model_id: str) -> str | None:
    """Return the maximum native effort for *model_id* from discovered data.

    Priority order: max > xhigh > high > medium > low.
    Returns ``None`` when no effort data is available.
    """
    efforts = catalog.effort_for_model(model_id)
    if not efforts:
        return None
    # max is the strongest native effort when advertised
    priority = {"max": 4, "xhigh": 3, "high": 2, "medium": 1, "low": 0}
    return max(efforts, key=lambda e: priority.get(e, -1), default=None)


class CodexAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="codex",
            support_tier=SUPPORT_TIER,
            backend="acp",
            supports_interactive=False,
            effort_map={"low": "low", "medium": "medium", "high": "high", "max": "max"},
            # codex-acp v1.1.7 exposes `reasoning_effort` as a
            # `thought_level` session config option.
            supports_acp_effort=True,
        )

    def discover_models(self, runner: Any) -> ModelCatalog:
        """Discover models live from the Codex App Server.

        Uses an injectable Codex session (``PopenCodexSession`` in production,
        ``FakeCodexSession`` in tests via ``_CODEX_SESSION_FACTORY``).
        """
        session = _create_codex_session()
        result = discover_codex_models(session)
        if not result.get("ok"):
            return ModelCatalog(
                models=(),
                default_model=None,
                native_efforts=(),
                source="codex app-server model/list",
                error=result.get("error", "unknown error"),
            )
        return parse_model_list_response({"data": result["models"]})

    def resolve_effort(
        self, effort: str, catalog: ModelCatalog | None, model_id: str | None
    ) -> str:
        """Resolve a public effort to a native effort using discovered model data.

        - ``"max"`` resolves to the strongest native effort advertised by the
          model (max > xhigh > high > medium > low).  It never fails merely
          because native ``"max"`` is absent.
        - ``"xhigh"`` is a Codex-specific effort that maps 1:1.
        - Without a catalog, falls back to the static ``effort_map``.
        """
        if effort == "max" and catalog is not None and model_id is not None:
            resolved = _max_effort_for_model(catalog, model_id)
            if resolved is not None:
                return resolved
        return self.map_effort(effort)

    def validate_effort_for_model(
        self, effort: str, catalog: ModelCatalog | None, model_id: str | None
    ) -> str | None:
        """Return an error message if *effort* is unsupported for *model_id*.

        Returns None when the effort is valid.
        """
        if catalog is None or model_id is None:
            return None
        native = self.resolve_effort(effort, catalog, model_id)
        supported = catalog.effort_for_model(model_id)
        if supported and native not in supported:
            return (
                f"Effort '{effort}' (native '{native}') is not supported "
                f"by model '{model_id}'. Supported: {', '.join(supported)}"
            )
        return None


adapter = CodexAdapter()
