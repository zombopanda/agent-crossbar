"""OpenCode native ACP adapter and CLI model discovery."""

from __future__ import annotations

import json
import re

from ..discovery_runner import DiscoveryProcess
from ..profiles.opencode import SUPPORT_TIER
from .base import PUBLIC_EFFORTS, ModelCatalog, ModelInfo, StaticAdapter

# Strict provider/model ID pattern: at least one '/', valid chars
# a-zA-Z0-9._~@/-.  Must not contain whitespace, JSON punctuation,
# or '://' (URLs), '//' (empty path segments).  Accepts nested slashes,
# @, ~, and : that real OpenCode models use
# (e.g. google-vertex/deepseek-ai/deepseek-v3.1-maas,

_MODEL_ID_RE = re.compile(r"^(?!.*://)(?!.*//)[a-zA-Z0-9~][-a-zA-Z0-9._~@/:]*/[-a-zA-Z0-9._~@:]+$")

# Internal qualified default for discovery-level default-model heuristic.
# This is NOT exposed as a public API default — model is always required.
_QUALIFIED_DEFAULT_MODEL = "opencode/deepseek-v4-flash-free"


def _select_default_model(model_ids: list[str]) -> str | None:
    """Pick the catalog default from live-discovered *model_ids*.

    Prefers the configured qualified default whenever it is present in the
    live list, regardless of its row order — the CLI's row order is not a
    qualification signal. If the qualified default is absent, falls back to
    the first discovered model: deterministic, but not a claim that it is
    itself a qualified/supported default.
    """
    if not model_ids:
        return None
    if _QUALIFIED_DEFAULT_MODEL in model_ids:
        return _QUALIFIED_DEFAULT_MODEL
    return model_ids[0]


def parse_models_output(output: str) -> list[str]:
    """Parse ``opencode models`` output into a list of model IDs.

    Each line matching the strict model-ID regex is treated as a model
    identifier.  Non-matching lines (comments, error output) are skipped.
    """
    result: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped and _MODEL_ID_RE.match(stripped):
            result.append(stripped)
    return result


def parse_opencode_verbose_output(output: str) -> ModelCatalog:
    """Parse ``opencode models --verbose`` output into a ModelCatalog.

    The verbose format emits alternating blocks: a model ID line
    (``provider/model``), then a (possibly multi-line pretty-printed)
    JSON object containing ``variants`` (a dict of effort→config).

    Rules:
    - Only lines matching the strict ``provider/model`` regex are treated as
      model IDs.  JSON-internal strings (URLs, paths) are never model IDs.
    - Multi-line JSON is accumulated until balanced ``{}`` braces.
    - Duplicate model IDs are deduplicated in first-seen order.
    - Variant keys are intersected with ``PUBLIC_EFFORTS``; internal-only keys
      (xhigh, turbo, etc.) are excluded from ``supported_efforts``.
    - ``default_effort`` is ``None`` unless the metadata explicitly marks a
      default — dict key order is not a default marker.
    - Malformed / truncated JSON blocks add a catalog-level ``error`` message
      so callers can distinguish partial discovery from clean success.
    """
    lines = output.splitlines()
    seen: set[str] = set()
    model_ids: list[str] = []
    model_info_list: list[ModelInfo] = []
    all_efforts: set[str] = set()
    errors: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Only strict provider/model lines are model IDs.
        if _MODEL_ID_RE.match(line):
            model_id = line
            if model_id in seen:
                i += 1
                continue
            seen.add(model_id)
            model_ids.append(model_id)

            # Accumulate multi-line JSON: find the balanced {} block starting
            # from the next non-blank line.
            i += 1
            json_lines: list[str] = []
            brace_depth = 0
            started = False
            json_start_i = i

            while i < len(lines):
                raw = lines[i]
                stripped = raw.strip()
                # If we haven't started seeing braces yet and this line looks
                # like another model ID, stop accumulating — the next model
                # starts here (no JSON block for the current model).
                if not started and _MODEL_ID_RE.match(stripped):
                    break
                for ch in raw:
                    if ch == "{":
                        brace_depth += 1
                        started = True
                    elif ch == "}":
                        brace_depth -= 1
                json_lines.append(raw)
                i += 1
                if started and brace_depth == 0:
                    break

            if json_lines:
                # Find the first line containing '{' to strip leading blank lines
                first_brace = next((j for j, jl in enumerate(json_lines) if "{" in jl), None)
                if first_brace is not None:
                    json_text = "\n".join(json_lines[first_brace:]).strip()
                else:
                    json_text = "\n".join(json_lines).strip()

                if json_text.startswith("{"):
                    try:
                        data = json.loads(json_text)
                    except json.JSONDecodeError:
                        data = None
                        errors.append(
                            f"Malformed JSON for model '{model_id}' at line {json_start_i + 1}"
                        )
                    if isinstance(data, dict):
                        variants = data.get("variants")
                        if isinstance(variants, dict) and variants:
                            supported = [k for k in variants if k in PUBLIC_EFFORTS]
                            all_efforts.update(supported)
                            model_info_list.append(
                                ModelInfo(
                                    id=model_id,
                                    supported_efforts=tuple(supported),
                                    default_effort=None,
                                )
                            )
                            continue

            # No valid JSON block found → model with empty capability data
            model_info_list.append(
                ModelInfo(
                    id=model_id,
                    supported_efforts=(),
                    default_effort=None,
                )
            )
        else:
            i += 1

    error_msg = "; ".join(errors) if errors else None

    return ModelCatalog(
        models=tuple(model_ids),
        default_model=_select_default_model(model_ids),
        native_efforts=tuple(sorted(all_efforts)),
        source="opencode models --verbose",
        model_info=tuple(model_info_list),
        error=error_msg,
    )


class OpencodeAdapter(StaticAdapter):
    def __init__(self) -> None:
        super().__init__(
            name="opencode",
            support_tier=SUPPORT_TIER,
            backend="acp",
            supports_interactive=False,
            effort_map={"low": "low", "medium": "medium", "high": "high", "max": "max"},
            # OpenCode ACP v1.18.4 advertises the per-model `effort` selector
            # (category `thought_level`) through session config options.
            supports_acp_effort=True,
            # OpenCode ACP also advertises a `mode` selector (category
            # `mode`) with values such as orchestrator/build/plan. "dev"
            # tasks require "build" so edits actually happen — the live
            # session must advertise and accept this value before prompting.
            dev_acp_mode="build",
        )

    def discover_models(self, runner: DiscoveryProcess) -> ModelCatalog:
        """Discover models live from ``opencode models --verbose``.

        Falls back to non-verbose ``opencode models`` when ``--verbose``
        fails, returning a catalog without effort metadata.
        """
        # Try --verbose first for effort metadata
        result = runner.run(["opencode", "models", "--verbose"], timeout=30)
        if result.returncode == 0:
            return parse_opencode_verbose_output(result.stdout)

        # Fallback: non-verbose for model IDs only
        result = runner.run(["opencode", "models"], timeout=30)
        if result.returncode != 0:
            return ModelCatalog(
                models=(),
                default_model=None,
                native_efforts=(),
                source="opencode models",
                error=f"opencode models exited {result.returncode}: {result.stderr[:200]}",
            )
        model_ids = parse_models_output(result.stdout)
        return ModelCatalog(
            models=tuple(model_ids),
            default_model=_select_default_model(model_ids),
            native_efforts=(),
            source="opencode models",
        )

    def validate_effort_for_model(
        self, effort: str, catalog: ModelCatalog | None, model_id: str | None
    ) -> str | None:
        """Return an error message if *effort* is unsupported for *model_id*.

        Returns None when the effort is valid.

        Rejects when:
        - catalog is unavailable (None).
        - catalog carries a discovery error.
        - model_info exists for *model_id* but supported_efforts is empty
          (no advertised variant means effort cannot be proven or mapped).
        - no model_info exists at all for this model (can't validate).
        """
        if catalog is None:
            return "Model catalog unavailable — cannot validate effort"
        if model_id is None:
            return None
        if catalog.error:
            return f"Model discovery error — cannot validate effort: {catalog.error}"
        native = self.map_effort(effort)
        supported = catalog.effort_for_model(model_id)
        # Check whether we have model_info for this model at all
        model_info_exists = any(mi.id == model_id for mi in catalog.model_info)
        if model_info_exists and not supported:
            return (
                f"Model '{model_id}' has no advertised effort variants. "
                f"Effort '{effort}' cannot be validated."
            )
        if not model_info_exists:
            return f"No capability data for model '{model_id}' — cannot validate effort"
        if native not in supported:
            return (
                f"Effort '{effort}' (native '{native}') is not supported "
                f"by model '{model_id}'. Supported: {', '.join(supported)}"
            )
        return None


adapter = OpencodeAdapter()
