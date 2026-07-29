"""Normalize and validate start requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_crossbar.models import Autonomy, Operation, Sensitivity, Transport
from agent_crossbar.profiles import (
    CODEX_DEFAULT_EFFORT,
    CODEX_EFFORT_ALIASES,
    CODEX_EFFORTS,
    allowed_models,
    profile_interactive,
    profile_operations,
    resolve_profile,
)

_REQUIRED_FIELDS = (
    "operation",
    "profile",
    "transport",
    "autonomy",
    "sensitivity",
)


def validate_start_request(
    req: dict[str, Any], state_root: Path | str | None = None
) -> dict[str, Any]:
    """Validate a start request dict and return a result dict.

    The result dict always contains:
      - ok: bool
      - error: str | None
      - message: str
      - warnings: list[str]
      - job_created: bool

    An unknown profile is rejected with error=="invalid_profile" and
    job_created==False — no job directory is created.
    """
    warnings: list[str] = []

    # Check required fields
    for field_name in _REQUIRED_FIELDS:
        if field_name not in req or req[field_name] is None:
            return {
                "ok": False,
                "error": "missing_required_field",
                "message": f"Required field '{field_name}' is missing",
                "warnings": warnings,
                "job_created": False,
            }

    def fail(error: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "error": error,
            "message": message,
            "warnings": warnings,
            "job_created": False,
        }

    # Model is required — no default model fallback for any profile.
    model_raw = req.get("model")
    if model_raw is None or not str(model_raw).strip():
        return fail("missing_model", "model is required for every agent_start invocation")

    req["model"] = str(model_raw).strip()

    # Resolve profile (aliases -> canonical)
    profile_raw: str = req["profile"]
    ok, resolved = resolve_profile(profile_raw)
    if not ok:
        return fail("invalid_profile", f"Unknown profile '{profile_raw}'")

    # Validate operation
    try:
        Operation(req["operation"])
    except ValueError:
        return fail("invalid_operation", f"Unknown operation '{req['operation']}'")

    # Validate transport
    try:
        Transport(req["transport"])
    except ValueError:
        return fail("invalid_transport", f"Unknown transport '{req['transport']}'")

    # Validate autonomy
    try:
        Autonomy(req["autonomy"])
    except ValueError:
        return fail("invalid_autonomy", f"Unknown autonomy '{req['autonomy']}'")

    # Validate sensitivity
    try:
        Sensitivity(req["sensitivity"])
    except ValueError:
        return fail("invalid_sensitivity", f"Unknown sensitivity '{req['sensitivity']}'")

    transport_val = Transport(req["transport"])
    operation_val = Operation(req["operation"])

    # Rule: operation must be supported by the profile.
    supported_ops = profile_operations(resolved)
    if operation_val.value not in supported_ops:
        return fail(
            "unsupported_operation",
            f"Profile '{resolved}' does not support operation '{operation_val.value}'",
        )

    if operation_val == Operation.ADVICE and not str(req.get("prompt", "")).strip():
        return fail("missing_required_field", "prompt is required for advice operations")

    # Rule: tmux transport requires interactive support.
    if transport_val == Transport.TMUX and not profile_interactive(resolved):
        return fail(
            "unsupported_transport",
            f"Profile '{resolved}' does not support interactive tmux transport",
        )

    normalized_model: str | None = None
    normalized_effort: str | None = None
    model = req["model"]
    models = allowed_models(resolved)

    # Validate model against the profile's known allowlist.
    # Profiles with no model allowlist (e.g. chatgpt_pro) skip this check —
    # model is still required but the value is accepted as-is.
    if resolved not in {"codex", "opencode", "claude"} and models and model not in models:
        return fail("invalid_model", f"Model '{model}' is not supported for profile '{resolved}'")
    normalized_model = model

    if resolved == "codex":
        if state_root is None:
            return fail("discovery_error", "state_root is required for Codex model discovery")

        from agent_crossbar.adapters.codex import adapter as codex_adapter
        from agent_crossbar.discovery import discover_profile_models

        sr = Path(state_root) if not isinstance(state_root, Path) else state_root
        try:
            catalog = discover_profile_models(sr, "codex")
        except Exception as exc:
            return fail("discovery_error", f"Codex model discovery failed: {exc}")

        if catalog.error:
            return fail("discovery_error", f"Codex model discovery failed: {catalog.error}")
        if not catalog.models:
            return fail("discovery_error", "No Codex models discovered")
        if model not in catalog.models:
            return fail(
                "invalid_model",
                f"Model '{model}' is not available in Codex "
                f"(discovered: {', '.join(catalog.models)})",
            )

        effort = req.get("effort") or CODEX_DEFAULT_EFFORT
        effort = CODEX_EFFORT_ALIASES.get(effort, effort)
        if effort not in CODEX_EFFORTS:
            return fail(
                "invalid_effort", f"Effort '{effort}' is not supported for profile '{resolved}'"
            )
        req["effort"] = effort
        normalized_effort = effort

        # Rule: Codex effort must be supported by the selected model's discovered capabilities.
        err = codex_adapter.validate_effort_for_model(effort, catalog, model)
        if err is not None:
            return fail("unsupported_effort_for_model", err)

    # Rule: OpenCode validates against the live model catalog obtained via
    # discover_profile_models — which uses the cache when fresh and performs
    # a bounded live refresh only when missing or stale.  There is no static
    # allowlist fallback; a missing/error/empty catalog fails preflight.
    if resolved == "opencode":
        if state_root is None:
            return fail("discovery_error", "state_root is required for OpenCode model discovery")

        from agent_crossbar.adapters.opencode import adapter as opencode_adapter
        from agent_crossbar.discovery import discover_profile_models

        sr = Path(state_root) if not isinstance(state_root, Path) else state_root

        try:
            catalog = discover_profile_models(sr, "opencode")
        except Exception as exc:
            return fail("discovery_error", f"OpenCode model discovery failed: {exc}")

        if catalog.error:
            return fail("discovery_error", f"OpenCode model discovery failed: {catalog.error}")

        if not catalog.models:
            return fail("discovery_error", "No OpenCode models discovered")

        # Match the required model against the live catalog. OpenCode's model
        # list is dynamic, so no static allowlist may reject it first.
        catalog_model_ids = set(catalog.models)
        if model in catalog_model_ids:
            # Model already matches a catalog entry — use as-is
            pass
        else:
            # Try suffix match: short name matches the part after
            # the last slash in a catalog model ID.
            matched = None
            for cid in catalog.models:
                if "/" in cid and cid.split("/", 1)[1] == model:
                    matched = cid
                    break
            if matched is not None:
                model = matched
                normalized_model = model
                req["model"] = model
            else:
                return fail(
                    "invalid_model",
                    f"Model '{model}' is not available in OpenCode "
                    f"(discovered: {', '.join(catalog.models)})",
                )

        # Validate effort against per-model discovery data
        effort = req.get("effort")
        if effort is not None:
            err = opencode_adapter.validate_effort_for_model(effort, catalog, model)
            if err is not None:
                return fail("unsupported_effort_for_model", err)
            normalized_effort = effort

    # Claude's public model values are live, versioned CLI IDs parsed from the
    # /model picker. Do not accept mutable family aliases or static fallbacks.
    if resolved == "claude":
        if state_root is None:
            return fail("discovery_error", "state_root is required for Claude model discovery")

        from agent_crossbar.discovery import discover_profile_models

        sr = Path(state_root) if not isinstance(state_root, Path) else state_root
        try:
            catalog = discover_profile_models(sr, "claude")
        except Exception as exc:
            return fail("discovery_error", f"Claude model discovery failed: {exc}")

        if catalog.error:
            return fail("discovery_error", f"Claude model discovery failed: {catalog.error}")
        if not catalog.models:
            return fail("discovery_error", "No Claude models discovered")
        if model not in catalog.models:
            return fail(
                "invalid_model",
                f"Model '{model}' is not available in Claude "
                f"(discovered: {', '.join(catalog.models)})",
            )

    return {
        "ok": True,
        "error": None,
        "message": "Validation passed",
        "warnings": warnings,
        "job_created": True,
        "profile": resolved,
        "operation": req["operation"],
        "model": normalized_model,
        "effort": normalized_effort,
    }
