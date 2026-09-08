"""Normalize and validate start requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_crossbar.adapters.registry import get_adapter
from agent_crossbar.models import Autonomy, Operation, Sensitivity, Transport
from agent_crossbar.profiles import (
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
    resolved_effort: str | None = None
    model = req["model"]
    models = allowed_models(resolved)
    adapter = get_adapter(resolved)

    if not adapter.live_model_discovery:
        # Profiles with no live model discovery (e.g. chatgpt_pro) validate
        # against a static allowlist when one is declared; model is still
        # required but is otherwise accepted as-is.
        if models and model not in models:
            return fail(
                "invalid_model", f"Model '{model}' is not supported for profile '{resolved}'"
            )
        normalized_model = model
    else:
        # Live-discovery profiles (Claude, Codex, OpenCode, Reasonix) never
        # fall back to a static allowlist — a missing/error/empty catalog
        # fails preflight closed. Model resolution (exact or qualified-suffix)
        # and effort resolution/validation are adapter-owned.
        if state_root is None:
            return fail("discovery_error", f"state_root is required for {resolved} model discovery")

        from agent_crossbar.discovery import discover_profile_models

        sr = Path(state_root) if not isinstance(state_root, Path) else state_root
        try:
            catalog = discover_profile_models(sr, resolved)
        except Exception as exc:
            return fail("discovery_error", f"{resolved} model discovery failed: {exc}")

        if catalog.error:
            return fail("discovery_error", f"{resolved} model discovery failed: {catalog.error}")
        if not catalog.models:
            return fail("discovery_error", f"No {resolved} models discovered")

        resolved_model, model_error = adapter.resolve_model_id(model, catalog)
        if model_error is not None:
            return fail(*model_error)
        model = resolved_model
        normalized_model = model
        req["model"] = model

        normalized_effort, resolved_effort, effort_error = adapter.validate_effort(
            req.get("effort"), catalog, model
        )
        if effort_error is not None:
            return fail(*effort_error)
        if normalized_effort is not None:
            req["effort"] = normalized_effort

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
        "resolved_effort": resolved_effort,
    }
