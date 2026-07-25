"""Secret redaction helpers for safe context bundles."""

from __future__ import annotations

import re

# Patterns that indicate secret/credential material in text.
# These are used to detect and redact sensitive values from context.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Authorization: Bearer <token> (HTTP auth headers, curl commands)
    (
        re.compile(r"(Bearer\s+)[^\s'\"]+", re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    # KEY=VALUE patterns (e.g., .env files)
    (
        re.compile(
            r"^([A-Z_]*(?:SECRET|PASSWORD|TOKEN|API_KEY|PRIVATE|CREDENTIAL|ACCESS_KEY|SECRET_KEY)[A-Z_]*)="
            r"(?!\[REDACTED\]$).*$",
            re.MULTILINE | re.IGNORECASE,
        ),
        r"\1=[REDACTED]",
    ),
    # Generic assignment / URL query-string secrets (e.g. ?api_key=..., JSON
    # "token": "..."). Stops at "&" so sibling query params survive, and
    # skips values that are already the redaction marker so re-running this
    # pass over previously-redacted text stays idempotent.
    (
        re.compile(
            r"(\"?(?:secret|password|token|api_key|private_key|credential)[\"']?\s*[:=]\s*)"
            r"(?!\[REDACTED\])[^\s,}\]&]+",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]",
    ),
]


def redact_secrets(text: str) -> tuple[str, bool]:
    """Redact secret patterns from *text*.

    Returns (redacted_text, was_redacted) where *was_redacted* is True if any
    substitution actually occurred.
    """
    redacted = text
    changed = False
    for pattern, replacement in _SECRET_PATTERNS:
        new_text, count = pattern.subn(replacement, redacted)
        if count > 0:
            redacted = new_text
            changed = True
    return redacted, changed


def has_secret_markers(text: str) -> bool:
    """Return True if *text* appears to contain secret/credential material."""
    _, changed = redact_secrets(text)
    return changed


def _stable_redacted_path(original_path: str) -> str:
    """Produce a stable opaque label for a denied/redacted path."""
    import hashlib

    digest = hashlib.sha256(original_path.encode("utf-8")).hexdigest()[:12]
    return f"[REDACTED:{digest}]"
