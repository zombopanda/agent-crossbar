"""Tests for centralized secret redaction — closing gaps from review feedback.

RED step: Authorization Bearer values are not redacted at all today, and
KEY=VALUE redaction leaves a double-bracket artifact when two patterns
both match the same substitution.
"""

from __future__ import annotations

from agent_crossbar.redaction import redact_secrets

SECRET = "sk-abcdef1234567890"


class TestBearerTokenRedaction:
    def test_authorization_bearer_header_redacted(self):
        text = f"Authorization: Bearer {SECRET}"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted
        assert "Bearer [REDACTED]" in redacted

    def test_bearer_token_inside_quoted_curl_command_redacted(self):
        text = f"curl -H 'Authorization: Bearer {SECRET}'"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted
        # the closing quote must survive — only the token value is stripped
        assert redacted.endswith("'")

    def test_lowercase_bearer_redacted(self):
        text = f"authorization: bearer {SECRET}"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted


class TestUrlQueryStringSecretRedaction:
    def test_api_key_query_param_redacted_other_params_preserved(self):
        text = f"https://api.example.com/v1/data?api_key={SECRET}&other=1"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted
        assert "other=1" in redacted

    def test_access_token_query_param_redacted_other_params_preserved(self):
        text = f"https://api.example.com/oauth?access_token={SECRET}&scope=read"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted
        assert "scope=read" in redacted

    def test_token_query_param_redacted(self):
        text = f"https://x.example.com?token={SECRET}"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted


class TestOrdinaryKeyValueSecretRedaction:
    def test_env_style_line_redacted_without_double_bracket_artifact(self):
        text = f"DB_PASSWORD={SECRET}"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted
        assert redacted == "DB_PASSWORD=[REDACTED]"

    def test_secret_key_line_redacted_without_double_bracket_artifact(self):
        text = f"MY_SECRET_KEY={SECRET}"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted
        assert redacted == "MY_SECRET_KEY=[REDACTED]"

    def test_inline_key_value_mid_sentence_redacted(self):
        text = f"config: TOKEN={SECRET} done"
        redacted, changed = redact_secrets(text)
        assert changed is True
        assert SECRET not in redacted
        assert redacted == "config: TOKEN=[REDACTED] done"


class TestRedactionIdempotency:
    def test_redacting_already_redacted_text_does_not_double_wrap(self):
        text = f"DB_PASSWORD={SECRET}"
        once, _ = redact_secrets(text)
        twice, changed_again = redact_secrets(once)
        assert twice == once
        assert changed_again is False
