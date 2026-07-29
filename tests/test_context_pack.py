"""Deterministic, bounded, safe context packing from generic cwd/scope."""

from __future__ import annotations

from agent_crossbar.context import MAX_ATTACHMENT_BYTES, pack_context


def build_repo(tmp_path):
    (tmp_path / "README.md").write_text("readme body", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('a')", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("print('b')", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("junk", encoding="utf-8")
    (tmp_path / "src" / "image.png").write_bytes(b"\x89PNG")
    return tmp_path


def test_no_scope_requests_no_context(tmp_path):
    packed = pack_context(tmp_path, None)

    assert packed["ok"] is True
    assert packed["text"] == ""
    assert packed["summary"]["requested"] is False


def test_packs_deterministically_with_priority_ordering(tmp_path):
    root = build_repo(tmp_path)

    packed = pack_context(root, {"paths": [".", "README.md"]})

    assert packed["ok"] is True
    assert packed["text"].startswith("BEGIN_AGENTS_MCP_CONTEXT")
    assert packed["text"].rstrip().endswith("END_AGENTS_MCP_CONTEXT")
    order = [line for line in packed["text"].splitlines() if line.startswith("--- ")]
    assert order == ["--- README.md ---", "--- src/a.py ---", "--- src/b.py ---"]
    assert packed["summary"]["files_included"] == 3
    assert packed["summary"]["chars_used"] > 0
    assert (
        pack_context(root, {"paths": ["."]})["text"] == pack_context(root, {"paths": ["."]})["text"]
    )


def test_generated_directories_and_binaries_are_omitted(tmp_path):
    root = build_repo(tmp_path)

    packed = pack_context(root, {"paths": ["."]})

    assert "junk" not in packed["text"]
    reasons = {entry["reason"] for entry in packed["summary"]["omissions"]}
    assert "generated_directory" in reasons
    assert "not_text" in reasons


def test_missing_path_fails_before_any_provider_work(tmp_path):
    packed = pack_context(tmp_path, {"paths": ["nope"]})

    assert packed["ok"] is False
    assert packed["error"] == "context_path_missing"


def test_symlinked_scope_path_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "link").symlink_to(outside)

    packed = pack_context(root, {"paths": ["link"]})

    assert packed["ok"] is False
    assert packed["error"] == "context_path_symlink"


def test_symlinks_inside_the_tree_are_skipped(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("SECRET-CONTENT", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "keep.md").write_text("kept", encoding="utf-8")
    (root / "sneaky.md").symlink_to(outside / "secret.md")

    packed = pack_context(root, {"paths": ["."]})

    assert "SECRET-CONTENT" not in packed["text"]
    assert "kept" in packed["text"]
    assert any(entry["reason"] == "symlink" for entry in packed["summary"]["omissions"])


def test_escaping_path_is_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "other.md").write_text("x", encoding="utf-8")

    packed = pack_context(root, {"paths": ["../other.md"]})

    assert packed["ok"] is False
    assert packed["error"] in {"context_path_missing", "context_path_outside_cwd"}


def test_large_file_is_chunked_within_budget(tmp_path):
    (tmp_path / "big.md").write_text("x" * 5000, encoding="utf-8")

    packed = pack_context(tmp_path, {"paths": ["big.md"], "max_file_chars": 500})

    assert packed["ok"] is True
    assert "[... truncated ...]" in packed["text"]
    assert packed["summary"]["chars_used"] < 5000
    assert any(entry["reason"] == "chunked_file_budget" for entry in packed["summary"]["omissions"])


def test_total_budget_stops_inclusion(tmp_path):
    for index in range(5):
        (tmp_path / f"f{index}.md").write_text("y" * 400, encoding="utf-8")

    packed = pack_context(tmp_path, {"paths": ["."], "max_chars": 600})

    assert packed["summary"]["files_included"] < 5
    assert any(
        entry["reason"] in {"total_budget_exhausted", "chunked_file_budget"}
        for entry in packed["summary"]["omissions"]
    )


def test_secrets_in_context_are_redacted(tmp_path):
    (tmp_path / ".env.sample").write_text("API_KEY=supersecretvalue\n", encoding="utf-8")
    (tmp_path / "conf.toml").write_text('token = "supersecretvalue"\n', encoding="utf-8")

    packed = pack_context(tmp_path, {"paths": ["conf.toml"]})

    assert "supersecretvalue" not in packed["text"]
    assert "[REDACTED]" in packed["text"]


def test_attachment_must_exist_and_fit_the_budget(tmp_path, monkeypatch):
    missing = pack_context(tmp_path, {"attachments": ["nope.png"]})
    assert missing["ok"] is False
    assert missing["error"] == "attachment_missing"

    payload = tmp_path / "small.png"
    payload.write_bytes(b"\x89PNG" * 10)
    ok = pack_context(tmp_path, {"attachments": ["small.png"]})
    assert ok["ok"] is True
    assert ok["attachments"][0]["bytes"] == payload.stat().st_size

    monkeypatch.setattr("agent_crossbar.context.MAX_ATTACHMENT_BYTES", 4)
    too_big = pack_context(tmp_path, {"attachments": ["small.png"]})
    assert too_big["ok"] is False
    assert too_big["error"] == "attachment_too_large"
    assert MAX_ATTACHMENT_BYTES > 0


def test_invalid_scope_shapes_are_rejected(tmp_path):
    assert pack_context(tmp_path, {"paths": [""]})["error"] == "invalid_scope"
    assert pack_context(tmp_path, {"paths": 5})["error"] == "invalid_scope"
    assert pack_context(tmp_path, {"attachments": 5})["error"] == "invalid_scope"
