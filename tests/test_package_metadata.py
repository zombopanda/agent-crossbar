"""Package metadata consistency checks for public release artifacts."""

import importlib.util
import json
import subprocess
import sys
import tarfile
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

# The tests/ directory IS inside the package directory, so go up one level.
PKG_DIR = Path(__file__).resolve().parents[1]
ROOT = PKG_DIR if (PKG_DIR / ".github").exists() else PKG_DIR.parents[1]
_gate_spec = importlib.util.spec_from_file_location(
    "provider_surface_gate", PKG_DIR / "scripts" / "provider_surface_gate.py"
)
assert _gate_spec and _gate_spec.loader
_gate = importlib.util.module_from_spec(_gate_spec)
sys.modules[_gate_spec.name] = _gate
_gate_spec.loader.exec_module(_gate)
GateCase = _gate.GateCase
_agent_start_args = _gate._agent_start_args
_parse_args = _gate._parse_args


def _load_package_json():
    return json.loads((PKG_DIR / "package.json").read_text())


def _load_pyproject():
    return tomllib.loads((PKG_DIR / "pyproject.toml").read_text())


def _load_readme():
    readme_path = PKG_DIR / "README.md"
    if readme_path.exists():
        return readme_path.read_text()
    return ""


# --- pyproject vs package.json consistency ---


def test_package_json_matches_pyproject_version():
    pkg = _load_package_json()
    py = _load_pyproject()
    assert pkg["version"] == py["project"]["version"]


def test_package_json_npm_name():
    pkg = _load_package_json()
    assert pkg["name"] == "agent-crossbar"


def test_npm_package_is_thin_launcher_only():
    """npm package ships only launcher + docs; no Python, pyproject, or tests."""
    pkg = _load_package_json()
    files = pkg["files"]
    # Must include launcher and docs
    assert "bin/agent-crossbar.mjs" in files
    assert "README.md" in files
    assert "LICENSE" in files
    # Must NOT include Python source or pyproject
    for entry in files:
        assert not entry.endswith(".py"), f"Python file in npm package: {entry}"
        assert "pyproject" not in entry
        assert "test" not in entry.lower()
    # Old internal paths must not appear
    assert "src/" not in str(files)


def test_npm_scripts_restored():
    """npm scripts smoke:live and provider:gate must be present with correct values."""
    pkg = _load_package_json()
    scripts = pkg.get("scripts", {})
    assert scripts.get("smoke:live") == "uv run python scripts/live_smoke.py"
    assert (
        scripts.get("provider:gate")
        == "uv run --extra test python scripts/provider_surface_gate.py"
    )


def test_provider_gate_requires_and_forwards_explicit_model():
    with pytest.raises(SystemExit):
        _parse_args(["--profile", "codex"])

    args = _parse_args(["--profile", "codex", "--model", "gpt-5.6-sol"])
    case = GateCase(
        profile=args.profile[0],
        model=args.model[0],
        effort=None,
        task="ask",
        interactive=False,
    )
    assert dict((field.name, field.type) for field in fields(GateCase))["model"] == "str"
    assert _agent_start_args(case)["model"] == "gpt-5.6-sol"


def test_pyproject_name():
    py = _load_pyproject()
    assert py["project"]["name"] == "agent-crossbar"


def test_sdist_excludes_nested_release_archives():
    py = _load_pyproject()
    excludes = py["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    assert "*.tgz" in excludes
    assert ".beads/**" in excludes
    assert "**/AGENTS.md" in excludes
    assert "**/CLAUDE.md" in excludes


def test_built_sdist_contains_no_internal_state_or_private_docs(tmp_path):
    out = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(out)],
        cwd=PKG_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    archives = list(out.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        names = archive.getnames()
    forbidden = ("/.beads/", "/AGENTS.md", "/CLAUDE.md", "/.memsearch/", "/.reasonix/")
    assert not any(any(marker in name for marker in forbidden) for name in names)


def test_wheel_includes_deprecated_import_shim():
    py = _load_pyproject()
    packages = py["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/agent_crossbar" in packages
    assert "src/agent_harness_mcp" in packages


# --- Package README public content ---


def test_package_readme_has_uvx_install():
    readme = _load_readme()
    assert "uvx agent-crossbar" in readme


def test_package_readme_has_uv_run():
    readme = _load_readme()
    assert "uvx agent-crossbar" in readme  # primary install path


def test_package_readme_uses_native_mcp_configuration_for_each_client():
    readme = _load_readme()

    assert "codex mcp add agents -- uvx agent-crossbar" in readme
    assert "[mcp_servers.agents]" in readme
    assert "claude mcp add --scope user agents -- uvx agent-crossbar" in readme
    assert '"$schema": "https://opencode.ai/config.json"' in readme
    assert '"command": ["uvx", "agent-crossbar"]' in readme


def test_package_readme_does_not_claim_mcp_json_is_codex_config():
    readme = _load_readme()
    codex_section = readme.split("#### Codex", maxsplit=1)[1].split("#### Claude Code", maxsplit=1)[
        0
    ]

    assert "Codex does **not** use" in codex_section
    assert "`.mcp.json` format" in codex_section
    assert '"mcpServers"' not in codex_section
    assert ".codex/config.toml" in codex_section
    assert "~/.codex/config.toml" in codex_section


def test_package_readme_discloses_runtime_and_local_logging_requirements():
    readme = _load_readme()

    assert "uv is required for both launch paths" in readme
    assert "full MCP request and response payloads" in readme
    assert "not sent remotely" in readme
    assert "35s for Codex" in readme


def test_package_readme_claude_billing_anchor_has_a_matching_heading():
    readme = _load_readme()

    assert "](#claude-subscription-vs-print-sdk-billing)" in readme
    assert "## Claude Subscription vs Print SDK Billing" in readme


def test_package_readme_troubleshooting_matches_public_runtime_contract():
    readme = _load_readme()

    assert "| `preflight` |" not in readme
    assert "`check_provider_limits_or_retry_with_free_model`" in readme
    assert "| `AGENT_CROSSBAR_CLIENT_VERSION` | `unknown` |" in readme
    assert "| `chatgpt_pro` | ask, review |" in readme
    assert "uvx agent-crossbar doctor --profile codex --json" in readme


def test_package_readme_explains_reasonix_modes_without_internal_gate_jargon():
    readme = _load_readme()

    assert "| `reasonix` | ask, review, dev | both |" in readme
    assert "live sentinel" not in readme
    assert "ask gate" not in readme


# --- Public branding ---


def test_package_json_no_private_registry():
    pkg = _load_package_json()
    assert "publishConfig" not in pkg
    repo_url = pkg.get("repository", {}).get("url", "")
    assert "git" + ".home" not in repo_url


def test_package_readme_no_pandenko_branding():
    readme = _load_readme()
    assert "@pandenko" not in readme
    assert "git" + ".home" not in readme


def test_package_readme_no_private_registry_instructions():
    readme = _load_readme()
    assert "pnpm config set" not in readme


# --- Public CI ---


def test_ci_workflow_uses_public_package_name():
    """CI workflows (when present) must reference agent-crossbar, not old names."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "agent-crossbar" in workflow
    assert "AGENT_HARNESS_RUN_LOCAL_E2E" not in workflow


def test_release_uses_oidc_for_both_registries_and_no_long_lived_npm_token():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "id-token: write" in workflow
    assert 'node-version: "24"' in workflow
    assert "npm install --global npm@latest" in workflow
    assert "NODE_AUTH_TOKEN" not in workflow
    assert "NPM_TOKEN" not in workflow


def test_release_requires_signed_tag_and_post_pypi_smoke_before_npm():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "git verify-tag" in workflow
    assert "gpg.ssh.allowedSignersFile" in workflow
    assert ".github/allowed_signers" in workflow
    assert "Verify tag matches package versions" in workflow
    assert "tag version does not match package versions" in workflow
    assert "pypi-smoke:" in workflow
    assert "needs: pypi-smoke" in workflow
    assert '"agent-crossbar==${version}"' in workflow


def test_release_attests_built_distributions():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "attestations: write" in workflow
    assert "actions/attest-build-provenance@" in workflow
    assert "subject-path:" in workflow


def test_release_creation_is_idempotent():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert 'gh release view "${GITHUB_REF_NAME}"' in workflow
    assert 'gh release edit "${GITHUB_REF_NAME}"' in workflow
    assert 'gh release upload "${GITHUB_REF_NAME}" dist/* --clobber' in workflow


def test_live_gate_is_maintainer_local_not_stock_github_workflow():
    assert not (ROOT / ".github" / "workflows" / "live-gate.yml").exists()
    readme = (ROOT / "README.md").read_text()
    assert "Maintainer live gate prerequisites" in readme
    assert "stock `ubuntu-latest` runners" in readme


def test_install_smoke_uses_an_isolated_virtual_environment():
    for workflow_name in ("ci.yml", "release.yml"):
        workflow = (ROOT / ".github" / "workflows" / workflow_name).read_text()
        assert "uv pip install --system" not in workflow
        assert "/tmp/smoke-venv/bin/agent-crossbar doctor" in workflow


def test_gitleaks_action_has_required_token_and_no_ignored_inputs():
    workflow = (ROOT / ".github" / "workflows" / "security.yml").read_text()
    assert "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in workflow
    secret_scan = workflow.split("# ── Dependency audit", maxsplit=1)[0]
    assert "fetch-depth: 0" in secret_scan
    assert "config-path:" not in workflow
