"""Tests for the ``opagent init`` interactive wizard.

Covers:
* ``no_interactive=True`` short-circuits without touching stdin
* a builtin-provider flow merges keys into the seed template in place
* the ``custom`` flow writes the four ``OPAGENT_CUSTOM_*`` keys
* the file is written with ``0o600`` so secrets stay user-only
* re-running with ``force=True`` overwrites without prompting
"""

from __future__ import annotations

import io
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPAGENT_HOME", str(tmp_path / "opagent_home"))
    # Wipe any provider key inherited from the dev shell so tests start
    # from a clean slate (otherwise ``has_any_provider_key`` would be
    # truthy and the wizard would prompt for overwrite).
    for key in (
        "MINIMAX_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
        "OPAGENT_CUSTOM_API_KEY", "OPAGENT_DEFAULT_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)

    # Reload runtime modules so the new OPAGENT_HOME wins.
    from importlib import reload
    from optiprofiler_agent.runtime import paths
    reload(paths)
    yield


class _FakeStdin(io.StringIO):
    """``StringIO`` that lies about being a tty so ``run_init`` proceeds."""

    def isatty(self) -> bool:
        return True


def _patch_stdin(monkeypatch, lines: str) -> None:
    monkeypatch.setattr("sys.stdin", _FakeStdin(lines))


def test_no_interactive_skips_cleanly(monkeypatch):
    from optiprofiler_agent import onboarding

    result = onboarding.run_init(force=False, no_interactive=True)
    assert result.skipped
    assert result.written_path is None
    assert result.reason == "no-interactive mode"


def test_non_tty_stdin_skips_cleanly(monkeypatch):
    """Piped / redirected stdin must NOT block on input()."""
    monkeypatch.setattr("sys.stdin", io.StringIO("ignored"))
    from optiprofiler_agent import onboarding

    result = onboarding.run_init(force=False, no_interactive=False)
    assert result.skipped
    assert result.reason == "stdin is not a tty"


def test_builtin_provider_writes_key_and_default(monkeypatch):
    """Picking '1' (minimax) + a key should write both lines in-place."""
    _patch_stdin(monkeypatch, "1\nsk-test-minimax\n")
    from optiprofiler_agent import onboarding
    from optiprofiler_agent.runtime import paths

    result = onboarding.run_init(force=True, no_interactive=False)
    assert not result.skipped
    assert result.provider == "minimax"
    assert result.written_path == paths.env_path()

    text = paths.env_path().read_text(encoding="utf-8")
    assert "MINIMAX_API_KEY=sk-test-minimax" in text
    assert "OPAGENT_DEFAULT_PROVIDER=minimax" in text
    # Original commented hints for OTHER providers must remain intact so
    # the user can still discover them by reading the file.
    assert "# KIMI_API_KEY=" in text
    # The seed template's leading banner should survive.
    assert text.startswith("# ====")


def test_custom_provider_writes_four_keys(monkeypatch):
    """The 6th choice (custom) collects base_url + model + key."""
    _patch_stdin(
        monkeypatch,
        "6\nhttps://api.foo.com/v1\nfoo-model\nsk-foo\n",
    )
    from optiprofiler_agent import onboarding
    from optiprofiler_agent.runtime import paths

    result = onboarding.run_init(force=True, no_interactive=False)
    assert not result.skipped
    assert result.provider == "custom"

    text = paths.env_path().read_text(encoding="utf-8")
    assert "OPAGENT_DEFAULT_PROVIDER=custom" in text
    assert "OPAGENT_CUSTOM_BASE_URL=https://api.foo.com/v1" in text
    assert "OPAGENT_CUSTOM_MODEL=foo-model" in text
    assert "OPAGENT_CUSTOM_API_KEY=sk-foo" in text


def test_secret_file_is_chmod_0600(monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX-only file-mode check")

    _patch_stdin(monkeypatch, "1\nsk-test\n")
    from optiprofiler_agent import onboarding
    from optiprofiler_agent.runtime import paths

    onboarding.run_init(force=True, no_interactive=False)
    mode = paths.env_path().stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"


def test_empty_key_aborts_without_writing(monkeypatch):
    """Empty input for the API key must abort cleanly, not write garbage."""
    _patch_stdin(monkeypatch, "1\n\n")  # provider 1, then empty key
    from optiprofiler_agent import onboarding
    from optiprofiler_agent.runtime import paths

    # Pre-create the seed file so we can confirm it is not mutated.
    from optiprofiler_agent.runtime import bootstrap
    bootstrap.ensure()
    pre = paths.env_path().read_text(encoding="utf-8")

    result = onboarding.run_init(force=True, no_interactive=False)
    assert result.skipped
    assert "Empty" in (result.reason or "")
    post = paths.env_path().read_text(encoding="utf-8")
    assert post == pre, "aborted wizard must not modify the .env file"


def test_has_any_provider_key_detects_env(monkeypatch):
    from optiprofiler_agent import onboarding

    assert not onboarding.has_any_provider_key()
    monkeypatch.setenv("MINIMAX_API_KEY", "x")
    assert onboarding.has_any_provider_key()


def test_detect_configured_providers_lists_every_match(monkeypatch):
    """Multiple keys should all show up; ordering follows registry order."""
    from optiprofiler_agent import onboarding

    assert onboarding.detect_configured_providers() == []
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    # PROVIDER_REGISTRY iteration order is kimi → minimax → ... so detection
    # should mirror that order regardless of which env var was set first.
    result = onboarding.detect_configured_providers()
    assert result[:2] == ["kimi", "minimax"]


def test_active_default_provider_prefers_explicit_when_key_exists(monkeypatch):
    """OPAGENT_DEFAULT_PROVIDER wins iff its env_key is also set."""
    from optiprofiler_agent import onboarding

    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("OPAGENT_DEFAULT_PROVIDER", "kimi")
    # Explicit default points at kimi but no KIMI_API_KEY → should fall back
    # to whatever IS configured rather than report a broken default.
    assert onboarding.active_default_provider() == "minimax"

    monkeypatch.setenv("KIMI_API_KEY", "k")
    assert onboarding.active_default_provider() == "kimi"


def test_active_default_provider_returns_none_when_empty(monkeypatch):
    from optiprofiler_agent import onboarding

    assert onboarding.active_default_provider() is None


def test_existing_key_prompts_specific_provider_name(monkeypatch, capsys):
    """Regression: the 'provider unknown' bug from terminals/1.txt:648."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-already-here")
    # User declines the overwrite — we just want to confirm the message
    # names 'minimax' explicitly rather than the legacy 'unknown'.
    _patch_stdin(monkeypatch, "n\n")
    from optiprofiler_agent import onboarding

    result = onboarding.run_init(force=False, no_interactive=False)
    assert result.skipped
    assert result.reason == "user declined overwrite"
    captured = capsys.readouterr().out
    assert "minimax" in captured
    assert "unknown" not in captured
