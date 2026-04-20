"""Behavioural tests for the CLI's auto-init hook.

Confirms that ``_maybe_run_first_time_init`` does the right thing across
the three relevant axes:

1. ``OPAGENT_NO_AUTO_INIT`` opts out completely (CI / Docker scenario).
2. Subcommands that don't need a key (``init``, ``home``, ``wiki`` etc.)
   are NEVER blocked by the hook.
3. Non-tty stdin degrades to a friendly warning, never blocks on input().
4. Having a key already configured short-circuits the hook silently.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPAGENT_HOME", str(tmp_path / "opagent_home"))
    for k in (
        "MINIMAX_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
        "OPAGENT_CUSTOM_API_KEY", "OPAGENT_DEFAULT_PROVIDER",
        "OPAGENT_NO_AUTO_INIT",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def _make_ctx(invoked: str | None = None) -> MagicMock:
    """Mimic just enough of click.Context for the hook."""
    ctx = MagicMock()
    ctx.invoked_subcommand = invoked
    ctx.exit = MagicMock(side_effect=SystemExit)
    return ctx


def test_no_auto_init_env_var_disables_hook(monkeypatch):
    monkeypatch.setenv("OPAGENT_NO_AUTO_INIT", "1")
    from optiprofiler_agent import cli, onboarding

    called = MagicMock()
    monkeypatch.setattr(onboarding, "run_init", called)

    cli._maybe_run_first_time_init(_make_ctx(invoked="agent"))
    called.assert_not_called()


@pytest.mark.parametrize("subcmd", sorted(["init", "wiki", "memory", "session", "home", "skills", "index", "check"]))
def test_no_key_required_subcommands_skip_hook(monkeypatch, subcmd):
    """Commands that don't need an LLM never trigger the wizard."""
    from optiprofiler_agent import cli, onboarding

    called = MagicMock()
    monkeypatch.setattr(onboarding, "run_init", called)

    cli._maybe_run_first_time_init(_make_ctx(invoked=subcmd))
    called.assert_not_called()


def test_existing_key_short_circuits_hook(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-already-here")
    from optiprofiler_agent import cli, onboarding

    called = MagicMock()
    monkeypatch.setattr(onboarding, "run_init", called)

    cli._maybe_run_first_time_init(_make_ctx(invoked="agent"))
    called.assert_not_called()


def test_non_tty_prints_friendly_warning_without_blocking(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    from optiprofiler_agent import cli, onboarding

    called = MagicMock()
    monkeypatch.setattr(onboarding, "run_init", called)

    captured: list[str] = []
    monkeypatch.setattr(cli.console, "print", lambda *a, **kw: captured.append(" ".join(str(x) for x in a)))

    cli._maybe_run_first_time_init(_make_ctx(invoked="agent"))
    called.assert_not_called()  # never block on input()
    assert any("opagent init" in line for line in captured)


def test_tty_with_no_key_runs_wizard(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    from optiprofiler_agent import cli, onboarding

    fake_result = MagicMock(skipped=False, written_path=MagicMock(__fspath__=lambda: "/tmp/x"), provider="kimi", reason=None)
    fake_run = MagicMock(return_value=fake_result)
    monkeypatch.setattr(onboarding, "run_init", fake_run)

    cli._maybe_run_first_time_init(_make_ctx(invoked="agent"))
    fake_run.assert_called_once_with(force=False, no_interactive=False)
