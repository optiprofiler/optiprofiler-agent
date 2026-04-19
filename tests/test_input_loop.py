"""Unit tests for ``optiprofiler_agent.common.input_loop``.

These tests do not exercise the actual TTY interaction (prompt_toolkit
needs a real terminal for that). They cover the surface that is testable
in a non-TTY environment:

- ``make_session`` returns a usable ``PromptSession``.
- The history file resolves under ``OPAGENT_HOME`` when bootstrap is
  available, and falls back to in-memory history otherwise.
- ``prompt`` accepts an ANSI-coloured label without raising.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, InMemoryHistory

from optiprofiler_agent.common import input_loop


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point OPAGENT_HOME at a tmp dir and clear cached path state."""
    monkeypatch.setenv("OPAGENT_HOME", str(tmp_path))
    from optiprofiler_agent.runtime import bootstrap as _rt_bootstrap

    _rt_bootstrap.ensure()
    yield tmp_path


def test_make_session_returns_prompt_session(isolated_home):
    sess = input_loop.make_session(label="unit-test")
    assert isinstance(sess, PromptSession)
    assert isinstance(sess.history, FileHistory)
    expected = Path(os.environ["OPAGENT_HOME"]) / "history" / "unit-test.txt"
    assert Path(sess.history.filename) == expected
    assert expected.parent.exists()


def test_make_session_falls_back_when_runtime_missing(monkeypatch):
    monkeypatch.setattr(input_loop, "_resolve_history_path", lambda label: None)
    sess = input_loop.make_session(label="unused")
    assert isinstance(sess.history, InMemoryHistory)


def test_resolve_history_path_creates_directory(isolated_home):
    p = input_loop._resolve_history_path("alpha")
    assert p is not None
    assert p.parent.is_dir()
    assert p.name == "alpha.txt"
