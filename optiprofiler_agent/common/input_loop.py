"""prompt_toolkit-based interactive prompt for the OPAGENT CLI.

We use ``prompt_toolkit`` instead of ``readline`` for one concrete reason:
``readline`` (and especially macOS ``libedit``) cannot reliably measure the
visible width of a Rich-rendered prompt that contains ANSI escape codes.
The result is that pressing ``Backspace`` early in the line can erase the
``You:`` label itself. ``prompt_toolkit`` owns the rendering loop so the
prompt is *immutable* — backspaces stop at the input column.

Adopting ``prompt_toolkit`` also matches the input layer used by
Claude Code, Cursor's terminal agent, and Aider, which gives us a
clear upgrade path later (multi-line edit, fuzzy history, completer
plugins) without rewriting the input layer again.

Public surface kept tiny on purpose:

- :func:`prompt`  — synchronous read of one line with optional history.
- :func:`make_session` — build a long-lived ``PromptSession``; reuse to
  preserve in-memory + on-disk history between calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory, InMemoryHistory


def _resolve_history_path(label: str) -> Optional[Path]:
    """Return the on-disk history file under ``OPAGENT_HOME``, or ``None``
    if the runtime directory is not yet usable (e.g. in pure-unit-test
    environments where bootstrap was skipped)."""
    try:
        from optiprofiler_agent.runtime import paths as _rt_paths
        hist_dir = _rt_paths.home() / "history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        return hist_dir / f"{label}.txt"
    except Exception:
        return None


def make_session(label: str = "default") -> PromptSession:
    """Build a ``PromptSession`` with persistent history (file-backed
    if ``OPAGENT_HOME`` is reachable, else in-memory)."""
    hist_path = _resolve_history_path(label)
    history = FileHistory(str(hist_path)) if hist_path else InMemoryHistory()
    return PromptSession(history=history)


def prompt(message_ansi: str, session: Optional[PromptSession] = None) -> str:
    """Read one line with arrow keys, history (↑/↓), Ctrl-A/E.

    Args:
        message_ansi: Pre-rendered ANSI string for the prompt label
            (e.g. produced by Rich's ``console.render(...)``). The
            visible width is computed by ``prompt_toolkit`` so the
            prompt itself can never be edited or deleted by the user.
        session: A reusable :class:`PromptSession`. If ``None``, a
            throwaway session with in-memory history is created.

    Raises ``EOFError`` on Ctrl-D and ``KeyboardInterrupt`` on Ctrl-C,
    matching the semantics of the previous ``console.input`` callsites.
    """
    sess = session or PromptSession(history=InMemoryHistory())
    return sess.prompt(ANSI(message_ansi))
