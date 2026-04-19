"""Tests for ``optiprofiler_agent.common.quiet_ml``.

Focuses on ``silence_fd``: the fd-level silencer must intercept output that
bypasses Python's ``sys.stdout`` / ``sys.stderr``, while still surfacing
captured output when the wrapped block raises.
"""

from __future__ import annotations

import os
import sys

import pytest

from optiprofiler_agent.common.quiet_ml import silence_fd, silence_stdio


def test_silence_fd_swallows_raw_os_writes(capfd):
    """OS-level writes inside ``silence_fd`` must not reach the terminal."""
    with silence_fd():
        os.write(1, b"NOISE_STDOUT\n")
        os.write(2, b"NOISE_STDERR\n")

    captured = capfd.readouterr()
    assert "NOISE_STDOUT" not in captured.out
    assert "NOISE_STDERR" not in captured.err


def test_silence_fd_also_swallows_python_prints(capfd):
    """Python-level ``print`` is captured by the layered Python redirect."""
    with silence_fd():
        print("python-stdout-noise")
        print("python-stderr-noise", file=sys.stderr)

    captured = capfd.readouterr()
    assert "python-stdout-noise" not in captured.out
    assert "python-stderr-noise" not in captured.err


def test_silence_fd_replays_buffer_on_exception(capfd):
    """When the wrapped block raises, captured output must be re-emitted."""
    with pytest.raises(RuntimeError):
        with silence_fd():
            os.write(2, b"important-failure-context\n")
            raise RuntimeError("boom")

    captured = capfd.readouterr()
    assert "important-failure-context" in captured.err


def test_silence_fd_restores_fds(capfd):
    """After the block exits, normal printing must work again."""
    with silence_fd():
        os.write(1, b"hidden\n")

    print("visible-after")
    captured = capfd.readouterr()
    assert "hidden" not in captured.out
    assert "visible-after" in captured.out


def test_silence_stdio_still_works():
    """Sanity: legacy ``silence_stdio`` continues to swallow Python prints."""
    with silence_stdio():
        print("legacy-noise")
        print("legacy-noise-err", file=sys.stderr)
