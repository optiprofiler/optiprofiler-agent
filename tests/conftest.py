"""Shared pytest fixtures and markers.

* ``requires_matlab`` — skips a test when no MATLAB binary can be resolved.
  Resolution order matches :func:`debugger.matlab_runner.resolve_matlab_bin`:
  explicit ``MATOP_MATLAB_BIN`` → ``matlab`` on ``PATH``.
"""

from __future__ import annotations

import pytest

from optiprofiler_agent.debugger.matlab_runner import is_matlab_available


def pytest_collection_modifyitems(config, items):
    skip_no_matlab = pytest.mark.skip(
        reason="MATLAB not available (set MATOP_MATLAB_BIN or add `matlab` to PATH)",
    )
    have_matlab = is_matlab_available()
    for item in items:
        if "requires_matlab" in item.keywords and not have_matlab:
            item.add_marker(skip_no_matlab)
