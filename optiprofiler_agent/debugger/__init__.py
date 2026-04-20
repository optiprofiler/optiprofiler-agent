"""Debugger agent — diagnoses and auto-fixes failing benchmark scripts.

Re-exports the public entry points so callers can do::

    from optiprofiler_agent.debugger import debug_script, run_and_debug

instead of the redundant ``optiprofiler_agent.debugger.debugger`` path.
"""

from optiprofiler_agent.debugger.debugger import (
    DebugResult,
    debug_script,
    run_and_debug,
)

__all__ = ["DebugResult", "debug_script", "run_and_debug"]
