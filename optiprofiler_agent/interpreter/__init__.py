"""Interpreter agent — turns benchmark results into structured reports.

Re-exports the public entry points so callers can do::

    from optiprofiler_agent.interpreter import interpret, generate_report_object

instead of the redundant ``optiprofiler_agent.interpreter.interpreter`` path.
"""

from optiprofiler_agent.interpreter.interpreter import (
    MAX_REPORT_RETRIES,
    generate_report_object,
    interpret,
    interpret_from_summary,
)

__all__ = [
    "MAX_REPORT_RETRIES",
    "generate_report_object",
    "interpret",
    "interpret_from_summary",
]
