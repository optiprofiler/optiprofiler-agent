"""Advisor agent — answers OptiProfiler usage questions and generates scripts.

Re-exports the main entry point so callers can do::

    from optiprofiler_agent.advisor import AdvisorAgent

instead of the redundant ``optiprofiler_agent.advisor.advisor`` path.
"""

from optiprofiler_agent.advisor.advisor import AdvisorAgent

__all__ = ["AdvisorAgent"]
