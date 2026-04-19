"""Business-invariant validator for ``BenchmarkReport``.

JSON Schema (enforced by ``with_structured_output``) covers types and
enum values. This module covers the cross-field invariants that a
schema cannot express:

- Solver names referenced anywhere in the report MUST exist in the
  experiment's ``summary.solver_names``.
- Per-solver failure counts MUST NOT exceed the actual count from
  the rule engine.
- Common-failure-problem names MUST be a subset of the names actually
  observed in the input.

Errors here trigger a single retry through the existing lint-loop
infrastructure (mirrors ``validators/lint_loop.py`` for code blocks).
Warnings are surfaced to the user but never fed back to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from optiprofiler_agent.agent_c.report_schema import BenchmarkReport
from optiprofiler_agent.agent_c.summary import BenchmarkSummary


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ReportIssue:
    """One business-invariant violation."""

    severity: Severity
    field_path: str
    message: str

    def format(self) -> str:
        prefix = "[error]" if self.severity == "error" else "[warning]"
        return f"{prefix} {self.field_path}: {self.message}"


@dataclass
class ReportValidationResult:
    """Aggregated validation result for one report."""

    issues: list[ReportIssue]

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return not self.issues


def _collect_failure_counts(summary: BenchmarkSummary) -> dict[str, int]:
    """Best-effort upper-bound on per-solver failure counts.

    Uses summary.failure_patterns when available; falls back to 0
    (no constraint) so that legacy summaries do not produce false
    positives.
    """
    counts: dict[str, int] = {name: 0 for name in summary.solver_names}
    for entry in summary.failure_patterns or []:
        solver = entry.get("solver") if isinstance(entry, dict) else None
        if solver in counts:
            counts[solver] = max(
                counts[solver],
                int(entry.get("failure_count", 0) or 0),
            )
    return counts


def _collect_known_problem_names(summary: BenchmarkSummary) -> set[str]:
    """Collect the set of problem names that appear anywhere in the
    summary's per-problem fields. Empty set means no constraint."""
    names: set[str] = set()
    for entry in summary.failure_patterns or []:
        if isinstance(entry, dict):
            for prob in entry.get("problems", []) or []:
                if isinstance(prob, str):
                    names.add(prob)
    for entry in summary.anomalies or []:
        if isinstance(entry, dict):
            prob = entry.get("problem")
            if isinstance(prob, str):
                names.add(prob)
    return names


def validate_report(
    report: BenchmarkReport,
    summary: BenchmarkSummary,
) -> ReportValidationResult:
    """Check report against the experiment summary."""
    solvers = set(summary.solver_names)
    failure_caps = _collect_failure_counts(summary)
    known_problems = _collect_known_problem_names(summary)

    issues: list[ReportIssue] = []

    def _check_solver(name: str, path: str, severity: Severity = "error") -> None:
        if name not in solvers:
            issues.append(
                ReportIssue(
                    severity=severity,
                    field_path=path,
                    message=(
                        f"unknown solver '{name}'; valid solvers: "
                        f"{sorted(solvers) or '[]'}"
                    ),
                )
            )

    _check_solver(
        report.performance_profile.winner_at_tau1,
        "performance_profile.winner_at_tau1",
    )
    _check_solver(
        report.performance_profile.most_robust,
        "performance_profile.most_robust",
    )
    _check_solver(
        report.data_profile.most_efficient,
        "data_profile.most_efficient",
    )

    for idx, entry in enumerate(report.convergence_issues.entries):
        path = f"convergence_issues.entries[{idx}]"
        _check_solver(entry.solver, f"{path}.solver")
        cap = failure_caps.get(entry.solver)
        if cap is not None and cap > 0 and entry.failure_count > cap:
            issues.append(
                ReportIssue(
                    severity="warning",
                    field_path=f"{path}.failure_count",
                    message=(
                        f"reported {entry.failure_count} failures for "
                        f"'{entry.solver}' but summary has at most {cap}"
                    ),
                )
            )

    if known_problems:
        for idx, name in enumerate(report.convergence_issues.common_failure_problems):
            if name not in known_problems:
                issues.append(
                    ReportIssue(
                        severity="warning",
                        field_path=f"convergence_issues.common_failure_problems[{idx}]",
                        message=(
                            f"problem '{name}' not present in the input "
                            "summary's failure_patterns"
                        ),
                    )
                )

    for idx, entry in enumerate(report.anomalies.entries):
        path = f"anomalies.entries[{idx}]"
        for jdx, name in enumerate(entry.affected_solvers):
            _check_solver(name, f"{path}.affected_solvers[{jdx}]")

    for idx, action in enumerate(report.recommendations.actions):
        if action.target_solver is not None:
            _check_solver(
                action.target_solver,
                f"recommendations.actions[{idx}].target_solver",
            )

    return ReportValidationResult(issues=issues)


def format_feedback_for_llm(result: ReportValidationResult) -> str:
    """Format error-only feedback for an LLM retry.

    Only ERRORS are included; warnings are intentionally suppressed to
    avoid the over-correction flip-flop discussed in §7.4.3 of the
    Hermes-inspired design doc.
    """
    errors = [i for i in result.issues if i.severity == "error"]
    if not errors:
        return ""
    body = "\n".join(f"- {issue.format()}" for issue in errors)
    return (
        "Validator found issues in your previous report. The following "
        "fields reference solvers or problems that do not exist in the "
        "experiment summary you were given. Please regenerate the report, "
        "fixing only these fields and keeping the rest unchanged:\n\n"
        f"{body}"
    )


def format_for_user(result: ReportValidationResult) -> list[str]:
    """Render surviving issues for a user-facing Rich panel."""
    return [issue.format() for issue in result.issues]


__all__ = [
    "ReportIssue",
    "ReportValidationResult",
    "Severity",
    "format_feedback_for_llm",
    "format_for_user",
    "validate_report",
]
