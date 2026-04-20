"""Tests for the BenchmarkReport business-invariant validator."""

from __future__ import annotations

from optiprofiler_agent.interpreter.report_schema import (
    AnomaliesSection,
    AnomalyEntry,
    BenchmarkReport,
    ConvergenceIssueEntry,
    ConvergenceIssuesSection,
    DataProfileSection,
    PerformanceProfileSection,
    RecommendationAction,
    RecommendationsSection,
    ReportOverview,
)
from optiprofiler_agent.interpreter.report_validator import (
    format_feedback_for_llm,
    format_for_user,
    validate_report,
)
from optiprofiler_agent.interpreter.summary import BenchmarkSummary


def _make_summary(
    *,
    failure_patterns: list[dict] | None = None,
) -> BenchmarkSummary:
    return BenchmarkSummary(
        experiment_dir="/tmp/exp",
        language="python",
        solver_names=["solver_a", "solver_b"],
        problem_types="u",
        dimension_range=(1, 5),
        feature_stamp="plain",
        problem_libraries=["cutest"],
        total_problems={"cutest": 10},
        wall_clock_times={"cutest": 12.3},
        solver_scores={"solver_a": 0.85, "solver_b": 0.62},
        rankings=[],
        head_to_head=[],
        precision_cliffs=[],
        failure_patterns=failure_patterns or [],
        timing_outliers=[],
        curve_crossovers=[],
        per_tolerance_scores={},
        anomalies=[],
        anomaly_counts={},
    )


def _make_report(
    *,
    winner: str = "solver_a",
    most_robust: str = "solver_a",
    most_efficient: str = "solver_a",
    convergence_entries: list[ConvergenceIssueEntry] | None = None,
    common_failures: list[str] | None = None,
    anomaly_solvers: list[str] | None = None,
    rec_target: str | None = "solver_a",
) -> BenchmarkReport:
    return BenchmarkReport(
        overview=ReportOverview(headline="x", setup="y"),
        performance_profile=PerformanceProfileSection(
            winner_at_tau1=winner,
            most_robust=most_robust,
            ranking_change="stable",
        ),
        data_profile=DataProfileSection(
            most_efficient=most_efficient,
            commentary="cheaper",
        ),
        convergence_issues=ConvergenceIssuesSection(
            entries=convergence_entries or [],
            common_failure_problems=common_failures or [],
        ),
        anomalies=AnomaliesSection(
            entries=[
                AnomalyEntry(
                    kind="timing_outlier",
                    affected_solvers=anomaly_solvers or ["solver_a"],
                    severity="low",
                    detail="d",
                ),
            ] if anomaly_solvers is not None else [],
        ),
        recommendations=RecommendationsSection(
            actions=[
                RecommendationAction(
                    kind="switch_solver",
                    target_solver=rec_target,
                    rationale="r",
                ),
            ],
        ),
    )


class TestValidateReport:

    def test_clean_report(self):
        result = validate_report(_make_report(), _make_summary())
        assert result.is_clean

    def test_unknown_winner_is_error(self):
        result = validate_report(
            _make_report(winner="solver_x"),
            _make_summary(),
        )
        assert result.has_errors
        assert any(
            i.field_path == "performance_profile.winner_at_tau1"
            for i in result.issues
        )

    def test_unknown_most_robust_is_error(self):
        result = validate_report(
            _make_report(most_robust="solver_z"),
            _make_summary(),
        )
        assert result.has_errors

    def test_unknown_recommendation_target_is_error(self):
        result = validate_report(
            _make_report(rec_target="ghost_solver"),
            _make_summary(),
        )
        assert result.has_errors

    def test_null_recommendation_target_ok(self):
        result = validate_report(
            _make_report(rec_target=None),
            _make_summary(),
        )
        assert result.is_clean

    def test_unknown_anomaly_solver_is_error(self):
        result = validate_report(
            _make_report(anomaly_solvers=["solver_a", "phantom"]),
            _make_summary(),
        )
        assert result.has_errors
        assert any(
            "phantom" in i.message for i in result.issues
        )

    def test_failure_count_overshoot_is_warning(self):
        summary = _make_summary(
            failure_patterns=[
                {"solver": "solver_b", "failure_count": 2, "problems": []},
            ],
        )
        report = _make_report(
            convergence_entries=[
                ConvergenceIssueEntry(
                    solver="solver_b",
                    failure_count=99,
                    severity="high",
                ),
            ],
        )
        result = validate_report(report, summary)
        assert result.has_warnings
        assert not result.has_errors

    def test_failure_count_within_cap_clean(self):
        summary = _make_summary(
            failure_patterns=[
                {"solver": "solver_b", "failure_count": 5, "problems": []},
            ],
        )
        report = _make_report(
            convergence_entries=[
                ConvergenceIssueEntry(
                    solver="solver_b",
                    failure_count=3,
                    severity="medium",
                ),
            ],
        )
        result = validate_report(report, summary)
        assert result.is_clean

    def test_unknown_problem_name_is_warning(self):
        summary = _make_summary(
            failure_patterns=[
                {"solver": "solver_b", "failure_count": 1, "problems": ["PROB1"]},
            ],
        )
        report = _make_report(
            common_failures=["PROB1", "FAKE_PROB"],
        )
        result = validate_report(report, summary)
        assert result.has_warnings
        assert any("FAKE_PROB" in i.message for i in result.issues)


class TestFeedbackFormatting:

    def test_feedback_includes_only_errors(self):
        result = validate_report(
            _make_report(winner="ghost"),
            _make_summary(
                failure_patterns=[
                    {"solver": "solver_b", "failure_count": 2, "problems": []},
                ],
            ),
        )
        # Add a manual warning by overshooting failure count too.
        report = _make_report(
            winner="ghost",
            convergence_entries=[
                ConvergenceIssueEntry(
                    solver="solver_b",
                    failure_count=50,
                    severity="high",
                ),
            ],
        )
        result = validate_report(
            report,
            _make_summary(
                failure_patterns=[
                    {"solver": "solver_b", "failure_count": 2, "problems": []},
                ],
            ),
        )
        feedback = format_feedback_for_llm(result)
        assert "ghost" in feedback
        # Warning content (overshoot) must NOT appear in retry feedback.
        assert "50 failures" not in feedback

    def test_empty_when_no_errors(self):
        result = validate_report(_make_report(), _make_summary())
        assert format_feedback_for_llm(result) == ""

    def test_format_for_user_lists_all(self):
        report = _make_report(
            winner="ghost",
            convergence_entries=[
                ConvergenceIssueEntry(
                    solver="solver_b",
                    failure_count=999,
                    severity="high",
                ),
            ],
        )
        summary = _make_summary(
            failure_patterns=[
                {"solver": "solver_b", "failure_count": 1, "problems": []},
            ],
        )
        result = validate_report(report, summary)
        lines = format_for_user(result)
        # Both error and warning surfaced to the user.
        assert any(line.startswith("[error]") for line in lines)
        assert any(line.startswith("[warning]") for line in lines)
