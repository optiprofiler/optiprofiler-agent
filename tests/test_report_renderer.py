"""Tests for the BenchmarkReport Markdown / HTML renderer."""

from __future__ import annotations

from optiprofiler_agent.interpreter.renderer import render_html, render_markdown
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
from optiprofiler_agent.interpreter.summary import BenchmarkSummary


def _make_summary() -> BenchmarkSummary:
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
        failure_patterns=[],
        timing_outliers=[],
        curve_crossovers=[],
        per_tolerance_scores={},
        anomalies=[],
        anomaly_counts={},
    )


def _make_full_report() -> BenchmarkReport:
    return BenchmarkReport(
        overview=ReportOverview(
            headline="solver_a dominates on cutest unconstrained.",
            setup="Two solvers compared, dim 1-5, 10 problems.",
        ),
        performance_profile=PerformanceProfileSection(
            winner_at_tau1="solver_a",
            most_robust="solver_a",
            ranking_change="Stable rankings across tolerances.",
        ),
        data_profile=DataProfileSection(
            most_efficient="solver_a",
            commentary="solver_a uses ~50% fewer evaluations.",
        ),
        convergence_issues=ConvergenceIssuesSection(
            entries=[
                ConvergenceIssueEntry(
                    solver="solver_b",
                    failure_count=3,
                    severity="medium",
                    notes="Fails on stiff problems.",
                ),
            ],
            common_failure_problems=["PROB1"],
        ),
        anomalies=AnomaliesSection(
            entries=[
                AnomalyEntry(
                    kind="timing_outlier",
                    affected_solvers=["solver_b"],
                    severity="low",
                    detail="solver_b takes 4x longer on PROB7.",
                ),
            ],
        ),
        recommendations=RecommendationsSection(
            actions=[
                RecommendationAction(
                    kind="switch_solver",
                    target_solver="solver_a",
                    rationale="Better on every metric.",
                ),
                RecommendationAction(
                    kind="gather_more_data",
                    target_solver=None,
                    rationale="Sample size of 10 is small.",
                ),
            ],
            caveats="Only one feature stamp tested.",
        ),
    )


class TestRenderMarkdown:

    def test_all_sections_present(self):
        md = render_markdown(_make_full_report(), _make_summary())
        for header in [
            "# Benchmark Analysis Report",
            "## Experiment Setup",
            "## Performance Profile Analysis",
            "## Data Profile Analysis",
            "## Convergence Issues",
            "## Anomalies and Warnings",
            "## Recommendations",
        ]:
            assert header in md, f"missing section: {header}"

    def test_headline_rendered_as_blockquote(self):
        md = render_markdown(_make_full_report(), _make_summary())
        assert "> solver_a dominates" in md

    def test_metadata_table_includes_solvers(self):
        md = render_markdown(_make_full_report(), _make_summary())
        assert "`solver_a`" in md
        assert "`solver_b`" in md
        assert "cutest" in md
        assert "1 – 5" in md  # dimension range

    def test_solver_scores_formatted(self):
        md = render_markdown(_make_full_report(), _make_summary())
        # Scores rendered with four decimals in a dedicated table.
        assert "0.8500" in md
        assert "0.6200" in md
        assert "Aggregate solver scores" in md

    def test_key_findings_rendered_as_bullets(self):
        report = _make_full_report()
        report.key_findings = [
            "solver_a wins at tau=1 with 0.85",
            "solver_b fails on stiff problems",
            "Sample size of 10 limits confidence",
        ]
        md = render_markdown(report, _make_summary())
        assert "## Key Findings" in md
        assert "- solver_a wins at tau=1 with 0.85" in md
        assert "- solver_b fails on stiff problems" in md

    def test_key_findings_section_omitted_when_empty(self):
        report = _make_full_report()
        report.key_findings = []
        md = render_markdown(report, _make_summary())
        assert "## Key Findings" not in md

    def test_recommendations_numbered(self):
        md = render_markdown(_make_full_report(), _make_summary())
        assert "1. **switch_solver**" in md
        assert "2. **gather_more_data**" in md

    def test_caveats_section(self):
        md = render_markdown(_make_full_report(), _make_summary())
        assert "Caveats" in md
        assert "Only one feature stamp tested" in md

    def test_empty_convergence_issues_fallback(self):
        report = _make_full_report()
        report.convergence_issues = ConvergenceIssuesSection()
        md = render_markdown(report, _make_summary())
        assert "No solver showed a notable failure pattern." in md

    def test_empty_anomalies_fallback(self):
        report = _make_full_report()
        report.anomalies = AnomaliesSection()
        md = render_markdown(report, _make_summary())
        assert "No anomalies detected by the rule engine." in md

    def test_no_recommendations_fallback(self):
        report = _make_full_report()
        report.recommendations = RecommendationsSection()
        md = render_markdown(report, _make_summary())
        assert "No further actions recommended." in md

    def test_schema_version_footer(self):
        md = render_markdown(_make_full_report(), _make_summary())
        assert "schema v1.0" in md


class TestRenderHtml:

    def test_html_wrapper_present(self):
        html = render_html(_make_full_report(), _make_summary())
        assert "<!doctype html>" in html
        assert "<title>OptiProfiler Benchmark Report</title>" in html

    def test_html_contains_report_text(self):
        html = render_html(_make_full_report(), _make_summary())
        # The headline should appear in either Markdown-rendered or
        # pre-wrapped form.
        assert "solver_a dominates" in html
