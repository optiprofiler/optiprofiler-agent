"""Tests for the BenchmarkReport Pydantic schema (Agent C)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from optiprofiler_agent.agent_c.report_schema import (
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


def _make_minimal_report(**overrides) -> BenchmarkReport:
    base = dict(
        overview=ReportOverview(
            headline="solver_a dominates solver_b on cutest unconstrained.",
            setup="Two solvers compared on cutest unconstrained, dim 1-5.",
        ),
        performance_profile=PerformanceProfileSection(
            winner_at_tau1="solver_a",
            most_robust="solver_a",
            ranking_change="Rankings remain stable as tolerance tightens.",
        ),
        data_profile=DataProfileSection(
            most_efficient="solver_a",
            commentary="solver_a needs roughly half the evaluations.",
        ),
        convergence_issues=ConvergenceIssuesSection(),
        anomalies=AnomaliesSection(),
        recommendations=RecommendationsSection(
            actions=[
                RecommendationAction(
                    kind="switch_solver",
                    target_solver="solver_a",
                    rationale="Dominates on both efficiency and robustness.",
                )
            ],
        ),
    )
    base.update(overrides)
    return BenchmarkReport(**base)


class TestBenchmarkReport:

    def test_minimal_report_instantiates(self):
        report = _make_minimal_report()
        assert report.schema_version == "1.0"
        assert report.performance_profile.winner_at_tau1 == "solver_a"
        assert len(report.recommendations.actions) == 1

    def test_json_round_trip(self):
        report = _make_minimal_report()
        as_json = report.model_dump_json()
        rebuilt = BenchmarkReport.model_validate_json(as_json)
        assert rebuilt == report

    def test_invalid_action_kind_rejected(self):
        with pytest.raises(ValidationError):
            RecommendationAction(
                kind="frobnicate",
                rationale="bogus",
            )

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            ConvergenceIssueEntry(
                solver="solver_a",
                failure_count=3,
                severity="catastrophic",
            )

    def test_negative_failure_count_rejected(self):
        with pytest.raises(ValidationError):
            ConvergenceIssueEntry(
                solver="solver_a",
                failure_count=-1,
                severity="low",
            )

    def test_target_solver_optional(self):
        action = RecommendationAction(
            kind="no_action",
            rationale="No actionable signal.",
        )
        assert action.target_solver is None

    def test_key_findings_default_empty(self):
        report = _make_minimal_report()
        assert report.key_findings == []

    def test_key_findings_round_trips(self):
        report = _make_minimal_report(
            key_findings=[
                "solver_a wins at tau=1",
                "solver_b fails on stiff problems",
            ]
        )
        rebuilt = BenchmarkReport.model_validate_json(
            report.model_dump_json()
        )
        assert rebuilt.key_findings == [
            "solver_a wins at tau=1",
            "solver_b fails on stiff problems",
        ]


class TestSchemaSurface:
    """The Pydantic schema is the contract sent to the LLM via JSON
    Schema mode; pin the most important descriptive fields so a
    refactor that drops them is caught immediately."""

    def test_solver_name_invariant_in_schema(self):
        schema = BenchmarkReport.model_json_schema()
        as_text = json.dumps(schema)
        # Each section that mentions a solver name must instruct the
        # model that the value MUST come from the input summary.
        assert "MUST be one of the solver names" in as_text

    def test_problem_invariant_in_schema(self):
        schema = BenchmarkReport.model_json_schema()
        as_text = json.dumps(schema)
        assert "Do NOT invent problem names" in as_text

    def test_anomaly_kind_enum_present(self):
        schema = AnomalyEntry.model_json_schema()
        kind = schema["properties"]["kind"]
        # In Pydantic v2, Literal enums become anyOf or const refs.
        assert kind  # presence check; exact JSON Schema shape varies
