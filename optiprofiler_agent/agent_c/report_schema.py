"""Pydantic schema for benchmark analysis reports (Agent C).

This is the structured contract between the LLM and the report renderer.
Each section is a flat Pydantic model with descriptions that double as
in-schema instructions for the model. Together with
``llm.with_structured_output(BenchmarkReport, method="json_schema")``
this lets the provider (OpenAI / MiniMax / Kimi) enforce the schema at
decode time rather than relying on free-form Markdown.

Business invariants beyond what JSON Schema can express
(e.g. "winner must be one of the experiment's solvers") live in
``report_validator.validate_report``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SchemaVersion = Literal["1.0"]

Severity = Literal["none", "low", "medium", "high", "critical"]

ActionKind = Literal[
    "tune_params",
    "switch_solver",
    "increase_budget",
    "gather_more_data",
    "investigate_anomaly",
    "no_action",
]

AnomalyKind = Literal[
    "extreme_value",
    "timing_outlier",
    "plateau",
    "divergence",
    "precision_cliff",
    "convergence_failure",
    "other",
]


class ReportOverview(BaseModel):
    """Two- to four-sentence framing of the experiment."""

    headline: str = Field(
        description=(
            "One-sentence headline (<= 25 words) summarising the experiment "
            "and the dominant finding."
        ),
    )
    setup: str = Field(
        description=(
            "Two to three sentences describing the solvers compared, the "
            "problem libraries, the dimension range, and the feature stamp."
        ),
    )


class PerformanceProfileSection(BaseModel):
    """Performance-profile narrative."""

    winner_at_tau1: str = Field(
        description=(
            "Name of the solver with the highest score at the loosest "
            "tolerance (tau=1). MUST be one of the solver names listed in "
            "the input summary."
        ),
    )
    most_robust: str = Field(
        description=(
            "Name of the solver with the highest score as tau -> infinity. "
            "MUST be one of the solver names listed in the input summary."
        ),
    )
    ranking_change: str = Field(
        description=(
            "One to two sentences on how rankings shift as tolerance "
            "tightens. Mention precision cliffs if any are present in the "
            "input."
        ),
    )


class DataProfileSection(BaseModel):
    """Data-profile narrative."""

    most_efficient: str = Field(
        description=(
            "Name of the solver that consistently uses fewer function "
            "evaluations to reach the target. MUST be one of the solver "
            "names listed in the input summary."
        ),
    )
    commentary: str = Field(
        description=(
            "Two to three sentences comparing evaluation efficiency across "
            "tolerances. Reference concrete numbers from the input summary "
            "where possible; do NOT invent figures."
        ),
    )


class ConvergenceIssueEntry(BaseModel):
    """One solver's convergence-failure profile."""

    solver: str = Field(
        description=(
            "Solver name. MUST be one of the solver names listed in the "
            "input summary."
        ),
    )
    failure_count: int = Field(
        ge=0,
        description=(
            "Number of problems on which this solver did not reach the "
            "tightest tolerance. MUST be <= the failure count present in "
            "the input summary; do NOT estimate."
        ),
    )
    severity: Severity = Field(
        description=(
            "Severity assessment relative to the other solvers in the same "
            "experiment."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "Optional one-sentence note on the failure pattern (e.g. "
            "'fails on stiff problems above dimension 50')."
        ),
    )


class ConvergenceIssuesSection(BaseModel):
    """Aggregated convergence-issue narrative."""

    entries: list[ConvergenceIssueEntry] = Field(
        default_factory=list,
        description=(
            "One entry per solver with a non-trivial failure count. Empty "
            "list is acceptable when no solver has notable failures."
        ),
    )
    common_failure_problems: list[str] = Field(
        default_factory=list,
        description=(
            "Names of problems on which ALL solvers failed, taken verbatim "
            "from the input summary. Do NOT invent problem names."
        ),
    )


class AnomalyEntry(BaseModel):
    """One anomaly the rule engine flagged or the model surfaced."""

    kind: AnomalyKind = Field(description="Anomaly category.")
    affected_solvers: list[str] = Field(
        description=(
            "Solver names affected. Each MUST be one of the solver names "
            "listed in the input summary."
        ),
    )
    severity: Severity = Field(description="Severity assessment.")
    detail: str = Field(
        description=(
            "One- to two-sentence description of the anomaly and its "
            "likely cause."
        ),
    )


class AnomaliesSection(BaseModel):
    """Aggregated anomaly narrative."""

    entries: list[AnomalyEntry] = Field(
        default_factory=list,
        description="Empty list is acceptable when no anomalies were flagged.",
    )


class RecommendationAction(BaseModel):
    """One concrete recommended action."""

    kind: ActionKind = Field(description="Action category.")
    target_solver: str | None = Field(
        default=None,
        description=(
            "Solver this action concerns, or null if the recommendation is "
            "experiment-wide. When non-null, MUST be one of the solver "
            "names listed in the input summary."
        ),
    )
    rationale: str = Field(
        description="One to two sentences justifying the action with data.",
    )


class RecommendationsSection(BaseModel):
    """Aggregated recommendations."""

    actions: list[RecommendationAction] = Field(
        default_factory=list,
        description=(
            "Ordered by importance. Use kind='no_action' when the "
            "experiment provides no actionable signal."
        ),
    )
    caveats: str = Field(
        default="",
        description=(
            "Optional caveats about the experiment setup itself "
            "(e.g. small sample size, narrow dimension range)."
        ),
    )


class BenchmarkReport(BaseModel):
    """Top-level structured benchmark analysis report.

    Filled by the LLM via ``llm.with_structured_output``; rendered to
    Markdown / HTML / future web by ``renderer.render_markdown`` etc.
    """

    schema_version: SchemaVersion = Field(
        default="1.0",
        description="Schema version. Always '1.0' for this revision.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Three to five short, scannable bullet points (each <= 20 "
            "words, no leading dash) summarising the most important "
            "takeaways. Render at the top of the report so a reader can "
            "skim the experiment in 10 seconds. Examples: "
            "'fminunc dominates at tau=1 with score 0.92', "
            "'fminsearch fails on 4/12 stiff problems above n=20'."
        ),
    )
    overview: ReportOverview
    performance_profile: PerformanceProfileSection
    data_profile: DataProfileSection
    convergence_issues: ConvergenceIssuesSection
    anomalies: AnomaliesSection
    recommendations: RecommendationsSection


__all__ = [
    "ActionKind",
    "AnomaliesSection",
    "AnomalyEntry",
    "AnomalyKind",
    "BenchmarkReport",
    "ConvergenceIssueEntry",
    "ConvergenceIssuesSection",
    "DataProfileSection",
    "PerformanceProfileSection",
    "RecommendationAction",
    "RecommendationsSection",
    "ReportOverview",
    "SchemaVersion",
    "Severity",
]
