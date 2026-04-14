"""Summary builder — combines all analyzers into a single BenchmarkSummary.

This is the interface contract between Stage 1 (pure Python rule engine)
and Stage 2 (LLM report generation).  The summary is a JSON-serializable
dict that contains only verified facts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from optiprofiler_agent.agent_c.result_loader import BenchmarkResults, load_results
from optiprofiler_agent.agent_c.profile_reader import read_all_profiles
from optiprofiler_agent.agent_c.score_analyzer import ScoreAnalysis, analyze
from optiprofiler_agent.agent_c.anomaly_detector import AnomalyReport, detect_anomalies


@dataclass
class BenchmarkSummary:
    """Complete structured summary of a benchmark experiment."""

    # Experiment metadata
    experiment_dir: str
    language: str
    solver_names: list[str]
    problem_types: str
    dimension_range: tuple[int, int]
    feature_stamp: str
    problem_libraries: list[str]
    total_problems: dict[str, int]
    wall_clock_times: dict[str, float]

    # Solver scores (from log.txt)
    solver_scores: dict[str, float]

    # Analysis results
    rankings: list[dict[str, Any]]
    head_to_head: list[dict[str, Any]]
    precision_cliffs: list[dict[str, Any]]
    failure_patterns: list[dict[str, Any]]
    timing_outliers: list[dict[str, Any]]
    curve_crossovers: list[dict[str, Any]]
    per_tolerance_scores: dict[str, dict[str, float]]

    # Anomalies
    anomalies: list[dict[str, Any]]
    anomaly_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        d = asdict(self)
        d["dimension_range"] = list(d["dimension_range"])
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def build_summary(
    results_dir: str | Path,
    read_profiles: bool = True,
) -> BenchmarkSummary:
    """Build a complete summary from a benchmark experiment directory.

    Parameters
    ----------
    results_dir : str or Path
        Path to the experiment output directory.
    read_profiles : bool
        Whether to read and analyze profile PDFs. Set to False for
        faster processing when only text-based analysis is needed.

    Returns
    -------
    BenchmarkSummary
        JSON-serializable summary of all analysis results.
    """
    results = load_results(results_dir)

    profiles = {}
    if read_profiles:
        try:
            profiles = read_all_profiles(results.profile_paths)
        except Exception:
            pass

    analysis = analyze(results, profiles)
    anomalies = detect_anomalies(results, profiles)

    # Count anomalies by severity
    anomaly_counts: dict[str, int] = {}
    for a in anomalies:
        anomaly_counts[a.severity] = anomaly_counts.get(a.severity, 0) + 1

    # Limit plateau anomalies to avoid noise
    filtered_anomalies = _filter_anomalies(anomalies)

    return BenchmarkSummary(
        experiment_dir=str(results.results_dir),
        language=results.language,
        solver_names=results.config.solver_names,
        problem_types=results.config.problem_types,
        dimension_range=(results.config.mindim, results.config.maxdim),
        feature_stamp=results.config.feature_stamp,
        problem_libraries=results.config.problem_libraries,
        total_problems=results.n_problems,
        wall_clock_times=results.wall_clock_times,
        solver_scores=results.solver_scores,
        rankings=[asdict(r) for r in analysis.rankings],
        head_to_head=[asdict(h) for h in analysis.head_to_head],
        precision_cliffs=[asdict(c) for c in analysis.precision_cliffs],
        failure_patterns=[asdict(f) for f in analysis.failure_patterns[:15]],
        timing_outliers=[asdict(t) for t in analysis.timing_outliers],
        curve_crossovers=[asdict(c) for c in analysis.curve_crossovers],
        per_tolerance_scores=analysis.per_tolerance_scores,
        anomalies=[asdict(a) for a in filtered_anomalies],
        anomaly_counts=anomaly_counts,
    )


def _filter_anomalies(
    anomalies: list[AnomalyReport],
    max_plateaus: int = 6,
    max_per_type: int = 10,
) -> list[AnomalyReport]:
    """Reduce noise by limiting repetitive anomaly reports."""
    from collections import defaultdict

    by_type: dict[str, list[AnomalyReport]] = defaultdict(list)
    for a in anomalies:
        by_type[a.anomaly_type].append(a)

    filtered: list[AnomalyReport] = []
    for atype, items in by_type.items():
        limit = max_plateaus if atype == "plateau" else max_per_type
        filtered.extend(items[:limit])

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    filtered.sort(key=lambda a: severity_order.get(a.severity, 3))
    return filtered
