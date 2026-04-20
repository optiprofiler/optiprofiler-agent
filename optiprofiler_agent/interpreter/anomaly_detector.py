"""Anomaly detector — flag issues humans might miss in benchmark results.

Scans for:
- Extreme function values (f = 1e+12 etc.) indicating solver failure
- Problems where ALL solvers fail across ALL tolerances/runs
- Timing outliers (one problem takes far longer than average)
- Profile curve plateaus (solver stops improving)
- Curve crossovers (solver A better at low precision, B at high)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from optiprofiler_agent.interpreter.result_loader import BenchmarkResults
from optiprofiler_agent.interpreter.profile_reader import ProfilePage, LogRatioPage


@dataclass
class AnomalyReport:
    """A single detected anomaly."""

    anomaly_type: str   # "extreme_value", "total_failure", "timing", "plateau", "crossover"
    severity: str       # "critical", "warning", "info"
    description: str
    evidence: dict = field(default_factory=dict)


def _detect_extreme_values(
    results: BenchmarkResults,
    threshold: float = 1e10,
) -> list[AnomalyReport]:
    """Detect solver runs that produced extreme function values."""
    anomalies: list[AnomalyReport] = []
    extreme_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rr in results.run_results:
        if rr.output_f >= threshold or math.isinf(rr.output_f) or math.isnan(rr.output_f):
            extreme_counts[rr.solver][rr.problem] += 1

    for solver, problems in extreme_counts.items():
        total = sum(problems.values())
        worst = sorted(problems.items(), key=lambda x: x[1], reverse=True)[:5]
        anomalies.append(AnomalyReport(
            anomaly_type="extreme_value",
            severity="warning" if total < 20 else "critical",
            description=(
                f"Solver '{solver}' produced extreme function values (>= {threshold:.0e}) "
                f"in {total} run(s) across {len(problems)} problem(s). "
                f"Most affected: {', '.join(f'{p}({c}x)' for p, c in worst)}."
            ),
            evidence={
                "solver": solver,
                "total_extreme_runs": total,
                "affected_problems": len(problems),
                "worst_problems": dict(worst),
            },
        ))

    return anomalies


def _detect_total_failures(results: BenchmarkResults) -> list[AnomalyReport]:
    """Detect problems where all solvers failed to evaluate a single point."""
    anomalies: list[AnomalyReport] = []

    if results.all_failed_problems:
        anomalies.append(AnomalyReport(
            anomaly_type="total_failure",
            severity="critical",
            description=(
                f"{len(results.all_failed_problems)} problem(s) could not be evaluated "
                f"by any solver: {', '.join(results.all_failed_problems)}."
            ),
            evidence={
                "problems": results.all_failed_problems,
            },
        ))

    return anomalies


def _detect_universal_convergence_failure(
    results: BenchmarkResults,
) -> list[AnomalyReport]:
    """Find problems that fail convergence in every tolerance and every run."""
    anomalies: list[AnomalyReport] = []

    if not results.convergence_failures:
        return anomalies

    # Count how many (tol, run, basis) combos each problem appears in
    problem_fail_count: Counter = Counter()
    total_combos = 0
    seen_combos: set[tuple[str, str, int]] = set()

    for f in results.convergence_failures:
        combo = (f.basis, f.tolerance, f.run)
        if combo not in seen_combos:
            seen_combos.add(combo)
            total_combos += 1
        for prob in f.problems:
            problem_fail_count[prob] += 1

    # Problems that fail in > 90% of all combos are "universally failing"
    threshold = total_combos * 0.9
    universal = [
        (prob, count) for prob, count in problem_fail_count.most_common()
        if count >= threshold
    ]

    if universal:
        anomalies.append(AnomalyReport(
            anomaly_type="universal_convergence_failure",
            severity="warning",
            description=(
                f"{len(universal)} problem(s) fail convergence in ≥90% of all "
                f"tolerance/run/basis combinations ({total_combos} total): "
                f"{', '.join(p for p, _ in universal[:10])}."
            ),
            evidence={
                "problems": {p: c for p, c in universal},
                "total_combos": total_combos,
            },
        ))

    return anomalies


def _detect_timing_anomalies(
    results: BenchmarkResults,
    ratio_threshold: float = 5.0,
) -> list[AnomalyReport]:
    """Detect problems with unusually high solving time."""
    anomalies: list[AnomalyReport] = []

    for plib, problems in results.problems.items():
        if not problems:
            continue
        times = [p.time_secs for p in problems]
        mean_time = sum(times) / len(times)
        if mean_time < 0.01:
            continue

        outliers = [
            (p.name, p.time_secs, p.time_secs / mean_time)
            for p in problems
            if p.time_secs / mean_time > ratio_threshold
        ]

        if outliers:
            outliers.sort(key=lambda x: x[2], reverse=True)
            anomalies.append(AnomalyReport(
                anomaly_type="timing",
                severity="info",
                description=(
                    f"In library '{plib}', {len(outliers)} problem(s) took "
                    f">{ratio_threshold}x the average time ({mean_time:.2f}s): "
                    f"{', '.join(f'{n}({t:.1f}s, {r:.1f}x)' for n, t, r in outliers)}."
                ),
                evidence={
                    "library": plib,
                    "mean_time": round(mean_time, 2),
                    "outliers": [
                        {"problem": n, "time": round(t, 2), "ratio": round(r, 2)}
                        for n, t, r in outliers
                    ],
                },
            ))

    return anomalies


def _detect_plateaus(
    profiles: dict[str, list[ProfilePage | LogRatioPage]],
    min_x_fraction: float = 0.3,
) -> list[AnomalyReport]:
    """Detect solver curves that plateau early (stop improving).

    A plateau is when the y-value doesn't increase over the last
    *min_x_fraction* of the x-range.
    """
    anomalies: list[AnomalyReport] = []

    for key, pages in profiles.items():
        for page in pages:
            if not isinstance(page, ProfilePage):
                continue
            for curve in page.curves:
                if len(curve.points) < 5:
                    continue

                x_vals = [x for x, _ in curve.points]
                y_vals = [y for _, y in curve.points]
                x_range = x_vals[-1] - x_vals[0]
                if x_range < 0.01:
                    continue

                cutoff_x = x_vals[-1] - min_x_fraction * x_range
                tail_ys = [y for x, y in curve.points if x >= cutoff_x]

                if not tail_ys:
                    continue

                y_change = max(tail_ys) - min(tail_ys)
                final_y = y_vals[-1]

                if y_change < 0.01 and final_y < 0.8:
                    anomalies.append(AnomalyReport(
                        anomaly_type="plateau",
                        severity="info",
                        description=(
                            f"Solver '{curve.solver_name}' plateaus at y={final_y:.3f} "
                            f"in the last {min_x_fraction*100:.0f}% of the x-range "
                            f"({page.profile_type} {page.basis}, tol={page.tolerance})."
                        ),
                        evidence={
                            "solver": curve.solver_name,
                            "profile": f"{page.profile_type}_{page.basis}",
                            "tolerance": page.tolerance,
                            "plateau_y": round(final_y, 4),
                        },
                    ))

    return anomalies


def _detect_solver_divergence(
    profiles: dict[str, list[ProfilePage | LogRatioPage]],
    divergence_threshold: float = 0.3,
) -> list[AnomalyReport]:
    """Detect cases where solvers diverge significantly at high precision.

    Compares the final y-values of solver curves at the tightest tolerance.
    """
    anomalies: list[AnomalyReport] = []

    for key, pages in profiles.items():
        profile_pages = [p for p in pages if isinstance(p, ProfilePage)]
        if not profile_pages:
            continue

        last_page = profile_pages[-1]
        if len(last_page.curves) < 2:
            continue

        finals = {c.solver_name: c.points[-1][1] for c in last_page.curves if c.points}
        if len(finals) < 2:
            continue

        names = list(finals.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = abs(finals[names[i]] - finals[names[j]])
                if diff > divergence_threshold:
                    anomalies.append(AnomalyReport(
                        anomaly_type="solver_divergence",
                        severity="warning",
                        description=(
                            f"At tightest tolerance ({last_page.tolerance}), "
                            f"'{names[i]}' ({finals[names[i]]:.3f}) and "
                            f"'{names[j]}' ({finals[names[j]]:.3f}) diverge by "
                            f"{diff:.3f} in {last_page.profile_type} {last_page.basis} profiles."
                        ),
                        evidence={
                            "profile": f"{last_page.profile_type}_{last_page.basis}",
                            "tolerance": last_page.tolerance,
                            "solver_scores": finals,
                            "difference": round(diff, 4),
                        },
                    ))

    return anomalies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(
    results: BenchmarkResults,
    profiles: dict[str, list[ProfilePage | LogRatioPage]] | None = None,
) -> list[AnomalyReport]:
    """Run all anomaly detectors.

    Parameters
    ----------
    results : BenchmarkResults
        From ``result_loader.load_results()``.
    profiles : dict, optional
        From ``profile_reader.read_all_profiles()``.

    Returns
    -------
    list[AnomalyReport]
        All detected anomalies, sorted by severity.
    """
    profiles = profiles or {}

    anomalies: list[AnomalyReport] = []
    anomalies.extend(_detect_extreme_values(results))
    anomalies.extend(_detect_total_failures(results))
    anomalies.extend(_detect_universal_convergence_failure(results))
    anomalies.extend(_detect_timing_anomalies(results))
    anomalies.extend(_detect_plateaus(profiles))
    anomalies.extend(_detect_solver_divergence(profiles))

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    anomalies.sort(key=lambda a: severity_order.get(a.severity, 3))

    return anomalies
