"""Score analyzer — extract statistics from benchmark results and profile curves.

Analyzes:
- Solver rankings (from log.txt scores)
- Head-to-head comparison (from profile curves)
- Precision cliff detection (score drop across tolerances)
- Convergence failure patterns (from report.txt)
- Timing analysis (from report.txt problem table)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from optiprofiler_agent.agent_c.result_loader import BenchmarkResults
from optiprofiler_agent.agent_c.profile_reader import ProfilePage, LogRatioPage


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SolverRanking:
    """Overall ranking of a solver."""

    name: str
    score: float
    rank: int


@dataclass
class HeadToHead:
    """Pairwise comparison between two solvers at a given tolerance."""

    solver_a: str
    solver_b: str
    tolerance: str
    basis: str
    a_better_fraction: float
    b_better_fraction: float
    tie_fraction: float


@dataclass
class PrecisionCliff:
    """Detected precision cliff — a solver's profile drops sharply."""

    solver: str
    profile_type: str
    basis: str
    from_tolerance: str
    to_tolerance: str
    score_drop: float


@dataclass
class FailurePattern:
    """A problem that frequently fails convergence across runs/tolerances."""

    problem: str
    total_appearances: int
    tolerances: list[str]
    bases: list[str]


@dataclass
class TimingOutlier:
    """A problem with unusually high or low solving time."""

    problem: str
    time_secs: float
    mean_time: float
    ratio: float  # time / mean


@dataclass
class CurveCrossover:
    """Point where two solvers' profile curves cross."""

    solver_a: str
    solver_b: str
    profile_type: str
    basis: str
    tolerance: str
    crossover_x: float
    a_leads_before: bool


@dataclass
class ScoreAnalysis:
    """Complete analysis results."""

    rankings: list[SolverRanking]
    head_to_head: list[HeadToHead]
    precision_cliffs: list[PrecisionCliff]
    failure_patterns: list[FailurePattern]
    timing_outliers: list[TimingOutlier]
    curve_crossovers: list[CurveCrossover]
    per_tolerance_scores: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _compute_rankings(results: BenchmarkResults) -> list[SolverRanking]:
    """Rank solvers by their overall scores from log.txt."""
    if not results.solver_scores:
        return []

    sorted_solvers = sorted(
        results.solver_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        SolverRanking(name=name, score=score, rank=i + 1)
        for i, (name, score) in enumerate(sorted_solvers)
    ]


def _compute_head_to_head(
    profiles: dict[str, list[ProfilePage | LogRatioPage]],
) -> list[HeadToHead]:
    """Compare solver pairs using profile curve data.

    For step-function profiles, at each x-value where both curves have data,
    the solver with the higher y-value (higher fraction of problems solved)
    is considered "better".
    """
    comparisons: list[HeadToHead] = []

    for key, pages in profiles.items():
        for page in pages:
            if not isinstance(page, ProfilePage) or len(page.curves) < 2:
                continue

            for i in range(len(page.curves)):
                for j in range(i + 1, len(page.curves)):
                    ca = page.curves[i]
                    cb = page.curves[j]

                    if not ca.points or not cb.points:
                        continue

                    a_pts = dict(ca.points)
                    b_pts = dict(cb.points)

                    # Sample at common x-values
                    all_x = sorted(set(a_pts.keys()) | set(b_pts.keys()))
                    if not all_x:
                        continue

                    a_wins = 0
                    b_wins = 0
                    ties = 0

                    a_y = 0.0
                    b_y = 0.0
                    for x in all_x:
                        if x in a_pts:
                            a_y = a_pts[x]
                        if x in b_pts:
                            b_y = b_pts[x]

                        if abs(a_y - b_y) < 0.005:
                            ties += 1
                        elif a_y > b_y:
                            a_wins += 1
                        else:
                            b_wins += 1

                    total = a_wins + b_wins + ties
                    if total > 0:
                        comparisons.append(HeadToHead(
                            solver_a=ca.solver_name,
                            solver_b=cb.solver_name,
                            tolerance=page.tolerance,
                            basis=page.basis,
                            a_better_fraction=a_wins / total,
                            b_better_fraction=b_wins / total,
                            tie_fraction=ties / total,
                        ))

    return comparisons


def _detect_precision_cliffs(
    profiles: dict[str, list[ProfilePage | LogRatioPage]],
    threshold: float = 0.15,
) -> list[PrecisionCliff]:
    """Detect sharp drops in solver performance across tolerance levels.

    A "cliff" is when the final y-value (fraction of problems solved)
    drops by more than *threshold* between consecutive tolerances.
    """
    cliffs: list[PrecisionCliff] = []

    for key, pages in profiles.items():
        profile_pages = [p for p in pages if isinstance(p, ProfilePage)]
        if len(profile_pages) < 2:
            continue

        # Group by solver
        solver_finals: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for page in profile_pages:
            for curve in page.curves:
                if curve.points:
                    final_y = curve.points[-1][1]
                    solver_finals[curve.solver_name].append(
                        (page.tolerance, final_y)
                    )

        for solver, tol_scores in solver_finals.items():
            for i in range(len(tol_scores) - 1):
                tol_a, score_a = tol_scores[i]
                tol_b, score_b = tol_scores[i + 1]
                drop = score_a - score_b
                if drop > threshold:
                    ptype = profile_pages[0].profile_type
                    basis = profile_pages[0].basis
                    cliffs.append(PrecisionCliff(
                        solver=solver,
                        profile_type=ptype,
                        basis=basis,
                        from_tolerance=tol_a,
                        to_tolerance=tol_b,
                        score_drop=round(drop, 4),
                    ))

    return cliffs


def _analyze_failure_patterns(
    results: BenchmarkResults,
    min_appearances: int = 5,
) -> list[FailurePattern]:
    """Find problems that frequently fail convergence."""
    problem_counts: Counter = Counter()
    problem_tols: dict[str, set[str]] = defaultdict(set)
    problem_bases: dict[str, set[str]] = defaultdict(set)

    for failure in results.convergence_failures:
        for prob in failure.problems:
            problem_counts[prob] += 1
            problem_tols[prob].add(failure.tolerance)
            problem_bases[prob].add(failure.basis)

    patterns = []
    for prob, count in problem_counts.most_common():
        if count < min_appearances:
            break
        patterns.append(FailurePattern(
            problem=prob,
            total_appearances=count,
            tolerances=sorted(problem_tols[prob]),
            bases=sorted(problem_bases[prob]),
        ))

    return patterns


def _analyze_timing(
    results: BenchmarkResults,
    outlier_ratio: float = 3.0,
) -> list[TimingOutlier]:
    """Find problems with unusually high solving time."""
    outliers: list[TimingOutlier] = []

    for plib, problems in results.problems.items():
        if not problems:
            continue
        times = [p.time_secs for p in problems]
        mean_time = sum(times) / len(times)
        if mean_time < 0.01:
            continue

        for p in problems:
            ratio = p.time_secs / mean_time
            if ratio > outlier_ratio:
                outliers.append(TimingOutlier(
                    problem=p.name,
                    time_secs=p.time_secs,
                    mean_time=round(mean_time, 2),
                    ratio=round(ratio, 2),
                ))

    outliers.sort(key=lambda o: o.ratio, reverse=True)
    return outliers


def _detect_curve_crossovers(
    profiles: dict[str, list[ProfilePage | LogRatioPage]],
) -> list[CurveCrossover]:
    """Detect points where two solver curves cross each other."""
    crossovers: list[CurveCrossover] = []

    for key, pages in profiles.items():
        for page in pages:
            if not isinstance(page, ProfilePage) or len(page.curves) < 2:
                continue

            for i in range(len(page.curves)):
                for j in range(i + 1, len(page.curves)):
                    ca = page.curves[i]
                    cb = page.curves[j]

                    if not ca.points or not cb.points:
                        continue

                    # Build interpolated y-values at common x-points
                    all_x = sorted(set(x for x, _ in ca.points) | set(x for x, _ in cb.points))
                    a_dict = dict(ca.points)
                    b_dict = dict(cb.points)

                    prev_diff = None
                    a_y = 0.0
                    b_y = 0.0
                    for x in all_x:
                        if x in a_dict:
                            a_y = a_dict[x]
                        if x in b_dict:
                            b_y = b_dict[x]
                        diff = a_y - b_y

                        if prev_diff is not None and abs(diff) > 0.01:
                            if (prev_diff > 0 and diff < 0) or (prev_diff < 0 and diff > 0):
                                crossovers.append(CurveCrossover(
                                    solver_a=ca.solver_name,
                                    solver_b=cb.solver_name,
                                    profile_type=page.profile_type,
                                    basis=page.basis,
                                    tolerance=page.tolerance,
                                    crossover_x=round(x, 4),
                                    a_leads_before=prev_diff > 0,
                                ))
                        if abs(diff) > 0.01:
                            prev_diff = diff

    return crossovers


def _compute_per_tolerance_scores(
    profiles: dict[str, list[ProfilePage | LogRatioPage]],
) -> dict[str, dict[str, float]]:
    """Extract the final y-value (fraction solved) per solver per tolerance.

    Returns {tolerance: {solver_name: final_y_value}}.
    Only considers performance profiles (history-based) as the primary metric.
    """
    scores: dict[str, dict[str, float]] = {}

    for key in ["perf_hist", "perf_out"]:
        pages = profiles.get(key, [])
        for page in pages:
            if not isinstance(page, ProfilePage):
                continue
            tol_scores: dict[str, float] = {}
            for curve in page.curves:
                if curve.points:
                    tol_scores[curve.solver_name] = round(curve.points[-1][1], 4)
            if tol_scores:
                label = f"{key}_{page.tolerance}"
                scores[label] = tol_scores

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(
    results: BenchmarkResults,
    profiles: dict[str, list[ProfilePage | LogRatioPage]] | None = None,
) -> ScoreAnalysis:
    """Run all analyses on benchmark results and profile data.

    Parameters
    ----------
    results : BenchmarkResults
        From ``result_loader.load_results()``.
    profiles : dict, optional
        From ``profile_reader.read_all_profiles()``. If not provided,
        curve-based analyses (head-to-head, crossovers, precision cliffs)
        will be empty.

    Returns
    -------
    ScoreAnalysis
        Complete analysis results.
    """
    profiles = profiles or {}

    rankings = _compute_rankings(results)
    head_to_head = _compute_head_to_head(profiles)
    precision_cliffs = _detect_precision_cliffs(profiles)
    failure_patterns = _analyze_failure_patterns(results)
    timing_outliers = _analyze_timing(results)
    curve_crossovers = _detect_curve_crossovers(profiles)
    per_tol_scores = _compute_per_tolerance_scores(profiles)

    return ScoreAnalysis(
        rankings=rankings,
        head_to_head=head_to_head,
        precision_cliffs=precision_cliffs,
        failure_patterns=failure_patterns,
        timing_outliers=timing_outliers,
        curve_crossovers=curve_crossovers,
        per_tolerance_scores=per_tol_scores,
    )
