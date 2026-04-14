"""Load human-readable benchmark outputs from an OptiProfiler experiment directory.

Parses:
- ``log.txt``   — experiment config, per-run solver results, final solver scores
- ``report.txt`` — experiment summary, problem table, convergence failure records

Ignores machine-only data files (pkl / mat / h5) which are used internally
by OptiProfiler for checkpoint/resume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SolverRunResult:
    """A single solver x problem x run result extracted from log.txt."""

    problem: str
    solver: str
    run: int
    total_runs: int
    elapsed_secs: float
    output_f: float
    best_f: float


@dataclass
class ProblemInfo:
    """A row from the problem table in report.txt."""

    name: str
    ptype: str
    dimension: int
    mb: int
    mlcon: int
    mnlcon: int
    mcon: int
    time_secs: float


@dataclass
class ConvergenceFailure:
    """A single convergence failure record from report.txt."""

    basis: str          # "History-based" or "Output-based"
    tolerance: str      # e.g. "1e-1"
    run: int
    problems: list[str]


@dataclass
class ExperimentConfig:
    """Experiment configuration extracted from log.txt / report.txt."""

    solver_names: list[str] = field(default_factory=list)
    problem_types: str = ""
    mindim: int = 0
    maxdim: int = 0
    feature_stamp: str = ""
    problem_libraries: list[str] = field(default_factory=list)
    problem_names_from_user: str = ""
    exclude_list: str = ""


@dataclass
class ProfilePaths:
    """Discovered PDF file paths for profiles and history plots."""

    summary_pdf: Path | None = None
    perf_hist: Path | None = None
    perf_out: Path | None = None
    data_hist: Path | None = None
    data_out: Path | None = None
    log_ratio_hist: Path | None = None
    log_ratio_out: Path | None = None
    detailed_profiles: dict[str, list[Path]] = field(default_factory=dict)
    history_plots: dict[str, list[Path]] = field(default_factory=dict)
    history_plots_summary: list[Path] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """All human-readable data loaded from a single experiment directory."""

    results_dir: Path
    language: str  # "python" or "matlab"
    config: ExperimentConfig
    solver_scores: dict[str, float]
    run_results: list[SolverRunResult]
    problems: dict[str, list[ProblemInfo]]  # keyed by problem library
    convergence_failures: list[ConvergenceFailure]
    all_failed_problems: list[str]
    profile_paths: ProfilePaths
    wall_clock_times: dict[str, float]  # plib -> total seconds
    n_problems: dict[str, int]          # plib -> count


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_LOG_PREFIX_RE = re.compile(r"^\[INFO\s*\]\s*")
_MATLAB_LOG_PREFIX_RE = re.compile(r"^INFO:\s*")


def _strip_log_prefix(line: str) -> str:
    """Remove ``[INFO    ] `` or ``INFO: `` prefix from a log line."""
    m = _LOG_PREFIX_RE.match(line)
    if m:
        return line[m.end():]
    m = _MATLAB_LOG_PREFIX_RE.match(line)
    if m:
        return line[m.end():]
    return line


def _parse_log_txt(log_path: Path) -> tuple[
    ExperimentConfig, dict[str, float], list[SolverRunResult]
]:
    """Parse ``log.txt`` to extract config, scores, and per-run results."""
    text = log_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    stripped = [_strip_log_prefix(l) for l in lines]

    config = ExperimentConfig()
    scores: dict[str, float] = {}
    run_results: list[SolverRunResult] = []

    # --- Extract config from the header ---
    for raw in stripped:
        s = raw.strip()
        if s.startswith("- Solvers:"):
            config.solver_names = [n.strip() for n in s.split(":", 1)[1].split(",")]
        elif s.startswith("- Problem libraries:"):
            config.problem_libraries = [n.strip() for n in s.split(":", 1)[1].split(",")]
        elif s.startswith("- Problem types:"):
            config.problem_types = s.split(":", 1)[1].strip()
        elif s.startswith("- Problem dimension range:"):
            m = re.search(r"\[(\d+),\s*(\d+)\]", s)
            if m:
                config.mindim = int(m.group(1))
                config.maxdim = int(m.group(2))
        elif s.startswith("- Feature stamp:"):
            config.feature_stamp = s.split(":", 1)[1].strip()

    # --- Extract solver scores (near end of file) ---
    # Python: "Scores of the solvers:"  MATLAB: "Scores of the solvers"
    in_scores = False
    for raw in stripped:
        s = raw.strip()
        if s.startswith("Scores of the solvers"):
            in_scores = True
            continue
        if in_scores:
            if not s:
                in_scores = False
                continue
            # Handles both "scipy_cobyla:    0.9468" and "fminunc   :    0.1402"
            m = re.match(r"^(\S+)\s*:\s+([\d.]+)$", s)
            if m:
                scores[m.group(1)] = float(m.group(2))

    # --- Extract per-run results ---
    finish_re = re.compile(
        r"Finish solving\s+(\S+)\s+with\s+(\S+)\s+\(run\s+(\d+)/(\d+)\)\s+\(in\s+([\d.]+)\s+seconds?\)"
    )
    output_re = re.compile(
        r"Output result for\s+(\S+)\s+with\s+(\S+)\s+\(run\s+(\d+)/(\d+)\):\s+f\s*=\s*([^\s]+)"
    )
    best_re = re.compile(
        r"Best\s+result for\s+(\S+)\s+with\s+(\S+)\s+\(run\s+(\d+)/(\d+)\):\s+f\s*=\s*([^\s]+)"
    )

    pending: dict[tuple[str, str, int], dict] = {}

    for raw in stripped:
        s = raw.strip()

        m = finish_re.search(s)
        if m:
            key = (m.group(1), m.group(2), int(m.group(3)))
            entry = pending.setdefault(key, {})
            entry["problem"] = m.group(1)
            entry["solver"] = m.group(2)
            entry["run"] = int(m.group(3))
            entry["total_runs"] = int(m.group(4))
            entry["elapsed_secs"] = float(m.group(5))
            continue

        m = output_re.search(s)
        if m:
            key = (m.group(1), m.group(2), int(m.group(3)))
            entry = pending.setdefault(key, {})
            entry["problem"] = m.group(1)
            entry["solver"] = m.group(2)
            entry["run"] = int(m.group(3))
            entry["total_runs"] = int(m.group(4))
            try:
                entry["output_f"] = float(m.group(5))
            except ValueError:
                entry["output_f"] = float("inf")
            continue

        m = best_re.search(s)
        if m:
            key = (m.group(1), m.group(2), int(m.group(3)))
            entry = pending.setdefault(key, {})
            entry["problem"] = m.group(1)
            entry["solver"] = m.group(2)
            entry["run"] = int(m.group(3))
            entry["total_runs"] = int(m.group(4))
            try:
                entry["best_f"] = float(m.group(5))
            except ValueError:
                entry["best_f"] = float("inf")

    for _key, d in pending.items():
        run_results.append(SolverRunResult(
            problem=d.get("problem", ""),
            solver=d.get("solver", ""),
            run=d.get("run", 0),
            total_runs=d.get("total_runs", 0),
            elapsed_secs=d.get("elapsed_secs", 0.0),
            output_f=d.get("output_f", float("nan")),
            best_f=d.get("best_f", float("nan")),
        ))

    return config, scores, run_results


def _parse_report_txt(report_path: Path) -> tuple[
    ExperimentConfig,
    dict[str, list[ProblemInfo]],
    list[ConvergenceFailure],
    list[str],
    dict[str, float],
    dict[str, int],
]:
    """Parse ``report.txt`` to extract config, problems, and failures."""
    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    config = ExperimentConfig()
    problems: dict[str, list[ProblemInfo]] = {}
    failures: list[ConvergenceFailure] = []
    all_failed: list[str] = []
    wall_times: dict[str, float] = {}
    n_problems: dict[str, int] = {}

    # --- Parse summary section ---
    for line in lines:
        s = line.strip()
        if s.startswith("Solver names:"):
            config.solver_names = [n.strip() for n in s.split(":", 1)[1].split(",")]
        elif s.startswith("Problem types:"):
            config.problem_types = s.split(":", 1)[1].strip()
        elif s.startswith("Problem mindim:"):
            config.mindim = int(s.split(":", 1)[1].strip())
        elif s.startswith("Problem maxdim:"):
            config.maxdim = int(s.split(":", 1)[1].strip())
        elif s.startswith("Feature stamp:"):
            config.feature_stamp = s.split(":", 1)[1].strip()
        elif s.startswith("Problem names from user:"):
            config.problem_names_from_user = s.split(":", 1)[1].strip()
        elif s.startswith("Exclude list from user:"):
            config.exclude_list = s.split(":", 1)[1].strip()

    # --- Parse problem library sections ---
    plib_header_re = re.compile(r'## Report for the problem library "(\w+)"')
    plib_count_re = re.compile(r"Number of problems selected:\s+(\d+)")
    plib_time_re = re.compile(r"Wall-clock time spent by all the solvers:\s+([\d.]+)\s+secs")
    problem_row_re = re.compile(
        r"^(\S+)\s+(u|b|l|n|q)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)"
    )

    current_plib = None
    for line in lines:
        m = plib_header_re.search(line)
        if m:
            current_plib = m.group(1)
            problems[current_plib] = []
            continue

        if current_plib:
            m = plib_count_re.search(line)
            if m:
                n_problems[current_plib] = int(m.group(1))
                continue

            m = plib_time_re.search(line)
            if m:
                wall_times[current_plib] = float(m.group(1))
                continue

            m = problem_row_re.match(line.strip())
            if m:
                problems[current_plib].append(ProblemInfo(
                    name=m.group(1),
                    ptype=m.group(2),
                    dimension=int(m.group(3)),
                    mb=int(m.group(4)),
                    mlcon=int(m.group(5)),
                    mnlcon=int(m.group(6)),
                    mcon=int(m.group(7)),
                    time_secs=float(m.group(8)),
                ))

    # --- Parse convergence failures ---
    fail_re = re.compile(
        r"^(History-based|Output-based)\s+tol\s*=\s*(\S+)\s+run\s*=\s*(\d+)\s*:\s+(.*)"
    )
    in_fail_section = False
    in_all_failed_section = False

    for line in lines:
        s = line.strip()

        if "that all the solvers failed to evaluate a single point" in s:
            in_all_failed_section = True
            in_fail_section = False
            continue

        if "that all the solvers failed to meet the convergence test" in s:
            in_fail_section = True
            in_all_failed_section = False
            continue

        if s.startswith("##"):
            in_fail_section = False
            in_all_failed_section = False
            continue

        if in_all_failed_section and s and s != "This part is empty.":
            all_failed.extend(s.split())

        if in_fail_section:
            m = fail_re.match(s)
            if m:
                probs = [p for p in m.group(4).split() if p]
                failures.append(ConvergenceFailure(
                    basis=m.group(1),
                    tolerance=m.group(2),
                    run=int(m.group(3)),
                    problems=probs,
                ))

    return config, problems, failures, all_failed, wall_times, n_problems


def _discover_profiles(results_dir: Path) -> ProfilePaths:
    """Discover PDF files in the experiment directory."""
    paths = ProfilePaths()

    # Summary PDF (matches summary_*.pdf)
    for p in results_dir.glob("summary_*.pdf"):
        paths.summary_pdf = p
        break

    # Top-level profile PDFs
    mapping = {
        "perf_hist.pdf": "perf_hist",
        "perf_out.pdf": "perf_out",
        "data_hist.pdf": "data_hist",
        "data_out.pdf": "data_out",
        "log-ratio_hist.pdf": "log_ratio_hist",
        "log-ratio_out.pdf": "log_ratio_out",
    }
    for fname, attr in mapping.items():
        p = results_dir / fname
        if p.exists():
            setattr(paths, attr, p)

    # Detailed profiles
    dp_dir = results_dir / "detailed_profiles"
    if dp_dir.is_dir():
        for subdir in sorted(dp_dir.iterdir()):
            if subdir.is_dir():
                pdfs = sorted(subdir.glob("*.pdf"))
                if pdfs:
                    paths.detailed_profiles[subdir.name] = pdfs

    # History plots
    hp_dir = results_dir / "history_plots"
    if hp_dir.is_dir():
        for item in sorted(hp_dir.iterdir()):
            if item.is_dir():
                pdfs = sorted(item.glob("*.pdf"))
                if pdfs:
                    paths.history_plots[item.name] = pdfs
            elif item.suffix == ".pdf" and "summary" in item.name:
                paths.history_plots_summary.append(item)

    return paths


def _detect_language(test_log_dir: Path) -> str:
    """Detect whether the experiment was run in Python or MATLAB."""
    if (test_log_dir / "_scratch.py").exists():
        return "python"
    if (test_log_dir / "scratch.m").exists():
        return "matlab"
    if any(test_log_dir.glob("*.pkl")):
        return "python"
    if any(test_log_dir.glob("*.mat")):
        return "matlab"
    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_results(results_dir: str | Path) -> BenchmarkResults:
    """Load all human-readable benchmark outputs from *results_dir*.

    Parameters
    ----------
    results_dir : str or Path
        Path to the experiment directory (the one containing ``README.txt``,
        profile PDFs, ``test_log/``, etc.).

    Returns
    -------
    BenchmarkResults
        Structured data extracted from log.txt, report.txt, and discovered
        PDF paths.
    """
    results_dir = Path(results_dir)
    test_log = results_dir / "test_log"

    if not test_log.is_dir():
        raise FileNotFoundError(
            f"No test_log/ directory found in {results_dir}. "
            "Is this a valid OptiProfiler experiment directory?"
        )

    language = _detect_language(test_log)

    # Parse log.txt
    log_path = test_log / "log.txt"
    if log_path.exists():
        log_config, scores, run_results = _parse_log_txt(log_path)
    else:
        log_config = ExperimentConfig()
        scores = {}
        run_results = []

    # Parse report.txt
    report_path = test_log / "report.txt"
    if report_path.exists():
        rpt_config, problems, failures, all_failed, wall_times, n_probs = (
            _parse_report_txt(report_path)
        )
    else:
        rpt_config = ExperimentConfig()
        problems = {}
        failures = []
        all_failed = []
        wall_times = {}
        n_probs = {}

    # Merge configs (report.txt has more structured data; log.txt has plibs)
    merged = ExperimentConfig(
        solver_names=rpt_config.solver_names or log_config.solver_names,
        problem_types=rpt_config.problem_types or log_config.problem_types,
        mindim=rpt_config.mindim or log_config.mindim,
        maxdim=rpt_config.maxdim or log_config.maxdim,
        feature_stamp=rpt_config.feature_stamp or log_config.feature_stamp,
        problem_libraries=log_config.problem_libraries or list(problems.keys()),
        problem_names_from_user=rpt_config.problem_names_from_user,
        exclude_list=rpt_config.exclude_list,
    )

    profile_paths = _discover_profiles(results_dir)

    return BenchmarkResults(
        results_dir=results_dir,
        language=language,
        config=merged,
        solver_scores=scores,
        run_results=run_results,
        problems=problems,
        convergence_failures=failures,
        all_failed_problems=all_failed,
        profile_paths=profile_paths,
        wall_clock_times=wall_times,
        n_problems=n_probs,
    )


def find_latest_experiment(base_dir: str | Path) -> Path:
    """Find the most recent experiment directory under *base_dir*.

    Experiment directories are identified by having a ``test_log/`` subdirectory.
    Among all matches, the one with the latest modification time is returned.
    """
    base = Path(base_dir)
    candidates = [
        d for d in base.iterdir()
        if d.is_dir() and (d / "test_log").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No experiment directories found in {base}")
    return max(candidates, key=lambda d: d.stat().st_mtime)
