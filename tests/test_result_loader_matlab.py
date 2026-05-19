"""Integration tests for MATLAB benchmark output parsing."""

import os
from pathlib import Path

import pytest

from optiprofiler_agent.interpreter.result_loader import (
    _parse_log_txt,
    load_results,
)
from optiprofiler_agent.interpreter.summary import build_summary

_FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "matlab_results" / "experiment_matlab"
)

# Snippet from a real MATLAB OptiProfiler run (spaces in "run  1/ 1").
_REAL_MATLAB_LOG_SNIPPET = """\
INFO: Finish solving    CLIFF      with fminsearch (run  1/ 1) in 2.87 seconds.
INFO: Output result for CLIFF      with fminsearch (run  1/ 1): f = 2.0069e-01.
INFO: Best result for CLIFF      with fminsearch (run  1/ 1): f = 2.0069e-01.
INFO: Finish solving    CLIFF      with fminunc    (run  1/ 1) in 0.64 seconds.
INFO: Output result for CLIFF      with fminunc    (run  1/ 1): f = 1.2345e-05.
INFO: Best result for CLIFF      with fminunc    (run  1/ 1): f = 1.2345e-05.
INFO: Scores of the solvers
INFO: fminsearch:    0.9806
INFO: fminunc   :    0.9483
"""


@pytest.fixture
def matlab_results_dir():
    if not _FIXTURE_DIR.is_dir():
        pytest.skip("MATLAB fixture directory not found")
    return _FIXTURE_DIR


class TestMatlabResultLoader:

    def test_parse_log_matlab_run_counter_spacing(self):
        """MATLAB uses '(run  1/ 1)' — must not yield zero run_results."""
        _, scores, runs = _parse_log_txt_from_text(_REAL_MATLAB_LOG_SNIPPET)
        assert scores["fminsearch"] == pytest.approx(0.9806)
        assert scores["fminunc"] == pytest.approx(0.9483)
        assert len(runs) == 2
        assert runs[0].problem == "CLIFF"
        assert runs[0].solver == "fminsearch"
        assert runs[0].elapsed_secs == pytest.approx(2.87)
        assert runs[0].output_f == pytest.approx(0.20069, rel=1e-3)
        assert runs[0].best_f == pytest.approx(0.20069, rel=1e-3)

    def test_load_results_detects_matlab(self, matlab_results_dir):
        results = load_results(matlab_results_dir)
        assert results.language == "matlab"

    def test_load_results_solver_scores(self, matlab_results_dir):
        results = load_results(matlab_results_dir)
        assert "fminunc" in results.solver_scores
        assert "nelder_mead" in results.solver_scores
        assert results.solver_scores["nelder_mead"] > results.solver_scores["fminunc"]

    def test_load_results_run_results(self, matlab_results_dir):
        results = load_results(matlab_results_dir)
        assert len(results.run_results) >= 2
        assert results.run_results[0].problem == "ROSENBR"

    def test_load_results_problems_table(self, matlab_results_dir):
        results = load_results(matlab_results_dir)
        assert "custom" in results.problems
        assert len(results.problems["custom"]) == 1
        assert results.problems["custom"][0].name == "ROSENBR"

    def test_build_summary_non_empty(self, matlab_results_dir):
        summary = build_summary(matlab_results_dir, read_profiles=False)
        assert summary.language == "matlab"
        assert len(summary.solver_names) >= 2
        assert summary.solver_scores
        assert summary.rankings


def _parse_log_txt_from_text(text: str):
    """Helper: parse log text via the same path as _parse_log_txt."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = Path(f.name)
    try:
        return _parse_log_txt(path)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.integration
class TestRealMatlabOutputOptional:
    """Optional: point MATOP_REAL_RESULTS_DIR at a full MATLAB experiment folder."""

    @pytest.fixture
    def real_dir(self):
        raw = os.environ.get("MATOP_REAL_RESULTS_DIR", "").strip()
        if not raw:
            pytest.skip("Set MATOP_REAL_RESULTS_DIR to a full MATLAB out/<experiment>/ path")
        p = Path(raw).expanduser()
        if not (p / "test_log" / "log.txt").is_file():
            pytest.skip(f"Not a valid experiment dir: {p}")
        return p

    def test_full_run_parses_scores_and_runs(self, real_dir):
        results = load_results(real_dir)
        assert results.language == "matlab"
        assert len(results.solver_scores) >= 2
        assert len(results.run_results) > 0
        assert len(results.problems.get("s2mpj", [])) >= 1

    def test_full_run_pdf_curves_when_present(self, real_dir):
        summary = build_summary(real_dir, read_profiles=True)
        # Real MATLAB PDFs from exportgraphics are often parseable; if not,
        # profile_curves_available should be False (graceful degradation).
        if summary.profile_curves_available:
            assert len(summary.curve_crossovers) > 0
