"""Tests for Agent C — interpreter and summary modules.

Tests the no-LLM path (rule engine) to avoid needing API keys.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from optiprofiler_agent.agent_c.result_loader import (
    BenchmarkResults,
    ExperimentConfig,
    ProfilePaths,
)
from optiprofiler_agent.agent_c.summary import BenchmarkSummary, build_summary
from optiprofiler_agent.config import AgentConfig


# ---------------------------------------------------------------------------
# Fixtures — create a minimal fake experiment directory
# ---------------------------------------------------------------------------

_LOG_TXT = """\
[INFO    ] Profiling solvers: solver_a, solver_b
[INFO    ] - Solvers: solver_a, solver_b
[INFO    ] - Problem libraries: cutest
[INFO    ] - Problem types: u
[INFO    ] - Problem dimension range: [1, 5]
[INFO    ] - Feature stamp: plain
[INFO    ] Finish solving PROB1 with solver_a (run 1/1) (in 0.5 seconds)
[INFO    ] Output result for PROB1 with solver_a (run 1/1): f = 1.0
[INFO    ] Best result for PROB1 with solver_a (run 1/1): f = 1.0
[INFO    ] Finish solving PROB1 with solver_b (run 1/1) (in 0.8 seconds)
[INFO    ] Output result for PROB1 with solver_b (run 1/1): f = 2.0
[INFO    ] Best result for PROB1 with solver_b (run 1/1): f = 2.0
[INFO    ] Scores of the solvers:
[INFO    ] solver_a:    0.8500
[INFO    ] solver_b:    0.6200
"""

_REPORT_TXT = """\
## Experiment Summary

Solver names: solver_a, solver_b
Problem types: u
Problem mindim: 1
Problem maxdim: 5
Feature stamp: plain

## Report for the problem library "cutest"

Number of problems selected: 1
Wall-clock time spent by all the solvers: 1.3 secs

Name    Type  Dim  mb  mlcon  mnlcon  mcon  Time(s)
PROB1   u     3    0   0      0       0     1.300
"""


@pytest.fixture
def fake_experiment(tmp_path):
    """Create a minimal experiment directory structure."""
    exp_dir = tmp_path / "experiment_20260101_120000"
    exp_dir.mkdir()
    test_log = exp_dir / "test_log"
    test_log.mkdir()

    (test_log / "log.txt").write_text(_LOG_TXT, encoding="utf-8")
    (test_log / "report.txt").write_text(_REPORT_TXT, encoding="utf-8")
    (test_log / "_scratch.py").write_text("# python experiment", encoding="utf-8")

    return exp_dir


# ---------------------------------------------------------------------------
# result_loader integration
# ---------------------------------------------------------------------------

class TestResultLoader:

    def test_load_results_basic(self, fake_experiment):
        from optiprofiler_agent.agent_c.result_loader import load_results
        results = load_results(fake_experiment)

        assert results.language == "python"
        assert "solver_a" in results.config.solver_names
        assert "solver_b" in results.config.solver_names
        assert results.solver_scores.get("solver_a") == pytest.approx(0.85)
        assert results.solver_scores.get("solver_b") == pytest.approx(0.62)
        assert results.config.problem_types == "u"
        assert results.config.mindim == 1
        assert results.config.maxdim == 5

    def test_load_results_problems(self, fake_experiment):
        from optiprofiler_agent.agent_c.result_loader import load_results
        results = load_results(fake_experiment)

        assert "cutest" in results.problems
        assert len(results.problems["cutest"]) == 1
        assert results.problems["cutest"][0].name == "PROB1"
        assert results.problems["cutest"][0].dimension == 3

    def test_load_results_wall_times(self, fake_experiment):
        from optiprofiler_agent.agent_c.result_loader import load_results
        results = load_results(fake_experiment)

        assert results.wall_clock_times.get("cutest") == pytest.approx(1.3)
        assert results.n_problems.get("cutest") == 1

    def test_missing_test_log_raises(self, tmp_path):
        from optiprofiler_agent.agent_c.result_loader import load_results
        with pytest.raises(FileNotFoundError, match="test_log"):
            load_results(tmp_path)


# ---------------------------------------------------------------------------
# build_summary (no LLM)
# ---------------------------------------------------------------------------

class TestBuildSummary:

    def test_build_summary_basic(self, fake_experiment):
        summary = build_summary(fake_experiment, read_profiles=False)

        assert isinstance(summary, BenchmarkSummary)
        assert summary.language == "python"
        assert set(summary.solver_names) == {"solver_a", "solver_b"}
        assert summary.solver_scores["solver_a"] > summary.solver_scores["solver_b"]
        assert summary.problem_types == "u"
        assert summary.dimension_range == (1, 5)

    def test_summary_to_json(self, fake_experiment):
        summary = build_summary(fake_experiment, read_profiles=False)
        json_str = summary.to_json()
        data = json.loads(json_str)
        assert "solver_names" in data
        assert "solver_scores" in data

    def test_summary_to_dict(self, fake_experiment):
        summary = build_summary(fake_experiment, read_profiles=False)
        d = summary.to_dict()
        assert isinstance(d["dimension_range"], list)


# ---------------------------------------------------------------------------
# interpret() — no-LLM mode
# ---------------------------------------------------------------------------

class TestInterpretNoLLM:

    def test_interpret_returns_json(self, fake_experiment):
        from optiprofiler_agent.agent_c.interpreter import interpret
        result = interpret(
            results_dir=fake_experiment,
            llm_enabled=False,
            read_profiles=False,
        )
        data = json.loads(result)
        assert "solver_names" in data
        assert "rankings" in data


# ---------------------------------------------------------------------------
# interpret() — mocked LLM
# ---------------------------------------------------------------------------

class TestInterpretWithLLM:

    @patch("optiprofiler_agent.common.llm_client.create_llm")
    def test_interpret_calls_llm(self, mock_create, fake_experiment):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="# Benchmark Report\n\nsolver_a is better."
        )
        mock_create.return_value = mock_llm

        from optiprofiler_agent.agent_c.interpreter import interpret
        result = interpret(
            results_dir=fake_experiment,
            config=AgentConfig(llm=MagicMock()),
            llm_enabled=True,
            read_profiles=False,
        )
        assert "solver_a" in result
        mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# profile_reader (mock fitz)
# ---------------------------------------------------------------------------

class TestProfileReader:

    def test_read_all_profiles_empty_paths(self):
        from optiprofiler_agent.agent_c.profile_reader import read_all_profiles
        paths = ProfilePaths()
        result = read_all_profiles(paths)
        assert result == {}

    def test_read_all_profiles_missing_file_skipped(self, tmp_path):
        from optiprofiler_agent.agent_c.profile_reader import read_all_profiles
        paths = ProfilePaths(perf_hist=tmp_path / "nonexistent.pdf")
        result = read_all_profiles(paths)
        assert "perf_hist" not in result
