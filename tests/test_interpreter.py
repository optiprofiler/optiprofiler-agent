"""Tests for Agent C — interpreter and summary modules.

Tests the no-LLM path (rule engine) to avoid needing API keys.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from optiprofiler_agent.agent_c.result_loader import ProfilePaths
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
    def test_interpret_falls_back_to_freeform(self, mock_create, fake_experiment):
        """When structured output is not supported AND the manual JSON
        path also fails to find JSON in the response, the legacy
        free-form Markdown path must still produce a usable report."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = NotImplementedError(
            "provider does not support structured output"
        )
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
        # invoke is called once by the manual-JSON attempt (which finds
        # no JSON) and once by the final legacy free-form fallback.
        assert mock_llm.invoke.call_count == 2

    @patch("optiprofiler_agent.common.llm_client.create_llm")
    def test_interpret_strips_thinking_in_freeform_fallback(
        self, mock_create, fake_experiment
    ):
        """Reasoning models such as MiniMax-M2 / DeepSeek-R1 emit
        ``<think>...</think>`` blocks. They MUST never leak into the
        report file."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = NotImplementedError("nope")
        mock_llm.invoke.return_value = MagicMock(
            content=(
                "<think>The user wants me to analyse...</think>\n\n"
                "# Benchmark Report\n\nsolver_a wins."
            )
        )
        mock_create.return_value = mock_llm

        from optiprofiler_agent.agent_c.interpreter import interpret
        result = interpret(
            results_dir=fake_experiment,
            config=AgentConfig(llm=MagicMock()),
            llm_enabled=True,
            read_profiles=False,
        )
        assert "<think>" not in result
        assert "</think>" not in result
        assert "user wants me to analyse" not in result
        assert "solver_a wins" in result

    @patch("optiprofiler_agent.common.llm_client.create_llm")
    def test_manual_json_path_recovers_from_thinking_model(
        self, mock_create, fake_experiment
    ):
        """When ``with_structured_output`` cannot parse a thinking-model
        response, the manual JSON path strips the reasoning block and
        unwraps a ```json ... ``` fence to produce a valid report."""
        mock_llm = MagicMock()
        # Provider rejects structured output (typical for generic
        # OpenAI-compatible endpoints behind thinking models).
        mock_llm.with_structured_output.side_effect = NotImplementedError(
            "json_schema not supported"
        )
        # The raw response a thinking model would emit. Note the
        # leading ``<think>`` block + fenced JSON.
        report_json = (
            '{"schema_version":"1.0",'
            '"key_findings":["solver_a wins at tau=1"],'
            '"overview":{"headline":"solver_a dominates.",'
            '"setup":"Two solvers, dim 1-5."},'
            '"performance_profile":{"winner_at_tau1":"solver_a",'
            '"most_robust":"solver_a",'
            '"ranking_change":"Stable across tolerances."},'
            '"data_profile":{"most_efficient":"solver_a",'
            '"commentary":"solver_a uses fewer evaluations."},'
            '"convergence_issues":{"entries":[],'
            '"common_failure_problems":[]},'
            '"anomalies":{"entries":[]},'
            '"recommendations":{"actions":[],"caveats":""}}'
        )
        mock_llm.invoke.return_value = MagicMock(
            content=(
                "<think>The user wants me to analyse the experiment. "
                "I need to identify the winner...</think>\n\n"
                f"```json\n{report_json}\n```"
            )
        )
        mock_create.return_value = mock_llm

        from optiprofiler_agent.agent_c.interpreter import interpret
        result = interpret(
            results_dir=fake_experiment,
            config=AgentConfig(llm=MagicMock()),
            llm_enabled=True,
            read_profiles=False,
        )
        # Structured-template output should appear, NOT the raw think
        # block, NOT the raw JSON.
        assert "<think>" not in result
        assert "schema_version" not in result  # raw JSON not leaked
        assert "## Key Findings" in result
        assert "solver_a wins at tau=1" in result
        assert "schema v1.0" in result  # rendered footer
        # The legacy free-form path was NOT used: only the single
        # manual-JSON invoke happened.
        assert mock_llm.invoke.call_count == 1


class TestThinkingHelpers:
    """Unit tests for the ``<think>`` / JSON-extraction utilities."""

    def test_strip_thinking_removes_paired_block(self):
        from optiprofiler_agent.agent_c.interpreter import _strip_thinking
        out = _strip_thinking("<think>plan</think>\nfinal answer")
        assert out == "final answer"

    def test_strip_thinking_handles_thinking_tag(self):
        from optiprofiler_agent.agent_c.interpreter import _strip_thinking
        out = _strip_thinking("<thinking>...</thinking>real")
        assert out == "real"

    def test_strip_thinking_handles_reasoning_tag(self):
        from optiprofiler_agent.agent_c.interpreter import _strip_thinking
        out = _strip_thinking("<reasoning>X</reasoning>Y")
        assert out == "Y"

    def test_strip_thinking_no_op_without_tags(self):
        from optiprofiler_agent.agent_c.interpreter import _strip_thinking
        assert _strip_thinking("just text") == "just text"

    def test_strip_thinking_handles_empty(self):
        from optiprofiler_agent.agent_c.interpreter import _strip_thinking
        assert _strip_thinking("") == ""
        assert _strip_thinking(None) == ""

    def test_extract_json_unwraps_fenced_block(self):
        from optiprofiler_agent.agent_c.interpreter import _extract_json_blob
        out = _extract_json_blob('prose ```json\n{"a": 1}\n``` more')
        assert out == '{"a": 1}'

    def test_extract_json_after_thinking_block(self):
        from optiprofiler_agent.agent_c.interpreter import _extract_json_blob
        out = _extract_json_blob('<think>x</think>{"a": 1, "b": [2, 3]}')
        assert out == '{"a": 1, "b": [2, 3]}'

    def test_extract_json_balances_nested_braces(self):
        from optiprofiler_agent.agent_c.interpreter import _extract_json_blob
        out = _extract_json_blob('lead {"x": {"y": 1}} trail')
        assert out == '{"x": {"y": 1}}'

    def test_extract_json_ignores_braces_inside_strings(self):
        from optiprofiler_agent.agent_c.interpreter import _extract_json_blob
        out = _extract_json_blob('{"s": "has } brace"}')
        assert out == '{"s": "has } brace"}'


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
