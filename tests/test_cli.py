"""Tests for the CLI module — uses Click's CliRunner (no real LLM needed)."""

import pytest
from click.testing import CliRunner

from optiprofiler_agent.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIVersion:

    def test_version_flag(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCheckCommand:

    def test_check_valid_script(self, runner, tmp_path):
        script = tmp_path / "good.py"
        script.write_text(
            "from optiprofiler import benchmark\n"
            "def a(fun, x0): return x0\n"
            "def b(fun, x0): return x0\n"
            "benchmark([a, b])\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["check", str(script)])
        assert result.exit_code == 0
        assert "looks good" in result.output.lower() or "✓" in result.output

    def test_check_syntax_error(self, runner, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("def f(\n", encoding="utf-8")
        result = runner.invoke(main, ["check", str(script)])
        assert result.exit_code != 0

    def test_check_single_solver_warning(self, runner, tmp_path):
        script = tmp_path / "one_solver.py"
        script.write_text(
            "from optiprofiler import benchmark\n"
            "def a(fun, x0): return x0\n"
            "benchmark([a])\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["check", str(script)])
        assert "1 provided" in result.output or result.exit_code != 0


class TestWikiStatsCommand:

    def test_wiki_stats_shows_output(self, runner):
        result = runner.invoke(main, ["wiki", "stats"])
        assert result.exit_code == 0
        assert "Total pages" in result.output or "pages" in result.output.lower()

    def test_wiki_stats_shows_categories(self, runner):
        result = runner.invoke(main, ["wiki", "stats"])
        assert "category" in result.output.lower() or "concepts" in result.output.lower()


class TestWikiLintCommand:

    def test_wiki_lint_runs(self, runner):
        result = runner.invoke(main, ["wiki", "lint"])
        assert result.exit_code == 0
        assert "lint" in result.output.lower() or "clean" in result.output.lower()


class TestInterpretCommand:

    def test_interpret_missing_dir(self, runner, tmp_path):
        result = runner.invoke(main, ["interpret", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0

    def test_interpret_no_llm_with_fake_experiment(self, runner, tmp_path):
        exp_dir = tmp_path / "exp_001"
        exp_dir.mkdir()
        test_log = exp_dir / "test_log"
        test_log.mkdir()
        (test_log / "log.txt").write_text(
            "[INFO    ] - Solvers: a, b\n"
            "[INFO    ] - Problem types: u\n"
            "[INFO    ] - Problem dimension range: [1, 5]\n"
            "[INFO    ] - Feature stamp: plain\n"
            "[INFO    ] Scores of the solvers:\n"
            "[INFO    ] a:    0.90\n"
            "[INFO    ] b:    0.10\n",
            encoding="utf-8",
        )
        (test_log / "report.txt").write_text(
            "Solver names: a, b\nProblem types: u\nProblem mindim: 1\nProblem maxdim: 5\n",
            encoding="utf-8",
        )
        (test_log / "_scratch.py").touch()

        result = runner.invoke(main, [
            "interpret", str(exp_dir), "--no-llm", "--no-profiles",
        ])
        assert result.exit_code == 0
        assert "solver_scores" in result.output or "a" in result.output


class TestDebugCommand:

    def test_debug_no_traceback_no_run_fails(self, runner, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("print(1)", encoding="utf-8")
        result = runner.invoke(main, ["debug", str(script)])
        assert result.exit_code != 0
        assert "traceback" in result.output.lower() or "error" in result.output.lower()
