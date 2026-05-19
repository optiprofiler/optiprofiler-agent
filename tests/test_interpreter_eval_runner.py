"""Tests for scripts/run_interpreter_eval.py."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_interpreter_eval import (  # noqa: E402
    _deterministic_report,
    fact_check_report,
    render_markdown_report,
    summarize,
)
from optiprofiler_agent.interpreter.summary import build_summary


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "matlab_results"
    / "experiment_matlab"
)


def test_deterministic_report_passes_fact_check_and_hides_language_field():
    summary = build_summary(FIXTURE, read_profiles=False)
    report = _deterministic_report(summary)
    result = fact_check_report(report, summary)
    assert result["passed"] is True
    assert result["checks"]["no_language_field"] is True
    assert "| Language |" not in report


def test_fact_check_rejects_unknown_solver():
    summary = build_summary(FIXTURE, read_profiles=False)
    report = _deterministic_report(summary) + "\n`fake_solver` unexpectedly wins.\n"
    result = fact_check_report(report, summary)
    assert result["passed"] is False
    assert "fake_solver" in result["unknown_solver_mentions"]


def test_summarize_reports_judge_coverage():
    summary = summarize(
        [
            {
                "id": "ok",
                "passed": True,
                "score": 1.0,
                "judge_avg": 0.9,
                "judge_scores": {"hallucination": 1.0},
            },
            {
                "id": "judge_failed",
                "passed": True,
                "score": 1.0,
                "judge_avg": None,
            },
        ]
    )
    assert summary["pass_rate"] == 1.0
    assert summary["judge_coverage"] == 0.5
    assert summary["judge_na_cases"] == ["judge_failed"]


def test_render_markdown_report_lists_failure_checks():
    md = render_markdown_report(
        {
            "timestamp_utc": "2026-05-19T00:00:00+00:00",
            "strategy": "deterministic",
            "summary": {
                "n_cases": 1,
                "pass_count": 0,
                "pass_rate": 0.0,
                "avg_fact_score": 0.5,
                "judge_avg": None,
            },
            "results": [
                {
                    "id": "bad",
                    "passed": False,
                    "score": 0.5,
                    "winner": "fminsearch",
                    "runner_up": "fminunc",
                    "checks": {"mentions_winner": False},
                }
            ],
        }
    )
    assert "# Interpreter Report Evaluation" in md
    assert "mentions_winner" in md
