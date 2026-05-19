"""Tests for scripts/run_eval_suite.py aggregation and gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_eval_suite import (  # noqa: E402
    CommandResult,
    _render_summary,
    _suite_passed,
    _summarize_debugger,
    _summarize_interpreter,
    _summarize_run_eval,
)


def test_summarize_run_eval_counts_judge_coverage_and_issues(tmp_path):
    path = tmp_path / "advisor.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "good",
                    "combined_score": 0.9,
                    "judge_avg": 0.88,
                    "judge_scores": {"hallucination": 1.0},
                },
                {
                    "id": "low",
                    "combined_score": 0.4,
                    "judge_avg": None,
                    "judge_reason": "Judge error",
                },
            ]
        ),
        encoding="utf-8",
    )
    summary = _summarize_run_eval(path)
    assert summary["n_cases"] == 2
    assert summary["pass_count"] == 1
    assert summary["judge_coverage"] == 0.5
    assert summary["judge_na_cases"] == ["low"]
    assert summary["low_score_cases"][0]["id"] == "low"


def test_summarize_debugger_collects_failures(tmp_path):
    path = tmp_path / "debugger.json"
    path.write_text(
        json.dumps(
            {
                "language": "python",
                "strategy": "golden",
                "n_total": 2,
                "n_pass": 1,
                "pass_rate": 0.5,
                "results": [
                    {"id": "ok", "broken_failed": True, "fix_ran": True},
                    {"id": "bad", "broken_failed": True, "fix_ran": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = _summarize_debugger(path)
    assert summary["pass_rate"] == 0.5
    assert summary["failures"] == ["bad"]


def test_summarize_interpreter_maps_fact_score_to_avg_score(tmp_path):
    path = tmp_path / "interpreter.json"
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "n_cases": 1,
                    "pass_count": 1,
                    "pass_rate": 1.0,
                    "avg_fact_score": 0.875,
                    "judge_coverage": None,
                    "judge_avg": None,
                    "judge_hallucination_avg": None,
                    "failures": [],
                    "judge_na_cases": [],
                },
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    summary = _summarize_interpreter(path)
    assert summary["kind"] == "interpreter"
    assert summary["avg_score"] == 0.875


def test_suite_passed_requires_command_and_thresholds():
    command = CommandResult(
        name="suite",
        command=["python", "script.py"],
        returncode=0,
        elapsed_s=0.1,
        timed_out=False,
        stdout_tail="",
        stderr_tail="",
    )
    assert _suite_passed(
        {
            "pass_rate": 0.95,
            "avg_score": 0.8,
            "judge_coverage": 0.9,
            "judge_hallucination_avg": 0.95,
        },
        command,
        min_pass_rate=0.9,
        min_avg_score=0.75,
        min_judge_coverage=0.8,
        min_hallucination=0.9,
    )
    assert not _suite_passed(
        {"pass_rate": 0.95, "avg_score": 0.8},
        CommandResult("suite", [], None, 10.0, True, "", ""),
        min_pass_rate=0.9,
        min_avg_score=0.75,
        min_judge_coverage=0.8,
        min_hallucination=0.9,
    )


def test_render_summary_lists_judge_na_cases():
    md = _render_summary(
        {
            "timestamp_utc": "2026-05-19T00:00:00+00:00",
            "provider": "minimax",
            "judge_provider": "minimax",
            "output_dir": "/tmp/eval",
            "passed": False,
            "thresholds": {
                "min_pass_rate": {"required": ">= 90.0%", "actual": "50.0%"}
            },
            "suites": [
                {
                    "name": "advisor",
                    "passed": False,
                    "summary": {
                        "kind": "advisor_or_unified",
                        "n_cases": 1,
                        "pass_count": 0,
                        "pass_rate": 0.0,
                        "avg_score": 0.4,
                        "judge_coverage": 0.0,
                        "judge_avg": None,
                        "judge_hallucination_avg": None,
                        "judge_na_cases": ["f01"],
                        "low_score_cases": [
                            {"id": "f01", "score": 0.4, "judge_avg": None}
                        ],
                    },
                    "command_result": {
                        "command": ["python", "scripts/run_eval.py"],
                        "returncode": 0,
                        "timed_out": False,
                    },
                }
            ],
        }
    )
    assert "judge N/A" in md
    assert "f01" in md
