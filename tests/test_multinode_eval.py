"""Tests for scripts/run_multinode_eval.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_multinode_eval import render_markdown, run_cases, summarize  # noqa: E402


def test_debugger_classify_case_passes():
    results = run_cases([
        {
            "id": "dep",
            "agent": "debugger",
            "task": "classify_error",
            "language": "matlab",
            "error_text": "Undefined function or variable 'cobyqa'.",
            "expected_category": "dependency_missing",
            "expected_module": "cobyqa",
        }
    ])
    assert results[0]["passed"] is True
    assert results[0]["details"]["actual_category"] == "dependency_missing"


def test_summary_groups_by_agent_and_task():
    summary = summarize([
        {"agent": "debugger", "task": "classify_error", "passed": True},
        {"agent": "debugger", "task": "validate_code", "passed": False},
    ])
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["by_agent"]["debugger"] == {"total": 2, "passed": 1}
    assert summary["by_task"]["classify_error"] == {"total": 1, "passed": 1}


def test_markdown_report_lists_failures():
    md = render_markdown({
        "timestamp_utc": "2026-05-18T00:00:00+00:00",
        "summary": {
            "total": 1,
            "passed": 0,
            "pass_rate": 0.0,
            "by_agent": {"debugger": {"total": 1, "passed": 0}},
            "by_task": {"classify_error": {"total": 1, "passed": 0}},
        },
        "results": [
            {
                "id": "bad",
                "agent": "debugger",
                "task": "classify_error",
                "passed": False,
                "elapsed_s": 0.01,
                "details": {"actual_category": "runtime_error"},
                "error": None,
            }
        ],
    })
    assert "# Multi-Node Agent Evaluation Report" in md
    assert "## Failures" in md
    assert "`bad`" in md
