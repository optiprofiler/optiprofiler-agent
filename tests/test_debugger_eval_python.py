"""Pass@1 evaluation for Agent B on broken Python scripts.

Uses the golden ``fix.py`` files in ``tests/fixtures/broken_python/`` so
the test is purely a local runner + classifier check (no LLM, no API key).

For the LLM-driven version, see ``scripts/run_debugger_eval.py --language python --strategy llm``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_debugger_eval import _load_cases, _render_markdown_report, _select_cases, _strategy_golden  # noqa: E402


pytestmark = [pytest.mark.integration]


def _cases():
    try:
        return _load_cases("python")
    except FileNotFoundError:
        return []


@pytest.fixture(
    params=_cases(),
    ids=lambda c: c["meta"]["id"],
)
def case(request):
    return request.param


def test_broken_fails_and_fix_runs(case):
    r = _strategy_golden(case, language="python", timeout=2)
    assert r["broken_failed"], (
        f"{r['id']}: broken.py must fail. notes={r.get('notes', '')}"
    )
    assert r["fix_ran"], f"{r['id']}: fix.py must run cleanly"
    assert r["category_match"], (
        f"{r['id']}: regex classifier returned {r['regex_category']}, "
        f"expected one of {r['expected_categories']}"
    )
    assert r["error_substr_match"], (
        f"{r['id']}: expected error substring(s) not found in stderr"
    )


def test_pass_rate_threshold_meets_l3(tmp_path):
    """Aggregate gate — must hit >=70% Pass@1 (L3 floor in docs/TASKS.md)."""
    cases = _cases()
    assert cases, "no broken_python fixtures discovered"
    results = [_strategy_golden(c, language="python", timeout=2) for c in cases]
    n_pass = sum(1 for r in results if r.get("broken_failed") and r.get("fix_ran"))
    rate = n_pass / len(results)
    (tmp_path / "debugger_eval_python.json").write_text(
        json.dumps(
            {
                "n_total": len(results),
                "n_pass": n_pass,
                "pass_rate": rate,
                "results": results,
            },
            indent=2,
        )
    )
    assert rate >= 0.70, f"Pass@1 {rate:.0%} < 70% floor; results: {results}"


def test_case_selection_by_name_and_limit():
    cases = _load_cases("python")
    selected = _select_cases(cases, "name_error,index_oob", limit=1)
    assert len(selected) == 1
    assert selected[0]["meta"]["id"] == "index_oob"


def test_markdown_report_includes_summary_table():
    md = _render_markdown_report({
        "agent": "debugger",
        "strategy": "golden",
        "language": "python",
        "provider": None,
        "model": None,
        "timestamp_utc": "2026-05-18T00:00:00+00:00",
        "n_total": 1,
        "n_pass": 1,
        "pass_rate": 1.0,
        "results": [
            {
                "id": "name_error",
                "broken_failed": True,
                "fix_ran": True,
                "regex_category": "runtime_error",
                "category_match": True,
                "elapsed_s": 0.1,
            }
        ],
    })
    assert "# Debugger Eval Last Run" in md
    assert "| `name_error` | PASS | `runtime_error` | yes | - | 0.1 |" in md


def test_golden_strategy_can_use_separate_fix_timeout():
    cases = _load_cases("python")
    timeout_case = next(c for c in cases if c["meta"]["id"] == "timeout_loop")
    result = _strategy_golden(
        timeout_case,
        language="python",
        timeout=1,
        fix_timeout=2,
    )
    assert result["broken_failed"]
    assert result["fix_ran"]
