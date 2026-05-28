"""Pass@1 evaluation for Agent B on broken MATLAB scripts.

Uses the golden ``fix.m`` files in ``tests/fixtures/broken_matlab/`` so
the test is purely a sandbox + classifier check (no LLM, no API key).
Requires a local MATLAB install.

For the LLM-driven version, see ``scripts/run_debugger_eval.py --strategy llm``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_debugger_eval import _load_cases, _strategy_golden  # noqa: E402


pytestmark = [pytest.mark.requires_matlab, pytest.mark.integration]
DEFAULT_TIMEOUT = 90
TIMEOUT_CASE_TIMEOUT = 8
FIX_TIMEOUT = 90


def _cases():
    try:
        return _load_cases("matlab")
    except FileNotFoundError:
        return []


@pytest.fixture(
    params=_cases(),
    ids=lambda c: c["meta"]["id"],
)
def case(request):
    return request.param


def _timeout_for(case):
    expected = case["meta"].get("expected_categories") or [
        case["meta"].get("expected_category")
    ]
    if "timeout" in expected:
        return TIMEOUT_CASE_TIMEOUT
    return DEFAULT_TIMEOUT


def test_broken_fails_and_fix_runs(case):
    r = _strategy_golden(
        case,
        language="matlab",
        timeout=_timeout_for(case),
        fix_timeout=FIX_TIMEOUT,
    )
    assert r["broken_failed"], (
        f"{r['id']}: broken.m must fail. notes={r.get('notes', '')}"
    )
    assert r["fix_ran"], f"{r['id']}: fix.m must run cleanly"
    assert r["category_match"], (
        f"{r['id']}: regex classifier returned {r['regex_category']}, "
        f"expected one of {r['expected_categories']}"
    )
    assert r["error_substr_match"], (
        f"{r['id']}: expected error substring(s) not found in stderr"
    )


def test_pass_rate_threshold_meets_l3(tmp_path):
    """Aggregate gate — must hit ≥70% Pass@1 (L3 floor in docs/TASKS.md)."""
    cases = _cases()
    assert cases, "no broken_matlab fixtures discovered"
    results = [
        _strategy_golden(
            c,
            language="matlab",
            timeout=_timeout_for(c),
            fix_timeout=FIX_TIMEOUT,
        )
        for c in cases
    ]
    n_pass = sum(1 for r in results if r.get("broken_failed") and r.get("fix_ran"))
    rate = n_pass / len(results)
    # Persist the run so CI / dev can inspect it.
    (tmp_path / "debugger_eval_matlab.json").write_text(
        json.dumps({"n_total": len(results), "n_pass": n_pass, "pass_rate": rate,
                    "results": results}, indent=2)
    )
    assert rate >= 0.70, f"Pass@1 {rate:.0%} < 70% floor; results: {results}"
