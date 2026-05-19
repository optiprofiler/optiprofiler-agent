"""Deterministic (no-LLM) task-level eval for Agent B/C.

These cases mirror ``tests/eval_cases/debugger_matlab.json`` and
``interpreter_matlab.json``. Full LLM end-to-end eval is tracked in ROADMAP D1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optiprofiler_agent.common.interface_adapter import analyze_solver
from optiprofiler_agent.debugger.error_classifier import classify_error
from optiprofiler_agent.interpreter.result_loader import _parse_log_txt, load_results
from optiprofiler_agent.interpreter.summary import build_summary
from optiprofiler_agent.validators.matlab_checker import check_matlab_code

_EVAL_DIR = Path(__file__).parent / "eval_cases"
_LOG_SNIPPET = """\
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


def _load_cases(name: str) -> list[dict]:
    path = _EVAL_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestDebuggerMatlabEval:
    @pytest.fixture(params=_load_cases("debugger_matlab.json"), ids=lambda c: c["id"])
    def case(self, request):
        return request.param

    def test_case(self, case):
        task = case["task"]
        lang = case.get("language", "matlab")

        if task == "classify_error":
            result = classify_error(case["error_text"], language=lang)
            assert result.error_type == case["expected_category"]
            if case.get("expected_module"):
                assert result.module_name == case["expected_module"]

        elif task == "validate_code":
            check = check_matlab_code(case["code"])
            if case.get("expect_valid"):
                assert not check.has_errors, check.errors
            else:
                assert check.has_errors
                needle = case.get("must_contain_issue", "").lower()
                if needle:
                    assert any(needle in e.lower() for e in check.errors)

        elif task == "analyze_solver":
            analysis = analyze_solver(case["code"], language=lang)
            assert analysis.needs_wrapper == case["expect_needs_wrapper"]
            if case.get("expect_func_name"):
                assert analysis.func_name == case["expect_func_name"]
        else:
            pytest.fail(f"Unknown debugger task: {task}")


class TestInterpreterMatlabEval:
    @pytest.fixture(params=_load_cases("interpreter_matlab.json"), ids=lambda c: c["id"])
    def case(self, request):
        return request.param

    def test_case(self, case):
        task = case["task"]
        expect = case.get("expect", {})

        if task == "load_results":
            root = Path(__file__).parent.parent / case["fixture"]
            if not root.is_dir():
                pytest.skip(f"Fixture missing: {root}")
            results = load_results(root)
            assert results.language == expect.get("language", "matlab")
            assert len(results.run_results) >= expect.get("min_run_results", 1)
            assert len(results.solver_scores) >= expect.get("min_solver_scores", 1)

        elif task == "parse_log_snippet":
            scores, runs = _parse_snippet(_LOG_SNIPPET)
            assert len(runs) >= expect.get("min_run_results", 1)
            assert len(scores) >= expect.get("min_scores", 1)

        elif task == "parse_log_snippet_padded":
            _, runs = _parse_snippet(case["log"])
            assert len(runs) >= expect.get("min_run_results", 1)

        elif task == "build_summary":
            root = Path(__file__).parent.parent / case["fixture"]
            if not root.is_dir():
                pytest.skip(f"Fixture missing: {root}")
            summary = build_summary(root, read_profiles=False)
            assert summary.language == expect.get("language", "matlab")
            assert len(summary.solver_names) >= expect.get("min_solvers", 1)
            if expect.get("has_rankings"):
                assert summary.rankings

        elif task == "summary_json_roundtrip":
            import json as _json
            root = Path(__file__).parent.parent / case["fixture"]
            if not root.is_dir():
                pytest.skip(f"Fixture missing: {root}")
            summary = build_summary(root, read_profiles=False)
            data = _json.loads(summary.to_json())
            assert data["language"] == expect.get("language", "matlab")
            for k in expect.get("must_have_keys", []):
                assert k in data, f"missing key {k} in summary JSON"

        elif task == "profile_curves_flag":
            root = Path(__file__).parent.parent / case["fixture"]
            if not root.is_dir():
                pytest.skip(f"Fixture missing: {root}")
            summary = build_summary(root, read_profiles=True)
            assert summary.profile_curves_available == expect["profile_curves_available"]

        else:
            pytest.fail(f"Unknown interpreter task: {task}")


def _parse_snippet(text: str):
    """Write ``text`` to a temp file and run it through _parse_log_txt."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = Path(f.name)
    try:
        _, scores, runs = _parse_log_txt(path)
    finally:
        path.unlink(missing_ok=True)
    return scores, runs
