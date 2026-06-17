#!/usr/bin/env python
"""Deterministic multi-node evaluation for OptiProfiler Agent.

This runner evaluates internal workflow nodes that are not plain chat
questions: debugger classifiers/checkers/adapters and interpreter
loaders/summaries. It complements ``run_eval.py`` (Advisor/Unified
question-answer eval) and ``run_debugger_eval.py`` (end-to-end Pass@1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optiprofiler_agent.common.interface_adapter import analyze_solver
from optiprofiler_agent.advisor.scaffold_feature import scaffold_custom_feature
from optiprofiler_agent.advisor.plib_scanner import scan_local_plib
from optiprofiler_agent.advisor.plib_wrapper import scaffold_plib_wrapper, smoke_test_plib_wrapper
from optiprofiler_agent.debugger.debugger import (
    _collect_web_debug_context,
    _format_web_debug_context,
)
from optiprofiler_agent.debugger.error_classifier import classify_error
from optiprofiler_agent.debugger.error_classifier import ErrorClassification
from optiprofiler_agent.interpreter.result_loader import _parse_log_txt, load_results
from optiprofiler_agent.interpreter.summary import build_summary
from optiprofiler_agent.validators.matlab_checker import check_matlab_code

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_CASES_DIR = REPO_ROOT / "tests" / "eval_cases"


def _load_cases(paths: list[str] | None) -> list[dict]:
    if paths:
        files = [Path(p) for p in paths]
    else:
        files = [
            EVAL_CASES_DIR / "advisor_scaffold_feature.json",
            EVAL_CASES_DIR / "advisor_plib_scan.json",
            EVAL_CASES_DIR / "debugger_matlab.json",
            EVAL_CASES_DIR / "debugger_web.json",
            EVAL_CASES_DIR / "interpreter_matlab.json",
        ]
    cases: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8") as f:
            cases.extend(json.load(f))
    return cases


def _select_cases(cases: list[dict], ids: str | None) -> list[dict]:
    if not ids:
        return cases
    wanted = {item.strip() for item in ids.split(",") if item.strip()}
    selected = [case for case in cases if case.get("id") in wanted]
    missing = wanted - {case.get("id") for case in selected}
    if missing:
        raise ValueError(f"Unknown case id(s): {', '.join(sorted(missing))}")
    return selected


def _fixture_path(case: dict) -> Path:
    return REPO_ROOT / case["fixture"]


def _parse_snippet(text: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = Path(f.name)
    try:
        _, scores, runs = _parse_log_txt(path)
    finally:
        path.unlink(missing_ok=True)
    return scores, runs


def _eval_debugger_case(case: dict) -> tuple[bool, dict]:
    task = case["task"]
    lang = case.get("language", "matlab")

    if task == "classify_error":
        result = classify_error(case["error_text"], language=lang)
        passed = result.error_type == case["expected_category"]
        if case.get("expected_module"):
            passed = passed and result.module_name == case["expected_module"]
        return passed, {
            "actual_category": result.error_type,
            "expected_category": case["expected_category"],
            "actual_module": result.module_name,
            "expected_module": case.get("expected_module"),
        }

    if task == "validate_code":
        check = check_matlab_code(case["code"])
        if case.get("expect_valid"):
            passed = not check.has_errors
        else:
            needle = case.get("must_contain_issue", "").lower()
            passed = check.has_errors and (
                not needle or any(needle in err.lower() for err in check.errors)
            )
        return passed, {"errors": check.errors}

    if task == "analyze_solver":
        analysis = analyze_solver(case["code"], language=lang)
        passed = analysis.needs_wrapper == case["expect_needs_wrapper"]
        if case.get("expect_func_name"):
            passed = passed and analysis.func_name == case["expect_func_name"]
        return passed, {
            "needs_wrapper": analysis.needs_wrapper,
            "func_name": analysis.func_name,
            "expected_needs_wrapper": case["expect_needs_wrapper"],
        }

    if task == "web_context":
        cls_data = case.get("classification", {})
        classification = ErrorClassification(
            error_type=cls_data.get("error_type", "runtime_error"),
            confidence=cls_data.get("confidence", 1.0),
            details=cls_data.get("details", "test case"),
            module_name=cls_data.get("module_name"),
        )
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "1"}, clear=False), patch(
            "optiprofiler_agent.debugger.debugger._run_debugger_web_search",
            return_value=case.get("mock_search_result", ""),
        ):
            context = _collect_web_debug_context(
                code=case.get("code", ""),
                error=case.get("error_text", ""),
                classification=classification,
                language=lang,
            )
        rendered = _format_web_debug_context(context)
        expect_context = bool(case.get("expect_context"))
        passed = bool(context) == expect_context
        missing = [
            needle for needle in case.get("must_contain", [])
            if needle not in rendered
        ]
        passed = passed and not missing
        return passed, {
            "has_context": bool(context),
            "query": context[0] if context else None,
            "missing": missing,
            "rendered_excerpt": rendered[:500],
        }

    raise ValueError(f"Unknown debugger task: {task}")


def _eval_advisor_case(case: dict) -> tuple[bool, dict]:
    task = case["task"]

    if task == "scaffold_feature":
        result = scaffold_custom_feature(
            description=case["description"],
            feature_name=case.get("feature_name", ""),
            n_runs=case.get("n_runs"),
        )
        code = result.code
        missing = [needle for needle in case.get("must_contain", []) if needle not in code]
        forbidden = [needle for needle in case.get("must_not_contain", []) if needle in code]
        expected_mods = case.get("expected_modifiers", [])
        missing_mods = [mod for mod in expected_mods if mod not in result.selected_modifiers]
        passed = result.ok and not missing and not forbidden and not missing_mods
        return passed, {
            "selected_modifiers": result.selected_modifiers,
            "missing": missing,
            "forbidden": forbidden,
            "missing_modifiers": missing_mods,
            "validation_errors": result.validation_errors,
            "validation_warnings": result.validation_warnings,
            "code_excerpt": code[:500],
        }

    if task == "scan_local_plib":
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel_path, content in case.get("files", {}).items():
                path = root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            evidence = scan_local_plib(root, library_name=case.get("library_name"))
        expect = case.get("expect", {})
        missing_languages = [
            item for item in expect.get("languages", [])
            if item not in evidence.languages
        ]
        missing_dependencies = [
            item for item in expect.get("dependencies", [])
            if item not in evidence.dependencies
        ]
        passed = (
            not missing_languages
            and not missing_dependencies
            and evidence.recommended_adapter_shape == expect.get(
                "recommended_adapter_shape",
                evidence.recommended_adapter_shape,
            )
            and len(evidence.loader_hints) >= expect.get("loader_hints_min", 0)
            and len(evidence.selector_hints) >= expect.get("selector_hints_min", 0)
        )
        return passed, {
            "languages": evidence.languages,
            "dependencies": evidence.dependencies,
            "loader_hints": evidence.loader_hints,
            "selector_hints": evidence.selector_hints,
            "recommended_adapter_shape": evidence.recommended_adapter_shape,
            "missing_languages": missing_languages,
            "missing_dependencies": missing_dependencies,
        }

    if task == "scaffold_plib_wrapper":
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel_path, content in case.get("files", {}).items():
                path = root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            scaffold = scaffold_plib_wrapper(
                root,
                library_name=case.get("library_name"),
                staging_dir=root / "stage",
            )
            smoke = smoke_test_plib_wrapper(scaffold.staging_dir, scaffold.library_name)
        expect = case.get("expect", {})
        expected_names = expect.get("tested_problem_names", [])
        missing_names = [
            name for name in expected_names
            if name not in smoke.tested_problem_names
        ]
        passed = smoke.ok == expect.get("smoke_ok", True) and not missing_names
        return passed, {
            "staging_dir": str(scaffold.staging_dir),
            "tools_path": str(scaffold.tools_path),
            "warnings": scaffold.warnings,
            "smoke_ok": smoke.ok,
            "tested_problem_names": smoke.tested_problem_names,
            "missing_names": missing_names,
            "stderr": smoke.stderr[-500:],
        }

    raise ValueError(f"Unknown advisor task: {task}")


def _eval_interpreter_case(case: dict) -> tuple[bool, dict]:
    task = case["task"]
    expect = case.get("expect", {})

    if task == "load_results":
        root = _fixture_path(case)
        results = load_results(root)
        passed = (
            results.language == expect.get("language", "matlab")
            and len(results.run_results) >= expect.get("min_run_results", 1)
            and len(results.solver_scores) >= expect.get("min_solver_scores", 1)
        )
        return passed, {
            "language": results.language,
            "run_results": len(results.run_results),
            "solver_scores": len(results.solver_scores),
        }

    if task == "parse_log_snippet":
        scores, runs = _parse_snippet(case.get("log") or _DEFAULT_LOG_SNIPPET)
        passed = (
            len(runs) >= expect.get("min_run_results", 1)
            and len(scores) >= expect.get("min_scores", 1)
        )
        return passed, {"run_results": len(runs), "scores": len(scores)}

    if task == "parse_log_snippet_padded":
        _, runs = _parse_snippet(case["log"])
        passed = len(runs) >= expect.get("min_run_results", 1)
        return passed, {"run_results": len(runs)}

    if task == "build_summary":
        summary = build_summary(_fixture_path(case), read_profiles=False)
        passed = (
            summary.language == expect.get("language", "matlab")
            and len(summary.solver_names) >= expect.get("min_solvers", 1)
        )
        if expect.get("has_rankings"):
            passed = passed and bool(summary.rankings)
        return passed, {
            "language": summary.language,
            "solver_count": len(summary.solver_names),
            "ranking_count": len(summary.rankings),
        }

    if task == "summary_json_roundtrip":
        data = json.loads(build_summary(_fixture_path(case), read_profiles=False).to_json())
        passed = data["language"] == expect.get("language", "matlab")
        missing = [key for key in expect.get("must_have_keys", []) if key not in data]
        passed = passed and not missing
        return passed, {"language": data.get("language"), "missing_keys": missing}

    if task == "profile_curves_flag":
        summary = build_summary(_fixture_path(case), read_profiles=True)
        passed = summary.profile_curves_available == expect["profile_curves_available"]
        return passed, {"profile_curves_available": summary.profile_curves_available}

    raise ValueError(f"Unknown interpreter task: {task}")


def run_cases(cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            if case.get("agent") == "advisor":
                passed, details = _eval_advisor_case(case)
            elif case.get("agent") == "debugger":
                passed, details = _eval_debugger_case(case)
            elif case.get("agent") == "interpreter":
                passed, details = _eval_interpreter_case(case)
            else:
                raise ValueError(f"Unsupported agent: {case.get('agent')}")
            error = None
        except Exception as exc:  # noqa: BLE001 - eval records errors as data.
            passed = False
            details = {}
            error = f"{type(exc).__name__}: {exc}"

        results.append({
            "id": case.get("id"),
            "agent": case.get("agent"),
            "task": case.get("task"),
            "passed": passed,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "details": details,
            "error": error,
        })
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    by_agent: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    by_task: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    for item in results:
        for bucket, key in ((by_agent, item["agent"]), (by_task, item["task"])):
            bucket[key]["total"] += 1
            bucket[key]["passed"] += int(item["passed"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "by_agent": dict(by_agent),
        "by_task": dict(by_task),
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Multi-Node Agent Evaluation Report",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Cases: `{summary['passed']}/{summary['total']}`",
        f"- Pass rate: `{summary['pass_rate'] * 100:.1f}%`",
        "",
        "## By Agent",
        "",
        "| Agent | Passed | Total | Pass Rate |",
        "|---|---:|---:|---:|",
    ]
    for agent, row in sorted(summary["by_agent"].items()):
        rate = row["passed"] / row["total"] if row["total"] else 0.0
        lines.append(f"| `{agent}` | {row['passed']} | {row['total']} | {rate*100:.1f}% |")

    lines.extend([
        "",
        "## Cases",
        "",
        "| Case | Agent | Task | Status | Time (s) |",
        "|---|---|---|---:|---:|",
    ])
    for item in payload["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(
            f"| `{item['id']}` | `{item['agent']}` | `{item['task']}` | "
            f"{status} | {item['elapsed_s']} |"
        )

    failures = [item for item in payload["results"] if not item["passed"]]
    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.append(f"### `{item['id']}`")
            if item.get("error"):
                lines.append(f"- Error: `{item['error']}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(item.get("details", {}), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", action="append", help="Case JSON file; repeatable")
    parser.add_argument("--case-ids", default=None, help="Comma-separated case ids")
    parser.add_argument("--output", default=None, help="Write JSON report")
    parser.add_argument("--report", default=None, help="Write Markdown report")
    args = parser.parse_args()

    try:
        cases = _select_cases(_load_cases(args.cases), args.case_ids)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results = run_cases(cases)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "summary": summarize(results),
        "results": results,
    }

    print(
        f"[run_multinode_eval] pass={payload['summary']['passed']}/"
        f"{payload['summary']['total']} ({payload['summary']['pass_rate']*100:.1f}%)"
    )
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  {status:4s} {item['id']}")

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Wrote {args.output}")
    if args.report:
        Path(args.report).write_text(render_markdown(payload))
        print(f"Wrote {args.report}")
    return 0 if payload["summary"]["pass_rate"] == 1.0 else 1


_DEFAULT_LOG_SNIPPET = """\
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


if __name__ == "__main__":
    raise SystemExit(main())
