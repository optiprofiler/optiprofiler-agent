#!/usr/bin/env python
"""Evaluation harness for Agent C (Interpreter) reports.

The deterministic gate checks whether rendered reports preserve the facts in
``BenchmarkSummary``. Optional LLM-as-Judge scores report quality on
correctness/completeness/grounding/hallucination.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optiprofiler_agent.config import AgentConfig, LLMConfig
from optiprofiler_agent.interpreter.interpreter import interpret
from optiprofiler_agent.interpreter.report_schema import (
    AnomaliesSection,
    ConvergenceIssuesSection,
    DataProfileSection,
    BenchmarkReport,
    PerformanceProfileSection,
    RecommendationsSection,
    ReportOverview,
)
from optiprofiler_agent.interpreter.renderer import render_markdown
from optiprofiler_agent.interpreter.summary import BenchmarkSummary, build_summary


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "matlab_results" / "experiment_matlab"
_JUDGE_DIMS = ["correctness", "completeness", "grounding", "hallucination"]


_JUDGE_SYSTEM = """\
You are an expert evaluator for OptiProfiler benchmark-analysis reports.

Score the report on each dimension from 0 to 10:
1. correctness: reported winners, solver names, scores, and caveats match the facts.
2. completeness: the report covers setup, rankings, convergence/anomalies, and recommendations.
3. grounding: claims are supported by the supplied BenchmarkSummary facts.
4. hallucination: 10 = no unsupported solver/problem/number claims; 0 = many fabricated claims.

Reply with ONLY a JSON object:
{"correctness": <0-10>, "completeness": <0-10>, "grounding": <0-10>,
 "hallucination": <0-10>, "reason": "<brief>"}
"""


def _extract_json_object(text: str) -> dict:
    """Import the shared judge parser lazily to keep this script standalone."""
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_eval import _extract_json_object as extract  # noqa: PLC0415

    return extract(text)


def _score(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if raw > 1.0:
        raw /= 10.0
    return max(0.0, min(1.0, raw))


def _top_solvers(summary: BenchmarkSummary) -> tuple[str | None, str | None]:
    scores = summary.solver_scores or {}
    if not scores:
        return None, None
    ordered = sorted(scores, key=scores.get, reverse=True)
    winner = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    return winner, runner_up


def _deterministic_report(summary: BenchmarkSummary) -> str:
    """Render a known-good report object for fast evaluator regression tests."""
    winner, runner_up = _top_solvers(summary)
    winner = winner or (summary.solver_names[0] if summary.solver_names else "n/a")
    runner_up = runner_up or winner
    score = summary.solver_scores.get(winner, 0.0)
    report = BenchmarkReport(
        key_findings=[
            f"{winner} has the highest aggregate score ({score:.4f}).",
            f"{runner_up} is the nearest comparator in this experiment.",
        ],
        overview=ReportOverview(
            headline=f"{winner} leads this OptiProfiler benchmark.",
            setup=(
                f"The experiment compares {', '.join(summary.solver_names)} on "
                f"{', '.join(summary.problem_libraries) or 'the selected'} problems "
                f"with dimensions {summary.dimension_range[0]}-{summary.dimension_range[1]} "
                f"and feature stamp {summary.feature_stamp}."
            ),
        ),
        performance_profile=PerformanceProfileSection(
            winner_at_tau1=winner,
            most_robust=winner,
            ranking_change=(
                f"The aggregate ranking is led by {winner}; {runner_up} remains the "
                "main baseline for comparison."
            ),
        ),
        data_profile=DataProfileSection(
            most_efficient=winner,
            commentary=(
                f"{winner} is the preferred solver by aggregate score. Inspect "
                "function-evaluation histories before generalising beyond this fixture."
            ),
        ),
        convergence_issues=ConvergenceIssuesSection(entries=[], common_failure_problems=[]),
        anomalies=AnomaliesSection(entries=[]),
        recommendations=RecommendationsSection(
            actions=[],
            caveats="This conclusion is based on the parsed log/report facts.",
        ),
    )
    return render_markdown(report, summary)


def _report_text(results_dir: Path, strategy: str, provider: str | None, model: str | None) -> str:
    if strategy == "deterministic":
        return _deterministic_report(build_summary(results_dir, read_profiles=False))
    cfg = AgentConfig(llm=LLMConfig(provider=provider, model=model))
    return interpret(
        results_dir=results_dir,
        config=cfg,
        language="English",
        read_profiles=False,
        llm_enabled=True,
        output_format="markdown",
    )


def _unknown_solver_mentions(report: str, summary: BenchmarkSummary) -> list[str]:
    """Best-effort detector for solver-like tokens not present in the summary."""
    allowed = set(summary.solver_names)
    known_non_solvers = {
        "Benchmark",
        "OptiProfiler",
        "Agent",
        "Profile",
        "Problem",
        "MATLAB",
        "Python",
        "DFO",
    }
    # Catch common solver-like tokens. We intentionally do not treat every
    # backticked word as a solver because the report also backticks feature
    # stamps such as `plain`.
    candidates = set(
        token
        for token in re.findall(
            r"\b(?:fmin\w+|solver_[A-Za-z0-9_]+|fake_solver|cobyla|nelder\w*)\b",
            report,
            flags=re.I,
        )
    )
    return sorted(
        token for token in candidates
        if token not in allowed and token not in known_non_solvers
    )


def _contains_score(report: str, score: float) -> bool:
    return f"{score:.4f}" in report or f"{score:.3f}" in report or f"{score:.2f}" in report


def fact_check_report(report: str, summary: BenchmarkSummary) -> dict[str, Any]:
    winner, runner_up = _top_solvers(summary)
    checks: dict[str, bool] = {}
    if winner:
        checks["mentions_winner"] = winner in report
        checks["mentions_winner_score"] = _contains_score(report, summary.solver_scores[winner])
    if runner_up:
        checks["mentions_runner_up"] = runner_up in report
    checks["mentions_all_solvers"] = all(name in report for name in summary.solver_names)
    checks["has_recommendations_section"] = "## Recommendations" in report
    checks["has_setup_section"] = "## Experiment Setup" in report
    unknown_solvers = _unknown_solver_mentions(report, summary)
    checks["no_unknown_solver_mentions"] = not unknown_solvers
    # User-facing report should not expose the internal execution-language field.
    checks["no_language_field"] = "| Language |" not in report and "**Language**" not in report
    passed = all(checks.values())
    return {
        "passed": passed,
        "score": sum(checks.values()) / len(checks) if checks else 0.0,
        "checks": checks,
        "unknown_solver_mentions": unknown_solvers,
        "winner": winner,
        "runner_up": runner_up,
    }


def score_with_judge(report: str, summary: BenchmarkSummary, provider: str | None, model: str | None) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    from optiprofiler_agent.common.llm_client import create_llm

    llm = create_llm(LLMConfig(provider=provider, model=model))
    compact_summary = {
        "solver_names": summary.solver_names,
        "solver_scores": summary.solver_scores,
        "rankings": summary.rankings[:5],
        "problem_libraries": summary.problem_libraries,
        "dimension_range": list(summary.dimension_range),
        "profile_curves_available": summary.profile_curves_available,
    }
    prompt = (
        "BenchmarkSummary facts:\n"
        f"{json.dumps(compact_summary, ensure_ascii=False, indent=2)}\n\n"
        "Report to grade:\n"
        f"{report}"
    )
    try:
        result = llm.invoke([
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        data = _extract_json_object(result.content)
        scores = {dim: _score(data.get(dim)) for dim in _JUDGE_DIMS}
        avg = sum(scores.values()) / len(scores)
        return {
            "judge_scores": scores,
            "judge_avg": round(avg, 3),
            "judge_reason": data.get("reason", ""),
        }
    except Exception as exc:  # noqa: BLE001 - eval captures provider/parser errors.
        return {
            "judge_scores": None,
            "judge_avg": None,
            "judge_reason": f"Judge error: {type(exc).__name__}: {exc}",
        }


def run_case(
    case: dict,
    strategy: str,
    provider: str | None,
    model: str | None,
    judge: bool,
    judge_provider: str | None,
    judge_model: str | None,
) -> dict:
    t0 = time.perf_counter()
    results_dir = Path(case.get("results_dir", DEFAULT_FIXTURE)).expanduser()
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    summary = build_summary(results_dir, read_profiles=case.get("read_profiles", False))
    report = _report_text(results_dir, strategy, provider, model)
    facts = fact_check_report(report, summary)
    row = {
        "id": case.get("id", results_dir.name),
        "results_dir": str(results_dir),
        "strategy": strategy,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "report_excerpt": report[:2000],
        **facts,
    }
    if judge:
        row.update(score_with_judge(report, summary, judge_provider or provider, judge_model))
    return row


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for row in results if row["passed"])
    judged = [row for row in results if row.get("judge_avg") is not None]
    hallucination = [
        row["judge_scores"]["hallucination"]
        for row in judged
        if row.get("judge_scores") and "hallucination" in row["judge_scores"]
    ]
    return {
        "n_cases": total,
        "pass_count": passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_fact_score": (
            sum(float(row["score"]) for row in results) / total if total else 0.0
        ),
        "judge_coverage": len(judged) / total if total and any("judge_avg" in row for row in results) else None,
        "judge_avg": (
            sum(row["judge_avg"] for row in judged) / len(judged) if judged else None
        ),
        "judge_hallucination_avg": (
            sum(hallucination) / len(hallucination) if hallucination else None
        ),
        "failures": [row["id"] for row in results if not row["passed"]],
        "judge_na_cases": [row["id"] for row in results if "judge_avg" in row and row.get("judge_avg") is None],
    }


def render_markdown_report(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Interpreter Report Evaluation",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Strategy: `{payload['strategy']}`",
        f"- Cases: `{summary['pass_count']}/{summary['n_cases']}`",
        f"- Pass rate: `{summary['pass_rate'] * 100:.1f}%`",
        f"- Average fact score: `{summary['avg_fact_score']:.3f}`",
    ]
    if summary.get("judge_avg") is not None:
        lines.append(f"- Judge average: `{summary['judge_avg']:.3f}`")
        lines.append(f"- Judge hallucination: `{summary['judge_hallucination_avg']:.3f}`")

    lines.extend([
        "",
        "| Case | Status | Fact Score | Winner | Runner-up | Judge |",
        "|---|---:|---:|---|---|---:|",
    ])
    for row in payload["results"]:
        judge = row.get("judge_avg")
        judge_text = f"{judge:.3f}" if judge is not None else "n/a"
        lines.append(
            f"| `{row['id']}` | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{row['score']:.3f} | `{row.get('winner')}` | "
            f"`{row.get('runner_up')}` | {judge_text} |"
        )

    failures = [row for row in payload["results"] if not row["passed"]]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            failed = [name for name, ok in row["checks"].items() if not ok]
            lines.append(f"- `{row['id']}` failed checks: {', '.join(failed)}")

    return "\n".join(lines).rstrip() + "\n"


def _default_cases() -> list[dict]:
    return [
        {
            "id": "matlab_fixture_report",
            "results_dir": str(DEFAULT_FIXTURE),
            "read_profiles": False,
        }
    ]


def _load_cases(path: str | None) -> list[dict]:
    if not path:
        return _default_cases()
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", default=None)
    parser.add_argument("--strategy", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    results = []
    for case in _load_cases(args.cases):
        print(f"[run_interpreter_eval] {case.get('id', 'case')} ...", flush=True)
        try:
            row = run_case(
                case,
                strategy=args.strategy,
                provider=args.provider,
                model=args.model,
                judge=args.judge,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
            )
        except Exception as exc:  # noqa: BLE001 - eval records errors.
            row = {
                "id": case.get("id", "case"),
                "passed": False,
                "score": 0.0,
                "checks": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(row)
        print(f"  {'PASS' if row['passed'] else 'FAIL'} fact_score={row['score']:.3f}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "strategy": args.strategy,
        "provider": args.provider,
        "judge_provider": args.judge_provider or (args.provider if args.judge else None),
        "summary": summarize(results),
        "results": results,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(render_markdown_report(payload), encoding="utf-8")
        print(f"Wrote {args.report}")
    return 0 if payload["summary"]["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
