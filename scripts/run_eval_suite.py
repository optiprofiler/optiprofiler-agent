#!/usr/bin/env python
"""Release-grade evaluation orchestrator for OptiProfiler Agent.

This runner composes the lower-level eval tools into a repeatable gate:

- Advisor/Unified question-answer suites run in subprocesses so a stuck LLM
  call cannot wedge the whole release run.
- Debugger golden Pass@1 gates run per language.
- Deterministic multi-node checks run across debugger/interpreter internals.
- Results are aggregated into JSON/Markdown with threshold decisions.

Use this for local release checks and CI jobs that can afford provider calls.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int | None
    elapsed_s: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_command(
    name: str,
    command: list[str],
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> CommandResult:
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandResult(
            name=name,
            command=command,
            returncode=completed.returncode,
            elapsed_s=round(time.perf_counter() - t0, 3),
            timed_out=False,
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return CommandResult(
            name=name,
            command=command,
            returncode=None,
            elapsed_s=round(time.perf_counter() - t0, 3),
            timed_out=True,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _summarize_run_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "kind": "advisor_or_unified",
            "n_cases": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "judge_coverage": None,
            "judge_avg": None,
            "judge_hallucination_avg": None,
            "low_score_cases": [],
            "judge_na_cases": [],
        }
    rows = _load_json(path)
    scores = [float(row.get("combined_score", 0.0)) for row in rows]
    passed = sum(1 for score in scores if score >= 0.5)
    judged = [row for row in rows if row.get("judge_avg") is not None]
    judge_na = [row.get("id") for row in rows if "judge_avg" in row and row.get("judge_avg") is None]
    hallucination = [
        row["judge_scores"]["hallucination"]
        for row in judged
        if row.get("judge_scores") and "hallucination" in row["judge_scores"]
    ]
    return {
        "kind": "advisor_or_unified",
        "n_cases": len(rows),
        "pass_count": passed,
        "pass_rate": passed / len(rows) if rows else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "judge_coverage": len(judged) / len(rows) if rows and any("judge_avg" in r for r in rows) else None,
        "judge_avg": (
            sum(float(row["judge_avg"]) for row in judged) / len(judged)
            if judged else None
        ),
        "judge_hallucination_avg": (
            sum(hallucination) / len(hallucination) if hallucination else None
        ),
        "low_score_cases": [
            {
                "id": row.get("id"),
                "score": row.get("combined_score"),
                "judge_avg": row.get("judge_avg"),
                "reason": row.get("judge_reason"),
            }
            for row in rows
            if float(row.get("combined_score", 0.0)) < 0.7
        ],
        "judge_na_cases": judge_na,
    }


def _summarize_debugger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"kind": "debugger", "n_cases": 0, "pass_count": 0, "pass_rate": 0.0}
    payload = _load_json(path)
    return {
        "kind": "debugger",
        "language": payload.get("language"),
        "strategy": payload.get("strategy"),
        "n_cases": payload.get("n_total", 0),
        "pass_count": payload.get("n_pass", 0),
        "pass_rate": payload.get("pass_rate", 0.0),
        "failures": [
            row.get("id")
            for row in payload.get("results", [])
            if not (row.get("broken_failed") and row.get("fix_ran"))
        ],
    }


def _summarize_multinode(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"kind": "multinode", "n_cases": 0, "pass_count": 0, "pass_rate": 0.0}
    payload = _load_json(path)
    summary = payload.get("summary", {})
    return {
        "kind": "multinode",
        "n_cases": summary.get("total", 0),
        "pass_count": summary.get("passed", 0),
        "pass_rate": summary.get("pass_rate", 0.0),
        "failures": [row.get("id") for row in payload.get("results", []) if not row.get("passed")],
    }


def _summarize_interpreter(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "kind": "interpreter",
            "n_cases": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
        }
    payload = _load_json(path)
    summary = payload.get("summary", {})
    return {
        "kind": "interpreter",
        "n_cases": summary.get("n_cases", 0),
        "pass_count": summary.get("pass_count", 0),
        "pass_rate": summary.get("pass_rate", 0.0),
        "avg_score": summary.get("avg_fact_score", 0.0),
        "judge_coverage": summary.get("judge_coverage"),
        "judge_avg": summary.get("judge_avg"),
        "judge_hallucination_avg": summary.get("judge_hallucination_avg"),
        "failures": summary.get("failures", []),
        "judge_na_cases": summary.get("judge_na_cases", []),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _min_metric(
    suite_results: list[dict[str, Any]],
    key: str,
    formatter,
) -> str:
    values = [
        suite["summary"].get(key)
        for suite in suite_results
        if suite["summary"].get(key) is not None
    ]
    if not values:
        return "n/a"
    return formatter(min(values))


def _render_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# OptiProfiler Agent Evaluation Suite",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Provider: `{payload.get('provider') or 'n/a'}`",
        f"- Judge provider: `{payload.get('judge_provider') or 'n/a'}`",
        f"- Overall status: `{'PASS' if payload['passed'] else 'FAIL'}`",
        f"- Output directory: `{payload['output_dir']}`",
        "",
        "## Thresholds",
        "",
        "| Metric | Required | Actual |",
        "|---|---:|---:|",
    ]
    for name, row in payload["thresholds"].items():
        lines.append(
            f"| `{name}` | {row['required']} | {row['actual']} |"
        )

    lines.extend([
        "",
        "## Suites",
        "",
        "| Suite | Kind | Cases | Pass Rate | Avg Score | Judge Coverage | Judge Avg | Hallucination | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for suite in payload["suites"]:
        summary = suite["summary"]
        lines.append(
            f"| `{suite['name']}` | `{summary.get('kind')}` | "
            f"{summary.get('pass_count', 0)}/{summary.get('n_cases', 0)} | "
            f"{_fmt_pct(summary.get('pass_rate'))} | "
            f"{_fmt_num(summary.get('avg_score'))} | "
            f"{_fmt_pct(summary.get('judge_coverage'))} | "
            f"{_fmt_num(summary.get('judge_avg'))} | "
            f"{_fmt_num(summary.get('judge_hallucination_avg'))} | "
            f"{'PASS' if suite['passed'] else 'FAIL'} |"
        )

    issues: list[str] = []
    for suite in payload["suites"]:
        summary = suite["summary"]
        if summary.get("low_score_cases"):
            issues.append(
                f"- `{suite['name']}` low-score cases: "
                + ", ".join(
                    f"{row['id']}={row['score']}" for row in summary["low_score_cases"]
                )
            )
        if summary.get("judge_na_cases"):
            issues.append(
                f"- `{suite['name']}` judge N/A: "
                + ", ".join(str(x) for x in summary["judge_na_cases"])
            )
        if summary.get("failures"):
            issues.append(
                f"- `{suite['name']}` failures: "
                + ", ".join(str(x) for x in summary["failures"])
            )
        if suite["command_result"].get("timed_out"):
            issues.append(f"- `{suite['name']}` command timed out.")
        elif suite["command_result"].get("returncode") not in (0, None):
            issues.append(
                f"- `{suite['name']}` exited with code "
                f"{suite['command_result'].get('returncode')}."
            )

    if issues:
        lines.extend(["", "## Issues", "", *issues])

    lines.extend([
        "",
        "## Commands",
        "",
    ])
    for suite in payload["suites"]:
        cmd = " ".join(suite["command_result"]["command"])
        lines.append(f"### `{suite['name']}`")
        lines.append("")
        lines.append("```bash")
        lines.append(cmd)
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _suite_passed(
    summary: dict[str, Any],
    command: CommandResult,
    min_pass_rate: float,
    min_avg_score: float,
    min_judge_coverage: float,
    min_hallucination: float,
) -> bool:
    if not command.ok:
        return False
    if summary.get("pass_rate", 0.0) < min_pass_rate:
        return False
    if summary.get("avg_score") is not None and summary.get("avg_score", 0.0) < min_avg_score:
        return False
    coverage = summary.get("judge_coverage")
    if coverage is not None and coverage < min_judge_coverage:
        return False
    hallucination = summary.get("judge_hallucination_avg")
    if hallucination is not None and hallucination < min_hallucination:
        return False
    return True


def _command_dict(result: CommandResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "command": result.command,
        "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "timed_out": result.timed_out,
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
    }


def _advisor_suites(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    suites: list[dict[str, Any]] = []
    case_files = [
        ("advisor_factual", "tests/eval_cases/factual.json"),
        ("advisor_adversarial", "tests/eval_cases/adversarial.json"),
        ("advisor_code_generation", "tests/eval_cases/code_generation.json"),
    ]
    if args.include_unified:
        case_files.append(("unified_tool_routing", "tests/eval_cases/tool_routing.json"))

    for name, cases in case_files:
        mode = "unified" if name.startswith("unified") else "advisor"
        output = out_dir / f"{name}.json"
        report = out_dir / f"{name}.md"
        cmd = [
            PYTHON,
            "scripts/run_eval.py",
            "--mode",
            mode,
            "--provider",
            args.provider,
            "--cases",
            cases,
            "--case-timeout",
            str(args.case_timeout),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
        if args.judge and mode == "advisor":
            cmd.extend(["--judge", "--judge-provider", args.judge_provider or args.provider])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        suites.append({"name": name, "command": cmd, "output": output, "summary_kind": "run_eval"})
    return suites


def _debugger_suites(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    suites = []
    for language in ("python", "matlab"):
        if language == "matlab" and args.skip_matlab:
            continue
        name = f"debugger_{language}_golden"
        output = out_dir / f"{name}.json"
        report = out_dir / f"{name}.md"
        cmd = [
            PYTHON,
            "scripts/run_debugger_eval.py",
            "--language",
            language,
            "--strategy",
            "golden",
            "--timeout",
            str(args.debugger_timeout),
            "--output",
            str(output),
            "--markdown-output",
            str(report),
        ]
        suites.append({"name": name, "command": cmd, "output": output, "summary_kind": "debugger"})
    return suites


def _multinode_suite(out_dir: Path) -> dict[str, Any]:
    output = out_dir / "multinode.json"
    report = out_dir / "multinode.md"
    return {
        "name": "multinode_deterministic",
        "command": [
            PYTHON,
            "scripts/run_multinode_eval.py",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        "output": output,
        "summary_kind": "multinode",
    }


def _interpreter_suite(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    output = out_dir / "interpreter_report.json"
    report = out_dir / "interpreter_report.md"
    cmd = [
        PYTHON,
        "scripts/run_interpreter_eval.py",
        "--strategy",
        args.interpreter_strategy,
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    if args.interpreter_strategy == "llm":
        cmd.extend(["--provider", args.provider])
    if args.judge_interpreter:
        cmd.extend(["--judge", "--judge-provider", args.judge_provider or args.provider])
    return {
        "name": "interpreter_report_factcheck",
        "command": cmd,
        "output": output,
        "summary_kind": "interpreter",
    }


def _summarize(kind: str, path: Path) -> dict[str, Any]:
    if kind == "run_eval":
        return _summarize_run_eval(path)
    if kind == "debugger":
        return _summarize_debugger(path)
    if kind == "multinode":
        return _summarize_multinode(path)
    if kind == "interpreter":
        return _summarize_interpreter(path)
    raise ValueError(f"unknown suite kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", default="minimax")
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--output-dir", default="docs/eval/latest")
    parser.add_argument("--case-timeout", type=int, default=90)
    parser.add_argument("--suite-timeout", type=int, default=900)
    parser.add_argument("--debugger-timeout", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None, help="Limit each advisor/unified suite")
    parser.add_argument("--include-unified", action="store_true")
    parser.add_argument("--skip-advisor", action="store_true")
    parser.add_argument("--skip-debugger", action="store_true")
    parser.add_argument("--skip-interpreter", action="store_true")
    parser.add_argument("--skip-multinode", action="store_true")
    parser.add_argument("--skip-matlab", action="store_true")
    parser.add_argument("--interpreter-strategy", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--judge-interpreter", action="store_true")
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--min-avg-score", type=float, default=0.75)
    parser.add_argument("--min-judge-coverage", type=float, default=0.8)
    parser.add_argument("--min-hallucination", type=float, default=0.9)
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if "MATOP_MATLAB_BIN" not in env:
        default_matlab = Path("/Applications/MATLAB_R2023b.app/bin/matlab")
        if default_matlab.exists():
            env["MATOP_MATLAB_BIN"] = str(default_matlab)

    pending: list[dict[str, Any]] = []
    if not args.skip_advisor:
        pending.extend(_advisor_suites(args, out_dir))
    if not args.skip_debugger:
        pending.extend(_debugger_suites(args, out_dir))
    if not args.skip_interpreter:
        pending.append(_interpreter_suite(args, out_dir))
    if not args.skip_multinode:
        pending.append(_multinode_suite(out_dir))

    suite_results: list[dict[str, Any]] = []
    print(f"[run_eval_suite] output_dir={out_dir}")
    for suite in pending:
        print(f"[run_eval_suite] running {suite['name']} ...", flush=True)
        command_result = _run_command(
            suite["name"],
            suite["command"],
            timeout_s=args.suite_timeout,
            env=env,
        )
        summary = _summarize(suite["summary_kind"], suite["output"])
        passed = _suite_passed(
            summary,
            command_result,
            min_pass_rate=args.min_pass_rate,
            min_avg_score=args.min_avg_score,
            min_judge_coverage=args.min_judge_coverage,
            min_hallucination=args.min_hallucination,
        )
        print(
            f"[run_eval_suite] {suite['name']}: "
            f"{'PASS' if passed else 'FAIL'} "
            f"pass_rate={_fmt_pct(summary.get('pass_rate'))} "
            f"avg={_fmt_num(summary.get('avg_score'))}",
            flush=True,
        )
        suite_results.append({
            "name": suite["name"],
            "output": str(suite["output"]),
            "summary": summary,
            "passed": passed,
            "command_result": _command_dict(command_result),
        })

    thresholds = {
        "min_pass_rate": {
            "required": f">= {_fmt_pct(args.min_pass_rate)}",
            "actual": _min_metric(suite_results, "pass_rate", _fmt_pct),
        },
        "min_avg_score": {
            "required": f">= {args.min_avg_score:.3f}",
            "actual": _min_metric(suite_results, "avg_score", _fmt_num),
        },
        "min_judge_coverage": {
            "required": f">= {_fmt_pct(args.min_judge_coverage)}",
            "actual": _min_metric(suite_results, "judge_coverage", _fmt_pct),
        },
        "min_hallucination": {
            "required": f">= {args.min_hallucination:.3f}",
            "actual": _min_metric(suite_results, "judge_hallucination_avg", _fmt_num),
        },
    }
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "provider": args.provider,
        "judge_provider": args.judge_provider or (args.provider if args.judge else None),
        "output_dir": str(out_dir),
        "passed": all(s["passed"] for s in suite_results),
        "thresholds": thresholds,
        "suites": suite_results,
    }

    _write_json(out_dir / "summary.json", payload)
    (out_dir / "summary.md").write_text(_render_summary(payload), encoding="utf-8")
    print(f"[run_eval_suite] wrote {out_dir / 'summary.json'}")
    print(f"[run_eval_suite] wrote {out_dir / 'summary.md'}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
