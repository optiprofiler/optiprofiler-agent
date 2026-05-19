#!/usr/bin/env python
"""Agent B (Debugger) task-level eval — Pass@1 on broken scripts.

Two modes, both produce the same JSON / Markdown shape so they're directly
comparable across runs:

* ``--strategy golden`` (default, no LLM, no API key)
  For each case in ``tests/fixtures/broken_<language>/`` run ``broken.m`` /
  ``broken.py`` in the language sandbox, classify the error, then run
  ``fix.m`` / ``fix.py`` and verify the run is now green. This isolates
  the runner + classifier from LLM variance; failures here mean a broken
  fixture, a broken runner, or a regressed regex.

* ``--strategy llm`` (requires API key)
  Same loop, but instead of ``fix.<ext>`` we call
  ``debug_script(code, error, language=...)`` and feed the LLM-generated
  fix back through the sandbox. This is the real Pass@1 metric.

Outputs::

    {
      "agent": "debugger",
      "strategy": "golden|llm",
      "language": "matlab|python",
      "n_total": 5,
      "n_pass": 5,
      "pass_rate": 1.0,
      "results": [
        {"id": "iface_reorder", "broken_failed": true, "fix_ran": true,
         "category_match": true, "regex_category": "runtime_error", ...},
        ...
      ]
    }

Usage::

    python scripts/run_debugger_eval.py --language matlab
    python scripts/run_debugger_eval.py --language matlab --strategy llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optiprofiler_agent.debugger.error_classifier import classify_error
from optiprofiler_agent.debugger.matlab_runner import (
    MatlabNotAvailable,
    is_matlab_available,
    run_matlab_script,
)
from optiprofiler_agent.debugger.local_runner import run_script as run_python_script

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fixture_dir(language: str) -> Path:
    return REPO_ROOT / "tests" / "fixtures" / f"broken_{language}"


def _load_cases(language: str) -> list[dict]:
    base = _fixture_dir(language)
    if not base.is_dir():
        raise FileNotFoundError(f"No fixture dir: {base}")
    cases = []
    for case_dir in sorted(base.iterdir()):
        if not case_dir.is_dir():
            continue
        meta_path = case_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        ext = "m" if language == "matlab" else "py"
        broken = case_dir / f"broken.{ext}"
        fix = case_dir / f"fix.{ext}"
        if not (broken.is_file() and fix.is_file()):
            print(f"  skip {case_dir.name}: missing broken/fix file", file=sys.stderr)
            continue
        cases.append({"meta": meta, "broken_path": broken, "fix_path": fix})
    return cases


def _select_cases(cases: list[dict], names: str | None, limit: int | None) -> list[dict]:
    """Return cases filtered by comma-separated ids and/or a leading limit."""
    selected = cases
    if names:
        wanted = {name.strip() for name in names.split(",") if name.strip()}
        selected = [case for case in selected if case["meta"]["id"] in wanted]
        missing = wanted - {case["meta"]["id"] for case in selected}
        if missing:
            raise ValueError(f"Unknown case id(s): {', '.join(sorted(missing))}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def _run(code: str, language: str, timeout: int):
    if language == "matlab":
        return run_matlab_script(code, timeout=timeout)
    return run_python_script(code, timeout=timeout)


def _strategy_golden(case: dict, language: str, timeout: int) -> dict:
    """Use the human-curated fix.<ext> as the proposed fix."""
    meta = case["meta"]
    broken_code = case["broken_path"].read_text()
    fix_code = case["fix_path"].read_text()

    t0 = time.perf_counter()
    broken_result = _run(broken_code, language, timeout)
    elapsed_broken = time.perf_counter() - t0

    if broken_result.success:
        return {
            "id": meta["id"],
            "broken_failed": False,
            "fix_ran": None,
            "regex_category": None,
            "category_match": False,
            "error_substr_match": False,
            "elapsed_s": elapsed_broken,
            "notes": "broken.<ext> did NOT fail — fixture is wrong",
        }

    err = broken_result.traceback or broken_result.stderr
    cls = classify_error(err, language=language)

    expected_cats = meta.get("expected_categories") or [meta.get("expected_category")]
    cat_match = cls.error_type in expected_cats
    err_substr_match = any(s in (err or "") for s in meta.get("expected_error_substr", []))

    t0 = time.perf_counter()
    fix_result = _run(fix_code, language, timeout)
    elapsed_fix = time.perf_counter() - t0

    return {
        "id": meta["id"],
        "broken_failed": True,
        "fix_ran": fix_result.success,
        "regex_category": cls.error_type,
        "expected_categories": expected_cats,
        "category_match": cat_match,
        "error_substr_match": err_substr_match,
        "elapsed_s": round(elapsed_broken + elapsed_fix, 2),
        "fix_stderr_excerpt": (fix_result.traceback or fix_result.stderr)[-1200:],
    }


def _strategy_llm(
    case: dict,
    language: str,
    timeout: int,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Use Agent B's debug_script to propose a fix, then re-run."""
    from optiprofiler_agent.config import AgentConfig, LLMConfig
    from optiprofiler_agent.debugger.debugger import debug_script

    meta = case["meta"]
    broken_code = case["broken_path"].read_text()

    t0 = time.perf_counter()
    broken_result = _run(broken_code, language, timeout)
    elapsed_broken = time.perf_counter() - t0

    if broken_result.success:
        return {
            "id": meta["id"],
            "broken_failed": False,
            "fix_ran": None,
            "notes": "broken.<ext> did NOT fail — fixture is wrong",
        }

    err = broken_result.traceback or broken_result.stderr
    config = AgentConfig(llm=LLMConfig(provider=provider, model=model))
    dr = debug_script(code=broken_code, error=err, config=config, language=language)
    proposed = dr.fixed_code

    if not proposed:
        return {
            "id": meta["id"],
            "broken_failed": True,
            "fix_proposed": False,
            "fix_ran": False,
            "classification": dr.classification.error_type,
            "category_match": dr.classification.error_type in (
                meta.get("expected_categories") or [meta.get("expected_category")]
            ),
            "elapsed_s": round(elapsed_broken, 2),
            "diagnostic_excerpt": dr.diagnostic_report[-1600:],
        }

    t0 = time.perf_counter()
    fix_result = _run(proposed, language, timeout)
    elapsed_fix = time.perf_counter() - t0

    expected_cats = meta.get("expected_categories") or [meta.get("expected_category")]
    return {
        "id": meta["id"],
        "broken_failed": True,
        "fix_proposed": True,
        "fix_ran": fix_result.success,
        "classification": dr.classification.error_type,
        "category_match": dr.classification.error_type in expected_cats,
        "elapsed_s": round(elapsed_broken + elapsed_fix, 2),
        "attempts": dr.attempts,
        "validation_passed": dr.validation_passed,
        "diagnostic_excerpt": dr.diagnostic_report[-1600:],
        "proposed_code_excerpt": proposed[:2500],
        "fix_stderr_excerpt": (fix_result.traceback or fix_result.stderr)[-1600:],
    }


def _render_markdown_report(payload: dict) -> str:
    """Render a compact Markdown report suitable for docs/eval/last_run.md."""
    lines = [
        "# Debugger Eval Last Run",
        "",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        f"- Agent: `{payload['agent']}`",
        f"- Strategy: `{payload['strategy']}`",
        f"- Language: `{payload['language']}`",
        f"- Provider: `{payload.get('provider') or 'n/a'}`",
        f"- Model: `{payload.get('model') or 'n/a'}`",
        f"- Cases: `{payload['n_pass']}/{payload['n_total']}`",
        f"- Pass@1: `{payload['pass_rate'] * 100:.1f}%`",
        "",
        "| Case | Status | Classification | Category ok | Fix proposed | Time (s) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for result in payload["results"]:
        status = "PASS" if result.get("broken_failed") and result.get("fix_ran") else "FAIL"
        cls = result.get("classification") or result.get("regex_category") or "-"
        cat_ok = result.get("category_match")
        cat_text = "-" if cat_ok is None else ("yes" if cat_ok else "no")
        proposed = result.get("fix_proposed")
        proposed_text = "-" if proposed is None else ("yes" if proposed else "no")
        elapsed = result.get("elapsed_s")
        lines.append(
            f"| `{result.get('id', '-')}` | {status} | `{cls}` | {cat_text} | "
            f"{proposed_text} | {elapsed if elapsed is not None else '-'} |"
        )

    failures = [
        result for result in payload["results"]
        if not (result.get("broken_failed") and result.get("fix_ran"))
    ]
    if failures:
        lines.extend(["", "## Failure Notes", ""])
        for result in failures:
            lines.append(f"### `{result.get('id', '-')}`")
            if result.get("error"):
                lines.append(f"- Error: `{result['error']}`")
            if result.get("diagnostic_excerpt"):
                lines.append("")
                lines.append("Diagnostic excerpt:")
                lines.append("")
                lines.append("```text")
                lines.append(result["diagnostic_excerpt"])
                lines.append("```")
            if result.get("fix_stderr_excerpt"):
                lines.append("")
                lines.append("Fix stderr excerpt:")
                lines.append("")
                lines.append("```text")
                lines.append(result["fix_stderr_excerpt"])
                lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--language", choices=["matlab", "python"], default="matlab")
    parser.add_argument("--strategy", choices=["golden", "llm"], default="golden")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", default=None, help="Write raw results JSON here")
    parser.add_argument("--markdown-output", default=None, help="Write Markdown summary here")
    parser.add_argument("--update-last-run", action="store_true",
                        help="Also write docs/eval/last_run.md")
    parser.add_argument("--provider", default=None, help="LLM provider for --strategy llm")
    parser.add_argument("--model", default=None, help="LLM model for --strategy llm")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated case ids to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N cases after filtering")
    args = parser.parse_args()

    resolved_provider = args.provider
    resolved_model = args.model
    if args.strategy == "llm":
        from optiprofiler_agent.config import LLMConfig

        resolved_llm = LLMConfig(provider=args.provider, model=args.model)
        resolved_provider = resolved_llm.provider
        resolved_model = resolved_llm.model

    if args.language == "matlab" and not is_matlab_available():
        print(
            "ERROR: MATLAB not available. Set MATOP_MATLAB_BIN or add `matlab` to PATH.",
            file=sys.stderr,
        )
        return 2

    try:
        cases = _load_cases(args.language)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        cases = _select_cases(cases, args.cases, args.limit)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not cases:
        print(f"ERROR: no cases found in {_fixture_dir(args.language)}", file=sys.stderr)
        return 2

    print(
        f"[run_debugger_eval] language={args.language} strategy={args.strategy} "
        f"n_cases={len(cases)} timeout={args.timeout}s"
    )

    results = []
    for case in cases:
        meta = case["meta"]
        print(f"  • {meta['id']:25s} … ", end="", flush=True)
        try:
            if args.strategy == "golden":
                r = _strategy_golden(case, args.language, args.timeout)
            else:
                r = _strategy_llm(
                    case,
                    args.language,
                    args.timeout,
                    provider=resolved_provider,
                    model=resolved_model,
                )
        except MatlabNotAvailable as exc:
            r = {"id": meta["id"], "error": f"MatlabNotAvailable: {exc}"}
        except Exception as exc:  # noqa: BLE001
            r = {"id": meta["id"], "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        status = "PASS" if (r.get("broken_failed") and r.get("fix_ran")) else "FAIL"
        print(status)

    n_pass = sum(1 for r in results if r.get("broken_failed") and r.get("fix_ran"))
    rate = n_pass / len(results) if results else 0.0
    print(f"\nPass@1 ({args.strategy}): {n_pass}/{len(results)} = {rate*100:.0f}%")

    payload = {
        "agent": "debugger",
        "strategy": args.strategy,
        "language": args.language,
        "provider": resolved_provider,
        "model": resolved_model,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "n_total": len(results),
        "n_pass": n_pass,
        "pass_rate": rate,
        "results": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")
    if args.markdown_output:
        Path(args.markdown_output).write_text(_render_markdown_report(payload))
        print(f"Wrote {args.markdown_output}")
    if args.update_last_run:
        out = REPO_ROOT / "docs" / "eval" / "last_run.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_render_markdown_report(payload))
        print(f"Wrote {out}")
    return 0 if rate >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
