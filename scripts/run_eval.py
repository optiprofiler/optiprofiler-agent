#!/usr/bin/env python
"""Automated evaluation harness for Agent A (Product Advisor).

Runs test cases against the agent and scores responses using:
1. **Keyword matching**: checks expected_keywords / must_contain / must_not_contain
2. **Syntax validation**: checks generated code for syntax errors
3. **API validation**: checks benchmark() calls for correctness
4. **LLM-as-Judge** (optional): uses a second LLM to grade response quality

Usage::

    python3.11 scripts/run_eval.py
    python3.11 scripts/run_eval.py --provider minimax --cases tests/eval_cases/factual.json
    python3.11 scripts/run_eval.py --judge          # enable LLM-as-Judge
    python3.11 scripts/run_eval.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from optiprofiler_agent.config import AgentConfig, LLMConfig
from optiprofiler_agent.agent_a.advisor import AdvisorAgent
from optiprofiler_agent.validators.syntax_checker import check_syntax
from optiprofiler_agent.validators.api_checker import validate_response_code

console = Console()

EVAL_CASES_DIR = Path(__file__).resolve().parent.parent / "tests" / "eval_cases"


def load_cases(path: Path | None = None) -> list[dict]:
    """Load test cases from JSON file(s)."""
    if path and path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    cases = []
    search_dir = path if (path and path.is_dir()) else EVAL_CASES_DIR
    for f in sorted(search_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            cases.extend(json.load(fh))
    return cases


def score_keyword(response: str, case: dict) -> dict:
    """Score based on keyword presence."""
    response_lower = response.lower()
    result = {"keyword_score": 0.0, "keyword_details": []}

    expected = case.get("expected_keywords", [])
    must_contain = case.get("must_contain", [])
    must_not_contain = case.get("must_not_contain", [])

    all_positive = expected + must_contain
    if all_positive:
        hits = sum(1 for kw in all_positive if kw.lower() in response_lower)
        result["keyword_score"] = hits / len(all_positive)
        for kw in all_positive:
            found = kw.lower() in response_lower
            result["keyword_details"].append({"keyword": kw, "found": found})

    if must_not_contain:
        for kw in must_not_contain:
            if kw.lower() in response_lower:
                result["keyword_score"] = max(0, result["keyword_score"] - 0.3)
                result["keyword_details"].append(
                    {"keyword": f"NOT:{kw}", "found": True, "penalty": True})

    return result


def score_code_quality(response: str) -> dict:
    """Score based on syntax and API validation."""
    syn = check_syntax(response)
    api = validate_response_code(response)

    code_score = 1.0
    details = []

    if syn.blocks_found > 0:
        if syn.has_errors:
            code_score -= 0.5
            details.append(f"syntax_errors={len(syn.errors)}")
        else:
            details.append(f"syntax_ok ({syn.blocks_valid}/{syn.blocks_found})")

    if api.benchmark_calls_found > 0:
        n_errors = sum(1 for i in api.issues if i.severity == "error")
        n_warnings = sum(1 for i in api.issues if i.severity == "warning")
        if n_errors:
            code_score -= 0.3
            details.append(f"api_errors={n_errors}")
        if n_warnings:
            code_score -= 0.1
            details.append(f"api_warnings={n_warnings}")
        if not n_errors and not n_warnings:
            details.append("api_ok")
    else:
        details.append("no_benchmark_call")

    return {"code_score": max(0, code_score), "code_details": details}


def score_with_judge(question: str, response: str, judge_llm) -> dict:
    """Use an LLM to grade the response quality (0-10)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "You are an expert evaluator for an AI assistant about OptiProfiler, "
        "a derivative-free optimization benchmarking tool.\n\n"
        "Score the assistant's response on a scale of 0-10 based on:\n"
        "- Factual accuracy (is the information correct?)\n"
        "- Completeness (does it answer the question fully?)\n"
        "- Code quality (if code is provided, is it correct and runnable?)\n"
        "- Clarity (is the response clear and well-structured?)\n\n"
        "Important rules the assistant must follow:\n"
        "- benchmark() requires at least 2 solvers\n"
        "- fun provides ONLY function values (no gradients) — DFO only\n"
        "- Python and MATLAB have different calling conventions\n"
        "- MATLAB uses struct for options, not name-value pairs\n\n"
        "Reply with ONLY a JSON object: {\"score\": <0-10>, \"reason\": \"<brief explanation>\"}"
    )

    prompt = f"Question: {question}\n\nAssistant's response:\n{response}"

    try:
        result = judge_llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        text = result.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return {
            "judge_score": data.get("score", 0) / 10.0,
            "judge_reason": data.get("reason", ""),
        }
    except Exception as e:
        return {"judge_score": None, "judge_reason": f"Judge error: {e}"}


def run_eval(
    cases: list[dict],
    agent: AdvisorAgent,
    judge_llm=None,
    verbose: bool = False,
) -> list[dict]:
    """Run all test cases and return scored results."""
    results = []

    for i, case in enumerate(cases):
        case_id = case.get("id", f"case_{i}")
        question = case["question"]

        console.print(f"[dim][{i+1}/{len(cases)}] {case_id}: {question[:60]}...[/]")

        agent.reset()
        t0 = time.time()
        try:
            response = agent.chat(question)
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.time() - t0

        kw_result = score_keyword(response, case)
        code_result = score_code_quality(response)

        combined_score = kw_result["keyword_score"] * 0.6 + code_result["code_score"] * 0.4

        entry = {
            "id": case_id,
            "question": question,
            "language": case.get("language"),
            "category": case.get("category"),
            "response_length": len(response),
            "elapsed_s": round(elapsed, 2),
            **kw_result,
            **code_result,
            "combined_score": round(combined_score, 3),
        }

        if judge_llm:
            judge_result = score_with_judge(question, response, judge_llm)
            entry.update(judge_result)
            if judge_result["judge_score"] is not None:
                entry["combined_score"] = round(
                    combined_score * 0.5 + judge_result["judge_score"] * 0.5, 3)

        if verbose:
            console.print(f"  Score: {entry['combined_score']:.2f} "
                          f"(kw={kw_result['keyword_score']:.2f}, "
                          f"code={code_result['code_score']:.2f})")

        results.append(entry)

    return results


def print_summary(results: list[dict]):
    """Print a summary table of evaluation results."""
    table = Table(title="Evaluation Results")
    table.add_column("ID", style="cyan")
    table.add_column("Category")
    table.add_column("KW Score", justify="right")
    table.add_column("Code Score", justify="right")
    table.add_column("Combined", justify="right")
    table.add_column("Time (s)", justify="right")

    has_judge = any(r.get("judge_score") is not None for r in results)
    if has_judge:
        table.add_column("Judge", justify="right")

    for r in results:
        kw = f"{r['keyword_score']:.2f}"
        code = f"{r['code_score']:.2f}"
        combined = f"{r['combined_score']:.2f}"
        time_s = f"{r['elapsed_s']:.1f}"

        style = "green" if r["combined_score"] >= 0.7 else (
            "yellow" if r["combined_score"] >= 0.4 else "red")

        row = [r["id"], r.get("category", ""), kw, code, combined, time_s]
        if has_judge:
            js = r.get("judge_score")
            row.append(f"{js:.2f}" if js is not None else "N/A")

        table.add_row(*row, style=style)

    console.print(table)

    scores = [r["combined_score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    console.print(f"\n[bold]Average score: {avg:.3f}[/] ({len(results)} cases)")

    pass_count = sum(1 for s in scores if s >= 0.5)
    console.print(f"Pass rate (>=0.5): {pass_count}/{len(results)} "
                  f"({pass_count/len(results)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent A")
    parser.add_argument("--provider", default="minimax", help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--cases", default=None, help="Path to test cases JSON")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-Judge scoring")
    parser.add_argument("--judge-provider", default=None,
                        help="Provider for judge LLM (defaults to same as agent)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rag", action="store_true", help="Enable RAG")
    args = parser.parse_args()

    cases_path = Path(args.cases) if args.cases else None
    cases = load_cases(cases_path)
    if not cases:
        console.print("[red]No test cases found![/]")
        sys.exit(1)

    console.print(f"[bold]Loaded {len(cases)} test cases[/]")

    config = AgentConfig(
        llm=LLMConfig(provider=args.provider, model=args.model),
        rag_enabled=args.rag,
    )
    agent = AdvisorAgent(config)
    console.print(f"Agent: provider={config.llm.provider}, model={config.llm.model}")

    judge_llm = None
    if args.judge:
        from optiprofiler_agent.common.llm_client import create_llm
        judge_provider = args.judge_provider or args.provider
        judge_cfg = LLMConfig(provider=judge_provider)
        judge_llm = create_llm(judge_cfg)
        console.print(f"Judge: provider={judge_cfg.provider}, model={judge_cfg.model}")

    console.print()
    results = run_eval(cases, agent, judge_llm=judge_llm, verbose=args.verbose)

    console.print()
    print_summary(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Results saved to {args.output}[/]")


if __name__ == "__main__":
    main()
