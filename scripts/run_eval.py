#!/usr/bin/env python
"""Automated evaluation harness for OptiProfiler Agent.

Supports **two modes**:
- ``advisor``: test Agent A (Product Advisor) via ``AdvisorAgent.chat()``
- ``unified``: test the full ReAct Unified Agent (default ``opagent`` path)

Scoring dimensions:
1. **Keyword recall**: expected_keywords / must_contain / must_not_contain
2. **Code quality**: syntax + API validation on generated code
3. **Tool routing accuracy**: did the agent call the expected tool? (unified mode)
4. **LLM-as-Judge** (optional, multi-dimensional rubric)

Usage::

    python scripts/run_eval.py                              # advisor mode, all cases
    python scripts/run_eval.py --mode unified               # unified agent
    python scripts/run_eval.py --judge                      # enable LLM-as-Judge
    python scripts/run_eval.py --cases tests/eval_cases/factual.json
    python scripts/run_eval.py --output results.json --report report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from optiprofiler_agent.config import AgentConfig, LLMConfig
from optiprofiler_agent.validators.syntax_checker import check_syntax
from optiprofiler_agent.validators.api_checker import validate_response_code

console = Console()

EVAL_CASES_DIR = Path(__file__).resolve().parent.parent / "tests" / "eval_cases"

# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_cases(path: Path | None = None) -> list[dict]:
    if path and path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    cases: list[dict] = []
    search_dir = path if (path and path.is_dir()) else EVAL_CASES_DIR
    for f in sorted(search_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            cases.extend(json.load(fh))
    return cases

# ---------------------------------------------------------------------------
# Scoring: keywords
# ---------------------------------------------------------------------------

def score_keyword(response: str, case: dict) -> dict:
    response_lower = response.lower()
    result: dict = {"keyword_score": 0.0, "keyword_hits": 0, "keyword_total": 0, "keyword_details": []}

    expected = case.get("expected_keywords", [])
    must_contain = case.get("must_contain", [])
    must_not_contain = case.get("must_not_contain", [])

    all_positive = expected + must_contain
    if all_positive:
        hits = sum(1 for kw in all_positive if kw.lower() in response_lower)
        result["keyword_score"] = hits / len(all_positive)
        result["keyword_hits"] = hits
        result["keyword_total"] = len(all_positive)
        for kw in all_positive:
            found = kw.lower() in response_lower
            result["keyword_details"].append({"keyword": kw, "found": found})

    if must_not_contain:
        for kw in must_not_contain:
            if kw.lower() in response_lower:
                result["keyword_score"] = max(0, result["keyword_score"] - 0.3)
                result["keyword_details"].append({"keyword": f"NOT:{kw}", "found": True, "penalty": True})

    return result

# ---------------------------------------------------------------------------
# Scoring: code quality
# ---------------------------------------------------------------------------

def score_code_quality(response: str, case: dict | None = None) -> dict:
    syn = check_syntax(response)
    api = validate_response_code(response)

    code_score = 1.0
    details: list[str] = []
    expect_code = (case or {}).get("expect_code", False)

    if syn.blocks_found > 0:
        if syn.has_errors:
            code_score -= 0.5
            details.append(f"syntax_errors={len(syn.errors)}")
        else:
            details.append(f"syntax_ok ({syn.blocks_valid}/{syn.blocks_found})")
    elif expect_code:
        code_score = 0.0
        details.append("NO_CODE (expected code)")

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

# ---------------------------------------------------------------------------
# Scoring: tool routing (unified mode only)
# ---------------------------------------------------------------------------

def extract_tool_calls(messages: list) -> list[str]:
    """Extract tool names from the LangGraph message list."""
    tools_called: list[str] = []
    for msg in messages:
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name:
                    tools_called.append(name)
    return tools_called


def score_tool_routing(tools_called: list[str], case: dict) -> dict:
    expected = case.get("expect_tool")
    if not expected:
        return {"tool_routing_score": None, "tools_called": tools_called, "expected_tool": None}

    hit = expected in tools_called
    return {
        "tool_routing_score": 1.0 if hit else 0.0,
        "tools_called": tools_called,
        "expected_tool": expected,
    }

# ---------------------------------------------------------------------------
# Scoring: LLM-as-Judge (multi-dimensional)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert evaluator for an AI assistant about OptiProfiler, \
a derivative-free optimization benchmarking tool.

Score the assistant's response on **each** of the following dimensions (0-10):

1. **accuracy**: Is the information factually correct per official docs?
2. **completeness**: Does it fully answer the question?
3. **code_quality**: If code is provided, is it correct, runnable, and idiomatic? \
If no code was expected, give 8.
4. **hallucination**: 10 = no fabrication; 0 = entirely made up.
5. **instruction_following**: Does it respect DFO-only, MATLAB struct convention, \
2-solver minimum, etc.?

Important rules:
- benchmark() requires at least 2 solvers
- fun provides ONLY function values (no gradients) — DFO only
- Python uses keyword args; MATLAB uses an options struct (never name-value pairs)
- MATLAB benchmark second arg is NEVER the user's objective function handle

Reply with ONLY a JSON object:
{"accuracy": <0-10>, "completeness": <0-10>, "code_quality": <0-10>, \
"hallucination": <0-10>, "instruction_following": <0-10>, "reason": "<brief>"}"""


_JUDGE_MAX_RETRIES = 3
_JUDGE_RETRY_BASE_DELAY = 2.0


def score_with_judge(question: str, response: str, judge_llm) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = f"Question: {question}\n\nAssistant's response:\n{response}"
    last_err = None

    for attempt in range(_JUDGE_MAX_RETRIES):
        try:
            if attempt > 0:
                delay = _JUDGE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)

            result = judge_llm.invoke([
                SystemMessage(content=_JUDGE_SYSTEM),
                HumanMessage(content=prompt),
            ])
            text = result.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)

            dims = ["accuracy", "completeness", "code_quality", "hallucination", "instruction_following"]
            scores = {d: data.get(d, 0) / 10.0 for d in dims}
            avg = sum(scores.values()) / len(scores)
            return {
                "judge_scores": scores,
                "judge_avg": round(avg, 3),
                "judge_reason": data.get("reason", ""),
            }
        except Exception as e:
            last_err = e
            if "overloaded" in str(e).lower() or "529" in str(e):
                continue
            break

    return {"judge_scores": None, "judge_avg": None, "judge_reason": f"Judge error: {last_err}"}

# ---------------------------------------------------------------------------
# Agent runners (with retry for transient API errors)
# ---------------------------------------------------------------------------

_AGENT_MAX_RETRIES = 3
_AGENT_RETRY_BASE_DELAY = 3.0


def _is_transient(e: Exception) -> bool:
    s = str(e).lower()
    return "overloaded" in s or "529" in s or "rate" in s or "timeout" in s


def run_advisor(question: str, agent) -> tuple[str, list[str]]:
    last_err: Exception | None = None
    for attempt in range(_AGENT_MAX_RETRIES):
        try:
            if attempt > 0:
                time.sleep(_AGENT_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            agent.reset()
            response = agent.chat(question)
            return response, []
        except Exception as e:
            last_err = e
            if _is_transient(e):
                continue
            raise
    raise last_err  # type: ignore[misc]


def run_unified(question: str, agent) -> tuple[str, list[str]]:
    last_err: Exception | None = None
    for attempt in range(_AGENT_MAX_RETRIES):
        try:
            if attempt > 0:
                time.sleep(_AGENT_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            result = agent.invoke({"messages": [("user", question)]})
            messages = result["messages"]
            reply = messages[-1].content
            tools_called = extract_tool_calls(messages)
            return reply, tools_called
        except Exception as e:
            last_err = e
            if _is_transient(e):
                continue
            raise
    raise last_err  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

_SHORT_RESPONSE_THRESHOLD = 250


def run_eval(
    cases: list[dict],
    agent,
    runner,
    judge_llm=None,
    verbose: bool = False,
    mode: str = "advisor",
) -> list[dict]:
    results: list[dict] = []
    is_unified = mode == "unified"

    for i, case in enumerate(cases):
        case_id = case.get("id", f"case_{i}")
        question = case["question"]
        tag = f"[{i+1}/{len(cases)}]"

        console.print(f"[dim]{tag} {case_id}: {question[:70]}{'...' if len(question) > 70 else ''}[/]")

        t0 = time.time()
        try:
            response, tools_called = runner(question, agent)
        except Exception as e:
            response, tools_called = f"ERROR: {e}", []
        elapsed = time.time() - t0

        short_response = len(response) < _SHORT_RESPONSE_THRESHOLD

        kw = score_keyword(response, case)
        code = score_code_quality(response, case)

        if is_unified:
            tool = score_tool_routing(tools_called, case)
        else:
            tool = {"tool_routing_score": None, "tools_called": [], "expected_tool": None}

        has_tool_dim = tool["tool_routing_score"] is not None
        if has_tool_dim:
            weights = {"keyword": 0.5, "code": 0.3, "tool": 0.2}
            combined = (kw["keyword_score"] * weights["keyword"]
                        + code["code_score"] * weights["code"]
                        + tool["tool_routing_score"] * weights["tool"])
        else:
            combined = kw["keyword_score"] * 0.6 + code["code_score"] * 0.4

        entry: dict = {
            "id": case_id,
            "question": question,
            "response": response[:2000],
            "language": case.get("language"),
            "category": case.get("category"),
            "response_length": len(response),
            "short_response": short_response,
            "elapsed_s": round(elapsed, 2),
            **kw,
            **code,
            **tool,
            "combined_score": round(combined, 3),
        }

        if judge_llm:
            judge = score_with_judge(question, response, judge_llm)
            entry.update(judge)
            if judge["judge_avg"] is not None:
                entry["combined_score"] = round(combined * 0.5 + judge["judge_avg"] * 0.5, 3)

        if verbose:
            parts = [f"kw={kw['keyword_score']:.2f}", f"code={code['code_score']:.2f}"]
            if has_tool_dim:
                parts.append(f"tool={'PASS' if tool['tool_routing_score'] == 1 else 'MISS'}")
            if short_response:
                parts.append("SHORT!")
            console.print(f"  Score: {entry['combined_score']:.2f} ({', '.join(parts)})")

        results.append(entry)

    return results

# ---------------------------------------------------------------------------
# Visual reporting
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "[green]" + "█" * filled + "[/][dim]" + "░" * (width - filled) + "[/]"


def print_summary(results: list[dict], mode: str):
    table = Table(title=f"Evaluation Results ({mode} mode, {len(results)} cases)")
    table.add_column("ID", style="cyan", min_width=5)
    table.add_column("Category", min_width=10)
    table.add_column("KW", justify="right", min_width=4)
    table.add_column("Code", justify="right", min_width=4)
    table.add_column("Tool", justify="right", min_width=4)
    table.add_column("Combined", justify="right", min_width=7)
    table.add_column("Time", justify="right", min_width=5)

    has_judge = any(r.get("judge_avg") is not None for r in results)
    if has_judge:
        table.add_column("Judge", justify="right", min_width=5)

    for r in results:
        style = "green" if r["combined_score"] >= 0.7 else (
            "yellow" if r["combined_score"] >= 0.4 else "red")

        tool_str = "—"
        if r.get("tool_routing_score") is not None:
            tool_str = "✓" if r["tool_routing_score"] == 1.0 else "✗"

        row = [
            r["id"],
            r.get("category", ""),
            f"{r['keyword_score']:.2f}",
            f"{r['code_score']:.2f}",
            tool_str,
            f"{r['combined_score']:.2f}",
            f"{r['elapsed_s']:.1f}s",
        ]
        if has_judge:
            ja = r.get("judge_avg")
            row.append(f"{ja:.2f}" if ja is not None else "N/A")

        table.add_row(*row, style=style)

    console.print(table)

    # --- Category breakdown ---
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r.get("category") or "other"].append(r["combined_score"])

    breakdown = Table(title="Per-Category Breakdown")
    breakdown.add_column("Category", style="bold")
    breakdown.add_column("Cases", justify="right")
    breakdown.add_column("Avg Score", justify="right")
    breakdown.add_column("Pass Rate", justify="right")
    breakdown.add_column("", min_width=22)

    for cat, scores in sorted(by_cat.items()):
        avg = sum(scores) / len(scores)
        passes = sum(1 for s in scores if s >= 0.5)
        bar = _bar(avg)
        style = "green" if avg >= 0.7 else ("yellow" if avg >= 0.4 else "red")
        breakdown.add_row(
            cat, str(len(scores)), f"{avg:.2f}",
            f"{passes}/{len(scores)} ({passes/len(scores)*100:.0f}%)",
            bar, style=style,
        )

    console.print(breakdown)

    # --- Tool routing summary (if applicable) ---
    tool_cases = [r for r in results if r.get("tool_routing_score") is not None]
    if tool_cases:
        hits = sum(1 for r in tool_cases if r["tool_routing_score"] == 1.0)
        total = len(tool_cases)
        console.print(Panel(
            f"Tool Routing Accuracy: [bold]{hits}/{total}[/] ({hits/total*100:.0f}%)",
            title="Tool Routing", border_style="blue",
        ))

    # --- Short response warning ---
    short_count = sum(1 for r in results if r.get("short_response"))
    if short_count:
        console.print(Panel(
            f"[bold red]{short_count}/{len(results)}[/] responses were under "
            f"{_SHORT_RESPONSE_THRESHOLD} chars — likely API errors or empty replies",
            title="Short Response Warning", border_style="red",
        ))

    # --- Overall ---
    all_scores = [r["combined_score"] for r in results]
    avg_all = sum(all_scores) / len(all_scores) if all_scores else 0
    pass_all = sum(1 for s in all_scores if s >= 0.5)

    console.print(Panel(
        f"Average Score: [bold]{avg_all:.3f}[/]  |  "
        f"Pass Rate (>=0.5): [bold]{pass_all}/{len(results)}[/] "
        f"({pass_all/len(results)*100:.0f}%)" if results else "No results",
        title="Overall", border_style="green" if avg_all >= 0.7 else (
            "yellow" if avg_all >= 0.4 else "red"),
    ))

# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], mode: str, config: AgentConfig) -> str:
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# OptiProfiler Agent Evaluation Report")
    lines.append(f"\n- **Date**: {ts}")
    lines.append(f"- **Mode**: {mode}")
    lines.append(f"- **Provider**: {config.llm.provider}")
    lines.append(f"- **Model**: {config.llm.model}")
    lines.append(f"- **Cases**: {len(results)}")

    all_scores = [r["combined_score"] for r in results]
    avg = sum(all_scores) / len(all_scores) if all_scores else 0
    passes = sum(1 for s in all_scores if s >= 0.5)
    lines.append(f"\n## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Average Score | **{avg:.3f}** |")
    lines.append(f"| Pass Rate (>=0.5) | **{passes}/{len(results)}** ({passes/len(results)*100:.0f}%) |")

    tool_cases = [r for r in results if r.get("tool_routing_score") is not None]
    if tool_cases:
        hits = sum(1 for r in tool_cases if r["tool_routing_score"] == 1.0)
        lines.append(f"| Tool Routing Accuracy | **{hits}/{len(tool_cases)}** ({hits/len(tool_cases)*100:.0f}%) |")

    # Per-category
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r.get("category") or "other"].append(r["combined_score"])

    lines.append(f"\n## Per-Category Breakdown\n")
    lines.append(f"| Category | Cases | Avg Score | Pass Rate |")
    lines.append(f"|----------|-------|-----------|-----------|")
    for cat, scores in sorted(by_cat.items()):
        a = sum(scores) / len(scores)
        p = sum(1 for s in scores if s >= 0.5)
        lines.append(f"| {cat} | {len(scores)} | {a:.2f} | {p}/{len(scores)} ({p/len(scores)*100:.0f}%) |")

    # Detail table
    lines.append(f"\n## Detailed Results\n")
    lines.append(f"| ID | Category | KW | Code | Tool | Combined | Time |")
    lines.append(f"|----|----------|----|------|------|----------|------|")
    for r in results:
        tool_str = "—"
        if r.get("tool_routing_score") is not None:
            tool_str = "PASS" if r["tool_routing_score"] == 1.0 else "MISS"
        emoji = "+" if r["combined_score"] >= 0.7 else ("~" if r["combined_score"] >= 0.4 else "-")
        lines.append(
            f"| {emoji} {r['id']} | {r.get('category','')} "
            f"| {r['keyword_score']:.2f} | {r['code_score']:.2f} "
            f"| {tool_str} | **{r['combined_score']:.2f}** | {r['elapsed_s']:.1f}s |"
        )

    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OptiProfiler Agent (advisor or unified mode)")
    parser.add_argument("--mode", choices=["advisor", "unified"], default="advisor",
                        help="Which agent to evaluate (default: advisor)")
    parser.add_argument("--provider", default="minimax", help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name (overrides provider default)")
    parser.add_argument("--cases", default=None, help="Path to test cases JSON or directory")
    parser.add_argument("--judge", action="store_true", help="Enable multi-dimensional LLM-as-Judge")
    parser.add_argument("--judge-provider", default=None,
                        help="Provider for judge LLM (defaults to same as agent)")
    parser.add_argument("--output", default=None, help="Save raw results to JSON file")
    parser.add_argument("--report", default=None, help="Save Markdown report to file")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--rag", action="store_true", help="Enable RAG (advisor mode)")
    args = parser.parse_args()

    cases_path = Path(args.cases) if args.cases else None
    cases = load_cases(cases_path)
    if not cases:
        console.print("[red]No test cases found![/]")
        sys.exit(1)

    console.print(Panel(
        f"Mode: [bold]{args.mode}[/]  |  Cases: [bold]{len(cases)}[/]  |  "
        f"Judge: [bold]{'ON' if args.judge else 'OFF'}[/]",
        title="OptiProfiler Agent Evaluation", border_style="blue",
    ))

    config = AgentConfig(
        llm=LLMConfig(provider=args.provider, model=args.model),
        rag_enabled=args.rag or args.mode == "unified",
    )

    if args.mode == "advisor":
        from optiprofiler_agent.agent_a.advisor import AdvisorAgent
        agent = AdvisorAgent(config)
        runner = run_advisor
    else:
        from optiprofiler_agent.unified_agent import create_unified_agent
        agent = create_unified_agent(config)
        runner = run_unified

    console.print(f"  Provider: {config.llm.provider} | Model: {config.llm.model}")

    judge_llm = None
    if args.judge:
        from optiprofiler_agent.common.llm_client import create_llm
        judge_provider = args.judge_provider or args.provider
        judge_cfg = LLMConfig(provider=judge_provider)
        judge_llm = create_llm(judge_cfg)
        console.print(f"  Judge: {judge_cfg.provider} / {judge_cfg.model}")

    console.print()
    results = run_eval(cases, agent, runner, judge_llm=judge_llm,
                       verbose=args.verbose, mode=args.mode)

    console.print()
    print_summary(results, args.mode)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Results saved to {args.output}[/]")

    if args.report:
        md = generate_report(results, args.mode, config)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(md)
        console.print(f"[green]Markdown report saved to {args.report}[/]")


if __name__ == "__main__":
    main()
