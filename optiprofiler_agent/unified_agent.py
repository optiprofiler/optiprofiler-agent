"""Unified tool-use Agent — a single ReAct agent that can invoke
Agent A (knowledge retrieval), Agent B (debug), and Agent C (interpret)
as tools, providing a natural conversational interface.

This agent automatically decides which tool to use based on the user's
question, combining advisory, debugging, and interpretation capabilities
into one loop.

Usage::

    from optiprofiler_agent.unified_agent import create_unified_agent
    agent = create_unified_agent(config)
    result = agent.invoke({"messages": [("user", "What is ptype?")]})
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from optiprofiler_agent.config import AgentConfig
from optiprofiler_agent.common.llm_client import create_llm

_SYSTEM_PROMPT = """\
You are the **OptiProfiler AI Assistant**, an expert in optimization benchmarking \
with OptiProfiler. You have three specialized capabilities available as tools:

1. **knowledge_search** — Search the OptiProfiler knowledge base to answer \
questions about the API, parameters, solver interface, features, problem libraries, \
performance profiles, data profiles, log-ratio profiles, and general usage.

2. **validate_script** — Check a Python benchmark script for syntax errors \
and API usage issues. Use when the user shares code and asks whether it is correct.

3. **debug_error** — Diagnose a benchmark error given the code and traceback. \
Use when the user pastes an error message or traceback.

4. **interpret_results** — Analyze benchmark output (from a results directory) \
and generate a summary report. Use when the user has run a benchmark and wants \
to understand the results.

**Guidelines:**
- OptiProfiler focuses on **Derivative-Free Optimization (DFO)**.
- `fun` provides ONLY function values, no gradients.
- `benchmark()` requires at least **2 solvers**.
- Always call a tool when specific information is needed rather than guessing.
- When using knowledge_search, formulate a clear, specific query.
- Respond in the same language as the user.

**Calling `benchmark()` — do not confuse solver vs benchmark API:**
- **Solvers** are called as `solver(fun, x0, ...)` — `fun` is the objective for one problem.
- **MATLAB** `benchmark` is `scores = benchmark({@s1, @s2})` or `scores = benchmark({@s1, @s2}, options)` where **`options` is a struct**. The second argument is **never** the user's objective function handle `f`. Test problems come from OptiProfiler problem libraries; you do **not** pass `f` into `benchmark`.
- **Python** `benchmark` takes keyword options: `benchmark([s1, s2], ptype='u', ...)`.
- Before writing MATLAB `benchmark` example code, call **knowledge_search** with a query such as \
"MATLAB benchmark function signature options struct" so retrieved docs override generic optimization habits.
"""


def _build_tools(config: AgentConfig) -> list:
    """Create the tool functions with bound config."""

    @tool
    def knowledge_search(
        query: Annotated[str, "A specific question about OptiProfiler"],
    ) -> str:
        """Search the OptiProfiler knowledge base for API docs, parameters,
        solver interface, features, profiles, and general usage information."""
        from optiprofiler_agent.common.rag import KnowledgeRAG

        rag = KnowledgeRAG(
            config.knowledge_dir,
            persist_dir=config.rag_persist_dir,
        )
        rag.build_index()
        results = rag.retrieve_with_index(query, top_k=5)

        if not results:
            return "No relevant information found in the knowledge base."

        parts = []
        for item in results:
            parts.append(f"[Source: {item['source']}]\n{item['text']}")
        return "\n\n---\n\n".join(parts)

    @tool
    def validate_script(
        code: Annotated[str, "Python source code to validate"],
        language: Annotated[str, "python or matlab"] = "python",
    ) -> str:
        """Validate a benchmark script for syntax errors and API usage issues.
        Returns a list of issues found or a success message."""
        from optiprofiler_agent.validators.syntax_checker import check_code_string
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        issues: list[str] = []

        syn = check_code_string(code)
        if syn.has_errors:
            for err in syn.errors:
                issues.append(f"Syntax error at line {err.line}: {err.message}")

        api = validate_benchmark_call(code, language=language)
        if api.has_errors or api.has_warnings:
            for issue in api.issues:
                issues.append(f"[{issue.severity}] {issue.message}")

        if not issues:
            calls = api.benchmark_calls_found
            return f"Script looks good! Found {calls} benchmark() call(s), no issues detected."

        return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)

    @tool
    def debug_error(
        code: Annotated[str, "The Python source code that produced the error"],
        error: Annotated[str, "The full traceback or error message"],
    ) -> str:
        """Diagnose a benchmark script error and suggest a fix.
        Provide the code and the error traceback."""
        from optiprofiler_agent.agent_b.debugger import debug_script

        result = debug_script(code=code, error=error, config=config)

        output = result.diagnostic_report
        if result.fixed_code:
            output += f"\n\n## Suggested Fix\n\n```python\n{result.fixed_code}\n```"
        return output

    @tool
    def interpret_results(
        results_dir: Annotated[str, "Path to the benchmark results directory"],
        use_latest: Annotated[bool, "Auto-detect latest experiment"] = True,
    ) -> str:
        """Analyze benchmark results and generate a summary report.
        Point to the output directory (e.g., 'out/') or a specific experiment folder."""
        from optiprofiler_agent.agent_c.interpreter import interpret
        from optiprofiler_agent.agent_c.result_loader import find_latest_experiment

        target = results_dir
        if use_latest:
            try:
                target = str(find_latest_experiment(results_dir))
            except FileNotFoundError:
                pass

        if not Path(target).exists():
            return f"Error: directory '{target}' does not exist."

        try:
            report = interpret(
                results_dir=target,
                config=config,
                language="English",
                read_profiles=True,
                llm_enabled=False,
            )
            return report
        except Exception as e:
            return f"Error analyzing results: {e}"

    return [knowledge_search, validate_script, debug_error, interpret_results]


def create_unified_agent(config: AgentConfig | None = None):
    """Create a LangGraph ReAct agent with all OptiProfiler tools.

    Returns a compiled graph that can be invoked with::

        result = agent.invoke({"messages": [("user", "...")]})
        print(result["messages"][-1].content)
    """
    config = config or AgentConfig()
    llm = create_llm(config.llm)
    tools = _build_tools(config)

    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=_SYSTEM_PROMPT,
    )

    return agent
