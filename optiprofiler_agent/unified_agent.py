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

_SYSTEM_PROMPT_BASE = """\
You are the **OptiProfiler AI Assistant**, an expert in optimization benchmarking \
with OptiProfiler. You have these specialized capabilities available as tools:

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

5. **remember** — Persist one short factual note to long-term memory. Use \
sparingly, only for facts the user explicitly asks you to keep, or stable \
preferences that will help across future sessions.

6. **update_user_profile** — Update one whitelisted profile field (name, role, \
preferred_solver, preferred_language, project_root). Use only when the user \
states a stable preference clearly.

7. **recall_past** — Full-text search of previous chat turns. Use when the \
user references something they said before that is not in the current context.

8. **add_wiki_page** — Create a small knowledge-base page for a fact you \
verified that is missing from the wiki. Pages are picked up by knowledge_search \
on the next index rebuild. Use rarely; do not duplicate existing pages.

9. **web_search** — Search the public web for OPEN-WORLD context that the \
OptiProfiler knowledge base cannot supply. Permitted scopes: external \
solvers (PRIMA, NLopt, etc.) and their known issues; recent papers; \
decoding tracebacks from third-party libraries (e.g. searching the exact \
error message from `scipy` or `pycutest`); installation tips for those \
external tools. STRICTLY NOT permitted: any question about the \
`optiprofiler` package itself — its API, parameters, examples, install \
flow, or `benchmark()` usage MUST come from `knowledge_search`. \
Routing rule: for OptiProfiler-internal questions, try `knowledge_search` \
first; for clearly open-world questions (e.g. "Search the web for ...", \
external library issues, recent papers), call `web_search` directly. \
**You MUST NEVER claim that a tool is "unavailable", "disabled", \
"not configured", or "I cannot search" without FIRST actually calling \
the tool.** The tool itself will return a message starting with \
"web_search disabled: ..." or "web_search error: ..." if it is genuinely \
not configured; only then may you relay that information to the user. \
Pre-emptively refusing without invoking the tool is a hallucination \
and is forbidden.

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

**Python imports — facts to repeat verbatim, never paraphrase:**
- The package is named **`optiprofiler`** (one word, lowercase, no hyphen, no underscore). \
Common typos to avoid: `optiprobe`, `opti_profiler`, `opti-profiler`.
- The public API is **flat**: `from optiprofiler import benchmark, Problem, Feature, FeaturedProblem, \
s2mpj_load, s2mpj_select, pycutest_load, pycutest_select, get_plib_config, set_plib_config`. \
There is **no** `optiprofiler.solvers`, `optiprofiler.algorithms`, or `optiprofiler.utils` submodule.
- Solvers are **third-party callables you import yourself** (e.g. `from prima import bobyqa`), \
not symbols from the `optiprofiler` package.
- If you are unsure whether a symbol exists, call **knowledge_search** with a query like \
"Python imports and exports public API" before writing the import line. Do not invent paths.
"""


def _compose_system_prompt() -> str:
    """Prepend the persistent-memory frozen snapshot (if any) to the base prompt.

    The snapshot is computed *at agent build time*, not per turn, so the
    LangGraph ReAct loop can keep its prompt static. CLI rebuilds the agent
    on every fresh chat session, so newly remembered facts surface on the
    next session without extra plumbing.
    """
    try:
        from optiprofiler_agent.runtime import memory as _rt_memory
        snapshot = _rt_memory.frozen_snapshot()
    except Exception:
        snapshot = ""
    if not snapshot:
        return _SYSTEM_PROMPT_BASE
    return snapshot + "\n" + _SYSTEM_PROMPT_BASE


_SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE  # backwards-compat for tests / imports


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
        from optiprofiler_agent.debugger.debugger import debug_script

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
        from optiprofiler_agent.interpreter.interpreter import interpret
        from optiprofiler_agent.interpreter.result_loader import find_latest_experiment

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

    @tool
    def remember(
        fact: Annotated[str, "One short factual sentence to persist"],
        tags: Annotated[list[str], "Optional list of short tags"] = [],
    ) -> str:
        """Append a fact to the agent's long-term MEMORY.md."""
        from optiprofiler_agent.runtime import memory as _mem

        line = _mem.append_fact(fact, tags=tags)
        if not line:
            return "Nothing to remember (empty fact)."
        return f"Stored: {line}"

    @tool
    def update_user_profile(
        field: Annotated[str, "One of: name, role, preferred_solver, preferred_language, project_root"],
        value: Annotated[str, "The new value"],
    ) -> str:
        """Set one whitelisted USER profile field."""
        from optiprofiler_agent.runtime import memory as _mem

        try:
            line = _mem.update_user_profile(field, value)
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Updated profile: {line}"

    @tool
    def recall_past(
        query: Annotated[str, "Keywords to search past chat turns"],
        limit: Annotated[int, "Max number of hits to return"] = 5,
    ) -> str:
        """Full-text search past chat turns from previous sessions."""
        from optiprofiler_agent.runtime import session_log as _sl

        hits = _sl.search(query, limit=limit)
        if not hits:
            return "No matching past turns."
        lines = []
        for h in hits:
            snippet = h.content if len(h.content) <= 240 else h.content[:240] + "..."
            lines.append(f"[{h.role} @ session {h.session_id[:8]}] {snippet}")
        return "\n".join(lines)

    @tool
    def add_wiki_page(
        slug: Annotated[str, "Short slug (will be sanitized)"],
        content: Annotated[str, "Markdown body of the page"],
        summary: Annotated[str, "One-line summary"] = "",
    ) -> str:
        """Create a new agent-authored wiki page under OPAGENT_HOME/wiki/auto/."""
        from optiprofiler_agent.runtime import wiki_local as _wl

        path = _wl.add_page(slug, content, summary=summary, source="agent")
        return f"Wrote {path}"

    from optiprofiler_agent.tools.web_search import web_search

    return [
        knowledge_search,
        validate_script,
        debug_error,
        interpret_results,
        remember,
        update_user_profile,
        recall_past,
        add_wiki_page,
        web_search,
    ]


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
        prompt=_compose_system_prompt(),
    )

    return agent
