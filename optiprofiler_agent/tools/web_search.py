"""Web search tool for the unified agent.

Backed by Tavily (optional dependency ``langchain-tavily``). Designed
for **graceful degradation**: if the dependency is missing or the API
key is unset, the tool returns a one-line explanation instead of
raising, so the rest of the agent keeps working.

Scope policy (enforced by the ``@tool`` docstring + the system prompt
in ``unified_agent``):

- ALLOWED: open-world facts about external solvers, recent papers,
  GitHub issues, error tracebacks from external libraries.
- DISALLOWED: any question about OptiProfiler's own API, parameters,
  installation, or examples — those must come from
  ``knowledge_search`` to avoid hallucination through the search
  surface.

The scope is enforced at the *prompt* layer, not in code, because the
LLM is the only component that can route questions semantically. The
docstring you see below is what the model receives as the tool
description.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Annotated

from langchain_core.tools import tool


_LOG = logging.getLogger(__name__)


def _disabled_no_dep() -> str:
    """Diagnostic message for missing langchain-tavily.

    Includes ``sys.executable`` so the user can immediately see which
    Python environment lacks the dependency — this is critical when
    multiple ``opagent`` shims exist (e.g. one in ``.venv/bin`` and one
    in ``/opt/homebrew/bin``) and PATH resolution is ambiguous.
    """
    return (
        "web_search disabled: `langchain-tavily` is not importable "
        f"from the active Python ({sys.executable}). Install it into "
        "THIS interpreter (not just any other env) with:\n"
        f"    {sys.executable} -m pip install 'langchain-tavily>=0.2'\n"
        "If you intended to run from a virtualenv, verify with "
        "`which opagent` that you are launching the venv binary, not a "
        "stale shim from /opt/homebrew/bin or ~/.local/bin."
    )


_DISABLED_NO_KEY = (
    "web_search disabled: set the TAVILY_API_KEY environment variable "
    "(get a free key at https://tavily.com) to enable web search."
)


def _format_results(payload: dict | list) -> str:
    """Format Tavily output into a compact LLM-friendly string."""
    if isinstance(payload, dict):
        results = payload.get("results", [])
    else:
        results = payload or []

    if not results:
        return "No web results found."

    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        snippet = (item.get("content") or item.get("snippet") or "").strip()
        url = (item.get("url") or "").strip()
        if len(snippet) > 500:
            snippet = snippet[:500].rstrip() + "..."
        block = f"[{idx}] {title}\n{snippet}\nurl: {url}"
        lines.append(block)
    return "\n\n".join(lines) if lines else "No web results found."


def _run_tavily_search(query: str, max_results: int = 3) -> str:
    """Invoke Tavily once, returning a formatted string.

    All exceptions are converted to a leading ``web_search error:`` line
    so the agent can decide whether to retry or fall back; nothing is
    raised to the caller.
    """
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return _disabled_no_dep()

    if not os.environ.get("TAVILY_API_KEY"):
        return _DISABLED_NO_KEY

    try:
        searcher = TavilySearch(max_results=max_results, search_depth="basic")
        raw = searcher.invoke({"query": query})
    except Exception as exc:
        _LOG.warning("web_search failed: %s", exc)
        return f"web_search error: {exc}"

    return _format_results(raw)


@tool
def web_search(
    query: Annotated[str, "An open-world question that the OptiProfiler knowledge base cannot answer"],
) -> str:
    """Search the public web for open-world context.

    USE this tool ONLY when the answer cannot come from
    `knowledge_search`. Good fits:
    - questions about external optimization solvers, their latest
      versions, or known bugs (e.g. "PRIMA bobyqa numerical issues")
    - recent papers or benchmarks mentioned by name
    - decoding an error traceback from an external library (e.g.
      a `scipy` or `pycutest` import error) by searching the exact
      message
    - operating system / installation tips for third-party tools

    DO NOT use this tool for:
    - any question about the `optiprofiler` package itself, its API,
      parameters, or examples (those must come from `knowledge_search`)
    - questions about how to call `benchmark()` or any OptiProfiler
      function

    Always try `knowledge_search` first. If it returns nothing useful
    AND the question is open-world, then call `web_search`.

    Returns a numbered list of search hits with titles, snippets, and
    URLs. Returns a one-line explanation when the tool is not
    configured (missing dependency or missing TAVILY_API_KEY); in that
    case, answer from your own knowledge with a clear caveat that web
    search was unavailable.
    """
    query = (query or "").strip()
    if not query:
        return "web_search error: empty query."
    return _run_tavily_search(query)


__all__ = ["web_search"]
