#!/usr/bin/env python3
"""Minimal one-shot test: unified ReAct agent must call ``validate_script``.

Usage (from repo root, after ``pip install -e .``)::

    SMOKE_PROVIDER=deepseek DEEPSEEK_API_KEY=... python scripts/smoke_react_tools.py
    SMOKE_PROVIDER=mimo MIMO_API_KEY=... python scripts/smoke_react_tools.py

Exit 0 only if ``validate_script`` appears in message ``tool_calls`` (proves
multi-turn tool replay, not plain chat). Set ``SMOKE_RAG=1`` to enable RAG for
``knowledge_search`` smoke (needs optional ``[rag]`` deps + index).

Do not paste API keys into shell history on shared machines; use ``export`` in
a private session or a local-only ``.env`` (never commit).
"""

from __future__ import annotations

import os
import sys


def _tool_names_from_messages(msgs: list) -> list[str]:
    names: list[str] = []
    for m in msgs:
        tcs = getattr(m, "tool_calls", None) or []
        for tc in tcs:
            n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if n:
                names.append(n)
    return names


def main() -> int:
    provider = os.environ.get("SMOKE_PROVIDER", "deepseek").strip().lower()
    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
    elif provider == "mimo":
        key = os.environ.get("MIMO_API_KEY")
    else:
        print("Set SMOKE_PROVIDER=deepseek or mimo", file=sys.stderr)
        return 2
    if not key:
        print("Missing API key (DEEPSEEK_API_KEY or MIMO_API_KEY)", file=sys.stderr)
        return 2

    rag = os.environ.get("SMOKE_RAG", "").strip().lower() in ("1", "true", "yes")

    from optiprofiler_agent.config import AgentConfig, LLMConfig
    from optiprofiler_agent.unified_agent import create_unified_agent

    llm = LLMConfig(provider=provider, api_key=key)
    cfg = AgentConfig(llm=llm, rag_enabled=rag)
    agent = create_unified_agent(cfg)

    # Strong routing: should invoke validate_script (no RAG / web required).
    prompt = (
        "You must use the validate_script tool exactly once before answering. "
        "Pass this as the code argument (verbatim):\n\n"
        "from optiprofiler import benchmark\n"
        "def a(fun, x0): return x0\n"
        "def b(fun, x0): return x0\n"
        "benchmark([a, b])\n\n"
        "Then summarize what validate_script returned in one short paragraph."
    )

    out = agent.invoke({"messages": [("user", prompt)]})
    msgs = out.get("messages", [])
    tools = _tool_names_from_messages(msgs)
    print("tool_calls (order):", tools)
    final = msgs[-1] if msgs else None
    body = getattr(final, "content", None) or ""
    print("--- last assistant (trim) ---\n", str(body)[:1600])
    if "validate_script" not in tools:
        print("\nFAIL: expected validate_script in tool_calls", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
