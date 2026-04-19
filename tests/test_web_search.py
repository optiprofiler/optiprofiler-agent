"""Tests for the web_search tool (Tavily-backed, optional)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from optiprofiler_agent.tools import web_search as ws_mod
from optiprofiler_agent.tools.web_search import web_search


class TestGracefulDegradation:
    """The tool MUST never raise; it returns a one-line explanation
    when the dependency or API key is unavailable."""

    def test_empty_query(self):
        result = web_search.invoke({"query": ""})
        assert result.startswith("web_search error: empty query")

    def test_missing_dependency_returns_disabled_message(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "langchain_tavily", raising=False)

        # Prime an import error by injecting a sentinel that raises on
        # import. We do this by patching builtins.__import__.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain_tavily":
                raise ImportError("simulated missing dep")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = web_search.invoke({"query": "anything"})

        assert "web_search disabled" in result
        assert "pip install" in result

    def test_missing_api_key_returns_disabled_message(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        # Stub the langchain_tavily module so the import succeeds.
        fake_module = MagicMock()
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_module)

        result = web_search.invoke({"query": "anything"})
        assert "web_search disabled" in result
        assert "TAVILY_API_KEY" in result


class TestSuccessfulSearch:

    def test_formatted_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")

        fake_searcher = MagicMock()
        fake_searcher.invoke.return_value = {
            "results": [
                {
                    "title": "PRIMA bobyqa README",
                    "content": "PRIMA implements bobyqa with Powell's method.",
                    "url": "https://github.com/example/prima",
                },
                {
                    "title": "scipy issue 12345",
                    "content": "ImportError when scipy.optimize is missing.",
                    "url": "https://github.com/scipy/scipy/issues/12345",
                },
            ]
        }
        fake_module = MagicMock()
        fake_module.TavilySearch.return_value = fake_searcher
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_module)

        result = web_search.invoke({"query": "prima bobyqa"})

        assert "[1] PRIMA bobyqa README" in result
        assert "url: https://github.com/example/prima" in result
        assert "[2] scipy issue 12345" in result

    def test_no_results_returns_friendly_message(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")

        fake_searcher = MagicMock()
        fake_searcher.invoke.return_value = {"results": []}
        fake_module = MagicMock()
        fake_module.TavilySearch.return_value = fake_searcher
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_module)

        result = web_search.invoke({"query": "no hits"})
        assert result == "No web results found."

    def test_provider_exception_swallowed(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-test")

        fake_searcher = MagicMock()
        fake_searcher.invoke.side_effect = RuntimeError("network down")
        fake_module = MagicMock()
        fake_module.TavilySearch.return_value = fake_searcher
        monkeypatch.setitem(sys.modules, "langchain_tavily", fake_module)

        result = web_search.invoke({"query": "anything"})
        assert result.startswith("web_search error:")
        assert "network down" in result


class TestSnippetTruncation:

    def test_long_snippet_truncated(self):
        payload = {
            "results": [
                {
                    "title": "long",
                    "content": "x" * 1000,
                    "url": "https://example.com",
                },
            ]
        }
        formatted = ws_mod._format_results(payload)
        assert "..." in formatted
        # 500-char body + "..." suffix; full 1000 chars must NOT appear.
        assert "x" * 1000 not in formatted


class TestScopeContract:
    """The system-prompt scope policy is enforced at the prompt layer,
    but we pin the docstring keywords so a refactor that drops them
    is caught immediately — those keywords are what tells the LLM
    when NOT to call this tool."""

    def test_docstring_forbids_optiprofiler_questions(self):
        doc = web_search.description or ""
        assert "DO NOT" in doc
        assert "optiprofiler" in doc.lower()
        assert "knowledge_search" in doc

    def test_docstring_lists_debug_use_case(self):
        doc = web_search.description or ""
        assert "traceback" in doc.lower()


class TestSystemPromptIntegration:
    """Pin the unified-agent system prompt scope language so a refactor
    that loses the RAG-first / debug-allowed contract is caught."""

    def test_prompt_mentions_web_search_with_guardrails(self):
        from optiprofiler_agent.unified_agent import _SYSTEM_PROMPT_BASE

        assert "web_search" in _SYSTEM_PROMPT_BASE
        assert "knowledge_search` first" in _SYSTEM_PROMPT_BASE
        assert "STRICTLY NOT permitted" in _SYSTEM_PROMPT_BASE
        assert "tracebacks" in _SYSTEM_PROMPT_BASE


@pytest.fixture(autouse=True)
def _isolate_tavily_module(monkeypatch):
    """Make sure tests do not contaminate each other via sys.modules."""
    monkeypatch.delitem(sys.modules, "langchain_tavily", raising=False)
    yield
    monkeypatch.delitem(sys.modules, "langchain_tavily", raising=False)
