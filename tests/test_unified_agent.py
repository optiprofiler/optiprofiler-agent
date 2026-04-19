"""Tests for the unified tool-use agent.

Verifies tool registration and agent creation without requiring an LLM.
"""

from unittest.mock import MagicMock, patch

from optiprofiler_agent.config import AgentConfig, LLMConfig
from optiprofiler_agent.unified_agent import _build_tools, create_unified_agent


class TestBuildTools:

    def test_returns_expected_tool_count(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        # 4 original optiprofiler tools + 4 Hermes-inspired runtime tools
        assert len(tools) == 8

    def test_tool_names(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        names = {t.name for t in tools}
        assert names == {
            "knowledge_search",
            "validate_script",
            "debug_error",
            "interpret_results",
            "remember",
            "update_user_profile",
            "recall_past",
            "add_wiki_page",
        }

    def test_validate_script_tool_works(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        validate = next(t for t in tools if t.name == "validate_script")

        good_code = (
            "from optiprofiler import benchmark\n"
            "def a(fun, x0): return x0\n"
            "def b(fun, x0): return x0\n"
            "benchmark([a, b])\n"
        )
        result = validate.invoke({"code": good_code})
        assert "looks good" in result.lower() or "no issues" in result.lower()

    def test_validate_script_detects_single_solver(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        validate = next(t for t in tools if t.name == "validate_script")

        bad_code = (
            "from optiprofiler import benchmark\n"
            "def a(fun, x0): return x0\n"
            "benchmark([a])\n"
        )
        result = validate.invoke({"code": bad_code})
        assert "1 provided" in result or "error" in result.lower()

    def test_interpret_results_nonexistent_dir(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        interp = next(t for t in tools if t.name == "interpret_results")

        result = interp.invoke({"results_dir": "/nonexistent/path/xyz"})
        assert "error" in result.lower() or "does not exist" in result.lower()


class TestCreateUnifiedAgent:

    @patch("optiprofiler_agent.unified_agent.create_llm")
    def test_agent_created_successfully(self, mock_create_llm):
        mock_create_llm.return_value = MagicMock()
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        agent = create_unified_agent(config)
        assert agent is not None

    @patch("optiprofiler_agent.unified_agent.create_llm")
    def test_agent_has_invoke(self, mock_create_llm):
        mock_create_llm.return_value = MagicMock()
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        agent = create_unified_agent(config)
        assert hasattr(agent, "invoke")
