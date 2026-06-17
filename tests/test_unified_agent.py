"""Tests for the unified tool-use agent.

Verifies tool registration and agent creation without requiring an LLM.
"""

from unittest.mock import MagicMock, patch

from optiprofiler_agent.config import AgentConfig, LLMConfig
from optiprofiler_agent.unified_agent import _SYSTEM_PROMPT, _build_tools, create_unified_agent


class TestBuildTools:

    def test_returns_expected_tool_count(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        # 4 original optiprofiler tools + 4 Hermes-inspired runtime tools
        # + scaffold_feature + write_scaffold_file + M4b plib tools + web_search.
        assert len(tools) == 14

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
            "scaffold_feature",
            "write_scaffold_file",
            "scan_local_plib",
            "scaffold_plib_wrapper",
            "smoke_test_plib_wrapper",
            "web_search",
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

    def test_validate_script_matlab_passes(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        validate = next(t for t in tools if t.name == "validate_script")

        code = (
            "function x = solver(fun, x0)\n"
            "    x = fminsearch(fun, x0);\n"
            "end\n"
        )
        result = validate.invoke({"code": code, "language": "matlab"})
        assert "looks good" in result.lower()

    def test_validate_script_matlab_detects_system(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        validate = next(t for t in tools if t.name == "validate_script")

        code = "function x = s(fun, x0)\n    system('ls');\nend\n"
        result = validate.invoke({"code": code, "language": "matlab"})
        assert "issues found" in result.lower() or "system" in result.lower()

    def test_interpret_results_nonexistent_dir(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        interp = next(t for t in tools if t.name == "interpret_results")

        result = interp.invoke({"results_dir": "/nonexistent/path/xyz"})
        assert "error" in result.lower() or "does not exist" in result.lower()

    def test_scaffold_feature_tool_generates_valid_custom_feature(self):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        scaffold = next(t for t in tools if t.name == "scaffold_feature")

        result = scaffold.invoke({
            "description": "heavy-tailed objective noise with level 1e-3",
            "feature_name": "heavy_tail_noise",
        })
        assert "feature_name=\"custom\"" in result
        assert "mod_fun=custom_mod_fun" in result
        assert "rng.standard_t" in result
        assert "Validation: passed" in result

    def test_scaffold_feature_tool_can_preview_file_write(self, tmp_path):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        scaffold = next(t for t in tools if t.name == "scaffold_feature")
        target = tmp_path / "features.py"

        result = scaffold.invoke({
            "description": "heavy-tailed objective noise with level 1e-3",
            "target_path": str(target),
            "dry_run": True,
        })

        assert "Scaffold File Preview" in result
        assert "```diff" in result
        assert not target.exists()

    def test_write_scaffold_file_tool_writes_when_requested(self, tmp_path):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        write_tool = next(t for t in tools if t.name == "write_scaffold_file")
        target = tmp_path / "snippet.py"

        result = write_tool.invoke({
            "path": str(target),
            "body": "print('ok')\n",
            "mode": "new",
            "dry_run": False,
        })

        assert "Scaffold File Written" in result
        assert target.exists()
        assert "print('ok')" in target.read_text(encoding="utf-8")

    def test_scan_local_plib_tool_returns_json_evidence(self, tmp_path):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        scan = next(t for t in tools if t.name == "scan_local_plib")
        (tmp_path / "toy.py").write_text(
            "def load_problem(name):\n    return name\n",
            encoding="utf-8",
        )

        result = scan.invoke({"src_dir": str(tmp_path), "library_name": "toy"})

        assert '"library_name": "toy"' in result
        assert '"toy.py"' in result

    def test_plib_wrapper_tools_generate_and_smoke_test(self, tmp_path):
        config = AgentConfig(llm=LLMConfig(provider="openai", api_key="fake"))
        tools = _build_tools(config)
        scaffold = next(t for t in tools if t.name == "scaffold_plib_wrapper")
        smoke = next(t for t in tools if t.name == "smoke_test_plib_wrapper")
        (tmp_path / "toy.py").write_text(
            "import numpy as np\n\n"
            "class P:\n"
            "    x0 = np.array([1.0])\n"
            "    def fun(self, x): return float(np.sum(np.asarray(x) ** 2))\n\n"
            "def load_problem(name): return P()\n"
            "def find_problems(options): return ['P1']\n",
            encoding="utf-8",
        )
        (tmp_path / "probinfo_toy.csv").write_text(
            "problem_name,ptype,dim,mb,mcon,mlcon,mnlcon\nP1,u,1,0,0,0,0\n",
            encoding="utf-8",
        )
        stage = tmp_path / "stage"

        scaffold_result = scaffold.invoke({
            "src_dir": str(tmp_path),
            "library_name": "toy",
            "staging_dir": str(stage),
        })
        (stage / "optiprofiler.py").write_text(
            "class Problem:\n"
            "    def __init__(self, fun, x0, name='', xl=None, xu=None):\n"
            "        self.fun = fun\n"
            "        self.x0 = x0\n"
            "        self.name = name\n"
            "        self.xl = xl\n"
            "        self.xu = xu\n",
            encoding="utf-8",
        )
        smoke_result = smoke.invoke({
            "staging_dir": str(stage),
            "library_name": "toy",
        })

        assert '"tools_path"' in scaffold_result
        assert '"ok": true' in smoke_result
        assert '"P1"' in smoke_result


def test_system_prompt_requires_debug_tool_for_error_plus_code():
    assert "error/traceback/exception message together with code" in _SYSTEM_PROMPT
    assert "MUST call **debug_error**" in _SYSTEM_PROMPT


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
