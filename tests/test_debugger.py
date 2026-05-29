"""Tests for Agent B — debugger module.

All tests use mocked LLM calls; no real API key needed.
"""

import os
from unittest.mock import MagicMock, patch

from optiprofiler_agent.debugger.debugger import (
    DebugResult,
    _build_debugger_web_query,
    _collect_web_debug_context,
    _debug_error_has_external_context,
    _extract_code_from_reply,
    _format_web_debug_context,
    _handle_python_static_fix,
    _handle_matlab_static_fix,
    _handle_interface_mismatch,
    _matlab_semantic_equal,
    _preservation_errors,
    _validate_code,
    debug_script,
    run_and_debug,
)
from optiprofiler_agent.common.interface_adapter import analyze_solver
from optiprofiler_agent.config import AgentConfig, LLMConfig


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

GOOD_CODE = """\
from optiprofiler import benchmark

def solver_a(fun, x0):
    return x0

def solver_b(fun, x0):
    return x0

benchmark([solver_a, solver_b])
"""

NAME_ERROR_TRACEBACK = """\
Traceback (most recent call last):
  File "script.py", line 10, in <module>
    benchmark([solver_a, solver_b])
NameError: name 'solver_b' is not defined
"""

IMPORT_ERROR_TRACEBACK = """\
Traceback (most recent call last):
  File "script.py", line 1, in <module>
    import nonexistent_module
ModuleNotFoundError: No module named 'nonexistent_module'
"""

SCIPY_TRACEBACK = """\
Traceback (most recent call last):
  File "script.py", line 3, in <module>
    from scipy.optimize import minimize
  File "/venv/lib/python3.11/site-packages/scipy/optimize/__init__.py", line 1, in <module>
    raise ImportError("cannot import name 'minimize'")
ImportError: cannot import name 'minimize' from 'scipy.optimize'
"""


def _make_config() -> AgentConfig:
    return AgentConfig(
        llm=LLMConfig(provider="openai", api_key="fake-key"),
        max_debug_retries=2,
    )


# ---------------------------------------------------------------------------
# _extract_code_from_reply
# ---------------------------------------------------------------------------

class TestExtractCode:

    def test_extracts_python_fenced_block(self):
        reply = "Here is the fix:\n\n```python\nprint('hello')\n```\n\nDone."
        assert _extract_code_from_reply(reply) == "print('hello')"

    def test_extracts_generic_fenced_block(self):
        reply = "Fix:\n```\nx = 1\n```"
        assert _extract_code_from_reply(reply) == "x = 1"

    def test_returns_none_for_no_code(self):
        assert _extract_code_from_reply("No code here.") is None

    def test_extracts_first_python_block(self):
        reply = "```python\nfirst()\n```\n```python\nsecond()\n```"
        assert _extract_code_from_reply(reply) == "first()"

    def test_extracts_matlab_fenced_block(self):
        reply = "Fix:\n```matlab\nfunction x = s(fun, x0)\nend\n```"
        code = _extract_code_from_reply(reply, language="matlab")
        assert code is not None
        assert "function x = s" in code

    def test_extracts_m_tagged_block(self):
        reply = "```m\nx = 1;\n```"
        assert _extract_code_from_reply(reply, language="matlab") == "x = 1;"


# ---------------------------------------------------------------------------
# _validate_code
# ---------------------------------------------------------------------------

class TestValidateCode:

    def test_valid_code_returns_empty(self):
        errors = _validate_code("x = 1\nprint(x)")
        assert errors == []

    def test_syntax_error_detected(self):
        errors = _validate_code("def f(\n")
        assert any("Syntax" in e or "syntax" in e.lower() for e in errors)

    def test_matlab_dangerous_call_detected(self):
        code = "function x = s(fun, x0)\n    system('ls');\nend\n"
        errors = _validate_code(code, language="matlab")
        assert len(errors) >= 1

    def test_matlab_valid_code_passes(self):
        code = "function x = s(fun, x0)\n    x = fminsearch(fun, x0);\nend\n"
        errors = _validate_code(code, language="matlab")
        assert errors == []

    def test_python_fix_must_preserve_top_level_defs(self):
        errors = _preservation_errors(
            "def validate_bounds(x0, lb, ub):\n    return True\n\nvalidate_bounds([], [], [])\n",
            "validate_bounds([], [], [])\n",
            language="python",
        )
        assert any("validate_bounds" in error for error in errors)

    def test_python_fix_may_change_body_while_preserving_def(self):
        errors = _preservation_errors(
            "def objective(x):\n    return 1 / x[0]\n",
            "def objective(x):\n    return 1 / (x[0] or 1e-12)\n",
            language="python",
        )
        assert errors == []

    def test_matlab_fix_must_preserve_local_functions(self):
        original = """\
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun, x0, options)
    x = options.scale * fminsearch(fun, x0);
end
"""
        fixed = """\
function x = my_solver_wrapper(fun, x0)
    x = my_solver(fun, x0);
end
"""
        errors = _preservation_errors(original, fixed, language="matlab")
        assert any("my_solver" in error for error in errors)
        assert any("top-level script" in error for error in errors)

    def test_matlab_fix_may_edit_local_function_signature(self):
        original = """\
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun)
    x = fminsearch(fun, [0; 0]);
end
"""
        fixed = """\
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun, x0)
    x = fminsearch(fun, x0);
end
"""
        assert _preservation_errors(original, fixed, language="matlab") == []


# ---------------------------------------------------------------------------
# interface mismatch (pre-flight bug #1)
# ---------------------------------------------------------------------------

class TestInterfaceMismatch:

    def test_wrapper_generated_for_reordered_params(self):
        code = "def my_solver(x0, fun):\n    return fun(x0)\n"
        analysis = analyze_solver(code)
        assert analysis.needs_wrapper
        fixed, report = _handle_interface_mismatch(code, "TypeError", language="python")
        assert fixed is not None
        assert "my_solver_wrapper" in fixed
        assert "Interface Mismatch" in report

    def test_matlab_wrapper_generated(self):
        code = "function x = my_solver(x0, fun)\n    x = fun(x0);\nend\n"
        analysis = analyze_solver(code, language="matlab")
        assert analysis.needs_wrapper
        fixed, report = _handle_interface_mismatch(
            code,
            "Error using my_solver\nToo many input arguments.",
            language="matlab",
        )
        assert fixed is not None
        assert "my_solver_wrapper" in fixed


# ---------------------------------------------------------------------------
# MATLAB static repairs
# ---------------------------------------------------------------------------

class TestMatlabStaticFixes:

    def test_repairs_options_missing_field_with_default_guard(self):
        code = "options.max_eval = 100;\ndisp(options.ptype);\n"
        fixed, report = _handle_matlab_static_fix(
            code,
            'Unrecognized field name "ptype".',
        )
        assert fixed is not None
        assert "if ~isfield(options, 'ptype')" in fixed
        assert "options.ptype = 'u';" in fixed
        assert "MATLAB Error Fixed" in report

    def test_repairs_result_missing_field_by_using_existing_field(self):
        code = "result.x = [0; 1];\nresult.f = 1.0;\ndisp(result.y);\n"
        fixed, _report = _handle_matlab_static_fix(
            code,
            'Unrecognized field name "y".',
        )
        assert fixed is not None
        assert "disp(result.x);" in fixed

    def test_repairs_too_many_inputs_by_expanding_local_signature(self):
        code = """\
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun)
    x = fminsearch(fun, [0; 0]);
end
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Too many input arguments.",
        )
        assert fixed is not None
        assert "function x = my_solver(fun, x0)" in fixed
        assert "fminsearch(fun, [0; 0])" in fixed

    def test_repairs_not_enough_inputs_by_defaulting_optional_struct(self):
        code = """\
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun, x0, options)
    x = options.scale * fminsearch(fun, x0);
end
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Not enough input arguments.",
        )
        assert fixed is not None
        assert "if nargin < 3" in fixed
        assert "options = struct();" in fixed
        assert "options.scale = 1;" in fixed

    def test_repairs_vector_concat_dimension_mismatch(self):
        code = """\
a = [1, 2, 3];
b = [4; 5; 6];
c = [a; b];
disp(c);
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Dimensions of arrays being concatenated are not consistent.",
        )
        assert fixed is not None
        assert "a = a(:).';" in fixed
        assert "b = b(:).';" in fixed
        assert "c = [a; b];" in fixed

    def test_repairs_constant_index_out_of_bounds(self):
        code = """\
a = [1, 2, 3];
disp(a(5));
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Index exceeds the number of array elements. Index must not exceed 3.",
        )
        assert fixed is not None
        assert "disp(a(min(5, numel(a))));" in fixed

    def test_index_bounds_fix_ignores_unassigned_function_calls(self):
        code = """\
rng(5);
a = [1, 2, 3];
disp(a(5));
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Index exceeds the number of array elements. Index must not exceed 3.",
        )
        assert fixed is not None
        assert "rng(5);" in fixed
        assert "disp(a(min(5, numel(a))));" in fixed

    def test_repairs_long_pause_timeout_repro(self):
        fixed, _report = _handle_matlab_static_fix(
            "pause(120);\n",
            "MATLAB script timed out after 45 seconds.",
        )
        assert fixed is not None
        assert "pause(0.1);" in fixed

    def test_repairs_single_missing_closing_parenthesis(self):
        code = """\
fun = @(z) sum(z.^2);
x0 = [1; 2; 3];
result = fminsearch(fun, x0;
disp(result);
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Invalid expression. When calling a function or indexing a variable, "
            "use parentheses. Otherwise, check for mismatched delimiters.",
        )
        assert fixed is not None
        assert "result = fminsearch(fun, x0);" in fixed

    def test_repairs_scalar_bound_shape_mismatch(self):
        code = """\
x0 = [0; 1];
lb = 0;
ub = [2; 3];
if numel(lb) ~= numel(x0) || numel(ub) ~= numel(x0)
    error('Bounds shape mismatch: expected length 2');
end
disp([lb(:), ub(:)]);
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Bounds shape mismatch: expected length 2",
        )
        assert fixed is not None
        assert "lb = [0; 0];" in fixed
        assert "ub = [2; 3];" in fixed

    def test_repairs_missing_optimizer_function_with_builtin_fallback(self):
        code = """\
x0 = [1; 2; 3];
fun = @(z) sum(z.^2);
result = cobyqa_mex(fun, x0);
disp(result);
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Undefined function 'cobyqa_mex' for input arguments of type 'function_handle'.",
        )
        assert fixed is not None
        assert "result = fminsearch(fun, x0);" in fixed

    def test_repairs_negative_start_for_complex_objective(self):
        code = """\
fun = @(x) sqrt(x(1));
x0 = -1;
y0 = fun(x0);
if ~isreal(y0) || ~isfinite(y0)
    error('Objective returned NaN/complex value at x0');
end
disp(y0);
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Objective returned NaN/complex value at x0",
        )
        assert fixed is not None
        assert "x0 = 1;" in fixed

    def test_repairs_x_start_typo_with_existing_x0(self):
        code = """\
x0 = [1; 2; 3];
fun = @(z) sum(z.^2);
result = fminsearch(fun, x_start);
disp(result);
"""
        fixed, _report = _handle_matlab_static_fix(
            code,
            "Unrecognized function or variable 'x_start'.",
        )
        assert fixed is not None
        assert "fminsearch(fun, x0)" in fixed
        assert "x_start" not in fixed

    def test_matlab_semantic_equal_ignores_comments(self):
        original = "% comment\nx0 = [1; 2];\nresult = cobyqa_mex(fun, x0);\n"
        candidate = "x0 = [1; 2];\nresult = cobyqa_mex(fun, x0);\n"
        assert _matlab_semantic_equal(original, candidate)


# ---------------------------------------------------------------------------
# Python static repairs
# ---------------------------------------------------------------------------

class TestPythonStaticFixes:

    def test_repairs_short_bounds_literals_to_match_x0(self):
        code = """\
def validate_bounds(x0, lb, ub):
    if len(lb) != len(x0) or len(ub) != len(x0):
        raise ValueError("bounds shape mismatch: expected length 2, got 1")


validate_bounds([0.0, 1.0], [0.0], [2.0, 3.0])
"""
        fixed, report = _handle_python_static_fix(
            code,
            "ValueError: bounds shape mismatch: expected length 2, got 1",
        )
        assert fixed is not None
        assert "validate_bounds([0.0, 1.0], [0.0, 0.0], [2.0, 3.0])" in fixed
        assert "Python Error Fixed" in report

    def test_repairs_obvious_timeout_loop(self):
        code = "import time\n\nwhile True:\n    time.sleep(0.1)\n"
        fixed, _report = _handle_python_static_fix(
            code,
            "Script timed out after 30 seconds.",
        )
        assert fixed == 'print("bounded run")\n'

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    @patch("optiprofiler_agent.debugger.debugger.classify_error_with_llm")
    def test_runtime_uses_python_static_fix_before_llm(self, mock_classify, mock_llm):
        mock_classify.return_value = MagicMock(error_type="runtime_error")
        code = """\
def validate_bounds(x0, lb, ub):
    if len(lb) != len(x0) or len(ub) != len(x0):
        raise ValueError("bounds shape mismatch: expected length 2, got 1")


validate_bounds([0.0, 1.0], [0.0], [2.0, 3.0])
"""
        result = debug_script(
            code=code,
            error=(
                "Traceback (most recent call last):\n"
                "ValueError: bounds shape mismatch: expected length 2, got 1\n"
            ),
            config=_make_config(),
        )
        assert result.fixed_code is not None
        assert "[0.0, 0.0]" in result.fixed_code
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Debugger web-search context
# ---------------------------------------------------------------------------

class TestDebuggerWebSearchContext:

    def test_external_dependency_triggers_web_context(self):
        classification = MagicMock(
            error_type="dependency_missing",
            module_name="scipy",
        )
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "1"}, clear=False):
            assert _debug_error_has_external_context(
                classification,
                SCIPY_TRACEBACK,
                "from scipy.optimize import minimize",
                "python",
            )

    def test_internal_optiprofiler_error_does_not_trigger_web_context(self):
        classification = MagicMock(
            error_type="dependency_missing",
            module_name="optiprofiler",
        )
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "1"}, clear=False):
            assert not _debug_error_has_external_context(
                classification,
                "ModuleNotFoundError: No module named 'optiprofiler'",
                "from optiprofiler import benchmark",
                "python",
            )

    def test_env_flag_disables_web_context(self):
        classification = MagicMock(
            error_type="dependency_missing",
            module_name="scipy",
        )
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "0"}, clear=False):
            assert not _debug_error_has_external_context(
                classification,
                "ModuleNotFoundError: No module named 'scipy'",
                "import scipy",
                "python",
            )

    def test_builds_focused_query_from_traceback(self):
        classification = MagicMock(
            error_type="runtime_error",
            module_name=None,
        )
        query = _build_debugger_web_query(
            classification,
            SCIPY_TRACEBACK,
            "from scipy.optimize import minimize",
            "python",
        )
        assert "scipy" in query
        assert "ImportError" in query
        assert "Python traceback fix" in query

    @patch("optiprofiler_agent.debugger.debugger._run_debugger_web_search")
    def test_collect_context_filters_disabled_search(self, mock_search):
        mock_search.return_value = "web_search disabled: set TAVILY_API_KEY"
        classification = MagicMock(
            error_type="dependency_missing",
            module_name="scipy",
        )
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "1"}, clear=False):
            assert _collect_web_debug_context(
                code="import scipy",
                error="ModuleNotFoundError: No module named 'scipy'",
                classification=classification,
                language="python",
            ) is None

    @patch("optiprofiler_agent.debugger.debugger._run_debugger_web_search")
    def test_collect_context_returns_query_and_results(self, mock_search):
        mock_search.return_value = "[1] scipy install issue\nUse scipy>=1.11\nurl: https://example.com"
        classification = MagicMock(
            error_type="dependency_missing",
            module_name="scipy",
        )
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "1"}, clear=False):
            context = _collect_web_debug_context(
                code="import scipy",
                error="ModuleNotFoundError: No module named 'scipy'",
                classification=classification,
                language="python",
            )
        assert context is not None
        assert "scipy" in context[0]
        assert "source=web" in _format_web_debug_context(context)

    @patch("optiprofiler_agent.common.llm_client.create_llm")
    def test_llm_prompt_receives_source_web_context(self, mock_create):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="```python\nimport scipy\nprint('fixed')\n```"
        )
        mock_create.return_value = mock_llm

        from optiprofiler_agent.debugger.debugger import _handle_runtime_with_llm

        fixed, report, attempts = _handle_runtime_with_llm(
            code="import scipy\nprint('broken')\n",
            error="ImportError: cannot import name 'minimize' from scipy.optimize",
            config=_make_config(),
            max_retries=1,
            language="python",
            web_context=(
                "scipy ImportError Python traceback fix",
                "[1] scipy issue\nUse a compatible scipy version.\nurl: https://example.com",
            ),
        )

        assert fixed is not None
        assert attempts == 1
        assert "source=web" in report
        user_msg = mock_llm.invoke.call_args.args[0][1].content
        assert "External Search Context (source=web)" in user_msg
        assert "Use the source=web context only as supporting external context" in user_msg


# ---------------------------------------------------------------------------
# debug_script
# ---------------------------------------------------------------------------

class TestDebugScript:

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    @patch("optiprofiler_agent.debugger.debugger._run_debugger_web_search")
    def test_dependency_missing_no_llm_needed(self, mock_search, mock_llm):
        mock_search.return_value = "web_search disabled: set TAVILY_API_KEY"
        mock_llm.return_value = (None, "attempted", 1)
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "1"}, clear=False):
            result = debug_script(
                code="import nonexistent_module",
                error=IMPORT_ERROR_TRACEBACK,
                config=_make_config(),
            )
        assert isinstance(result, DebugResult)
        assert result.classification.error_type == "dependency_missing"
        assert result.classification.module_name == "nonexistent_module"
        assert "pip install" in result.diagnostic_report

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    @patch("optiprofiler_agent.debugger.debugger._run_debugger_web_search")
    def test_dependency_missing_appends_web_context_to_fallback_report(self, mock_search, mock_llm):
        mock_search.return_value = "[1] scipy install\nInstall scipy wheels.\nurl: https://example.com"
        mock_llm.return_value = (None, "attempted", 1)
        with patch.dict(os.environ, {"OPAGENT_DEBUGGER_WEB_SEARCH": "1"}, clear=False):
            result = debug_script(
                code="import scipy",
                error="ModuleNotFoundError: No module named 'scipy'",
                config=_make_config(),
            )
        assert result.fixed_code is None
        assert "pip install scipy" in result.diagnostic_report
        assert "source=web" in result.diagnostic_report
        assert "url: https://example.com" in result.diagnostic_report
        assert "scipy" in mock_search.call_args.args[0]

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_name_error_classified_as_runtime(self, mock_llm):
        mock_llm.return_value = (None, "attempted", 1)
        result = debug_script(
            code=GOOD_CODE,
            error=NAME_ERROR_TRACEBACK,
            config=_make_config(),
        )
        assert result.classification.error_type == "runtime_error"

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_timeout_error_classified(self, mock_llm):
        mock_llm.return_value = (None, "attempted", 1)
        result = debug_script(
            code=GOOD_CODE,
            error="TimeoutError: execution timed out after 120s",
            config=_make_config(),
        )
        assert result.classification.error_type == "timeout"
        assert "time limit" in result.diagnostic_report.lower() or "timeout" in result.diagnostic_report.lower()

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_numerical_error_classified(self, mock_llm):
        mock_llm.return_value = (None, "attempted", 1)
        result = debug_script(
            code=GOOD_CODE,
            error="RuntimeWarning: overflow encountered in double_scalars",
            config=_make_config(),
        )
        assert result.classification.error_type == "numerical"

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_matlab_dependency_missing(self, mock_llm):
        mock_llm.return_value = (None, "attempted", 1)
        result = debug_script(
            code="x = foo(1);",
            error="Undefined function or variable 'foo'.",
            config=_make_config(),
            language="matlab",
        )
        assert result.classification.error_type == "dependency_missing"
        assert "addpath" in result.diagnostic_report.lower()

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_matlab_interface_mismatch(self, mock_llm):
        mock_llm.return_value = (None, "attempted", 1)
        result = debug_script(
            code="function x = s(a, b, c)\nend\n",
            error="Error using s\nToo many input arguments.",
            config=_make_config(),
            language="matlab",
        )
        assert result.classification.error_type == "interface_mismatch"

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    @patch("optiprofiler_agent.debugger.debugger.classify_error_with_llm")
    def test_runtime_error_calls_llm_handler(self, mock_classify, mock_llm):
        mock_classify.return_value = MagicMock(error_type="runtime_error")
        mock_llm.return_value = (GOOD_CODE, "Fixed it", 1)
        result = debug_script(
            code=GOOD_CODE,
            error="ValueError: something went wrong",
            config=_make_config(),
        )
        assert result.classification.error_type == "runtime_error"

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_specialized_diagnostic_categories_try_llm_fix_first(self, mock_llm):
        mock_llm.return_value = ("print('fixed')", "Fixed it", 1)
        result = debug_script(
            code="import nonexistent_module",
            error=IMPORT_ERROR_TRACEBACK,
            config=_make_config(),
        )
        assert result.classification.error_type == "dependency_missing"
        assert result.fixed_code == "print('fixed')"
        mock_llm.assert_called_once()

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_dependency_missing_keeps_diagnostic_when_llm_has_no_fix(self, mock_llm):
        mock_llm.return_value = (None, "attempted", 2)
        result = debug_script(
            code="import nonexistent_module",
            error=IMPORT_ERROR_TRACEBACK,
            config=_make_config(),
        )
        assert result.classification.error_type == "dependency_missing"
        assert result.fixed_code is None
        assert "pip install" in result.diagnostic_report

    @patch("optiprofiler_agent.debugger.debugger._handle_runtime_with_llm")
    def test_interface_fallback_tries_llm_when_adapter_has_no_wrapper(self, mock_llm):
        mock_llm.return_value = ("def solver(fun, x0, **kwargs):\n    return x0", "Fixed it", 1)
        result = debug_script(
            code="def solver(fun, x0):\n    return x0\n",
            error="TypeError: solver() got an unexpected keyword argument 'max_eval'",
            config=_make_config(),
        )
        assert result.classification.error_type == "interface_mismatch"
        assert result.fixed_code is not None
        mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# run_and_debug
# ---------------------------------------------------------------------------

class TestRunAndDebug:

    @patch("optiprofiler_agent.debugger.local_runner.run_script")
    def test_success_on_first_run(self, mock_run):
        mock_run.return_value = MagicMock(
            success=True, stdout="OK", stderr="", traceback=None, timed_out=False,
        )
        result = run_and_debug(
            code=GOOD_CODE,
            config=_make_config(),
        )
        assert result.classification.error_type == "none"
        assert result.attempts == 1
        assert result.validation_passed is True

    @patch("optiprofiler_agent.debugger.debugger.debug_script")
    @patch("optiprofiler_agent.debugger.local_runner.run_script")
    def test_fix_and_rerun_success(self, mock_run, mock_debug):
        fail_result = MagicMock(
            success=False, stdout="", stderr="NameError: x",
            traceback="NameError: x", timed_out=False,
        )
        success_result = MagicMock(
            success=True, stdout="OK", stderr="", traceback=None, timed_out=False,
        )
        mock_run.side_effect = [fail_result, success_result]

        mock_debug.return_value = DebugResult(
            classification=MagicMock(error_type="runtime_error"),
            fixed_code="x = 1\nprint(x)",
            diagnostic_report="Fixed NameError",
            attempts=1,
            validation_passed=True,
        )

        result = run_and_debug(code="print(x)", config=_make_config())
        assert result.classification.error_type == "none"
        assert result.attempts == 2

    @patch("optiprofiler_agent.debugger.debugger.debug_script")
    @patch("optiprofiler_agent.debugger.local_runner.run_script")
    def test_no_fix_available_stops(self, mock_run, mock_debug):
        fail_result = MagicMock(
            success=False, stdout="", stderr="Error",
            traceback="Error", timed_out=False,
        )
        mock_run.return_value = fail_result

        mock_debug.return_value = DebugResult(
            classification=MagicMock(error_type="runtime_error"),
            fixed_code=None,
            diagnostic_report="Could not fix",
            attempts=1,
            validation_passed=False,
        )

        result = run_and_debug(code="bad code", config=_make_config())
        assert result.validation_passed is False

    @patch("optiprofiler_agent.debugger.debugger.debug_script")
    @patch("optiprofiler_agent.debugger.local_runner.run_script")
    def test_timeout_on_first_run(self, mock_run, mock_debug):
        mock_run.return_value = MagicMock(
            success=False, stdout="", stderr="",
            traceback="TimeoutError", timed_out=True,
        )
        mock_debug.return_value = DebugResult(
            classification=MagicMock(error_type="timeout"),
            fixed_code=None,
            diagnostic_report="Timed out",
            attempts=1,
            validation_passed=False,
        )
        result = run_and_debug(code=GOOD_CODE, config=_make_config())
        assert result.classification.error_type in ("timeout", "runtime_error")

    @patch("optiprofiler_agent.debugger.local_runner.run_script")
    def test_progress_callback_called(self, mock_run):
        mock_run.return_value = MagicMock(
            success=True, stdout="OK", stderr="", traceback=None, timed_out=False,
        )
        messages = []
        run_and_debug(
            code=GOOD_CODE,
            config=_make_config(),
            progress_callback=lambda msg: messages.append(msg),
        )
        assert len(messages) >= 1
        assert any("Round" in m for m in messages)


# ---------------------------------------------------------------------------
# MATLAB run_and_debug control flow (mirrors TestRunAndDebug above).
# The local runner today is Python-only (B-10 sandbox is a platform task);
# these mocked tests just validate that the language parameter is threaded
# correctly through the round loop, save_fixed, and progress reporting.
# ---------------------------------------------------------------------------

MATLAB_GOOD = (
    "function x = solver(fun, x0)\n"
    "    x = fminsearch(fun, x0);\n"
    "    x = x(:);\n"
    "end\n"
)

MATLAB_RUNTIME_ERROR = (
    "Error using solver\n"
    "Index exceeds the number of array elements (2)."
)


class TestRunAndDebugMatlab:
    """End-to-end MATLAB control-flow with the runner patched at the dispatch layer.

    We patch ``debugger._run_code_for_language`` so the test works whether
    or not a real MATLAB binary is installed and so it parallels the
    Python mocks above (which patch ``local_runner.run_script``).
    """

    @patch("optiprofiler_agent.debugger.debugger._run_code_for_language")
    def test_matlab_success_on_first_run(self, mock_run):
        mock_run.return_value = MagicMock(
            success=True, stdout="OK", stderr="", traceback=None, timed_out=False,
        )
        result = run_and_debug(
            code=MATLAB_GOOD,
            config=_make_config(),
            language="matlab",
        )
        assert result.classification.error_type == "none"
        assert result.attempts == 1
        assert result.validation_passed is True
        # Confirm the dispatcher saw ``language="matlab"``.
        assert mock_run.call_args.kwargs.get("language") == "matlab"

    @patch("optiprofiler_agent.debugger.debugger.debug_script")
    @patch("optiprofiler_agent.debugger.debugger._run_code_for_language")
    def test_matlab_fix_and_rerun_success(self, mock_run, mock_debug):
        fail_result = MagicMock(
            success=False, stdout="", stderr=MATLAB_RUNTIME_ERROR,
            traceback=MATLAB_RUNTIME_ERROR, timed_out=False,
        )
        success_result = MagicMock(
            success=True, stdout="OK", stderr="", traceback=None, timed_out=False,
        )
        mock_run.side_effect = [fail_result, success_result]

        mock_debug.return_value = DebugResult(
            classification=MagicMock(error_type="runtime_error"),
            fixed_code=MATLAB_GOOD,
            diagnostic_report="Fixed index error",
            attempts=1,
            validation_passed=True,
        )

        result = run_and_debug(
            code="x = a(3);",
            config=_make_config(),
            language="matlab",
        )
        assert result.classification.error_type == "none"
        assert result.attempts == 2
        # The debug_script call must be told it's MATLAB.
        kwargs = mock_debug.call_args.kwargs
        assert kwargs.get("language") == "matlab"

    @patch("optiprofiler_agent.debugger.debugger.debug_script")
    @patch("optiprofiler_agent.debugger.debugger._run_code_for_language")
    def test_matlab_save_fixed_writes_file(self, mock_run, mock_debug, tmp_path):
        fail_result = MagicMock(
            success=False, stdout="", stderr=MATLAB_RUNTIME_ERROR,
            traceback=MATLAB_RUNTIME_ERROR, timed_out=False,
        )
        success_result = MagicMock(
            success=True, stdout="OK", stderr="", traceback=None, timed_out=False,
        )
        mock_run.side_effect = [fail_result, success_result]

        mock_debug.return_value = DebugResult(
            classification=MagicMock(error_type="runtime_error"),
            fixed_code=MATLAB_GOOD,
            diagnostic_report="Fixed",
            attempts=1,
            validation_passed=True,
        )

        out = tmp_path / "fixed_solver.m"
        run_and_debug(
            code="x = a(3);",
            config=_make_config(),
            language="matlab",
            save_fixed=str(out),
        )
        assert out.exists()
        assert "function x = solver" in out.read_text(encoding="utf-8")

    @patch("optiprofiler_agent.debugger.debugger._run_code_for_language")
    def test_matlab_progress_callback_reports_rounds(self, mock_run):
        mock_run.return_value = MagicMock(
            success=True, stdout="OK", stderr="", traceback=None, timed_out=False,
        )
        messages: list[str] = []
        run_and_debug(
            code=MATLAB_GOOD,
            config=_make_config(),
            language="matlab",
            progress_callback=lambda m: messages.append(m),
        )
        assert any("Round" in m for m in messages)
