"""Tests for Agent B — debugger module.

All tests use mocked LLM calls; no real API key needed.
"""

from unittest.mock import MagicMock, patch

from optiprofiler_agent.agent_b.debugger import (
    DebugResult,
    _extract_code_from_reply,
    _validate_code,
    debug_script,
    run_and_debug,
)
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


# ---------------------------------------------------------------------------
# debug_script
# ---------------------------------------------------------------------------

class TestDebugScript:

    def test_dependency_missing_no_llm_needed(self):
        result = debug_script(
            code="import nonexistent_module",
            error=IMPORT_ERROR_TRACEBACK,
            config=_make_config(),
        )
        assert isinstance(result, DebugResult)
        assert result.classification.error_type == "dependency_missing"
        assert result.classification.module_name == "nonexistent_module"
        assert "pip install" in result.diagnostic_report

    def test_name_error_classified_as_runtime(self):
        result = debug_script(
            code=GOOD_CODE,
            error=NAME_ERROR_TRACEBACK,
            config=_make_config(),
        )
        assert result.classification.error_type == "runtime_error"

    def test_timeout_error_classified(self):
        result = debug_script(
            code=GOOD_CODE,
            error="TimeoutError: execution timed out after 120s",
            config=_make_config(),
        )
        assert result.classification.error_type == "timeout"
        assert "time limit" in result.diagnostic_report.lower() or "timeout" in result.diagnostic_report.lower()

    def test_numerical_error_classified(self):
        result = debug_script(
            code=GOOD_CODE,
            error="RuntimeWarning: overflow encountered in double_scalars",
            config=_make_config(),
        )
        assert result.classification.error_type == "numerical"

    @patch("optiprofiler_agent.agent_b.debugger._handle_runtime_with_llm")
    def test_runtime_error_calls_llm_handler(self, mock_llm):
        mock_llm.return_value = (GOOD_CODE, "Fixed it", 1)
        result = debug_script(
            code=GOOD_CODE,
            error="ValueError: something went wrong",
            config=_make_config(),
        )
        assert result.classification.error_type == "runtime_error"


# ---------------------------------------------------------------------------
# run_and_debug
# ---------------------------------------------------------------------------

class TestRunAndDebug:

    @patch("optiprofiler_agent.agent_b.local_runner.run_script")
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

    @patch("optiprofiler_agent.agent_b.debugger.debug_script")
    @patch("optiprofiler_agent.agent_b.local_runner.run_script")
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

    @patch("optiprofiler_agent.agent_b.debugger.debug_script")
    @patch("optiprofiler_agent.agent_b.local_runner.run_script")
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

    @patch("optiprofiler_agent.agent_b.local_runner.run_script")
    def test_timeout_on_first_run(self, mock_run):
        mock_run.return_value = MagicMock(
            success=False, stdout="", stderr="",
            traceback="TimeoutError", timed_out=True,
        )
        result = run_and_debug(code=GOOD_CODE, config=_make_config())
        assert result.classification.error_type in ("timeout", "runtime_error")

    @patch("optiprofiler_agent.agent_b.local_runner.run_script")
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
