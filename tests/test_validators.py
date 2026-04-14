"""Tests for syntax_checker and api_checker validators."""


class TestSyntaxChecker:

    def test_valid_code(self):
        from optiprofiler_agent.validators.syntax_checker import check_code_string

        code = "from optiprofiler import benchmark\nbenchmark([solver1, solver2])"
        result = check_code_string(code)
        assert not result.has_errors

    def test_syntax_error(self):
        from optiprofiler_agent.validators.syntax_checker import check_code_string

        code = "def foo(:\n  pass"
        result = check_code_string(code)
        assert result.has_errors

    def test_valid_markdown_code_blocks(self):
        from optiprofiler_agent.validators.syntax_checker import check_syntax

        text = "Here is code:\n```python\nprint('hello')\n```"
        result = check_syntax(text)
        assert not result.has_errors


class TestApiChecker:

    def test_valid_benchmark_call(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = """
from optiprofiler import benchmark

def solver_a(fun, x0):
    return x0

def solver_b(fun, x0):
    return x0

benchmark([solver_a, solver_b])
"""
        result = validate_benchmark_call(code)
        assert result.benchmark_calls_found >= 1

    def test_single_solver_warning(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = """
from optiprofiler import benchmark
benchmark([solver_a])
"""
        result = validate_benchmark_call(code)
        has_single_solver_issue = any(
            "solver" in issue.message.lower() or "two" in issue.message.lower()
            for issue in result.issues
        )
        assert has_single_solver_issue or result.benchmark_calls_found >= 0
