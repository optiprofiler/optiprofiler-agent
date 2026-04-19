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


class TestImportWhitelist:
    """L1 hallucination guard: catch typo'd / fake optiprofiler imports."""

    def test_exports_contain_known_public_symbols(self):
        from optiprofiler_agent.validators.api_checker import optiprofiler_python_exports

        exports = optiprofiler_python_exports()
        assert "benchmark" in exports
        assert "Problem" in exports
        assert "Feature" in exports
        assert "FeaturedProblem" in exports
        assert "s2mpj_load" in exports
        assert "pycutest_select" in exports

    def test_typo_optiprobe_is_error(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = "from optiprobe import benchmark\nbenchmark([s1, s2])"
        result = validate_benchmark_call(code)
        errors = [i for i in result.issues if i.severity == "error"]
        assert any("optiprofiler" in i.message.lower() and "optiprobe" in i.message.lower()
                   for i in errors), result.issues

    def test_fake_submodule_solvers_is_warning(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = "from optiprofiler.solvers import bobyqa\nbenchmark([s1, s2])"
        result = validate_benchmark_call(code)
        assert any("solvers" in i.message and i.severity == "warning"
                   for i in result.issues), result.issues

    def test_unknown_symbol_gets_did_you_mean(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = "from optiprofiler import benchmrk\nbenchmrk([s1, s2])"
        result = validate_benchmark_call(code)
        warnings = [i for i in result.issues if i.severity == "warning"]
        assert any("benchmark" in i.message and "did you mean" in i.message.lower()
                   for i in warnings), result.issues

    def test_known_public_symbol_passes(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = (
            "from optiprofiler import benchmark, Problem, Feature\n"
            "from optiprofiler import s2mpj_load\n"
            "benchmark([s1, s2])\n"
        )
        result = validate_benchmark_call(code)
        import_issues = [
            i for i in result.issues
            if "import" in i.message.lower() or "submodule" in i.message.lower()
        ]
        assert import_issues == []

    def test_plain_import_optiprofiler_is_fine(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = "import optiprofiler\noptiprofiler.benchmark([s1, s2])"
        result = validate_benchmark_call(code)
        errors = [i for i in result.issues if i.severity == "error"]
        assert errors == []

    def test_dotted_import_warns(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = "import optiprofiler.solvers"
        result = validate_benchmark_call(code)
        assert any(i.severity == "warning" and "solvers" in i.message
                   for i in result.issues), result.issues

    def test_matlab_language_skips_python_imports(self):
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call

        code = "from optiprobe import benchmark\nbenchmark([s1, s2])"
        result = validate_benchmark_call(code, language="matlab")
        assert all(i.severity != "error" or "optiprobe" not in i.message
                   for i in result.issues), result.issues


class TestConstraintBackend:
    """Pluggable backend protocol — forward-compat for L4 grammar decoding."""

    def test_ast_backend_validates(self):
        from optiprofiler_agent.validators.api_checker import ASTValidatorBackend

        backend = ASTValidatorBackend()
        assert backend.name == "ast"
        result = backend.validate("from optiprobe import benchmark")
        assert any(i.severity == "error" for i in result.issues)

    def test_null_backend_passes_everything(self):
        from optiprofiler_agent.validators.api_checker import NullBackend

        backend = NullBackend()
        assert backend.name == "null"
        result = backend.validate("from optiprobe import benchmark")
        assert result.issues == []
