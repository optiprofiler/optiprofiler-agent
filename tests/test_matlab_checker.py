"""Tests for ``validators/matlab_checker.py``.

Mirror the breadth of ``test_validators.py`` on the Python side so that
MATLAB safety/structure validation is held to the same bar.
"""

from optiprofiler_agent.validators.matlab_checker import check_matlab_code


class TestMatlabCheckerValid:
    """Code that should pass the checker."""

    def test_simple_solver_passes(self):
        code = (
            "function x = solver(fun, x0)\n"
            "    x = fminsearch(fun, x0);\n"
            "    x = x(:);\n"
            "end\n"
        )
        result = check_matlab_code(code)
        assert not result.has_errors

    def test_nested_brackets_balanced(self):
        code = (
            "function x = s(fun, x0)\n"
            "    A = [1, 2; 3, 4];\n"
            "    b = {{'a','b'}, {'c'}};\n"
            "    x = fun(A * x0);\n"
            "end\n"
        )
        result = check_matlab_code(code)
        assert not result.has_errors

    def test_comments_with_dangerous_words_pass(self):
        code = (
            "% This solver does NOT call system or eval\n"
            "function x = s(fun, x0)\n"
            "    x = fminsearch(fun, x0);\n"
            "end\n"
        )
        result = check_matlab_code(code)
        assert not result.has_errors


class TestMatlabCheckerDangerous:
    """Dangerous calls must be flagged with a line number."""

    def test_detects_system_call(self):
        code = "function x = solver(fun, x0)\n    system('rm -rf /');\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("system" in e for e in result.errors)
        assert any("Line 2" in e for e in result.errors)

    def test_detects_unix_call(self):
        code = "function x = s(fun, x0)\n    unix('whoami');\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("unix" in e for e in result.errors)

    def test_detects_dos_call(self):
        code = "function x = s(fun, x0)\n    dos('dir');\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("dos" in e for e in result.errors)

    def test_detects_eval(self):
        code = "function x = s(fun, x0)\n    eval('x = 1 + 1');\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("eval" in e for e in result.errors)

    def test_detects_urlread(self):
        code = "function x = s(fun, x0)\n    data = urlread('http://x');\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("urlread" in e for e in result.errors)

    def test_detects_java_lang(self):
        code = "function x = s(fun, x0)\n    java.lang.Runtime.getRuntime();\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("java" in e.lower() for e in result.errors)

    def test_detects_py_importlib(self):
        code = (
            "function x = s(fun, x0)\n"
            "    m = py.importlib.import_module('os');\n"
            "    x = x0;\nend\n"
        )
        result = check_matlab_code(code)
        assert result.has_errors


class TestMatlabCheckerShellEscape:
    """Bang-prefixed lines run shell commands and must be rejected."""

    def test_detects_shell_escape_at_start(self):
        code = "!rm -rf /\nfunction x = solver(fun, x0)\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("shell escape" in e for e in result.errors)

    def test_detects_shell_escape_with_leading_whitespace(self):
        code = "function x = s(fun, x0)\n  !echo hi\n    x = x0;\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("shell escape" in e for e in result.errors)


class TestMatlabCheckerStructure:
    """Structural validation: balanced parens/brackets/braces."""

    def test_detects_unbalanced_parens(self):
        code = "function x = solver(fun, x0\n    x = fminsearch(fun, x0;\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("Unbalanced" in e for e in result.errors)

    def test_detects_unbalanced_braces(self):
        code = "function x = s(fun, x0)\n    c = {1, 2;\n    x = x0;\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("Unbalanced" in e for e in result.errors)

    def test_detects_unbalanced_brackets(self):
        code = "function x = s(fun, x0)\n    A = [1 2;\n    x = x0;\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("Unbalanced" in e for e in result.errors)

    def test_accepts_script_with_local_functions_at_end(self):
        code = (
            "options.ptype = 'u';\n"
            "scores = benchmark({@s1, @s2}, options);\n"
            "\n"
            "function x = s1(fun, x0)\n"
            "    x = fminsearch(fun, x0);\n"
            "end\n"
            "\n"
            "function x = s2(fun, x0)\n"
            "    x = fminsearch(fun, x0);\n"
            "end\n"
        )
        result = check_matlab_code(code)
        assert not result.has_errors

    def test_rejects_script_statements_after_leading_function(self):
        code = (
            "function x = s1(fun, x0)\n"
            "    x = fminsearch(fun, x0);\n"
            "end\n"
            "\n"
            "options.ptype = 'u';\n"
            "scores = benchmark({@s1, @s2}, options);\n"
        )
        result = check_matlab_code(code)
        assert result.has_errors
        assert any("script statements" in e for e in result.errors)


class TestMatlabCheckerLineNumbers:
    """Diagnostics must point at the right line."""

    def test_error_includes_line_number(self):
        code = "function x = s(fun, x0)\n    x = x0;\n    eval('x = 1');\nend\n"
        result = check_matlab_code(code)
        assert result.has_errors
        # eval is on line 3
        assert any("Line 3" in e for e in result.errors)
