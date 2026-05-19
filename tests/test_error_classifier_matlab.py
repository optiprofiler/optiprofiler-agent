"""MATLAB-side error classifier tests.

Mirrors ``tests/test_error_classifier.py`` so that every Python error
category exercised on the Python side has at least one MATLAB
counterpart, plus MATLAB-specific patterns that don't exist on Python.
"""

from optiprofiler_agent.debugger.error_classifier import (
    classify_error,
    classify_error_with_llm,
)


class TestMatlabClassifierAllCategories:

    # --- interface_mismatch (parity with Python TypeError) -----------------

    def test_too_many_input_arguments(self):
        tb = "Error using my_solver\nToo many input arguments."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "interface_mismatch"
        assert result.confidence >= 0.9

    def test_not_enough_input_arguments(self):
        tb = "Error using my_solver\nNot enough input arguments."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "interface_mismatch"

    def test_bare_not_enough_input_arguments(self):
        tb = (
            "Not enough input arguments.\n\n"
            "Error in opagent_script>my_solver (line 9)\n"
            "    x = options.scale * fminsearch(fun, x0);"
        )
        result = classify_error(tb, language="matlab")
        assert result.error_type == "interface_mismatch"

    # --- dependency_missing (parity with Python ModuleNotFoundError) -------

    def test_undefined_function_or_variable(self):
        tb = "Undefined function or variable 'cobyqa'."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "dependency_missing"
        assert result.module_name == "cobyqa"

    def test_unrecognized_function_or_variable(self):
        tb = "Unrecognized function or variable 'pdfo'."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "dependency_missing"
        assert result.module_name == "pdfo"

    # --- runtime_error (parity with Python NameError / IndexError) ---------

    def test_index_exceeds(self):
        tb = "Index exceeds the number of array elements (2)."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "runtime_error"

    def test_dimension_mismatch(self):
        tb = "Dimensions of arrays being concatenated are not consistent."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "runtime_error"

    def test_error_using_generic(self):
        tb = "Error using fminsearch\nObjective function returned NaN."
        result = classify_error(tb, language="matlab")
        # MATLAB "Error using" without a specific pattern routes either to
        # interface_mismatch (if numeric pattern matches) or runtime_error;
        # the numerical word "NaN" wins here. Both are acceptable.
        assert result.error_type in ("runtime_error", "numerical")

    # --- timeout (parity with Python TimeoutError) -------------------------

    def test_timeout(self):
        tb = "Script timed out after 120 seconds."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "timeout"

    # --- numerical (parity with Python overflow/NaN) -----------------------

    def test_numerical_nan(self):
        tb = "Warning: NaN detected in solver output."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "numerical"

    def test_numerical_inf(self):
        tb = "Warning: Inf returned by objective function."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "numerical"

    def test_file_path_with_inf_substring_is_not_numerical(self):
        tb = (
            "File: /tmp/opagent_script.m\n"
            "Invalid expression. When calling a function or indexing a variable, "
            "use parentheses."
        )
        result = classify_error(tb, language="matlab")
        assert result.error_type == "runtime_error"

    def test_numerical_overflow(self):
        tb = "Overflow encountered in step computation."
        result = classify_error(tb, language="matlab")
        assert result.error_type == "numerical"


class TestMatlabClassifierFallback:
    """When no pattern matches, fall through to ``runtime_error``."""

    def test_unknown_text_defaults_to_runtime(self):
        result = classify_error("Some completely opaque message", language="matlab")
        assert result.error_type == "runtime_error"

    def test_llm_path_without_config_returns_regex(self):
        # No config => LLM is not consulted; should yield regex result.
        result = classify_error_with_llm(
            "Undefined function or variable 'foo'.",
            language="matlab",
        )
        assert result.error_type == "dependency_missing"
