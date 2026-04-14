"""Tests for Agent B's error classifier — all 5 error types."""

from optiprofiler_agent.agent_b.error_classifier import classify_error


class TestErrorClassifier:

    def test_interface_mismatch(self):
        tb = (
            "Traceback (most recent call last):\n"
            "  File 'test.py', line 10\n"
            "TypeError: my_solver() takes 1 positional argument but 2 were given"
        )
        result = classify_error(tb)
        assert result.error_type == "interface_mismatch"
        assert result.confidence > 0.5

    def test_dependency_missing(self):
        tb = (
            "Traceback (most recent call last):\n"
            "  File 'test.py', line 1\n"
            "ModuleNotFoundError: No module named 'pdfo'"
        )
        result = classify_error(tb)
        assert result.error_type == "dependency_missing"
        assert result.module_name == "pdfo"

    def test_timeout(self):
        tb = "Script timed out after 120 seconds."
        result = classify_error(tb)
        assert result.error_type == "timeout"

    def test_numerical(self):
        tb = (
            "RuntimeWarning: overflow encountered in scalar multiply\n"
            "ValueError: array must not contain infs or NaNs"
        )
        result = classify_error(tb)
        assert result.error_type == "numerical"

    def test_runtime_error(self):
        tb = (
            "Traceback (most recent call last):\n"
            "  File 'test.py', line 5\n"
            "RuntimeError: something unexpected happened"
        )
        result = classify_error(tb)
        assert result.error_type == "runtime_error"

    def test_value_error_at_least_two_solvers(self):
        tb = (
            "Traceback (most recent call last):\n"
            "  File 'test.py', line 10\n"
            "ValueError: At least two solvers must be given."
        )
        result = classify_error(tb)
        assert result.error_type in ("interface_mismatch", "runtime_error")
