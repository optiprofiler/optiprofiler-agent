"""Tests for the L2 lint loop (post-generation hallucination guard)."""

from __future__ import annotations

from optiprofiler_agent.validators import lint_loop
from optiprofiler_agent.validators.api_checker import (
    NullBackend,
    ValidationIssue,
    ValidationResult,
)


class TestLintReply:

    def test_clean_reply_has_no_issues(self):
        reply = (
            "Here is a working example:\n\n"
            "```python\n"
            "from optiprofiler import benchmark\n"
            "benchmark([s1, s2])\n"
            "```\n"
        )
        report = lint_loop.lint_reply(reply)
        assert not report.has_errors
        assert report.code_blocks_checked == 1

    def test_optiprobe_typo_is_error(self):
        reply = "```python\nfrom optiprobe import benchmark\nbenchmark([s1, s2])\n```"
        report = lint_loop.lint_reply(reply)
        assert report.has_errors

    def test_fake_submodule_is_warning_not_error(self):
        reply = "```python\nfrom optiprofiler.solvers import bobyqa\nbenchmark([s1, s2])\n```"
        report = lint_loop.lint_reply(reply)
        assert not report.has_errors
        assert report.has_warnings

    def test_no_code_blocks_returns_empty(self):
        report = lint_loop.lint_reply("Just prose, no code.")
        assert report.code_blocks_checked == 0
        assert report.issues == []

    def test_null_backend_disables_checks(self):
        reply = "```python\nfrom optiprobe import benchmark\n```"
        report = lint_loop.lint_reply(reply, backend=NullBackend())
        assert report.issues == []


class TestFormatFeedbackForLlm:

    def test_errors_produce_actionable_feedback(self):
        report = lint_loop.LintReport(
            issues=[
                ValidationIssue("error", "package typo", line=1),
                ValidationIssue("warning", "unknown kwarg foo", line=3),
            ],
            code_blocks_checked=1,
        )
        feedback = lint_loop.format_feedback_for_llm(report)
        assert "package typo" in feedback
        assert "unknown kwarg foo" not in feedback
        assert "knowledge_search" in feedback

    def test_warnings_only_produces_no_feedback(self):
        report = lint_loop.LintReport(
            issues=[ValidationIssue("warning", "unknown kwarg", line=1)],
            code_blocks_checked=1,
        )
        assert lint_loop.format_feedback_for_llm(report) == ""

    def test_empty_report_produces_no_feedback(self):
        report = lint_loop.LintReport(issues=[], code_blocks_checked=0)
        assert lint_loop.format_feedback_for_llm(report) == ""


class TestFormatForUser:

    def test_renders_severity_and_line(self):
        report = lint_loop.LintReport(
            issues=[
                ValidationIssue("error", "bad import", line=4),
                ValidationIssue("warning", "weird kwarg", line=None),
            ],
            code_blocks_checked=1,
        )
        lines = lint_loop.format_for_user(report)
        assert any("[error" in line and "line 4" in line for line in lines)
        assert any("[warning" in line and "weird kwarg" in line for line in lines)


class TestPluggableBackend:
    """Smoke test that the loop honours an arbitrary backend (forward-compat
    for the L4 grammar backend)."""

    def test_custom_backend_is_called(self):
        captured = []

        class _StubBackend:
            name = "stub"

            def validate(self, code: str, *, language: str = "python") -> ValidationResult:
                captured.append((code, language))
                return ValidationResult()

        lint_loop.lint_reply("```python\nfoo()\n```", backend=_StubBackend())
        assert captured and captured[0][1] == "python"
