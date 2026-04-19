"""L2 hallucination guard — post-generation lint loop for the agent CLI.

Pipeline (Cursor / Claude Code pattern):

1. Extract Python code blocks from the agent's reply.
2. Run the configured ``CodeConstraintBackend`` over each block.
3. If any *errors* are found, synthesise a ``ToolMessage`` describing them
   and re-invoke the agent **once** so it self-corrects within the same
   conversational turn.
4. Re-validate. Whatever errors remain are surfaced to the user as a
   transparent yellow warning (we do not silently delete bad code — that
   would break trust).

Design notes
------------
* The backend is pluggable through :class:`~optiprofiler_agent.validators.api_checker.CodeConstraintBackend`.
  Today only ``ASTValidatorBackend`` is wired in; a future L4 backend
  using vLLM grammar-constrained decoding can replace it with no other
  changes.
* The retry budget defaults to **1**: matches Cursor (≈95% repair rate)
  and keeps tail latency bounded. Tune via :data:`MAX_VALIDATION_RETRIES`.
* We only inject feedback for **errors** (e.g. typo of `optiprofiler`,
  invalid syntax). Warnings (unknown kwarg, fake submodule path) are
  shown to the user but never trigger a retry — feeding them back tends
  to make the LLM over-correct on legitimate kwargs it actually wants.
"""

from __future__ import annotations

from dataclasses import dataclass

from optiprofiler_agent.validators.api_checker import (
    ASTValidatorBackend,
    CodeConstraintBackend,
    ValidationIssue,
)
from optiprofiler_agent.validators.syntax_checker import extract_code_blocks

MAX_VALIDATION_RETRIES = 1


@dataclass
class LintReport:
    """Outcome of one full lint pass over an agent reply."""

    issues: list[ValidationIssue]
    code_blocks_checked: int

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)


def lint_reply(
    reply: str,
    backend: CodeConstraintBackend | None = None,
    *,
    language: str = "python",
) -> LintReport:
    """Run the configured backend over every code block in ``reply``."""
    backend = backend or ASTValidatorBackend()
    blocks = extract_code_blocks(reply)
    issues: list[ValidationIssue] = []
    for block in blocks:
        result = backend.validate(block, language=language)
        issues.extend(result.issues)
    return LintReport(issues=issues, code_blocks_checked=len(blocks))


def format_feedback_for_llm(report: LintReport) -> str:
    """Render an error report into a tool-message body the LLM can act on.

    Returns the empty string if there are no errors (warnings alone do not
    trigger a retry — see module docstring).
    """
    errors = [i for i in report.issues if i.severity == "error"]
    if not errors:
        return ""
    lines = [
        "Validator found errors in your last reply's Python code blocks.",
        "Please rewrite the reply with the same intent, fixing every "
        "issue below. Do not invent symbols; if you are unsure, use "
        "`knowledge_search('Python imports and exports public API')` first.",
        "",
    ]
    for issue in errors:
        loc = f" (line {issue.line})" if issue.line else ""
        lines.append(f"- ERROR{loc}: {issue.message}")
    return "\n".join(lines)


def format_for_user(report: LintReport) -> list[str]:
    """Render remaining issues as one-liners suitable for a Rich panel."""
    out = []
    for issue in report.issues:
        loc = f" (line {issue.line})" if issue.line else ""
        out.append(f"[{issue.severity}{loc}] {issue.message}")
    return out
