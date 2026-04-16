"""AST-based syntax validation for LLM-generated Python code.

Extracts code blocks from LLM responses (markdown-fenced or bare),
parses them with ``ast.parse``, and reports syntax errors with
line numbers and suggestions.

Usage::

    from optiprofiler_agent.validators.syntax_checker import check_syntax
    result = check_syntax(llm_response_text)
    if result.has_errors:
        for err in result.errors:
            print(err)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

_PYTHON_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL,
)

_UNTAGGED_BLOCK_RE = re.compile(
    r"```\s*\n(.*?)```",
    re.DOTALL,
)

_NON_PYTHON_BLOCK_RE = re.compile(
    r"```(?:matlab|m|bash|shell|sh|json|text|txt|r)\s*\n",
    re.IGNORECASE,
)

_BARE_CODE_HEURISTIC = re.compile(
    r"^(?:import |from |def |class |benchmark\(|scores\s*=)",
    re.MULTILINE,
)


@dataclass
class SyntaxError_:
    """A single syntax error found in a code block."""

    block_index: int
    line: int
    col: int
    message: str
    code_snippet: str


@dataclass
class SyntaxCheckResult:
    """Result of syntax-checking all code blocks in a response."""

    blocks_found: int = 0
    blocks_valid: int = 0
    errors: list[SyntaxError_] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def all_valid(self) -> bool:
        return self.blocks_found > 0 and not self.has_errors


def extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown-formatted text.

    Looks for explicitly tagged ``python`` blocks first, then falls
    back to untagged blocks (skipping those tagged as other languages).
    """
    blocks = _PYTHON_BLOCK_RE.findall(text)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]

    if not _NON_PYTHON_BLOCK_RE.search(text):
        untagged = _UNTAGGED_BLOCK_RE.findall(text)
        if untagged:
            return [b.strip() for b in untagged if b.strip()]

    text_no_fences = re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.DOTALL)
    if _BARE_CODE_HEURISTIC.search(text_no_fences):
        lines = text_no_fences.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if _BARE_CODE_HEURISTIC.match(line) or in_code:
                code_lines.append(line)
                in_code = True
                if line.strip() == "" and code_lines:
                    in_code = bool(line.startswith(" ") or line.startswith("\t"))
            elif code_lines and (line.startswith(" ") or line.startswith("\t")):
                code_lines.append(line)

        if code_lines:
            return ["\n".join(code_lines)]

    return []


def check_syntax(text: str) -> SyntaxCheckResult:
    """Check all Python code blocks in *text* for syntax errors.

    Args:
        text: LLM response text, potentially containing markdown code blocks.

    Returns:
        SyntaxCheckResult with details about each block.
    """
    blocks = extract_code_blocks(text)
    result = SyntaxCheckResult(blocks_found=len(blocks))

    for i, code in enumerate(blocks):
        try:
            ast.parse(code)
            result.blocks_valid += 1
        except SyntaxError as e:
            snippet_lines = code.split("\n")
            err_line = (e.lineno or 1) - 1
            start = max(0, err_line - 1)
            end = min(len(snippet_lines), err_line + 2)
            snippet = "\n".join(snippet_lines[start:end])

            result.errors.append(SyntaxError_(
                block_index=i,
                line=e.lineno or 0,
                col=e.offset or 0,
                message=str(e.msg) if hasattr(e, "msg") else str(e),
                code_snippet=snippet,
            ))

    return result


def check_code_string(code: str) -> SyntaxCheckResult:
    """Check a single code string (not embedded in markdown)."""
    result = SyntaxCheckResult(blocks_found=1)
    try:
        ast.parse(code)
        result.blocks_valid = 1
    except SyntaxError as e:
        snippet_lines = code.split("\n")
        err_line = (e.lineno or 1) - 1
        start = max(0, err_line - 1)
        end = min(len(snippet_lines), err_line + 2)
        snippet = "\n".join(snippet_lines[start:end])

        result.errors.append(SyntaxError_(
            block_index=0,
            line=e.lineno or 0,
            col=e.offset or 0,
            message=str(e.msg) if hasattr(e, "msg") else str(e),
            code_snippet=snippet,
        ))
    return result
