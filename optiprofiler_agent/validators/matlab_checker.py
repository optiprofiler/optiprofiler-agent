"""Basic MATLAB code validation — keyword blacklist + structural checks.

Unlike Python's AST-based syntax_checker, MATLAB has no stdlib parser.
We do lightweight checks: dangerous function blacklist + basic structural
validation (matching function/end, balanced parentheses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_DANGEROUS_CALLS = re.compile(
    r"\b(?:system|unix|dos|eval|feval\s*\(\s*['\"]system|"
    r"web\s*\(\s*['\"]http|urlread|urlwrite|"
    r"java\.lang|py\.importlib)\b",
    re.IGNORECASE,
)
_SHELL_ESCAPE = re.compile(r"^\s*!", re.MULTILINE)
_FUNCTION_DEF = re.compile(r"^\s*function\b", re.IGNORECASE)


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("%")


def _looks_like_script_statement(line: str) -> bool:
    stripped = line.strip()
    if _is_comment_or_blank(stripped):
        return False
    lower = stripped.lower()
    if lower in {"end", "end;"}:
        return False
    if lower.startswith(("function", "if ", "for ", "while ", "switch ", "try", "catch", "else", "elseif ")):
        return False
    return True


@dataclass
class MatlabCheckResult:
    has_errors: bool = False
    errors: list[str] = field(default_factory=list)


def check_matlab_code(code: str) -> MatlabCheckResult:
    """Validate MATLAB source for sandbox safety and basic structure."""
    result = MatlabCheckResult()

    for m in _DANGEROUS_CALLS.finditer(code):
        result.has_errors = True
        line_num = code[: m.start()].count("\n") + 1
        result.errors.append(
            f"Line {line_num}: potentially dangerous call '{m.group()}'"
        )

    for m in _SHELL_ESCAPE.finditer(code):
        result.has_errors = True
        line_num = code[: m.start()].count("\n") + 1
        result.errors.append(
            f"Line {line_num}: shell escape '!' is not allowed"
        )

    opens = code.count("(") + code.count("[") + code.count("{")
    closes = code.count(")") + code.count("]") + code.count("}")
    if opens != closes:
        result.has_errors = True
        result.errors.append("Unbalanced parentheses/brackets/braces")

    lines = code.splitlines()
    first_code_idx = next(
        (idx for idx, line in enumerate(lines) if not _is_comment_or_blank(line)),
        None,
    )
    if first_code_idx is not None and _FUNCTION_DEF.match(lines[first_code_idx]):
        seen_end = False
        for line in lines[first_code_idx + 1:]:
            stripped = line.strip()
            if stripped.lower() in {"end", "end;"}:
                seen_end = True
                continue
            if seen_end and _looks_like_script_statement(line):
                result.has_errors = True
                result.errors.append(
                    "MATLAB script statements appear after a leading function definition; "
                    "put script code first and local functions at the end."
                )
                break

    return result
