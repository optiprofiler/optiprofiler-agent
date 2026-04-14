"""Validate benchmark() calls in LLM-generated Python code.

Checks that:
- At least 2 solvers are passed
- Parameter names are valid
- Enum values are valid (ptype, feature_name, etc.)
- Required arguments are present

Uses AST inspection — no code execution needed.

Usage::

    from optiprofiler_agent.validators.api_checker import validate_benchmark_call
    issues = validate_benchmark_call(code_string, knowledge_base)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from optiprofiler_agent.common.knowledge_base import KnowledgeBase, OPTION_CATEGORIES


@dataclass
class ValidationIssue:
    severity: str  # "error", "warning", "info"
    message: str
    line: int | None = None


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)
    benchmark_calls_found: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return not self.has_errors and not self.has_warnings


def _get_valid_params(kb: KnowledgeBase, lang: str) -> set[str]:
    """Collect all valid parameter names from the knowledge base."""
    bm = kb.get_benchmark(lang)
    params: set[str] = set()
    for cat in OPTION_CATEGORIES:
        cat_data = bm.get(cat, {})
        if isinstance(cat_data, dict):
            params.update(cat_data.keys())
    return params


def _get_valid_enums(kb: KnowledgeBase) -> dict[str, set[str]]:
    """Get valid enum values keyed by lowercase enum class name."""
    enums: dict[str, set[str]] = {}
    raw = kb._enums
    for cls_name, members in raw.items():
        enums[cls_name.lower()] = {v.lower() for v in members.values()}
    return enums


class _BenchmarkCallVisitor(ast.NodeVisitor):
    """AST visitor that finds and validates benchmark() calls."""

    def __init__(self, valid_params: set[str], valid_enums: dict[str, set[str]]):
        self.valid_params = valid_params
        self.valid_enums = valid_enums
        self.issues: list[ValidationIssue] = []
        self.call_count = 0

    def visit_Call(self, node: ast.Call):
        func_name = self._get_func_name(node)
        if func_name != "benchmark":
            self.generic_visit(node)
            return

        self.call_count += 1
        self._check_solvers_arg(node)
        self._check_keyword_args(node)
        self.generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _check_solvers_arg(self, node: ast.Call):
        """First positional arg should be a list/tuple with >= 2 solvers."""
        if not node.args:
            self.issues.append(ValidationIssue(
                "error", "benchmark() called without positional arguments; "
                "first argument must be a list of at least 2 solvers.",
                line=node.lineno,
            ))
            return

        solvers_arg = node.args[0]
        if isinstance(solvers_arg, (ast.List, ast.Tuple)):
            n = len(solvers_arg.elts)
            if n < 2:
                self.issues.append(ValidationIssue(
                    "error",
                    f"benchmark() requires at least 2 solvers, but only {n} provided.",
                    line=node.lineno,
                ))
        elif isinstance(solvers_arg, ast.Name):
            self.issues.append(ValidationIssue(
                "info",
                f"Solvers passed as variable '{solvers_arg.id}'; "
                "cannot statically verify count >= 2.",
                line=node.lineno,
            ))

    def _check_keyword_args(self, node: ast.Call):
        """Validate keyword argument names and known enum values."""
        enum_param_map = {
            "ptype": None,
            "feature_name": "featurename",
            "noise_type": "noisetype",
        }

        for kw in node.keywords:
            if kw.arg is None:
                continue

            if kw.arg not in self.valid_params:
                self.issues.append(ValidationIssue(
                    "warning",
                    f"Unknown parameter '{kw.arg}' in benchmark() call.",
                    line=node.lineno,
                ))

            if kw.arg in enum_param_map:
                self._check_enum_value(kw, enum_param_map[kw.arg] or kw.arg)

    def _check_enum_value(self, kw: ast.keyword, enum_key: str):
        """Check if a keyword's value matches known enum values."""
        if enum_key not in self.valid_enums:
            return

        valid = self.valid_enums[enum_key]

        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            val = kw.value.value.lower()
            if val not in valid:
                self.issues.append(ValidationIssue(
                    "warning",
                    f"'{kw.value.value}' may not be a valid value for '{kw.arg}'. "
                    f"Known values: {', '.join(sorted(valid))}",
                    line=kw.value.lineno if hasattr(kw.value, "lineno") else None,
                ))


def validate_benchmark_call(
    code: str,
    kb: KnowledgeBase | None = None,
    language: str = "python",
) -> ValidationResult:
    """Validate benchmark() calls in Python source code.

    Args:
        code: Python source code string.
        kb: KnowledgeBase instance. If None, loads default.
        language: "python" or "matlab" (for param lookup).

    Returns:
        ValidationResult with any issues found.
    """
    if kb is None:
        kb = KnowledgeBase()

    result = ValidationResult()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result.issues.append(ValidationIssue(
            "error", f"Syntax error: {e}", line=e.lineno))
        return result

    valid_params = _get_valid_params(kb, language)
    valid_enums = _get_valid_enums(kb)

    visitor = _BenchmarkCallVisitor(valid_params, valid_enums)
    visitor.visit(tree)

    result.benchmark_calls_found = visitor.call_count
    result.issues = visitor.issues

    if visitor.call_count == 0:
        result.issues.append(ValidationIssue(
            "info", "No benchmark() call found in the code."))

    return result


def validate_response_code(
    response_text: str,
    kb: KnowledgeBase | None = None,
    language: str = "python",
) -> ValidationResult:
    """Extract code blocks from an LLM response and validate them."""
    from optiprofiler_agent.validators.syntax_checker import extract_code_blocks

    blocks = extract_code_blocks(response_text)
    if not blocks:
        return ValidationResult()

    combined = ValidationResult()
    for block in blocks:
        result = validate_benchmark_call(block, kb=kb, language=language)
        combined.benchmark_calls_found += result.benchmark_calls_found
        combined.issues.extend(result.issues)

    return combined
