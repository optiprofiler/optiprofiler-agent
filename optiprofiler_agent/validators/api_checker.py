"""Validate benchmark() calls and ``optiprofiler`` imports in LLM-generated code.

Two layers:

* **Call-shape validation** (`_BenchmarkCallVisitor`): solver count, kwarg
  names, enum values for ``benchmark()`` calls.
* **Import validation** (`_ImportVisitor`): catches the most common
  reference-hallucination pattern we observe in the wild — ``optiprobe``
  typos and fake submodules (``optiprofiler.solvers`` does not exist).
  The whitelist is **derived automatically** from
  ``knowledge/_sources/python/*.json`` so it stays in sync with the upstream
  API without a separate maintenance task.

Design note (hallucination guard, L1):
  This module is the *post-generation verifier* in our L0-L2 stack. It is
  intentionally pluggable through :class:`CodeConstraintBackend` so that a
  later L4 implementation (vLLM grammar-constrained decoding) can slot in
  alongside without touching the AST path.

Usage::

    from optiprofiler_agent.validators.api_checker import validate_benchmark_call
    result = validate_benchmark_call(code_string)
    if result.has_errors:
        for issue in result.issues:
            print(issue.severity, issue.message)
"""

from __future__ import annotations

import ast
import difflib
import functools
from dataclasses import dataclass, field
from typing import Protocol

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


# ───────────────────────────────────────────────────────────────────────────
# Import whitelist (auto-derived from knowledge/_sources/python/*.json)
#
# We deliberately do NOT enumerate `optiprofiler` internals (private
# helpers, submodule paths). Anything not in this set fires a warning with
# a did-you-mean suggestion. This catches the two most common mistakes:
#
#     from optiprobe import benchmark           # typo of optiprofiler
#     from optiprofiler.solvers import bobyqa   # solvers/ is not a module
#
# Note: the module name itself ("optiprofiler") is implicit; the whitelist
# tracks importable *symbols*. A plain `import optiprofiler` is always OK.
# ───────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=4)
def _load_optiprofiler_python_exports(_kb_id: int = 0) -> frozenset[str]:
    """Return the set of public symbols importable from ``optiprofiler``.

    The cache is keyed on a synthetic id (default 0 = singleton KB) so
    tests can pass an alternative KB without polluting prod cache.
    """
    kb = KnowledgeBase()
    exports: set[str] = set()

    classes = kb.get_classes("python") or {}
    exports.update(classes.keys())

    plib = kb.get_plib_tools("python") or {}
    exports.update(plib.keys())

    if kb.get_benchmark("python"):
        exports.add("benchmark")

    return frozenset(exports)


def optiprofiler_python_exports() -> frozenset[str]:
    """Public accessor (used by tests and by the lint loop in cli.py)."""
    return _load_optiprofiler_python_exports()


def _suggest(name: str, candidates: frozenset[str]) -> str | None:
    """Single best did-you-mean suggestion, or None if nothing close enough."""
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


class _ImportVisitor(ast.NodeVisitor):
    """Flag ``optiprofiler``-shaped imports that won't actually work."""

    _PACKAGE = "optiprofiler"
    _PACKAGE_TYPOS = {"optiprobe", "opti_profiler", "opti-profiler", "optiprofile", "optiproflier"}

    def __init__(self, exports: frozenset[str]):
        self.exports = exports
        self.issues: list[ValidationIssue] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in self._PACKAGE_TYPOS:
                self.issues.append(ValidationIssue(
                    "error",
                    f"`import {alias.name}` looks like a typo of "
                    f"`import {self._PACKAGE}` (the package is named "
                    f"`{self._PACKAGE}`, not `{top}`).",
                    line=node.lineno,
                ))
                continue
            if top == self._PACKAGE and "." in alias.name:
                sub = alias.name.split(".", 1)[1]
                self.issues.append(ValidationIssue(
                    "warning",
                    f"`import {alias.name}` references `{self._PACKAGE}.{sub}` — "
                    f"`{self._PACKAGE}` exposes its public API as flat top-level "
                    f"symbols (`from {self._PACKAGE} import {sorted(self.exports)[0]}`), "
                    f"there is no `{sub}` submodule.",
                    line=node.lineno,
                ))

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        if not mod:
            return
        top = mod.split(".")[0]

        if top in self._PACKAGE_TYPOS:
            suggestion = (
                f"from {self._PACKAGE} import "
                + ", ".join(n.name for n in node.names)
            )
            self.issues.append(ValidationIssue(
                "error",
                f"`from {mod} import ...` — package name typo. The correct "
                f"package is `{self._PACKAGE}`. Did you mean: `{suggestion}`?",
                line=node.lineno,
            ))
            return

        if top != self._PACKAGE:
            return

        if "." in mod:
            sub = mod.split(".", 1)[1]
            self.issues.append(ValidationIssue(
                "warning",
                f"`from {mod} import ...` references submodule `{sub}` — "
                f"`{self._PACKAGE}` does not expose `{sub}` as a public submodule. "
                f"Use `from {self._PACKAGE} import <symbol>` instead.",
                line=node.lineno,
            ))
            return

        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name in self.exports:
                continue
            suggestion = _suggest(alias.name, self.exports)
            msg = (
                f"`from {self._PACKAGE} import {alias.name}` — `{alias.name}` is not "
                f"a known public export of `{self._PACKAGE}`."
            )
            if suggestion:
                msg += f" Did you mean `{suggestion}`?"
            self.issues.append(ValidationIssue(
                "warning", msg, line=node.lineno,
            ))


# ───────────────────────────────────────────────────────────────────────────
# Pluggable constraint backend (forward-compat for L4 / vLLM grammar)
#
# The CLI lint loop talks to a backend through this Protocol. Today only
# the AST backend is wired in; future grammar-constrained decoding lands
# as a sibling class without touching cli.py or this validator.
# ───────────────────────────────────────────────────────────────────────────


class CodeConstraintBackend(Protocol):
    """Validates a chunk of code, returning issues to feed back to the LLM."""

    name: str

    def validate(self, code: str, *, language: str = "python") -> "ValidationResult":
        ...


class ASTValidatorBackend:
    """Default backend — AST-based whitelist + benchmark() shape check."""

    name = "ast"

    def validate(self, code: str, *, language: str = "python") -> "ValidationResult":
        return validate_benchmark_call(code, language=language)


class NullBackend:
    """No-op backend (used to disable the lint loop in tests / dev)."""

    name = "null"

    def validate(self, code: str, *, language: str = "python") -> "ValidationResult":
        return ValidationResult()


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
    result.issues = list(visitor.issues)

    if language == "python":
        import_visitor = _ImportVisitor(_load_optiprofiler_python_exports())
        import_visitor.visit(tree)
        result.issues.extend(import_visitor.issues)

    if visitor.call_count == 0 and not result.issues:
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
