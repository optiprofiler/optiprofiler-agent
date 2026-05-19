"""Solver interface adapter — signature analysis and wrapper generation.

This shared module is used by:
- Agent A (proactive): guide users to adapt their solver before testing
- Agent B (reactive): auto-generate wrapper when interface mismatch is detected

The adapter inspects a user's solver function signature (via AST)
and compares it against the benchmark() requirements, then generates a thin
wrapper function that bridges the gap.

Usage::

    from optiprofiler_agent.common.interface_adapter import (
        analyze_solver, generate_wrapper, SolverAnalysis,
    )
    analysis = analyze_solver("def my_opt(f, x0, lb, ub): ...")
    if analysis.needs_wrapper:
        wrapper_code = generate_wrapper(analysis, "bound_constrained")
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field

EXPECTED_SIGNATURES: dict[str, list[str]] = {
    "unconstrained": ["fun", "x0"],
    "bound_constrained": ["fun", "x0", "xl", "xu"],
    "linearly_constrained": ["fun", "x0", "xl", "xu", "aub", "bub", "aeq", "beq"],
    "nonlinearly_constrained": [
        "fun", "x0", "xl", "xu", "aub", "bub", "aeq", "beq", "cub", "ceq",
    ],
}

_PARAM_ALIASES: dict[str, list[str]] = {
    "fun": ["fun", "f", "func", "objective", "obj", "cost", "fobj", "objfun"],
    "x0": ["x0", "x_init", "xinit", "x_start", "start", "initial_point"],
    "xl": ["xl", "lb", "lower", "lower_bound", "bounds_lower", "xmin"],
    "xu": ["xu", "ub", "upper", "upper_bound", "bounds_upper", "xmax"],
    "aub": ["aub", "a_ub", "a_ineq", "aineq"],
    "bub": ["bub", "b_ub", "b_ineq", "bineq"],
    "aeq": ["aeq", "a_eq", "aeq_"],
    "beq": ["beq", "b_eq", "beq_"],
    "cub": ["cub", "c_ub", "c_ineq", "cineq", "nonlcon_ub"],
    "ceq": ["ceq", "c_eq", "ceq_", "nonlcon_eq"],
}


@dataclass
class SolverAnalysis:
    """Result of analyzing a solver function's signature."""

    func_name: str
    params: list[str]
    matched_params: dict[str, str] = field(default_factory=dict)
    missing_params: list[str] = field(default_factory=list)
    extra_params: list[str] = field(default_factory=list)
    reorder_needed: bool = False
    needs_wrapper: bool = False
    problem_type: str | None = None
    notes: list[str] = field(default_factory=list)


def _resolve_alias(param_name: str) -> str | None:
    """Map a user parameter name to the canonical OptiProfiler name."""
    lower = param_name.lower()
    for canonical, aliases in _PARAM_ALIASES.items():
        if lower in aliases:
            return canonical
    return None


def _extract_function_def(source: str) -> ast.FunctionDef | None:
    """Parse source and return the first function definition."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _normalize_language(language: str) -> str:
    lang = (language or "python").strip().lower()
    return "matlab" if lang in ("matlab", "m") else "python"


def _extract_matlab_function_def(source: str) -> tuple[str, list[str], list[str]] | None:
    """Extract MATLAB function name, output args, and input args."""
    pattern = re.compile(
        r"^\s*function\s+"
        r"(?:(\[?\w+(?:\s*,\s*\w+)*\]?)\s*=\s*)?"
        r"(\w+)"
        r"\s*\(([^)]*)\)",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return None
    func_name = m.group(2)
    input_args = [a.strip() for a in m.group(3).split(",") if a.strip()]
    output_str = m.group(1) or ""
    output_args = [
        a.strip() for a in output_str.strip("[]").split(",") if a.strip()
    ]
    return func_name, output_args, input_args


def _analyze_matlab_solver(
    source: str,
    problem_type: str = "unconstrained",
) -> SolverAnalysis:
    """Analyze a MATLAB solver function signature."""
    parsed = _extract_matlab_function_def(source)
    if parsed is None:
        return SolverAnalysis(
            func_name="<parse_error>",
            params=[],
            needs_wrapper=True,
            problem_type=problem_type,
            notes=["Could not parse MATLAB function definition. Check syntax."],
        )

    func_name, _output_args, user_params = parsed
    expected = EXPECTED_SIGNATURES.get(problem_type, EXPECTED_SIGNATURES["unconstrained"])

    matched: dict[str, str] = {}
    used_user_params: set[int] = set()

    for exp_name in expected:
        for i, up in enumerate(user_params):
            if i in used_user_params:
                continue
            canonical = _resolve_alias(up)
            if canonical == exp_name:
                matched[exp_name] = up
                used_user_params.add(i)
                break

    missing = [e for e in expected if e not in matched]
    extra = [user_params[i] for i in range(len(user_params)) if i not in used_user_params]

    reorder = False
    if not missing and not extra:
        matched_order = [matched[e] for e in expected]
        if matched_order != user_params:
            reorder = True

    needs_wrapper = bool(missing) or bool(extra) or reorder

    notes: list[str] = []
    if missing:
        notes.append(f"Missing required parameters: {', '.join(missing)}")
    if extra:
        notes.append(f"Extra parameters not in benchmark spec: {', '.join(extra)}")
    if reorder and not missing and not extra:
        notes.append("Parameters are in wrong order; a thin wrapper can fix this.")

    for exp_name, user_name in matched.items():
        if user_name != exp_name:
            notes.append(f"'{user_name}' maps to '{exp_name}' (alias)")

    return SolverAnalysis(
        func_name=func_name,
        params=user_params,
        matched_params=matched,
        missing_params=missing,
        extra_params=extra,
        reorder_needed=reorder,
        needs_wrapper=needs_wrapper,
        problem_type=problem_type,
        notes=notes,
    )


def analyze_solver(
    source: str,
    problem_type: str = "unconstrained",
    language: str = "python",
) -> SolverAnalysis:
    """Analyze a solver function's signature against benchmark requirements.

    Args:
        source: Source code containing the solver function definition.
        problem_type: One of the keys in EXPECTED_SIGNATURES.
        language: ``"python"`` or ``"matlab"``.

    Returns:
        SolverAnalysis with detailed mismatch information.
    """
    if _normalize_language(language) == "matlab":
        return _analyze_matlab_solver(source, problem_type)

    func_def = _extract_function_def(source)
    if func_def is None:
        return SolverAnalysis(
            func_name="<parse_error>",
            params=[],
            needs_wrapper=True,
            problem_type=problem_type,
            notes=["Could not parse function definition. Check syntax."],
        )

    user_params = [arg.arg for arg in func_def.args.args]
    if user_params and user_params[0] == "self":
        user_params = user_params[1:]

    expected = EXPECTED_SIGNATURES.get(problem_type, EXPECTED_SIGNATURES["unconstrained"])

    matched: dict[str, str] = {}
    used_user_params: set[int] = set()

    for exp_name in expected:
        for i, up in enumerate(user_params):
            if i in used_user_params:
                continue
            canonical = _resolve_alias(up)
            if canonical == exp_name:
                matched[exp_name] = up
                used_user_params.add(i)
                break

    missing = [e for e in expected if e not in matched]
    extra = [user_params[i] for i in range(len(user_params)) if i not in used_user_params]

    reorder = False
    if not missing and not extra:
        matched_order = [matched[e] for e in expected]
        if matched_order != user_params:
            reorder = True

    needs_wrapper = bool(missing) or bool(extra) or reorder

    notes = []
    if missing:
        notes.append(f"Missing required parameters: {', '.join(missing)}")
    if extra:
        notes.append(f"Extra parameters not in benchmark spec: {', '.join(extra)}")
    if reorder and not missing and not extra:
        notes.append("Parameters are in wrong order; a thin wrapper can fix this.")

    for exp_name, user_name in matched.items():
        if user_name != exp_name:
            notes.append(f"'{user_name}' maps to '{exp_name}' (alias)")

    return SolverAnalysis(
        func_name=func_def.name,
        params=user_params,
        matched_params=matched,
        missing_params=missing,
        extra_params=extra,
        reorder_needed=reorder,
        needs_wrapper=needs_wrapper,
        problem_type=problem_type,
        notes=notes,
    )


def _generate_matlab_wrapper(
    analysis: SolverAnalysis,
    problem_type: str | None = None,
) -> str:
    """Generate a MATLAB wrapper function."""
    ptype = problem_type or analysis.problem_type or "unconstrained"
    expected = EXPECTED_SIGNATURES.get(ptype, EXPECTED_SIGNATURES["unconstrained"])

    wrapper_name = f"{analysis.func_name}_wrapper"
    sig_params = ", ".join(expected)
    has_missing = any(exp not in analysis.matched_params for exp in expected)

    lines = [
        f"function x = {wrapper_name}({sig_params})",
        f"    % Wrapper adapting {analysis.func_name} to OptiProfiler benchmark interface.",
    ]

    if has_missing:
        lines.append(f"    % NOTE: {analysis.func_name} does not accept all parameters.")
        lines.append(
            f"    % Unused parameters from benchmark: {', '.join(analysis.missing_params)}"
        )

    inner_args = [exp for exp in expected if exp in analysis.matched_params]
    lines.append(f"    x = {analysis.func_name}({', '.join(inner_args)});")
    lines.append("    x = x(:);  % ensure column vector")
    lines.append("end")
    lines.append("")

    return "\n".join(lines)


def generate_wrapper(
    analysis: SolverAnalysis,
    problem_type: str | None = None,
    language: str = "python",
) -> str:
    """Generate a wrapper function that adapts the user's solver."""
    if _normalize_language(language) == "matlab":
        return _generate_matlab_wrapper(analysis, problem_type)

    ptype = problem_type or analysis.problem_type or "unconstrained"
    expected = EXPECTED_SIGNATURES.get(ptype, EXPECTED_SIGNATURES["unconstrained"])

    wrapper_name = f"{analysis.func_name}_wrapper"
    sig_params = ", ".join(expected)

    has_missing = any(exp not in analysis.matched_params for exp in expected)

    lines = [
        f"def {wrapper_name}({sig_params}):",
        f'    """Wrapper adapting {analysis.func_name} to OptiProfiler benchmark interface."""',
    ]

    if has_missing:
        lines.append(f"    # NOTE: {analysis.func_name} does not accept all parameters.")
        lines.append(
            f"    # Unused parameters from benchmark: {', '.join(analysis.missing_params)}"
        )

    inner_args = [exp for exp in expected if exp in analysis.matched_params]
    lines.append(f"    return {analysis.func_name}({', '.join(inner_args)})")
    lines.append("")

    return "\n".join(lines)


def generate_wrapper_with_context(
    source: str,
    problem_type: str = "unconstrained",
    language: str = "python",
) -> tuple[SolverAnalysis, str]:
    """Convenience: analyze + generate wrapper in one call."""
    analysis = analyze_solver(source, problem_type, language=language)
    if not analysis.needs_wrapper:
        return analysis, ""
    wrapper = generate_wrapper(analysis, problem_type, language=language)
    return analysis, wrapper
