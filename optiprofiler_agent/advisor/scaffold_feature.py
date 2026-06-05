"""Deterministic scaffolding for Python custom OptiProfiler features."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from optiprofiler_agent.validators.api_checker import validate_benchmark_call
from optiprofiler_agent.validators.syntax_checker import check_code_string


SETUP_MODIFIERS = {
    "mod_x0": ("rng", "problem"),
    "mod_affine": ("rng", "problem"),
    "mod_bounds": ("rng", "problem"),
    "mod_linear_ub": ("rng", "problem"),
    "mod_linear_eq": ("rng", "problem"),
}

EVAL_MODIFIERS = {
    "mod_fun": ("x", "rng", "problem"),
    "mod_cub": ("x", "rng", "problem"),
    "mod_ceq": ("x", "rng", "problem"),
}

MODIFIER_SIGNATURES = {**SETUP_MODIFIERS, **EVAL_MODIFIERS}


@dataclass(frozen=True)
class ScaffoldFeatureResult:
    """Generated custom-feature scaffold and validation metadata."""

    code: str
    selected_modifiers: list[str]
    assumptions: list[str]
    validation_errors: list[str]
    validation_warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.validation_errors

    def to_markdown(self) -> str:
        status = "passed" if self.ok else "failed"
        assumptions = "\n".join(f"- {item}" for item in self.assumptions)
        modifiers = ", ".join(f"`{item}`" for item in self.selected_modifiers)
        md = (
            "## Custom Feature Scaffold\n\n"
            f"Selected modifiers: {modifiers}\n\n"
            "```python\n"
            f"{self.code.rstrip()}\n"
            "```\n\n"
            "## Assumptions\n\n"
            f"{assumptions}\n\n"
            f"Validation: {status}"
        )
        if self.validation_errors:
            md += "\n\nErrors:\n" + "\n".join(f"- {item}" for item in self.validation_errors)
        if self.validation_warnings:
            md += "\n\nWarnings:\n" + "\n".join(f"- {item}" for item in self.validation_warnings)
        return md


def _slug(text: str, fallback: str = "custom_feature") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:48] or fallback


def _literal_float(value: float) -> str:
    return f"{value:.12g}"


def _infer_level(description: str, default: float = 1e-3) -> float:
    """Best-effort magnitude extraction from a user description."""
    match = re.search(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?![A-Za-z0-9_.])", description, re.I)
    if not match:
        return default
    try:
        value = float(match.group(1))
    except ValueError:
        return default
    if value <= 0 or value > 1e6:
        return default
    return value


def _has_word(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def _select_templates(description: str) -> tuple[list[str], list[str], list[str]]:
    """Return code blocks, selected modifier names, and assumptions."""
    desc = (description or "").lower()
    level = _literal_float(_infer_level(desc))
    blocks: list[str] = []
    selected: list[str] = []
    assumptions: list[str] = []

    wants_noise = any(term in desc for term in ("noise", "noisy", "heavy", "tail", "t-distribution", "student"))
    wants_gradient_scaled = any(term in desc for term in ("gradient", "slope", "sensitivity"))
    wants_quantized = any(term in desc for term in ("quant", "mesh", "discrete", "round"))
    wants_x0 = any(term in desc for term in ("x0", "initial", "start", "restart", "ellipsoid", "ellipsoid"))
    wants_bounds = any(term in desc for term in ("bound", "box", "feasible region"))
    wants_affine = any(term in desc for term in ("affine", "rotate", "rotation", "linear transform", "coordinate"))
    wants_cub = _has_word(desc, r"\binequality\b|\bcub\b|infeasible-side")
    wants_ceq = _has_word(desc, r"\bequality\s+constraint\b|\bceq\b")

    if wants_noise or wants_gradient_scaled or wants_quantized or not any(
        (wants_x0, wants_bounds, wants_affine, wants_cub, wants_ceq)
    ):
        selected.append("mod_fun")
        if wants_gradient_scaled:
            blocks.append(
                f"""def custom_mod_fun(x, rng, problem):
    \"\"\"Add noise scaled by a finite-difference gradient estimate.\"\"\"
    x = np.asarray(x, dtype=float)
    f = float(problem.fun(x))
    if not np.isfinite(f):
        return f
    eps = 1e-6
    grad_norm_sq = 0.0
    for i in range(problem.n):
        step = np.zeros(problem.n)
        step[i] = eps
        grad_i = (float(problem.fun(x + step)) - f) / eps
        grad_norm_sq += grad_i * grad_i
    scale = {level} * (1.0 + np.sqrt(grad_norm_sq))
    return f + scale * rng.standard_normal()
"""
            )
            assumptions.append("Objective noise is scaled by a forward finite-difference gradient estimate.")
        elif wants_quantized:
            blocks.append(
                f"""def custom_mod_fun(x, rng, problem):
    \"\"\"Evaluate the objective on a quantized mesh, then add small noise.\"\"\"
    x = np.asarray(x, dtype=float)
    mesh_size = {level}
    x_quantized = mesh_size * np.round(x / mesh_size)
    f = float(problem.fun(x_quantized))
    if not np.isfinite(f):
        return f
    return f + mesh_size * rng.standard_normal()
"""
            )
            assumptions.append("The requested quantization is implemented by snapping x to a mesh before evaluation.")
        elif any(term in desc for term in ("heavy", "tail", "t-distribution", "student", "cauchy")):
            blocks.append(
                f"""def custom_mod_fun(x, rng, problem):
    \"\"\"Add heavy-tailed Student-t noise to objective evaluations.\"\"\"
    f = float(problem.fun(x))
    if not np.isfinite(f):
        return f
    noise = {level} * rng.standard_t(df=3)
    return f + noise
"""
            )
            assumptions.append("Heavy-tailed noise uses a Student-t distribution with 3 degrees of freedom.")
        else:
            blocks.append(
                f"""def custom_mod_fun(x, rng, problem):
    \"\"\"Add reproducible Gaussian noise to objective evaluations.\"\"\"
    f = float(problem.fun(x))
    if not np.isfinite(f):
        return f
    return f + {level} * rng.standard_normal()
"""
            )
            assumptions.append("Objective noise is Gaussian and additive.")

    if wants_x0:
        selected.append("mod_x0")
        blocks.append(
            f"""def custom_mod_x0(rng, problem):
    \"\"\"Perturb the initial point while respecting finite bounds.\"\"\"
    step = {level} * rng.standard_normal(problem.n)
    x_new = np.asarray(problem.x0, dtype=float) + step
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    lower = np.where(np.isfinite(xl), xl, -np.inf)
    upper = np.where(np.isfinite(xu), xu, np.inf)
    return np.clip(x_new, lower, upper)
"""
        )
        assumptions.append("Initial-point perturbation is isotropic; finite bounds are respected with np.clip.")

    if wants_bounds:
        selected.append("mod_bounds")
        blocks.append(
            f"""def custom_mod_bounds(rng, problem):
    \"\"\"Shrink finite bounds toward their midpoint.\"\"\"
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    finite = np.isfinite(xl) & np.isfinite(xu)
    new_xl = xl.copy()
    new_xu = xu.copy()
    width = xu[finite] - xl[finite]
    shrink = min({level}, 0.45)
    new_xl[finite] = xl[finite] + shrink * width
    new_xu[finite] = xu[finite] - shrink * width
    return new_xl, new_xu
"""
        )
        assumptions.append("Bound modification shrinks only finite two-sided bounds.")

    if wants_affine:
        selected.append("mod_affine")
        blocks.append(
            """def custom_mod_affine(rng, problem):
    \"\"\"Apply a reproducible random orthogonal coordinate rotation.\"\"\"
    gaussian = rng.standard_normal((problem.n, problem.n))
    q, _ = np.linalg.qr(gaussian)
    b = np.zeros(problem.n)
    return q, b, q.T
"""
        )
        assumptions.append("Affine transformation is an orthogonal rotation, so the inverse is q.T.")

    if wants_cub:
        selected.append("mod_cub")
        blocks.append(
            f"""def custom_mod_cub(x, rng, problem):
    \"\"\"Relax nonlinear inequality constraints by a small margin.\"\"\"
    return np.asarray(problem.cub(x), dtype=float) - {level}
"""
        )
        assumptions.append("Nonlinear inequality perturbation relaxes cub(x) <= 0 by subtracting a margin.")

    if wants_ceq:
        selected.append("mod_ceq")
        blocks.append(
            f"""def custom_mod_ceq(x, rng, problem):
    \"\"\"Add a small reproducible offset to nonlinear equality constraints.\"\"\"
    return np.asarray(problem.ceq(x), dtype=float) + {level} * rng.standard_normal(np.asarray(problem.ceq(x)).shape)
"""
        )
        assumptions.append("Nonlinear equality perturbation adds stochastic offsets with the same output shape.")

    selected = list(dict.fromkeys(selected))
    return blocks, selected, assumptions


def _render_code(description: str, feature_name: str, n_runs: int | None) -> tuple[str, list[str], list[str]]:
    blocks, selected, assumptions = _select_templates(description)
    feature_name = _slug(feature_name or description)
    stochastic_mods = {"mod_fun", "mod_x0", "mod_ceq"}
    n_runs = n_runs if n_runs and n_runs > 0 else (5 if stochastic_mods.intersection(selected) else 1)
    ptype = "n" if {"mod_cub", "mod_ceq"}.intersection(selected) else "b" if "mod_bounds" in selected else "u"

    kwargs = ['    feature_name="custom"', f"    n_runs={n_runs}"]
    for mod in selected:
        func_name = "custom_" + mod
        kwargs.append(f"    {mod}={func_name}")
    kwargs_s = ",\n".join(kwargs)

    benchmark_lines = (
        "def main():\n"
        f"    # Custom feature scaffold: {feature_name}\n"
        "    scores = benchmark(\n"
        "        [solver_a, solver_b],\n"
        f"{_indent(kwargs_s, 4)},\n"
        f"        ptype=\"{ptype}\",\n"
        "        maxdim=2,\n"
        "    )\n"
        "    return scores\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n"
    )

    code = (
        "import numpy as np\n"
        "from optiprofiler import benchmark\n\n\n"
        "def solver_a(fun, x0):\n"
        "    return x0\n\n\n"
        "def solver_b(fun, x0):\n"
        "    return x0\n\n\n"
        + "\n\n".join(block.rstrip() for block in blocks)
        + "\n\n\n"
        + benchmark_lines
    )
    return code, selected, assumptions


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def validate_custom_feature_code(code: str) -> tuple[list[str], list[str]]:
    """Validate generated code and custom-feature modifier signatures."""
    errors: list[str] = []
    warnings: list[str] = []

    syntax = check_code_string(code)
    if syntax.has_errors:
        for err in syntax.errors:
            errors.append(f"Syntax error at line {err.line}: {err.message}")
        return errors, warnings

    tree = ast.parse(code)
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    benchmark_mods = _benchmark_custom_modifiers(tree)
    api = validate_benchmark_call(code, language="python")
    stochastic_mods = {"mod_fun", "mod_x0", "mod_ceq"}
    for issue in api.issues:
        if (
            issue.severity == "warning"
            and "n_runs <= 1" in issue.message
            and not stochastic_mods.intersection(benchmark_mods)
        ):
            continue
        target = errors if issue.severity == "error" else warnings
        target.append(issue.message)

    for mod_name, func_name in benchmark_mods.items():
        expected = MODIFIER_SIGNATURES[mod_name]
        node = funcs.get(func_name)
        if node is None:
            errors.append(f"{mod_name} references `{func_name}`, but no module-level function defines it.")
            continue
        args = tuple(arg.arg for arg in node.args.args)
        if args != expected:
            errors.append(
                f"{func_name} has signature ({', '.join(args)}), expected "
                f"({', '.join(expected)}) for {mod_name}."
            )
        if any(isinstance(child, (ast.Lambda, ast.FunctionDef)) for child in ast.walk(node) if child is not node):
            warnings.append(f"{func_name} contains a nested function or lambda; keep modifiers pickle-safe.")

    if not benchmark_mods:
        errors.append("No custom mod_* benchmark arguments found.")

    return errors, warnings


def _benchmark_custom_modifiers(tree: ast.Module) -> dict[str, str]:
    modifiers: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name != "benchmark":
            continue
        is_custom = False
        for kw in node.keywords:
            if kw.arg == "feature_name" and isinstance(kw.value, ast.Constant) and kw.value.value == "custom":
                is_custom = True
                break
        if not is_custom:
            continue
        for kw in node.keywords:
            if kw.arg in MODIFIER_SIGNATURES and isinstance(kw.value, ast.Name):
                modifiers[kw.arg] = kw.value.id
    return modifiers


def scaffold_custom_feature(
    description: str,
    feature_name: str = "",
    n_runs: int | None = None,
) -> ScaffoldFeatureResult:
    """Generate a validated Python custom-feature scaffold."""
    code, selected, assumptions = _render_code(description, feature_name, n_runs)
    errors, warnings = validate_custom_feature_code(code)
    return ScaffoldFeatureResult(
        code=code,
        selected_modifiers=selected,
        assumptions=assumptions,
        validation_errors=errors,
        validation_warnings=warnings,
    )
