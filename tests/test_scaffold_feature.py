"""Tests for custom-feature scaffolding."""

from __future__ import annotations

import ast
import importlib.util
import pickle
import sys

import pytest

from optiprofiler_agent.advisor.scaffold_feature import (
    scaffold_custom_feature,
    validate_custom_feature_code,
)


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("heavy-tailed objective noise with level 1e-3", ["mod_fun", "standard_t"]),
        ("ellipsoidal x0 perturbation", ["mod_x0", "np.clip"]),
        ("gradient-scaled noise", ["mod_fun", "finite-difference"]),
        ("quantized noisy composite with mesh 1e-2", ["mod_fun", "x_quantized"]),
        ("infeasible-side nonlinear inequality constraint perturbation", ["mod_cub", "ptype=\"n\""]),
    ],
)
def test_canned_custom_feature_scaffolds_validate(description, expected):
    result = scaffold_custom_feature(description)

    assert result.ok, result.validation_errors
    for needle in expected:
        assert needle in result.code
    assert "feature_name=\"custom\"" in result.code
    assert "lambda" not in result.code
    ast.parse(result.code)


def test_bound_scaffold_uses_bound_problem_type():
    result = scaffold_custom_feature("tighten finite bounds by 0.05")

    assert result.ok
    assert "mod_bounds=custom_mod_bounds" in result.code
    assert "ptype=\"b\"" in result.code
    assert result.validation_warnings == []


def test_generated_modifier_functions_are_pickleable(tmp_path):
    result = scaffold_custom_feature("heavy-tailed objective noise")
    module_path = tmp_path / "generated_feature.py"
    module_path.write_text(result.code, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("generated_feature", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_feature"] = module
    spec.loader.exec_module(module)

    pickle.dumps(module.custom_mod_fun)


def test_signature_checker_rejects_wrong_mod_fun_args():
    code = """\
from optiprofiler import benchmark

def solver_a(fun, x0):
    return x0

def solver_b(fun, x0):
    return x0

def bad_mod_fun(rng, problem):
    return 1.0

benchmark([solver_a, solver_b], feature_name="custom", mod_fun=bad_mod_fun)
"""
    errors, warnings = validate_custom_feature_code(code)

    assert any("expected (x, rng, problem)" in err for err in errors)
    assert warnings == []


def test_signature_checker_warns_on_nested_lambda():
    code = """\
from optiprofiler import benchmark

def solver_a(fun, x0):
    return x0

def solver_b(fun, x0):
    return x0

def custom_mod_fun(x, rng, problem):
    helper = lambda y: y
    return helper(problem.fun(x))

benchmark([solver_a, solver_b], feature_name="custom", mod_fun=custom_mod_fun)
"""
    errors, warnings = validate_custom_feature_code(code)

    assert errors == []
    assert any("pickle-safe" in warning for warning in warnings)
