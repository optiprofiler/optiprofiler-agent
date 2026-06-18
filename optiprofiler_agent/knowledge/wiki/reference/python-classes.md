---
tags: [reference, source-backed, python, classes]
sources: [_sources/python/classes.json]
related: [../api/python/problem-class.md]
last_updated: 2026-06-18
generated: true
---

# Source Reference: Python classes.json

This page is auto-generated from `_sources/python/classes.json`. It is the lossless wiki mirror for this source.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.

## Source Metadata

- Source path: `_sources/python/classes.json`
- Canonical SHA256: `c21f66a836af1efa12daf4bde48fdb4d103c77fa084d4886e172799f8758c593`
- Top-level keys: `Problem`, `Feature`, `FeaturedProblem`

## Path Index

| Path | Kind |
|---|---|
| `Problem` | dict[7] |
| `Feature` | dict[8] |
| `FeaturedProblem` | dict[8] |

## Problem

```json
{
  "description": "Optimization problem to be used in the benchmarking. ``Problem`` describes an optimization problem with the following structure:  .. math::  \\min \\quad & \\mathrm{fun}(x) \\\\ \\text{s.t.} \\quad & x_l \\le x \\le x_u, \\\\ & A_{\\mathrm{ub}} x \\le b_{\\mathrm{ub}}, \\\\ & A_{\\mathrm{eq}} x = b_{\\mathrm{eq}}, \\\\ & c_{\\mathrm{ub}}(x) \\le 0, \\\\ & c_{\\mathrm{eq}}(x) = 0, \\\\ & \\text{with initial point } x_0,  where ``fun`` is the objective function, ``x`` is the variable to optimize, ``xl`` and ``xu`` are the lower and upper bounds, ``aub`` and ``bub`` are the coefficient matrix and right-hand side vector of the linear inequality constraints, ``aeq`` and ``beq`` are the coefficient matrix and right-hand side vector of the linear equality constraints, ``cub`` is the function of nonlinear inequality constraints, and ``ceq`` is the function of nonlinear equality constraints.",
  "methods": {
    "ceq": {
      "description": "Evaluate the nonlinear constraints ``ceq(x) == 0``.",
      "signature": "(self, x)"
    },
    "cub": {
      "description": "Evaluate the nonlinear constraints ``cub(x) <= 0``.",
      "signature": "(self, x)"
    },
    "fun": {
      "description": "Evaluate the objective function.",
      "signature": "(self, x)"
    },
    "grad": {
      "description": "Evaluate the gradient of the objective function.",
      "signature": "(self, x)"
    },
    "hceq": {
      "description": "Evaluate the Hessian of the nonlinear equality constraints.",
      "signature": "(self, x)"
    },
    "hcub": {
      "description": "Evaluate the Hessian of the nonlinear inequality constraints.",
      "signature": "(self, x)"
    },
    "hess": {
      "description": "Evaluate the Hessian of the objective function.",
      "signature": "(self, x)"
    },
    "jceq": {
      "description": "Evaluate the Jacobian of the nonlinear equality constraints.",
      "signature": "(self, x)"
    },
    "jcub": {
      "description": "Evaluate the Jacobian of the nonlinear inequality constraints.",
      "signature": "(self, x)"
    },
    "maxcv": {
      "description": "Evaluate the maximum constraint violation.",
      "signature": "(self, x)"
    },
    "project_x0": {
      "description": "Project the initial guess onto the feasible region.",
      "signature": "(self)"
    }
  },
  "name": "Problem",
  "parameters": {
    "aeq": {
      "default": "an empty matrix",
      "description": "Coefficient matrix of the linear equality constraints ``aeq @ x == beq``. Default is an empty matrix.",
      "type": "array_like, shape (m_linear_eq, n)"
    },
    "aub": {
      "default": "an empty matrix",
      "description": "Coefficient matrix of the linear inequality constraints ``aub @ x <= bub``. Default is an empty matrix.",
      "type": "array_like, shape (m_linear_ub, n)"
    },
    "beq": {
      "default": "an empty vector",
      "description": "Right-hand side of the linear equality constraints ``aeq @ x == beq``. Default is an empty vector.",
      "type": "array_like, shape (m_linear_eq,)"
    },
    "bub": {
      "default": "an empty vector",
      "description": "Right-hand side of the linear inequality constraints ``aub @ x <= bub``. Default is an empty vector.",
      "type": "array_like, shape (m_linear_ub,)"
    },
    "ceq": {
      "description": "Nonlinear equality constraints ``ceq(x) == 0``: ``ceq(x) -> array_like, shape (m_nonlinear_eq,)``. Default returns an empty array.",
      "type": "callable"
    },
    "cub": {
      "description": "Nonlinear inequality constraints ``cub(x) <= 0``: ``cub(x) -> array_like, shape (m_nonlinear_ub,)``. Default returns an empty array.",
      "type": "callable"
    },
    "fun": {
      "description": "Objective function to be minimized: ``fun(x) -> float``, where ``x`` is an array with shape ``(n,)``.",
      "type": "callable"
    },
    "grad": {
      "description": "Gradient of the objective function: ``grad(x) -> array, shape (n,)``. Default returns an empty array.",
      "type": "callable"
    },
    "hceq": {
      "description": "Hessians of the nonlinear equality constraints: ``hceq(x) -> list of arrays, each shape (n, n)``. Default returns an empty list.",
      "type": "callable"
    },
    "hcub": {
      "description": "Hessians of the nonlinear inequality constraints: ``hcub(x) -> list of arrays, each shape (n, n)``. The *i*-th element is the Hessian of the *i*-th constraint in ``cub``. Default returns an empty list.",
      "type": "callable"
    },
    "hess": {
      "description": "Hessian of the objective function: ``hess(x) -> array, shape (n, n)``. Default returns an empty matrix.",
      "type": "callable"
    },
    "jceq": {
      "description": "Jacobian of the nonlinear equality constraints: ``jceq(x) -> array, shape (m_nonlinear_eq, n)``. Default returns an empty matrix.",
      "type": "callable"
    },
    "jcub": {
      "description": "Jacobian of the nonlinear inequality constraints: ``jcub(x) -> array, shape (m_nonlinear_ub, n)``. The number of columns must equal ``n`` and the number of rows must equal ``m_nonlinear_ub``. Default returns an empty matrix.",
      "type": "callable"
    },
    "name": {
      "default": "``'Unnamed Problem'``",
      "description": "Name of the problem. Default is ``'Unnamed Problem'``.",
      "type": "str"
    },
    "x0": {
      "description": "Initial guess.",
      "type": "array_like, shape (n,)"
    },
    "xl": {
      "default": "``-numpy.inf`` for each component",
      "description": "Lower bounds on the variables ``xl <= x``. Default is ``-numpy.inf`` for each component.",
      "type": "array_like, shape (n,)"
    },
    "xu": {
      "default": "``numpy.inf`` for each component",
      "description": "Upper bounds on the variables ``x <= xu``. Default is ``numpy.inf`` for each component.",
      "type": "array_like, shape (n,)"
    }
  },
  "properties": {
    "aeq": {
      "description": "Coefficient matrix of the linear constraints ``aeq @ x == beq``."
    },
    "aub": {
      "description": "Coefficient matrix of the linear constraints ``aub @ x <= bub``."
    },
    "beq": {
      "description": "Right-hand side of the linear constraints ``aeq @ x == beq``."
    },
    "bub": {
      "description": "Right-hand side of the linear constraints ``aub @ x <= bub``."
    },
    "m_linear_eq": {
      "description": "Number of linear equality constraints."
    },
    "m_linear_ub": {
      "description": "Number of linear inequality constraints."
    },
    "m_nonlinear_eq": {
      "description": "Number of nonlinear equality constraints."
    },
    "m_nonlinear_ub": {
      "description": "Number of nonlinear inequality constraints."
    },
    "mb": {
      "description": "Number of bound constraints."
    },
    "mcon": {
      "description": "Total number of constraints (linear and nonlinear)."
    },
    "mlcon": {
      "description": "Total number of linear constraints (inequality and equality)."
    },
    "mnlcon": {
      "description": "Total number of nonlinear constraints (inequality and equality)."
    },
    "n": {
      "description": "Dimension of the problem."
    },
    "name": {
      "description": "Name of the problem."
    },
    "ptype": {
      "description": "Type of the problem."
    },
    "x0": {
      "description": "Initial guess."
    },
    "xl": {
      "description": "Lower bounds on the variables."
    },
    "xu": {
      "description": "Upper bounds on the variables."
    }
  },
  "see_also": [
    {
      "description": "Feature applied to problems during benchmarking.",
      "name": [
        "Feature",
        null
      ]
    },
    {
      "description": "Problem equipped with a specific feature.",
      "name": [
        "FeaturedProblem",
        null
      ]
    },
    {
      "description": "Main benchmarking function.",
      "name": [
        "benchmark",
        null
      ]
    }
  ],
  "signature": "(fun, x0, name=None, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, grad=None, hess=None, jcub=None, jceq=None, hcub=None, hceq=None)"
}
```

## Feature

```json
{
  "description": "Mapping from an optimization problem to a new one with specified features. We are interested in testing solvers on problems with different features. For example, we may want to test the performance of solvers when the objective function is noisy. For this purpose, we define the ``Feature`` class.  Suppose we have an optimization problem  .. math::  \\min \\quad & \\mathrm{fun}(x) \\\\ \\text{s.t.} \\quad & x_l \\le x \\le x_u, \\\\ & A_{\\mathrm{ub}} x \\le b_{\\mathrm{ub}}, \\\\ & A_{\\mathrm{eq}} x = b_{\\mathrm{eq}}, \\\\ & c_{\\mathrm{ub}}(x) \\le 0, \\\\ & c_{\\mathrm{eq}}(x) = 0, \\\\ & \\text{with initial point } x_0.  Then ``Feature`` maps the above problem to the following one:  .. math::  \\min \\quad & \\mathrm{fun\\_mod}(Ax + b) \\\\ \\text{s.t.} \\quad & x_{l,\\mathrm{mod}} \\le Ax + b \\le x_{u,\\mathrm{mod}}, \\\\ & A_{\\mathrm{ub,mod}} (Ax + b) \\le b_{\\mathrm{ub,mod}}, \\\\ & A_{\\mathrm{eq,mod}} (Ax + b) = b_{\\mathrm{eq,mod}}, \\\\ & c_{\\mathrm{ub,mod}}(Ax + b) \\le 0, \\\\ & c_{\\mathrm{eq,mod}}(Ax + b) = 0, \\\\ & \\text{with initial guess } x_{0,\\mathrm{mod}},  where the modified quantities are determined by the chosen feature name and options.",
  "methods": {
    "chebyshev_noise_map": {
      "description": "",
      "signature": "(x)"
    },
    "get_default_rng": {
      "description": "Generate a random number generator.",
      "signature": "(seed, *args)"
    },
    "modifier_affine": {
      "description": "Generate an invertible matrix A and a vector b for the affine transformation applied to the variables.",
      "signature": "(self, seed, problem)"
    },
    "modifier_bounds": {
      "description": "Modify the bounds.",
      "signature": "(self, seed, problem)"
    },
    "modifier_ceq": {
      "description": "Modify the values of the nonlinear equality constraints.",
      "signature": "(self, x, seed, problem, n_eval_ceq)"
    },
    "modifier_cub": {
      "description": "Modify the values of the nonlinear inequality constraints.",
      "signature": "(self, x, seed, problem, n_eval_cub)"
    },
    "modifier_fun": {
      "description": "Modify the objective function value.",
      "signature": "(self, x, seed, problem, n_eval)"
    },
    "modifier_linear_eq": {
      "description": "Modify the linear equality constraints.",
      "signature": "(self, seed, problem)"
    },
    "modifier_linear_ub": {
      "description": "Modify the linear inequality constraints.",
      "signature": "(self, seed, problem)"
    },
    "modifier_x0": {
      "description": "Modify the initial point.",
      "signature": "(self, seed, problem)"
    }
  },
  "name": "Feature",
  "notes": "Different feature names accept different subsets of options. The valid options for each feature name are:  1. ``'plain'`` : ``n_runs``. 2. ``'perturbed_x0'`` : ``n_runs``, ``distribution``, ``perturbation_level``. 3. ``'noisy'`` : ``n_runs``, ``distribution``, ``noise_level``, ``noise_type``, ``noise_mode``, ``noise_map``. 4. ``'truncated'`` : ``n_runs``, ``significant_digits``, ``perturbed_trailing_digits``. 5. ``'permuted'`` : ``n_runs``. 6. ``'linearly_transformed'`` : ``n_runs``, ``rotated``, ``condition_factor``. 7. ``'random_nan'`` : ``n_runs``, ``nan_rate``. 8. ``'unrelaxable_constraints'`` : ``n_runs``, ``unrelaxable_bounds``, ``unrelaxable_linear_constraints``, ``unrelaxable_nonlinear_constraints``. 9. ``'nonquantifiable_constraints'`` : ``n_runs``. 10. ``'quantized'`` : ``n_runs``, ``mesh_size``, ``mesh_type``, ``ground_truth``. 11. ``'custom'`` : ``n_runs``, ``mod_x0``, ``mod_affine``, ``mod_bounds``, ``mod_linear_ub``, ``mod_linear_eq``, ``mod_fun``, ``mod_cub``, ``mod_ceq``.",
  "parameters": {
    "feature_options": {
      "choices": [
        "spherical",
        "gaussian"
      ],
      "default": "``5`` for stochastic features and ``1`` for deterministic features",
      "description": "Keyword arguments passed after ``name``. The available options depend on the chosen ``name``:  - **n_runs** (*int*) -- Number of runs of the experiment under the given feature. Default is ``5`` for stochastic features and ``1`` for deterministic features. Valid for all features. - **distribution** (*str or callable*) -- Distribution of perturbation (``'perturbed_x0'``) or noise (``'noisy'``). For ``'perturbed_x0'``, it should be ``'spherical'`` (default) or ``'gaussian'``. For ``'noisy'``, it should be ``'gaussian'`` (default) or ``'uniform'``. It can also be a callable ``(rng, dimension) -> array``. - **perturbation_level** (*float*) -- Magnitude of the perturbation in ``'perturbed_x0'``. Default is ``1e-3``. - **noise_level** (*float*) -- Magnitude of the noise in ``'noisy'``. Default is ``1e-3``. - **noise_type** (*str*) -- Type of the noise in ``'noisy'``. Must be ``'absolute'``, ``'relative'``, or ``'mixed'`` (default). - **noise_mode** (*str*) -- Mode of the noise in ``'noisy'``. Must be ``'random'`` (default) or ``'deterministic'``. - **noise_map** (*str or callable*) -- Deterministic scalar noise map in ``'noisy'``. It should be ``'chebyshev'`` (default) or a callable ``x -> noise`` returning a real scalar. It is used only when ``noise_mode`` is ``'deterministic'``. The built-in ``'chebyshev'`` map follows the deterministic noise model in Moré and Wild, \"Benchmarking derivative-free optimization algorithms\" (2009). - **significant_digits** (*int*) -- Number of significant digits in ``'truncated'``. Default is ``6``. - **perturbed_trailing_digits** (*bool*) -- Whether to randomize the trailing digits in ``'truncated'``. Default is ``False``. - **rotated** (*bool*) -- Whether to use a random rotation matrix in ``'linearly_transformed'``. Default is ``True``. - **condition_factor** (*float*) -- Scaling factor of the condition number of the linear transformation in ``'linearly_transformed'``. The condition number will be ``2^(condition_factor * n / 2)``. Default is ``0``. - **nan_rate** (*float*) -- Probability that an evaluation returns ``NaN`` in ``'random_nan'``. Default is ``0.05``. - **unrelaxable_bounds** (*bool*) -- Whether bound constraints are unrelaxable in ``'unrelaxable_constraints'``. Default is ``True``. - **unrelaxable_linear_constraints** (*bool*) -- Whether linear constraints are unrelaxable. Default is ``False``. - **unrelaxable_nonlinear_constraints** (*bool*) -- Whether nonlinear constraints are unrelaxable. Default is ``False``. - **mesh_size** (*float*) -- Size of the mesh in ``'quantized'``. Default is ``1e-3``. - **mesh_type** (*str*) -- Type of the mesh in ``'quantized'``. Must be ``'absolute'`` (default) or ``'relative'``. - **ground_truth** (*bool*) -- Whether the featured problem is the ground truth in ``'quantized'``. Default is ``True``. - **mod_x0** (*callable*) -- Modifier for the initial guess in ``'custom'``: ``(rng, problem) -> modified_x0``. - **mod_affine** (*callable*) -- Modifier for the affine transformation in ``'custom'``: ``(rng, problem) -> (A, b, inv)``. - **mod_bounds** (*callable*) -- Modifier for the bounds in ``'custom'``: ``(rng, problem) -> (xl, xu)``. - **mod_linear_ub** (*callable*) -- Modifier for the linear inequality constraints in ``'custom'``: ``(rng, problem) -> (aub, bub)``. - **mod_linear_eq** (*callable*) -- Modifier for the linear equality constraints in ``'custom'``: ``(rng, problem) -> (aeq, beq)``. - **mod_fun** (*callable*) -- Modifier for the objective function in ``'custom'``: ``(x, rng, problem) -> modified_fun``. - **mod_cub** (*callable*) -- Modifier for the nonlinear inequality constraints in ``'custom'``: ``(x, rng, problem) -> modified_cub``. - **mod_ceq** (*callable*) -- Modifier for the nonlinear equality constraints in ``'custom'``: ``(x, rng, problem) -> modified_ceq``.",
      "type": "dict"
    },
    "name": {
      "description": "Name of the feature. Must be one of the following:  1. ``'plain'`` : do nothing to the optimization problem. 2. ``'perturbed_x0'`` : perturb the initial guess ``x0``. 3. ``'noisy'`` : add noise to the objective function and nonlinear constraints. 4. ``'truncated'`` : truncate values of the objective function and nonlinear constraints to a given number of significant digits. 5. ``'permuted'`` : randomly permute the variables. The bounds and linear constraints are modified accordingly so that the new problem is mathematically equivalent to the original one. 6. ``'linearly_transformed'`` : apply an invertible linear transformation ``D @ Q'`` (with ``D`` diagonal and ``Q`` orthogonal) to the variables. Bounds and linear constraints are modified accordingly. 7. ``'random_nan'`` : randomly replace values of the objective function and nonlinear constraints with ``NaN``. 8. ``'unrelaxable_constraints'`` : set the objective function to ``Inf`` outside the feasible region. 9. ``'nonquantifiable_constraints'`` : replace values of nonlinear constraints with either ``0`` (satisfied) or ``1`` (violated). 10. ``'quantized'`` : quantize the objective function and nonlinear constraints. 11. ``'custom'`` : user-defined feature.",
      "type": "str"
    }
  },
  "properties": {
    "is_stochastic": {
      "description": "Whether the feature is stochastic."
    },
    "name": {
      "description": "Name of the feature."
    },
    "options": {
      "description": "Options of the feature."
    }
  },
  "see_also": [
    {
      "description": "Optimization problem.",
      "name": [
        "Problem",
        null
      ]
    },
    {
      "description": "Problem equipped with a specific feature.",
      "name": [
        "FeaturedProblem",
        null
      ]
    },
    {
      "description": "Main benchmarking function.",
      "name": [
        "benchmark",
        null
      ]
    }
  ],
  "signature": "(name, **feature_options)"
}
```

## FeaturedProblem

```json
{
  "description": "Subclass of `Problem` that equips an optimization problem with a feature. ``Problem`` and its subclass ``FeaturedProblem`` describe the following optimization problem:  .. math::  \\min \\quad & \\mathrm{fun}(x) \\\\ \\text{s.t.} \\quad & x_l \\le x \\le x_u, \\\\ & A_{\\mathrm{ub}} x \\le b_{\\mathrm{ub}}, \\\\ & A_{\\mathrm{eq}} x = b_{\\mathrm{eq}}, \\\\ & c_{\\mathrm{ub}}(x) \\le 0, \\\\ & c_{\\mathrm{eq}}(x) = 0, \\\\ & \\text{with initial point } x_0.",
  "methods": {
    "ceq": {
      "description": "Evaluate the nonlinear constraints ``ceq(x) == 0``.",
      "signature": "(self, x, record_hist=True)"
    },
    "cub": {
      "description": "Evaluate the nonlinear constraints ``cub(x) <= 0``.",
      "signature": "(self, x, record_hist=True)"
    },
    "fun": {
      "description": "Evaluate the objective function.",
      "signature": "(self, x)"
    },
    "grad": {
      "description": "Evaluate the gradient of the objective function.",
      "signature": "(self, x)"
    },
    "hceq": {
      "description": "Evaluate the Hessian of the nonlinear equality constraints.",
      "signature": "(self, x)"
    },
    "hcub": {
      "description": "Evaluate the Hessian of the nonlinear inequality constraints.",
      "signature": "(self, x)"
    },
    "hess": {
      "description": "Evaluate the Hessian of the objective function.",
      "signature": "(self, x)"
    },
    "jceq": {
      "description": "Evaluate the Jacobian of the nonlinear equality constraints.",
      "signature": "(self, x)"
    },
    "jcub": {
      "description": "Evaluate the Jacobian of the nonlinear inequality constraints.",
      "signature": "(self, x)"
    },
    "maxcv": {
      "description": "Evaluate the maximum constraint violation.",
      "signature": "(self, x)"
    },
    "project_x0": {
      "description": "Project the initial guess onto the feasible region.",
      "signature": "(self)"
    }
  },
  "name": "FeaturedProblem",
  "notes": "``FeaturedProblem`` inherits all methods of ``Problem``, but the methods ``fun``, ``cub``, ``ceq``, and ``maxcv`` are modified by the input ``Feature``.  1. When the number of function evaluations reaches ``max_eval``, the methods ``fun``, ``cub``, and ``ceq`` will return the values at the point where the maximum number of function evaluations was reached. 2. When the number of function evaluations reaches ``termination_eval`` (used internally by ``benchmark``), the methods ``fun``, ``cub``, and ``ceq`` will raise an error to terminate the optimization process.  .. note::  For consistency with the rest of OptiProfiler, we recommend defining callables (such as ``fun``) with ``def`` rather than ``lambda``. Lambda expressions are not picklable, which prevents parallel execution when such callables are eventually passed to :func:`~optiprofiler.benchmark` with ``n_jobs > 1``. See :ref:`py_callable_picklability` for details.",
  "parameters": {
    "feature": {
      "description": "The feature to apply to the problem.",
      "type": "Feature"
    },
    "max_eval": {
      "description": "Maximum number of function evaluations.",
      "type": "int"
    },
    "problem": {
      "description": "The original optimization problem.",
      "type": "Problem"
    },
    "seed": {
      "description": "Nonnegative integer seed for the random number generator.",
      "type": "int"
    }
  },
  "properties": {
    "aeq": {
      "description": "Coefficient matrix of the linear constraints ``aeq @ x == beq``."
    },
    "aub": {
      "description": "Coefficient matrix of the linear constraints ``aub @ x <= bub``."
    },
    "beq": {
      "description": "Right-hand side of the linear constraints ``aeq @ x == beq``."
    },
    "bub": {
      "description": "Right-hand side of the linear constraints ``aub @ x <= bub``."
    },
    "ceq_hist": {
      "description": "History of nonlinear equality constraints."
    },
    "cub_hist": {
      "description": "History of nonlinear inequality constraints."
    },
    "fun_hist": {
      "description": "History of objective function values."
    },
    "fun_init": {
      "description": "Objective function value at the initial point."
    },
    "m_linear_eq": {
      "description": "Number of linear equality constraints."
    },
    "m_linear_ub": {
      "description": "Number of linear inequality constraints."
    },
    "m_nonlinear_eq": {
      "description": "Number of nonlinear equality constraints."
    },
    "m_nonlinear_ub": {
      "description": "Number of nonlinear inequality constraints."
    },
    "maxcv_hist": {
      "description": "History of maximum constraint violations."
    },
    "maxcv_init": {
      "description": "Maximum constraint violation at the initial point."
    },
    "mb": {
      "description": "Number of bound constraints."
    },
    "mcon": {
      "description": "Total number of constraints (linear and nonlinear)."
    },
    "mlcon": {
      "description": "Total number of linear constraints (inequality and equality)."
    },
    "mnlcon": {
      "description": "Total number of nonlinear constraints (inequality and equality)."
    },
    "n": {
      "description": "Dimension of the problem."
    },
    "n_eval_ceq": {
      "description": "Number of nonlinear equality constraint evaluations."
    },
    "n_eval_cub": {
      "description": "Number of nonlinear inequality constraint evaluations."
    },
    "n_eval_fun": {
      "description": "Number of objective function evaluations."
    },
    "name": {
      "description": "Name of the problem."
    },
    "ptype": {
      "description": "Type of the problem."
    },
    "x0": {
      "description": "Initial guess."
    },
    "xl": {
      "description": "Lower bounds on the variables."
    },
    "xu": {
      "description": "Upper bounds on the variables."
    }
  },
  "see_also": [
    {
      "description": "Optimization problem.",
      "name": [
        "Problem",
        null
      ]
    },
    {
      "description": "Feature applied to problems during benchmarking.",
      "name": [
        "Feature",
        null
      ]
    },
    {
      "description": "Main benchmarking function.",
      "name": [
        "benchmark",
        null
      ]
    }
  ],
  "signature": "(problem, feature, max_eval, seed=None)"
}
```

## Canonical JSON Mirror

```json
{
  "Feature": {
    "description": "Mapping from an optimization problem to a new one with specified features. We are interested in testing solvers on problems with different features. For example, we may want to test the performance of solvers when the objective function is noisy. For this purpose, we define the ``Feature`` class.  Suppose we have an optimization problem  .. math::  \\min \\quad & \\mathrm{fun}(x) \\\\ \\text{s.t.} \\quad & x_l \\le x \\le x_u, \\\\ & A_{\\mathrm{ub}} x \\le b_{\\mathrm{ub}}, \\\\ & A_{\\mathrm{eq}} x = b_{\\mathrm{eq}}, \\\\ & c_{\\mathrm{ub}}(x) \\le 0, \\\\ & c_{\\mathrm{eq}}(x) = 0, \\\\ & \\text{with initial point } x_0.  Then ``Feature`` maps the above problem to the following one:  .. math::  \\min \\quad & \\mathrm{fun\\_mod}(Ax + b) \\\\ \\text{s.t.} \\quad & x_{l,\\mathrm{mod}} \\le Ax + b \\le x_{u,\\mathrm{mod}}, \\\\ & A_{\\mathrm{ub,mod}} (Ax + b) \\le b_{\\mathrm{ub,mod}}, \\\\ & A_{\\mathrm{eq,mod}} (Ax + b) = b_{\\mathrm{eq,mod}}, \\\\ & c_{\\mathrm{ub,mod}}(Ax + b) \\le 0, \\\\ & c_{\\mathrm{eq,mod}}(Ax + b) = 0, \\\\ & \\text{with initial guess } x_{0,\\mathrm{mod}},  where the modified quantities are determined by the chosen feature name and options.",
    "methods": {
      "chebyshev_noise_map": {
        "description": "",
        "signature": "(x)"
      },
      "get_default_rng": {
        "description": "Generate a random number generator.",
        "signature": "(seed, *args)"
      },
      "modifier_affine": {
        "description": "Generate an invertible matrix A and a vector b for the affine transformation applied to the variables.",
        "signature": "(self, seed, problem)"
      },
      "modifier_bounds": {
        "description": "Modify the bounds.",
        "signature": "(self, seed, problem)"
      },
      "modifier_ceq": {
        "description": "Modify the values of the nonlinear equality constraints.",
        "signature": "(self, x, seed, problem, n_eval_ceq)"
      },
      "modifier_cub": {
        "description": "Modify the values of the nonlinear inequality constraints.",
        "signature": "(self, x, seed, problem, n_eval_cub)"
      },
      "modifier_fun": {
        "description": "Modify the objective function value.",
        "signature": "(self, x, seed, problem, n_eval)"
      },
      "modifier_linear_eq": {
        "description": "Modify the linear equality constraints.",
        "signature": "(self, seed, problem)"
      },
      "modifier_linear_ub": {
        "description": "Modify the linear inequality constraints.",
        "signature": "(self, seed, problem)"
      },
      "modifier_x0": {
        "description": "Modify the initial point.",
        "signature": "(self, seed, problem)"
      }
    },
    "name": "Feature",
    "notes": "Different feature names accept different subsets of options. The valid options for each feature name are:  1. ``'plain'`` : ``n_runs``. 2. ``'perturbed_x0'`` : ``n_runs``, ``distribution``, ``perturbation_level``. 3. ``'noisy'`` : ``n_runs``, ``distribution``, ``noise_level``, ``noise_type``, ``noise_mode``, ``noise_map``. 4. ``'truncated'`` : ``n_runs``, ``significant_digits``, ``perturbed_trailing_digits``. 5. ``'permuted'`` : ``n_runs``. 6. ``'linearly_transformed'`` : ``n_runs``, ``rotated``, ``condition_factor``. 7. ``'random_nan'`` : ``n_runs``, ``nan_rate``. 8. ``'unrelaxable_constraints'`` : ``n_runs``, ``unrelaxable_bounds``, ``unrelaxable_linear_constraints``, ``unrelaxable_nonlinear_constraints``. 9. ``'nonquantifiable_constraints'`` : ``n_runs``. 10. ``'quantized'`` : ``n_runs``, ``mesh_size``, ``mesh_type``, ``ground_truth``. 11. ``'custom'`` : ``n_runs``, ``mod_x0``, ``mod_affine``, ``mod_bounds``, ``mod_linear_ub``, ``mod_linear_eq``, ``mod_fun``, ``mod_cub``, ``mod_ceq``.",
    "parameters": {
      "feature_options": {
        "choices": [
          "spherical",
          "gaussian"
        ],
        "default": "``5`` for stochastic features and ``1`` for deterministic features",
        "description": "Keyword arguments passed after ``name``. The available options depend on the chosen ``name``:  - **n_runs** (*int*) -- Number of runs of the experiment under the given feature. Default is ``5`` for stochastic features and ``1`` for deterministic features. Valid for all features. - **distribution** (*str or callable*) -- Distribution of perturbation (``'perturbed_x0'``) or noise (``'noisy'``). For ``'perturbed_x0'``, it should be ``'spherical'`` (default) or ``'gaussian'``. For ``'noisy'``, it should be ``'gaussian'`` (default) or ``'uniform'``. It can also be a callable ``(rng, dimension) -> array``. - **perturbation_level** (*float*) -- Magnitude of the perturbation in ``'perturbed_x0'``. Default is ``1e-3``. - **noise_level** (*float*) -- Magnitude of the noise in ``'noisy'``. Default is ``1e-3``. - **noise_type** (*str*) -- Type of the noise in ``'noisy'``. Must be ``'absolute'``, ``'relative'``, or ``'mixed'`` (default). - **noise_mode** (*str*) -- Mode of the noise in ``'noisy'``. Must be ``'random'`` (default) or ``'deterministic'``. - **noise_map** (*str or callable*) -- Deterministic scalar noise map in ``'noisy'``. It should be ``'chebyshev'`` (default) or a callable ``x -> noise`` returning a real scalar. It is used only when ``noise_mode`` is ``'deterministic'``. The built-in ``'chebyshev'`` map follows the deterministic noise model in Moré and Wild, \"Benchmarking derivative-free optimization algorithms\" (2009). - **significant_digits** (*int*) -- Number of significant digits in ``'truncated'``. Default is ``6``. - **perturbed_trailing_digits** (*bool*) -- Whether to randomize the trailing digits in ``'truncated'``. Default is ``False``. - **rotated** (*bool*) -- Whether to use a random rotation matrix in ``'linearly_transformed'``. Default is ``True``. - **condition_factor** (*float*) -- Scaling factor of the condition number of the linear transformation in ``'linearly_transformed'``. The condition number will be ``2^(condition_factor * n / 2)``. Default is ``0``. - **nan_rate** (*float*) -- Probability that an evaluation returns ``NaN`` in ``'random_nan'``. Default is ``0.05``. - **unrelaxable_bounds** (*bool*) -- Whether bound constraints are unrelaxable in ``'unrelaxable_constraints'``. Default is ``True``. - **unrelaxable_linear_constraints** (*bool*) -- Whether linear constraints are unrelaxable. Default is ``False``. - **unrelaxable_nonlinear_constraints** (*bool*) -- Whether nonlinear constraints are unrelaxable. Default is ``False``. - **mesh_size** (*float*) -- Size of the mesh in ``'quantized'``. Default is ``1e-3``. - **mesh_type** (*str*) -- Type of the mesh in ``'quantized'``. Must be ``'absolute'`` (default) or ``'relative'``. - **ground_truth** (*bool*) -- Whether the featured problem is the ground truth in ``'quantized'``. Default is ``True``. - **mod_x0** (*callable*) -- Modifier for the initial guess in ``'custom'``: ``(rng, problem) -> modified_x0``. - **mod_affine** (*callable*) -- Modifier for the affine transformation in ``'custom'``: ``(rng, problem) -> (A, b, inv)``. - **mod_bounds** (*callable*) -- Modifier for the bounds in ``'custom'``: ``(rng, problem) -> (xl, xu)``. - **mod_linear_ub** (*callable*) -- Modifier for the linear inequality constraints in ``'custom'``: ``(rng, problem) -> (aub, bub)``. - **mod_linear_eq** (*callable*) -- Modifier for the linear equality constraints in ``'custom'``: ``(rng, problem) -> (aeq, beq)``. - **mod_fun** (*callable*) -- Modifier for the objective function in ``'custom'``: ``(x, rng, problem) -> modified_fun``. - **mod_cub** (*callable*) -- Modifier for the nonlinear inequality constraints in ``'custom'``: ``(x, rng, problem) -> modified_cub``. - **mod_ceq** (*callable*) -- Modifier for the nonlinear equality constraints in ``'custom'``: ``(x, rng, problem) -> modified_ceq``.",
        "type": "dict"
      },
      "name": {
        "description": "Name of the feature. Must be one of the following:  1. ``'plain'`` : do nothing to the optimization problem. 2. ``'perturbed_x0'`` : perturb the initial guess ``x0``. 3. ``'noisy'`` : add noise to the objective function and nonlinear constraints. 4. ``'truncated'`` : truncate values of the objective function and nonlinear constraints to a given number of significant digits. 5. ``'permuted'`` : randomly permute the variables. The bounds and linear constraints are modified accordingly so that the new problem is mathematically equivalent to the original one. 6. ``'linearly_transformed'`` : apply an invertible linear transformation ``D @ Q'`` (with ``D`` diagonal and ``Q`` orthogonal) to the variables. Bounds and linear constraints are modified accordingly. 7. ``'random_nan'`` : randomly replace values of the objective function and nonlinear constraints with ``NaN``. 8. ``'unrelaxable_constraints'`` : set the objective function to ``Inf`` outside the feasible region. 9. ``'nonquantifiable_constraints'`` : replace values of nonlinear constraints with either ``0`` (satisfied) or ``1`` (violated). 10. ``'quantized'`` : quantize the objective function and nonlinear constraints. 11. ``'custom'`` : user-defined feature.",
        "type": "str"
      }
    },
    "properties": {
      "is_stochastic": {
        "description": "Whether the feature is stochastic."
      },
      "name": {
        "description": "Name of the feature."
      },
      "options": {
        "description": "Options of the feature."
      }
    },
    "see_also": [
      {
        "description": "Optimization problem.",
        "name": [
          "Problem",
          null
        ]
      },
      {
        "description": "Problem equipped with a specific feature.",
        "name": [
          "FeaturedProblem",
          null
        ]
      },
      {
        "description": "Main benchmarking function.",
        "name": [
          "benchmark",
          null
        ]
      }
    ],
    "signature": "(name, **feature_options)"
  },
  "FeaturedProblem": {
    "description": "Subclass of `Problem` that equips an optimization problem with a feature. ``Problem`` and its subclass ``FeaturedProblem`` describe the following optimization problem:  .. math::  \\min \\quad & \\mathrm{fun}(x) \\\\ \\text{s.t.} \\quad & x_l \\le x \\le x_u, \\\\ & A_{\\mathrm{ub}} x \\le b_{\\mathrm{ub}}, \\\\ & A_{\\mathrm{eq}} x = b_{\\mathrm{eq}}, \\\\ & c_{\\mathrm{ub}}(x) \\le 0, \\\\ & c_{\\mathrm{eq}}(x) = 0, \\\\ & \\text{with initial point } x_0.",
    "methods": {
      "ceq": {
        "description": "Evaluate the nonlinear constraints ``ceq(x) == 0``.",
        "signature": "(self, x, record_hist=True)"
      },
      "cub": {
        "description": "Evaluate the nonlinear constraints ``cub(x) <= 0``.",
        "signature": "(self, x, record_hist=True)"
      },
      "fun": {
        "description": "Evaluate the objective function.",
        "signature": "(self, x)"
      },
      "grad": {
        "description": "Evaluate the gradient of the objective function.",
        "signature": "(self, x)"
      },
      "hceq": {
        "description": "Evaluate the Hessian of the nonlinear equality constraints.",
        "signature": "(self, x)"
      },
      "hcub": {
        "description": "Evaluate the Hessian of the nonlinear inequality constraints.",
        "signature": "(self, x)"
      },
      "hess": {
        "description": "Evaluate the Hessian of the objective function.",
        "signature": "(self, x)"
      },
      "jceq": {
        "description": "Evaluate the Jacobian of the nonlinear equality constraints.",
        "signature": "(self, x)"
      },
      "jcub": {
        "description": "Evaluate the Jacobian of the nonlinear inequality constraints.",
        "signature": "(self, x)"
      },
      "maxcv": {
        "description": "Evaluate the maximum constraint violation.",
        "signature": "(self, x)"
      },
      "project_x0": {
        "description": "Project the initial guess onto the feasible region.",
        "signature": "(self)"
      }
    },
    "name": "FeaturedProblem",
    "notes": "``FeaturedProblem`` inherits all methods of ``Problem``, but the methods ``fun``, ``cub``, ``ceq``, and ``maxcv`` are modified by the input ``Feature``.  1. When the number of function evaluations reaches ``max_eval``, the methods ``fun``, ``cub``, and ``ceq`` will return the values at the point where the maximum number of function evaluations was reached. 2. When the number of function evaluations reaches ``termination_eval`` (used internally by ``benchmark``), the methods ``fun``, ``cub``, and ``ceq`` will raise an error to terminate the optimization process.  .. note::  For consistency with the rest of OptiProfiler, we recommend defining callables (such as ``fun``) with ``def`` rather than ``lambda``. Lambda expressions are not picklable, which prevents parallel execution when such callables are eventually passed to :func:`~optiprofiler.benchmark` with ``n_jobs > 1``. See :ref:`py_callable_picklability` for details.",
    "parameters": {
      "feature": {
        "description": "The feature to apply to the problem.",
        "type": "Feature"
      },
      "max_eval": {
        "description": "Maximum number of function evaluations.",
        "type": "int"
      },
      "problem": {
        "description": "The original optimization problem.",
        "type": "Problem"
      },
      "seed": {
        "description": "Nonnegative integer seed for the random number generator.",
        "type": "int"
      }
    },
    "properties": {
      "aeq": {
        "description": "Coefficient matrix of the linear constraints ``aeq @ x == beq``."
      },
      "aub": {
        "description": "Coefficient matrix of the linear constraints ``aub @ x <= bub``."
      },
      "beq": {
        "description": "Right-hand side of the linear constraints ``aeq @ x == beq``."
      },
      "bub": {
        "description": "Right-hand side of the linear constraints ``aub @ x <= bub``."
      },
      "ceq_hist": {
        "description": "History of nonlinear equality constraints."
      },
      "cub_hist": {
        "description": "History of nonlinear inequality constraints."
      },
      "fun_hist": {
        "description": "History of objective function values."
      },
      "fun_init": {
        "description": "Objective function value at the initial point."
      },
      "m_linear_eq": {
        "description": "Number of linear equality constraints."
      },
      "m_linear_ub": {
        "description": "Number of linear inequality constraints."
      },
      "m_nonlinear_eq": {
        "description": "Number of nonlinear equality constraints."
      },
      "m_nonlinear_ub": {
        "description": "Number of nonlinear inequality constraints."
      },
      "maxcv_hist": {
        "description": "History of maximum constraint violations."
      },
      "maxcv_init": {
        "description": "Maximum constraint violation at the initial point."
      },
      "mb": {
        "description": "Number of bound constraints."
      },
      "mcon": {
        "description": "Total number of constraints (linear and nonlinear)."
      },
      "mlcon": {
        "description": "Total number of linear constraints (inequality and equality)."
      },
      "mnlcon": {
        "description": "Total number of nonlinear constraints (inequality and equality)."
      },
      "n": {
        "description": "Dimension of the problem."
      },
      "n_eval_ceq": {
        "description": "Number of nonlinear equality constraint evaluations."
      },
      "n_eval_cub": {
        "description": "Number of nonlinear inequality constraint evaluations."
      },
      "n_eval_fun": {
        "description": "Number of objective function evaluations."
      },
      "name": {
        "description": "Name of the problem."
      },
      "ptype": {
        "description": "Type of the problem."
      },
      "x0": {
        "description": "Initial guess."
      },
      "xl": {
        "description": "Lower bounds on the variables."
      },
      "xu": {
        "description": "Upper bounds on the variables."
      }
    },
    "see_also": [
      {
        "description": "Optimization problem.",
        "name": [
          "Problem",
          null
        ]
      },
      {
        "description": "Feature applied to problems during benchmarking.",
        "name": [
          "Feature",
          null
        ]
      },
      {
        "description": "Main benchmarking function.",
        "name": [
          "benchmark",
          null
        ]
      }
    ],
    "signature": "(problem, feature, max_eval, seed=None)"
  },
  "Problem": {
    "description": "Optimization problem to be used in the benchmarking. ``Problem`` describes an optimization problem with the following structure:  .. math::  \\min \\quad & \\mathrm{fun}(x) \\\\ \\text{s.t.} \\quad & x_l \\le x \\le x_u, \\\\ & A_{\\mathrm{ub}} x \\le b_{\\mathrm{ub}}, \\\\ & A_{\\mathrm{eq}} x = b_{\\mathrm{eq}}, \\\\ & c_{\\mathrm{ub}}(x) \\le 0, \\\\ & c_{\\mathrm{eq}}(x) = 0, \\\\ & \\text{with initial point } x_0,  where ``fun`` is the objective function, ``x`` is the variable to optimize, ``xl`` and ``xu`` are the lower and upper bounds, ``aub`` and ``bub`` are the coefficient matrix and right-hand side vector of the linear inequality constraints, ``aeq`` and ``beq`` are the coefficient matrix and right-hand side vector of the linear equality constraints, ``cub`` is the function of nonlinear inequality constraints, and ``ceq`` is the function of nonlinear equality constraints.",
    "methods": {
      "ceq": {
        "description": "Evaluate the nonlinear constraints ``ceq(x) == 0``.",
        "signature": "(self, x)"
      },
      "cub": {
        "description": "Evaluate the nonlinear constraints ``cub(x) <= 0``.",
        "signature": "(self, x)"
      },
      "fun": {
        "description": "Evaluate the objective function.",
        "signature": "(self, x)"
      },
      "grad": {
        "description": "Evaluate the gradient of the objective function.",
        "signature": "(self, x)"
      },
      "hceq": {
        "description": "Evaluate the Hessian of the nonlinear equality constraints.",
        "signature": "(self, x)"
      },
      "hcub": {
        "description": "Evaluate the Hessian of the nonlinear inequality constraints.",
        "signature": "(self, x)"
      },
      "hess": {
        "description": "Evaluate the Hessian of the objective function.",
        "signature": "(self, x)"
      },
      "jceq": {
        "description": "Evaluate the Jacobian of the nonlinear equality constraints.",
        "signature": "(self, x)"
      },
      "jcub": {
        "description": "Evaluate the Jacobian of the nonlinear inequality constraints.",
        "signature": "(self, x)"
      },
      "maxcv": {
        "description": "Evaluate the maximum constraint violation.",
        "signature": "(self, x)"
      },
      "project_x0": {
        "description": "Project the initial guess onto the feasible region.",
        "signature": "(self)"
      }
    },
    "name": "Problem",
    "parameters": {
      "aeq": {
        "default": "an empty matrix",
        "description": "Coefficient matrix of the linear equality constraints ``aeq @ x == beq``. Default is an empty matrix.",
        "type": "array_like, shape (m_linear_eq, n)"
      },
      "aub": {
        "default": "an empty matrix",
        "description": "Coefficient matrix of the linear inequality constraints ``aub @ x <= bub``. Default is an empty matrix.",
        "type": "array_like, shape (m_linear_ub, n)"
      },
      "beq": {
        "default": "an empty vector",
        "description": "Right-hand side of the linear equality constraints ``aeq @ x == beq``. Default is an empty vector.",
        "type": "array_like, shape (m_linear_eq,)"
      },
      "bub": {
        "default": "an empty vector",
        "description": "Right-hand side of the linear inequality constraints ``aub @ x <= bub``. Default is an empty vector.",
        "type": "array_like, shape (m_linear_ub,)"
      },
      "ceq": {
        "description": "Nonlinear equality constraints ``ceq(x) == 0``: ``ceq(x) -> array_like, shape (m_nonlinear_eq,)``. Default returns an empty array.",
        "type": "callable"
      },
      "cub": {
        "description": "Nonlinear inequality constraints ``cub(x) <= 0``: ``cub(x) -> array_like, shape (m_nonlinear_ub,)``. Default returns an empty array.",
        "type": "callable"
      },
      "fun": {
        "description": "Objective function to be minimized: ``fun(x) -> float``, where ``x`` is an array with shape ``(n,)``.",
        "type": "callable"
      },
      "grad": {
        "description": "Gradient of the objective function: ``grad(x) -> array, shape (n,)``. Default returns an empty array.",
        "type": "callable"
      },
      "hceq": {
        "description": "Hessians of the nonlinear equality constraints: ``hceq(x) -> list of arrays, each shape (n, n)``. Default returns an empty list.",
        "type": "callable"
      },
      "hcub": {
        "description": "Hessians of the nonlinear inequality constraints: ``hcub(x) -> list of arrays, each shape (n, n)``. The *i*-th element is the Hessian of the *i*-th constraint in ``cub``. Default returns an empty list.",
        "type": "callable"
      },
      "hess": {
        "description": "Hessian of the objective function: ``hess(x) -> array, shape (n, n)``. Default returns an empty matrix.",
        "type": "callable"
      },
      "jceq": {
        "description": "Jacobian of the nonlinear equality constraints: ``jceq(x) -> array, shape (m_nonlinear_eq, n)``. Default returns an empty matrix.",
        "type": "callable"
      },
      "jcub": {
        "description": "Jacobian of the nonlinear inequality constraints: ``jcub(x) -> array, shape (m_nonlinear_ub, n)``. The number of columns must equal ``n`` and the number of rows must equal ``m_nonlinear_ub``. Default returns an empty matrix.",
        "type": "callable"
      },
      "name": {
        "default": "``'Unnamed Problem'``",
        "description": "Name of the problem. Default is ``'Unnamed Problem'``.",
        "type": "str"
      },
      "x0": {
        "description": "Initial guess.",
        "type": "array_like, shape (n,)"
      },
      "xl": {
        "default": "``-numpy.inf`` for each component",
        "description": "Lower bounds on the variables ``xl <= x``. Default is ``-numpy.inf`` for each component.",
        "type": "array_like, shape (n,)"
      },
      "xu": {
        "default": "``numpy.inf`` for each component",
        "description": "Upper bounds on the variables ``x <= xu``. Default is ``numpy.inf`` for each component.",
        "type": "array_like, shape (n,)"
      }
    },
    "properties": {
      "aeq": {
        "description": "Coefficient matrix of the linear constraints ``aeq @ x == beq``."
      },
      "aub": {
        "description": "Coefficient matrix of the linear constraints ``aub @ x <= bub``."
      },
      "beq": {
        "description": "Right-hand side of the linear constraints ``aeq @ x == beq``."
      },
      "bub": {
        "description": "Right-hand side of the linear constraints ``aub @ x <= bub``."
      },
      "m_linear_eq": {
        "description": "Number of linear equality constraints."
      },
      "m_linear_ub": {
        "description": "Number of linear inequality constraints."
      },
      "m_nonlinear_eq": {
        "description": "Number of nonlinear equality constraints."
      },
      "m_nonlinear_ub": {
        "description": "Number of nonlinear inequality constraints."
      },
      "mb": {
        "description": "Number of bound constraints."
      },
      "mcon": {
        "description": "Total number of constraints (linear and nonlinear)."
      },
      "mlcon": {
        "description": "Total number of linear constraints (inequality and equality)."
      },
      "mnlcon": {
        "description": "Total number of nonlinear constraints (inequality and equality)."
      },
      "n": {
        "description": "Dimension of the problem."
      },
      "name": {
        "description": "Name of the problem."
      },
      "ptype": {
        "description": "Type of the problem."
      },
      "x0": {
        "description": "Initial guess."
      },
      "xl": {
        "description": "Lower bounds on the variables."
      },
      "xu": {
        "description": "Upper bounds on the variables."
      }
    },
    "see_also": [
      {
        "description": "Feature applied to problems during benchmarking.",
        "name": [
          "Feature",
          null
        ]
      },
      {
        "description": "Problem equipped with a specific feature.",
        "name": [
          "FeaturedProblem",
          null
        ]
      },
      {
        "description": "Main benchmarking function.",
        "name": [
          "benchmark",
          null
        ]
      }
    ],
    "signature": "(fun, x0, name=None, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, grad=None, hess=None, jcub=None, jceq=None, hcub=None, hceq=None)"
  }
}
```
