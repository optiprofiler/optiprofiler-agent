---
tags: [reference, source-backed, python, benchmark]
sources: [_sources/python/benchmark.json]
related: [../api/python/benchmark.md]
last_updated: 2026-06-07
generated: true
---

# Source Reference: Python benchmark.json

This page is auto-generated from `_sources/python/benchmark.json`. It is the lossless wiki mirror for this source.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.

## Source Metadata

- Source path: `_sources/python/benchmark.json`
- Canonical SHA256: `3344b987704f5783c868e6fb42fc9cac37e016efa6d33383f0e1db06fa3c8440`
- Top-level keys: `description`, `signature`, `calling_convention`, `parameters`, `feature_options`, `profile_options`, `problem_options`, `returns`, `output_artifacts`, `raises`, `notes`, `see_also`, `name`, `solver_signatures`, `solver_notes`

## Path Index

| Path | Kind |
|---|---|
| `description` | str |
| `signature` | str |
| `calling_convention` | dict[3] |
| `parameters` | dict[1] |
| `parameters.solvers` | dict[2] |
| `feature_options` | dict[25] |
| `feature_options.feature_name` | dict[4] |
| `feature_options.n_runs` | dict[3] |
| `feature_options.distribution` | dict[3] |
| `feature_options.perturbation_level` | dict[3] |
| `feature_options.noise_level` | dict[3] |
| `feature_options.noise_type` | dict[4] |
| `feature_options.significant_digits` | dict[3] |
| `feature_options.perturbed_trailing_digits` | dict[3] |
| `feature_options.rotated` | dict[3] |
| `feature_options.condition_factor` | dict[3] |
| `feature_options.nan_rate` | dict[3] |
| `feature_options.unrelaxable_bounds` | dict[3] |
| `feature_options.unrelaxable_linear_constraints` | dict[3] |
| `feature_options.unrelaxable_nonlinear_constraints` | dict[3] |
| `feature_options.mesh_size` | dict[3] |
| `feature_options.mesh_type` | dict[4] |
| `feature_options.ground_truth` | dict[3] |
| `feature_options.mod_x0` | dict[2] |
| `feature_options.mod_affine` | dict[2] |
| `feature_options.mod_bounds` | dict[2] |
| `feature_options.mod_linear_ub` | dict[2] |
| `feature_options.mod_linear_eq` | dict[2] |
| `feature_options.mod_fun` | dict[2] |
| `feature_options.mod_cub` | dict[2] |
| `feature_options.mod_ceq` | dict[2] |
| `profile_options` | dict[38] |
| `profile_options.bar_colors` | dict[4] |
| `profile_options.benchmark_id` | dict[3] |
| `profile_options.draw_hist_plots` | dict[4] |
| `profile_options.errorbar_type` | dict[4] |
| `profile_options.feature_stamp` | dict[2] |
| `profile_options.hist_aggregation` | dict[4] |
| `profile_options.line_colors` | dict[2] |
| `profile_options.line_styles` | dict[3] |
| `profile_options.line_widths` | dict[3] |
| `profile_options.load` | dict[3] |
| `profile_options.max_eval_factor` | dict[3] |
| `profile_options.max_tol_order` | dict[3] |
| `profile_options.merit_fun` | dict[2] |
| `profile_options.n_jobs` | dict[3] |
| `profile_options.normalized_scores` | dict[3] |
| `profile_options.project_x0` | dict[3] |
| `profile_options.run_plain` | dict[3] |
| `profile_options.savepath` | dict[3] |
| `profile_options.score_fun` | dict[2] |
| `profile_options.score_only` | dict[3] |
| `profile_options.score_weight_fun` | dict[3] |
| `profile_options.seed` | dict[3] |
| `profile_options.semilogx` | dict[3] |
| `profile_options.silent` | dict[3] |
| `profile_options.solver_isrand` | dict[3] |
| `profile_options.solver_names` | dict[3] |
| `profile_options.solver_verbose` | dict[3] |
| `profile_options.solvers_to_load` | dict[3] |
| `profile_options.summarize_data_profiles` | dict[3] |
| `profile_options.summarize_log_ratio_profiles` | dict[3] |
| `profile_options.summarize_output_based_profiles` | dict[3] |
| `profile_options.summarize_performance_profiles` | dict[3] |
| `profile_options.xlabel_data_profile` | dict[3] |
| `profile_options.xlabel_log_ratio_profile` | dict[3] |
| `profile_options.xlabel_performance_profile` | dict[3] |
| `profile_options.ylabel_data_profile` | dict[3] |
| `profile_options.ylabel_log_ratio_profile` | dict[3] |
| `profile_options.ylabel_performance_profile` | dict[3] |
| `problem_options` | dict[16] |
| `problem_options.plibs` | dict[3] |
| `problem_options.ptype` | dict[4] |
| `problem_options.mindim` | dict[3] |
| `problem_options.maxdim` | dict[3] |
| `problem_options.minb` | dict[3] |
| `problem_options.maxb` | dict[3] |
| `problem_options.minlcon` | dict[3] |
| `problem_options.maxlcon` | dict[3] |
| `problem_options.minnlcon` | dict[3] |
| `problem_options.maxnlcon` | dict[3] |
| `problem_options.mincon` | dict[3] |
| `problem_options.maxcon` | dict[3] |
| `problem_options.custom_problem_libs_path` | dict[4] |
| `problem_options.excludelist` | dict[3] |
| `problem_options.problem_names` | dict[3] |
| `problem_options.problem` | dict[3] |
| `returns` | dict[3] |
| `returns.solver_scores` | dict[2] |
| `returns.profile_scores` | dict[2] |
| `returns.curves` | dict[2] |
| `output_artifacts` | dict[4] |
| `raises` | list[2] |
| `notes` | str |
| `see_also` | list[3] |
| `name` | str |
| `solver_signatures` | dict[4] |
| `solver_signatures.unconstrained` | str |
| `solver_signatures.bound_constrained` | str |
| `solver_signatures.linearly_constrained` | str |
| `solver_signatures.nonlinearly_constrained` | str |
| `solver_notes` | list[3] |

## description

```text
Benchmark optimization solvers on a set of problems with specified features. This function creates multiple profiles for benchmarking optimization solvers on a set of problems with different features. It generates performance profiles, data profiles, and log-ratio profiles [1]_, [2]_, [4]_, [5]_ for the given solvers on various test suites, returning solver scores based on the profiles.
```

## signature

```text
(solvers: 'list[callable] | None' = None, /, **kwargs) -> 'tuple[np.ndarray, np.ndarray | None, list[dict] | None]'
```

## calling_convention

```json
{
  "options": "keyword arguments to benchmark(). Example: benchmark(solvers, ptype='u', mindim=2)",
  "solvers": "list of callables: [solver1, solver2]",
  "syntax": "scores = benchmark([solver1, solver2], ptype='u', mindim=2, maxdim=20)"
}
```

## parameters

```json
{
  "solvers": {
    "description": "Solvers to benchmark. Each solver must be a callable accepting corresponding arguments depending on the test suite you choose:  - for an unconstrained problem, ``solver(fun, x0) -> numpy.ndarray, shape (n,)``, where ``fun`` is the objective function accepting a 1-D array and returning a float, and ``x0`` is the initial guess (1-D array); - for a bound-constrained problem, ``solver(fun, x0, xl, xu) -> numpy.ndarray, shape (n,)``, where ``xl`` and ``xu`` are the lower and upper bounds (1-D arrays, may contain ``-numpy.inf`` or ``numpy.inf``); - for a linearly constrained problem, ``solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray, shape (n,)``, where ``aub`` and ``aeq`` are the coefficient matrices of the linear inequality and equality constraints, and ``bub`` and ``beq`` are the right-hand side vectors; - for a nonlinearly constrained problem, ``solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray, shape (n,)``, where ``cub`` and ``ceq`` are the nonlinear inequality and equality constraint functions accepting a 1-D array and returning a 1-D array.  All vectors and matrices mentioned above are `numpy.ndarray`.  If the 'load' option is provided in ``**kwargs``, solvers can be None, in which case data from a previous experiment will be loaded to generate profiles.",
    "type": "list of callable if 'load' in ``**kwargs``"
  }
}
```

## parameters.solvers

```json
{
  "description": "Solvers to benchmark. Each solver must be a callable accepting corresponding arguments depending on the test suite you choose:  - for an unconstrained problem, ``solver(fun, x0) -> numpy.ndarray, shape (n,)``, where ``fun`` is the objective function accepting a 1-D array and returning a float, and ``x0`` is the initial guess (1-D array); - for a bound-constrained problem, ``solver(fun, x0, xl, xu) -> numpy.ndarray, shape (n,)``, where ``xl`` and ``xu`` are the lower and upper bounds (1-D arrays, may contain ``-numpy.inf`` or ``numpy.inf``); - for a linearly constrained problem, ``solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray, shape (n,)``, where ``aub`` and ``aeq`` are the coefficient matrices of the linear inequality and equality constraints, and ``bub`` and ``beq`` are the right-hand side vectors; - for a nonlinearly constrained problem, ``solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray, shape (n,)``, where ``cub`` and ``ceq`` are the nonlinear inequality and equality constraint functions accepting a 1-D array and returning a 1-D array.  All vectors and matrices mentioned above are `numpy.ndarray`.  If the 'load' option is provided in ``**kwargs``, solvers can be None, in which case data from a previous experiment will be loaded to generate profiles.",
  "type": "list of callable if 'load' in ``**kwargs``"
}
```

## feature_options

```json
{
  "condition_factor": {
    "default": "0",
    "description": "The scaling factor of the condition number of the linear transformation in the 'linearly_transformed' feature. More specifically, the condition number of the linear transformation will be 2 ** (condition_factor * n / 2), where n is the dimension of the problem. Default is 0.",
    "type": "float"
  },
  "distribution": {
    "default": "'spherical')",
    "description": "The distribution of perturbation in 'perturbed_x0' feature or noise in 'noisy' feature. It should be either a str (or char), or a callable ``(random_stream, dimension) -> random vector``, accepting a random_stream and the dimension of a problem and returning a random vector with the given dimension. In 'perturbed_x0' case, the str should be either 'spherical' or 'gaussian' (default is 'spherical'). In 'noisy' case, the str should be either 'gaussian' or 'uniform' (default is 'gaussian').",
    "type": "str or callable"
  },
  "feature_name": {
    "choices": [
      "plain",
      "perturbed_x0",
      "noisy",
      "truncated",
      "permuted",
      "linearly_transformed",
      "random_nan",
      "unrelaxable_constraints",
      "nonquantifiable_constraints",
      "quantized",
      "custom"
    ],
    "default": "'plain'",
    "description": "Name of the feature to apply to problems. The available features are 'plain', 'perturbed_x0', 'noisy', 'truncated', 'permuted', 'linearly_transformed', 'random_nan', 'unrelaxable_constraints', 'nonquantifiable_constraints', 'quantized', and 'custom'. Default is 'plain'.",
    "type": "str"
  },
  "ground_truth": {
    "default": "True",
    "description": "Whether the featured problem is the ground truth or not in the 'quantized' feature. Default is True.",
    "type": "bool"
  },
  "mesh_size": {
    "default": "1e-3",
    "description": "The size of the mesh in the 'quantized' feature. Default is 1e-3.",
    "type": "float"
  },
  "mesh_type": {
    "choices": [
      "absolute",
      "relative"
    ],
    "default": "'absolute'",
    "description": "The type of the mesh in the 'quantized' feature. It should be either 'absolute' or 'relative'. Default is 'absolute'.",
    "type": "str"
  },
  "mod_affine": {
    "description": "The modifier function to generate the affine transformation applied to the variables in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (A, b, inv)``, where problem is an instance of the class Problem, A is the matrix of the affine transformation, b is the vector of the affine transformation, and inv is the inverse of matrix A. No default.",
    "type": "callable"
  },
  "mod_bounds": {
    "description": "The modifier function to modify the bound constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_xl, modified_xu)``, where problem is an instance of the class Problem, modified_xl is the modified lower bound, and modified_xu is the modified upper bound. No default.",
    "type": "callable"
  },
  "mod_ceq": {
    "description": "The modifier function to modify the nonlinear equality constraints in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_ceq``, where x is the evaluation point, problem is an instance of the class Problem, and modified_ceq is the modified vector of the nonlinear equality constraints. No default.",
    "type": "callable"
  },
  "mod_cub": {
    "description": "The modifier function to modify the nonlinear inequality constraints in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_cub``, where x is the evaluation point, problem is an instance of the class Problem, and modified_cub is the modified vector of the nonlinear inequality constraints. No default.",
    "type": "callable"
  },
  "mod_fun": {
    "description": "The modifier function to modify the objective function in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_fun``, where x is the evaluation point, problem is an instance of the class Problem, and modified_fun is the modified objective function value. No default.",
    "type": "callable"
  },
  "mod_linear_eq": {
    "description": "The modifier function to modify the linear equality constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_aeq, modified_beq)``, where problem is an instance of the class Problem, modified_aeq is the modified matrix of the linear equality constraints, and modified_beq is the modified vector of the linear equality constraints. No default.",
    "type": "callable"
  },
  "mod_linear_ub": {
    "description": "The modifier function to modify the linear inequality constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_aub, modified_bub)``, where problem is an instance of the class Problem, modified_aub is the modified matrix of the linear inequality constraints, and modified_bub is the modified vector of the linear inequality constraints. No default.",
    "type": "callable"
  },
  "mod_x0": {
    "description": "The modifier function to modify the initial guess in the 'custom' feature. It should be a callable ``(random_stream, problem) -> modified_x0``, where problem is an instance of the class Problem, and modified_x0 is the modified initial guess. No default.",
    "type": "callable"
  },
  "n_runs": {
    "default": "5 for stochastic features and 1 for deterministic features",
    "description": "The number of runs of the experiments with the given feature. Default is 5 for stochastic features and 1 for deterministic features.",
    "type": "int"
  },
  "nan_rate": {
    "default": "0.05",
    "description": "The probability that the evaluation of the objective function will return np.nan in the 'random_nan' feature. Default is 0.05.",
    "type": "float"
  },
  "noise_level": {
    "default": "1e-3",
    "description": "The magnitude of the noise in the 'noisy' feature. Default is 1e-3.",
    "type": "float"
  },
  "noise_type": {
    "choices": [
      "absolute",
      "relative",
      "mixed"
    ],
    "default": "'mixed'",
    "description": "The type of the noise in the 'noisy' features. It should be either 'absolute', 'relative', or 'mixed'. Default is 'mixed'.",
    "type": "str"
  },
  "perturbation_level": {
    "default": "1e-3",
    "description": "The magnitude of the perturbation to the initial guess in the 'perturbed_x0' feature. Default is 1e-3.",
    "type": "float"
  },
  "perturbed_trailing_digits": {
    "default": "False",
    "description": "Whether we will randomize the trailing digits of the objective function value in the 'truncated' feature. Default is False.",
    "type": "bool"
  },
  "rotated": {
    "default": "True",
    "description": "Whether to use a random or given rotation matrix to rotate the coordinates of a problem in the 'linearly_transformed' feature. Default is True.",
    "type": "bool"
  },
  "significant_digits": {
    "default": "6",
    "description": "The number of significant digits in the 'truncated' feature. Default is 6.",
    "type": "int"
  },
  "unrelaxable_bounds": {
    "default": "True",
    "description": "Whether the bound constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is True.",
    "type": "bool"
  },
  "unrelaxable_linear_constraints": {
    "default": "False",
    "description": "Whether the linear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is False.",
    "type": "bool"
  },
  "unrelaxable_nonlinear_constraints": {
    "default": "False",
    "description": "Whether the nonlinear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is False.",
    "type": "bool"
  }
}
```

## feature_options.feature_name

```json
{
  "choices": [
    "plain",
    "perturbed_x0",
    "noisy",
    "truncated",
    "permuted",
    "linearly_transformed",
    "random_nan",
    "unrelaxable_constraints",
    "nonquantifiable_constraints",
    "quantized",
    "custom"
  ],
  "default": "'plain'",
  "description": "Name of the feature to apply to problems. The available features are 'plain', 'perturbed_x0', 'noisy', 'truncated', 'permuted', 'linearly_transformed', 'random_nan', 'unrelaxable_constraints', 'nonquantifiable_constraints', 'quantized', and 'custom'. Default is 'plain'.",
  "type": "str"
}
```

## feature_options.n_runs

```json
{
  "default": "5 for stochastic features and 1 for deterministic features",
  "description": "The number of runs of the experiments with the given feature. Default is 5 for stochastic features and 1 for deterministic features.",
  "type": "int"
}
```

## feature_options.distribution

```json
{
  "default": "'spherical')",
  "description": "The distribution of perturbation in 'perturbed_x0' feature or noise in 'noisy' feature. It should be either a str (or char), or a callable ``(random_stream, dimension) -> random vector``, accepting a random_stream and the dimension of a problem and returning a random vector with the given dimension. In 'perturbed_x0' case, the str should be either 'spherical' or 'gaussian' (default is 'spherical'). In 'noisy' case, the str should be either 'gaussian' or 'uniform' (default is 'gaussian').",
  "type": "str or callable"
}
```

## feature_options.perturbation_level

```json
{
  "default": "1e-3",
  "description": "The magnitude of the perturbation to the initial guess in the 'perturbed_x0' feature. Default is 1e-3.",
  "type": "float"
}
```

## feature_options.noise_level

```json
{
  "default": "1e-3",
  "description": "The magnitude of the noise in the 'noisy' feature. Default is 1e-3.",
  "type": "float"
}
```

## feature_options.noise_type

```json
{
  "choices": [
    "absolute",
    "relative",
    "mixed"
  ],
  "default": "'mixed'",
  "description": "The type of the noise in the 'noisy' features. It should be either 'absolute', 'relative', or 'mixed'. Default is 'mixed'.",
  "type": "str"
}
```

## feature_options.significant_digits

```json
{
  "default": "6",
  "description": "The number of significant digits in the 'truncated' feature. Default is 6.",
  "type": "int"
}
```

## feature_options.perturbed_trailing_digits

```json
{
  "default": "False",
  "description": "Whether we will randomize the trailing digits of the objective function value in the 'truncated' feature. Default is False.",
  "type": "bool"
}
```

## feature_options.rotated

```json
{
  "default": "True",
  "description": "Whether to use a random or given rotation matrix to rotate the coordinates of a problem in the 'linearly_transformed' feature. Default is True.",
  "type": "bool"
}
```

## feature_options.condition_factor

```json
{
  "default": "0",
  "description": "The scaling factor of the condition number of the linear transformation in the 'linearly_transformed' feature. More specifically, the condition number of the linear transformation will be 2 ** (condition_factor * n / 2), where n is the dimension of the problem. Default is 0.",
  "type": "float"
}
```

## feature_options.nan_rate

```json
{
  "default": "0.05",
  "description": "The probability that the evaluation of the objective function will return np.nan in the 'random_nan' feature. Default is 0.05.",
  "type": "float"
}
```

## feature_options.unrelaxable_bounds

```json
{
  "default": "True",
  "description": "Whether the bound constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is True.",
  "type": "bool"
}
```

## feature_options.unrelaxable_linear_constraints

```json
{
  "default": "False",
  "description": "Whether the linear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is False.",
  "type": "bool"
}
```

## feature_options.unrelaxable_nonlinear_constraints

```json
{
  "default": "False",
  "description": "Whether the nonlinear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is False.",
  "type": "bool"
}
```

## feature_options.mesh_size

```json
{
  "default": "1e-3",
  "description": "The size of the mesh in the 'quantized' feature. Default is 1e-3.",
  "type": "float"
}
```

## feature_options.mesh_type

```json
{
  "choices": [
    "absolute",
    "relative"
  ],
  "default": "'absolute'",
  "description": "The type of the mesh in the 'quantized' feature. It should be either 'absolute' or 'relative'. Default is 'absolute'.",
  "type": "str"
}
```

## feature_options.ground_truth

```json
{
  "default": "True",
  "description": "Whether the featured problem is the ground truth or not in the 'quantized' feature. Default is True.",
  "type": "bool"
}
```

## feature_options.mod_x0

```json
{
  "description": "The modifier function to modify the initial guess in the 'custom' feature. It should be a callable ``(random_stream, problem) -> modified_x0``, where problem is an instance of the class Problem, and modified_x0 is the modified initial guess. No default.",
  "type": "callable"
}
```

## feature_options.mod_affine

```json
{
  "description": "The modifier function to generate the affine transformation applied to the variables in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (A, b, inv)``, where problem is an instance of the class Problem, A is the matrix of the affine transformation, b is the vector of the affine transformation, and inv is the inverse of matrix A. No default.",
  "type": "callable"
}
```

## feature_options.mod_bounds

```json
{
  "description": "The modifier function to modify the bound constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_xl, modified_xu)``, where problem is an instance of the class Problem, modified_xl is the modified lower bound, and modified_xu is the modified upper bound. No default.",
  "type": "callable"
}
```

## feature_options.mod_linear_ub

```json
{
  "description": "The modifier function to modify the linear inequality constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_aub, modified_bub)``, where problem is an instance of the class Problem, modified_aub is the modified matrix of the linear inequality constraints, and modified_bub is the modified vector of the linear inequality constraints. No default.",
  "type": "callable"
}
```

## feature_options.mod_linear_eq

```json
{
  "description": "The modifier function to modify the linear equality constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_aeq, modified_beq)``, where problem is an instance of the class Problem, modified_aeq is the modified matrix of the linear equality constraints, and modified_beq is the modified vector of the linear equality constraints. No default.",
  "type": "callable"
}
```

## feature_options.mod_fun

```json
{
  "description": "The modifier function to modify the objective function in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_fun``, where x is the evaluation point, problem is an instance of the class Problem, and modified_fun is the modified objective function value. No default.",
  "type": "callable"
}
```

## feature_options.mod_cub

```json
{
  "description": "The modifier function to modify the nonlinear inequality constraints in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_cub``, where x is the evaluation point, problem is an instance of the class Problem, and modified_cub is the modified vector of the nonlinear inequality constraints. No default.",
  "type": "callable"
}
```

## feature_options.mod_ceq

```json
{
  "description": "The modifier function to modify the nonlinear equality constraints in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_ceq``, where x is the evaluation point, problem is an instance of the class Problem, and modified_ceq is the modified vector of the nonlinear equality constraints. No default.",
  "type": "callable"
}
```

## profile_options

```json
{
  "bar_colors": {
    "choices": [
      "r",
      "g",
      "b",
      "c",
      "m",
      "y",
      "k"
    ],
    "default": "set to the first two colors in the 'line_colors' option",
    "description": "Two different colors for the bars of two solvers in the log-ratio profiles. It can be a list of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a 2-by-3 array with each row being a RGB triplet. Default is set to the first two colors in the 'line_colors' option.",
    "type": "list or numpy.ndarray"
  },
  "benchmark_id": {
    "default": "'out' if the option 'load' is not provided, otherwise default is '.'",
    "description": "The identifier of the test. It is used to create the specific directory to store the results. Default is 'out' if the option 'load' is not provided, otherwise default is '.'.",
    "type": "str"
  },
  "draw_hist_plots": {
    "choices": [
      "none",
      "sequential",
      "parallel"
    ],
    "default": "'parallel'",
    "description": "Whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', we will not draw the history plots. If it is 'parallel', we will draw the history plots at the same time when solvers are solving the problems. If it is 'sequential', we will draw the history plots after all the problems are solved. Default is 'parallel'.",
    "type": "str"
  },
  "errorbar_type": {
    "choices": [
      "minmax",
      "meanstd"
    ],
    "default": "'minmax', meaning that we takes the pointwise minimum and maximum of the curves",
    "description": "The type of the uncertainty interval that can be either 'minmax' or 'meanstd'. When 'n_runs' is greater than 1, we run several times of the experiments and get average curves and uncertainty intervals. Default is 'minmax', meaning that we takes the pointwise minimum and maximum of the curves.",
    "type": "str"
  },
  "feature_stamp": {
    "description": "The stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features.",
    "type": "str"
  },
  "hist_aggregation": {
    "choices": [
      "min",
      "mean",
      "max"
    ],
    "default": "'min'",
    "description": "The aggregation method we use to reduce the number of points in the history plots. It can be 'min', 'mean', or 'max'. Default is 'min'.",
    "type": "str"
  },
  "line_colors": {
    "description": "The colors of the lines in the plots. It can be a list of any valid matplotlib colors (short names, hex strings, RGB tuples, etc.). Default line colors are from the matplotlib tab10 color cycle. Note that if the number of solvers is greater than the number of colors, we will cycle through the colors.",
    "type": "list"
  },
  "line_styles": {
    "choices": [
      "-",
      "-.",
      "--",
      ":",
      "o",
      "+",
      "*",
      ".",
      "x",
      "s",
      "d",
      "^",
      "v",
      ">",
      "<",
      "p",
      "h"
    ],
    "description": "The styles of the lines in the plots. It can be a list of strs that are the combinations of line styles ('-', '-.', '--', ':') and markers ('o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'). Default line style order is ['-', '-.', '--', ':']. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles.",
    "type": "list of str"
  },
  "line_widths": {
    "default": "1.5",
    "description": "The widths of the lines in the plots. It should be a positive float or a list. Default is 1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths.",
    "type": "float or list"
  },
  "load": {
    "choices": [
      "latest",
      "yyyyMMdd_HHmmss"
    ],
    "description": "Loading the stored data from a completed experiment and draw profiles. It can be either 'latest' or a time stamp of an experiment in the format of 'yyyyMMdd_HHmmss'. No default. Note that if solvers is None, this key must be provided to load data from a previous experiment and generate profiles.",
    "type": "str"
  },
  "max_eval_factor": {
    "default": "500",
    "description": "The factor multiplied to each problem's dimension to get the maximum number of evaluations for each problem. Default is 500.",
    "type": "int"
  },
  "max_tol_order": {
    "default": "10",
    "description": "The maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of 'tolerances' to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as ``10**(-k)`` for ``k = 1, 2, ..., max_tol_order``. Default is 10.",
    "type": "int"
  },
  "merit_fun": {
    "description": "The merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a callable ``(fun_value, maxcv_value, maxcv_init) -> merit_value``, where fun_value is the objective function value, maxcv_value is the maximum constraint violation, and maxcv_init is the maximum constraint violation at the initial guess. The default merit function varphi(x) is defined by the objective function f(x) and the maximum constraint violation v(x) as::  varphi(x) = f(x)                        if v(x) <= v1 varphi(x) = f(x) + 1e5 * (v(x) - v1)   if v1 < v(x) <= v2 varphi(x) = np.inf                       if v(x) > v2  where v1 = min(0.01, 1e-10 * max(1, v0)), v2 = max(0.1, 2 * v0), and v0 is the maximum constraint violation at the initial guess. If varphi(x_0) is inf for a problem/run, all solvers are declared to pass that degenerate convergence test, and the case is listed in test_log/report.txt.",
    "type": "callable"
  },
  "n_jobs": {
    "default": "about half of available workers, at least 2 when possible",
    "description": "The number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers, with at least 2 when more than one worker is available.",
    "type": "int"
  },
  "normalized_scores": {
    "default": "True",
    "description": "Whether to normalize the scores of the solvers by the maximum score of the solvers. Default is True.",
    "type": "bool"
  },
  "project_x0": {
    "default": "False",
    "description": "Whether to project the initial point to the feasible set. Default is False.",
    "type": "bool"
  },
  "run_plain": {
    "default": "False",
    "description": "Whether to run an extra experiment with the 'plain' feature. Default is False.",
    "type": "bool"
  },
  "savepath": {
    "default": "the current working directory",
    "description": "The path to store the results. Default is the current working directory.",
    "type": "str"
  },
  "score_fun": {
    "description": "The scoring function to calculate the scores of the solvers. It should be a callable ``profile_scores -> solver_scores``, where profile_scores is a 4D array containing scores for all profiles. The first dimension of profile_scores corresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring function takes the average of the history-based performance profiles under all the tolerances.",
    "type": "callable"
  },
  "score_only": {
    "default": "False",
    "description": "Whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default is False.",
    "type": "bool"
  },
  "score_weight_fun": {
    "default": "a constant function returning 1",
    "description": "The weight function to calculate the scores of the solvers in the performance and data profiles. It should be a callable representing a nonnegative function in R^+. Default is a constant function returning 1.",
    "type": "callable"
  },
  "seed": {
    "default": "0",
    "description": "The seed of the random number generator. Default is 0.",
    "type": "int"
  },
  "semilogx": {
    "default": "True",
    "description": "Whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default is True.",
    "type": "bool"
  },
  "silent": {
    "default": "False",
    "description": "Whether to show the information of the progress. Default is False.",
    "type": "bool"
  },
  "solver_isrand": {
    "default": "a list of bools of the same length as the number of solvers, where the value is True if the solver is randomized, and False otherwise",
    "description": "Whether the solvers are randomized or not. Default is a list of bools of the same length as the number of solvers, where the value is True if the solver is randomized, and False otherwise. Note that if 'n_runs' is not specified, we will set it 5 for the randomized solvers.",
    "type": "list of bool"
  },
  "solver_names": {
    "default": "the names of the callables in solvers",
    "description": "The names of the solvers. Default is the names of the callables in solvers.",
    "type": "list of str"
  },
  "solver_verbose": {
    "default": "1",
    "description": "The level of the verbosity of the solvers. 0 means no verbosity, 1 means some verbosity, and 2 means full verbosity. Default is 1.",
    "type": "int"
  },
  "solvers_to_load": {
    "default": "all the solvers",
    "description": "The indices of the solvers to load when the 'load' option is provided. It can be a list of different integers selected from 0 to the total number of solvers minus 1 of the loading experiment. At least two indices should be provided. Default is all the solvers.",
    "type": "list of int"
  },
  "summarize_data_profiles": {
    "default": "True",
    "description": "Whether to add all the data profiles to the summary PDF. Default is True.",
    "type": "bool"
  },
  "summarize_log_ratio_profiles": {
    "default": "False",
    "description": "Whether to add all the log-ratio profiles to the summary PDF. Default is False.",
    "type": "bool"
  },
  "summarize_output_based_profiles": {
    "default": "True",
    "description": "Whether to add all the output-based profiles of the selected profiles to the summary PDF. Default is True.",
    "type": "bool"
  },
  "summarize_performance_profiles": {
    "default": "True",
    "description": "Whether to add all the performance profiles to the summary PDF. Default is True.",
    "type": "bool"
  },
  "xlabel_data_profile": {
    "default": "'Number of simplex gradients'",
    "description": "The label of the x-axis of the data profiles. Default is 'Number of simplex gradients'. Note: LaTeX formatting is supported. The same applies to the options 'xlabel_log_ratio_profile', 'xlabel_performance_profile', 'ylabel_data_profile', 'ylabel_log_ratio_profile', and 'ylabel_performance_profile'.",
    "type": "str"
  },
  "xlabel_log_ratio_profile": {
    "default": "'Problem'",
    "description": "The label of the x-axis of the log-ratio profiles. Default is 'Problem'.",
    "type": "str"
  },
  "xlabel_performance_profile": {
    "default": "'Performance ratio'",
    "description": "The label of the x-axis of the performance profiles. Default is 'Performance ratio'.",
    "type": "str"
  },
  "ylabel_data_profile": {
    "default": "'Data profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
    "description": "The label of the y-axis of the data profiles. Default is 'Data profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format. You can also use '%s' in your custom label, and it will be replaced accordingly. The same applies to the options 'ylabel_log_ratio_profile' and 'ylabel_performance_profile'.",
    "type": "str"
  },
  "ylabel_log_ratio_profile": {
    "default": "'Log-ratio profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
    "description": "The label of the y-axis of the log-ratio profiles. Default is 'Log-ratio profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format.",
    "type": "str"
  },
  "ylabel_performance_profile": {
    "default": "'Performance profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
    "description": "The label of the y-axis of the performance profiles. Default is 'Performance profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format.",
    "type": "str"
  }
}
```

## profile_options.bar_colors

```json
{
  "choices": [
    "r",
    "g",
    "b",
    "c",
    "m",
    "y",
    "k"
  ],
  "default": "set to the first two colors in the 'line_colors' option",
  "description": "Two different colors for the bars of two solvers in the log-ratio profiles. It can be a list of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a 2-by-3 array with each row being a RGB triplet. Default is set to the first two colors in the 'line_colors' option.",
  "type": "list or numpy.ndarray"
}
```

## profile_options.benchmark_id

```json
{
  "default": "'out' if the option 'load' is not provided, otherwise default is '.'",
  "description": "The identifier of the test. It is used to create the specific directory to store the results. Default is 'out' if the option 'load' is not provided, otherwise default is '.'.",
  "type": "str"
}
```

## profile_options.draw_hist_plots

```json
{
  "choices": [
    "none",
    "sequential",
    "parallel"
  ],
  "default": "'parallel'",
  "description": "Whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', we will not draw the history plots. If it is 'parallel', we will draw the history plots at the same time when solvers are solving the problems. If it is 'sequential', we will draw the history plots after all the problems are solved. Default is 'parallel'.",
  "type": "str"
}
```

## profile_options.errorbar_type

```json
{
  "choices": [
    "minmax",
    "meanstd"
  ],
  "default": "'minmax', meaning that we takes the pointwise minimum and maximum of the curves",
  "description": "The type of the uncertainty interval that can be either 'minmax' or 'meanstd'. When 'n_runs' is greater than 1, we run several times of the experiments and get average curves and uncertainty intervals. Default is 'minmax', meaning that we takes the pointwise minimum and maximum of the curves.",
  "type": "str"
}
```

## profile_options.feature_stamp

```json
{
  "description": "The stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features.",
  "type": "str"
}
```

## profile_options.hist_aggregation

```json
{
  "choices": [
    "min",
    "mean",
    "max"
  ],
  "default": "'min'",
  "description": "The aggregation method we use to reduce the number of points in the history plots. It can be 'min', 'mean', or 'max'. Default is 'min'.",
  "type": "str"
}
```

## profile_options.line_colors

```json
{
  "description": "The colors of the lines in the plots. It can be a list of any valid matplotlib colors (short names, hex strings, RGB tuples, etc.). Default line colors are from the matplotlib tab10 color cycle. Note that if the number of solvers is greater than the number of colors, we will cycle through the colors.",
  "type": "list"
}
```

## profile_options.line_styles

```json
{
  "choices": [
    "-",
    "-.",
    "--",
    ":",
    "o",
    "+",
    "*",
    ".",
    "x",
    "s",
    "d",
    "^",
    "v",
    ">",
    "<",
    "p",
    "h"
  ],
  "description": "The styles of the lines in the plots. It can be a list of strs that are the combinations of line styles ('-', '-.', '--', ':') and markers ('o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'). Default line style order is ['-', '-.', '--', ':']. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles.",
  "type": "list of str"
}
```

## profile_options.line_widths

```json
{
  "default": "1.5",
  "description": "The widths of the lines in the plots. It should be a positive float or a list. Default is 1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths.",
  "type": "float or list"
}
```

## profile_options.load

```json
{
  "choices": [
    "latest",
    "yyyyMMdd_HHmmss"
  ],
  "description": "Loading the stored data from a completed experiment and draw profiles. It can be either 'latest' or a time stamp of an experiment in the format of 'yyyyMMdd_HHmmss'. No default. Note that if solvers is None, this key must be provided to load data from a previous experiment and generate profiles.",
  "type": "str"
}
```

## profile_options.max_eval_factor

```json
{
  "default": "500",
  "description": "The factor multiplied to each problem's dimension to get the maximum number of evaluations for each problem. Default is 500.",
  "type": "int"
}
```

## profile_options.max_tol_order

```json
{
  "default": "10",
  "description": "The maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of 'tolerances' to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as ``10**(-k)`` for ``k = 1, 2, ..., max_tol_order``. Default is 10.",
  "type": "int"
}
```

## profile_options.merit_fun

```json
{
  "description": "The merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a callable ``(fun_value, maxcv_value, maxcv_init) -> merit_value``, where fun_value is the objective function value, maxcv_value is the maximum constraint violation, and maxcv_init is the maximum constraint violation at the initial guess. The default merit function varphi(x) is defined by the objective function f(x) and the maximum constraint violation v(x) as::  varphi(x) = f(x)                        if v(x) <= v1 varphi(x) = f(x) + 1e5 * (v(x) - v1)   if v1 < v(x) <= v2 varphi(x) = np.inf                       if v(x) > v2  where v1 = min(0.01, 1e-10 * max(1, v0)), v2 = max(0.1, 2 * v0), and v0 is the maximum constraint violation at the initial guess. If varphi(x_0) is inf for a problem/run, all solvers are declared to pass that degenerate convergence test, and the case is listed in test_log/report.txt.",
  "type": "callable"
}
```

## profile_options.n_jobs

```json
{
  "default": "about half of available workers, at least 2 when possible",
  "description": "The number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers, with at least 2 when more than one worker is available.",
  "type": "int"
}
```

## profile_options.normalized_scores

```json
{
  "default": "True",
  "description": "Whether to normalize the scores of the solvers by the maximum score of the solvers. Default is True.",
  "type": "bool"
}
```

## profile_options.project_x0

```json
{
  "default": "False",
  "description": "Whether to project the initial point to the feasible set. Default is False.",
  "type": "bool"
}
```

## profile_options.run_plain

```json
{
  "default": "False",
  "description": "Whether to run an extra experiment with the 'plain' feature. Default is False.",
  "type": "bool"
}
```

## profile_options.savepath

```json
{
  "default": "the current working directory",
  "description": "The path to store the results. Default is the current working directory.",
  "type": "str"
}
```

## profile_options.score_fun

```json
{
  "description": "The scoring function to calculate the scores of the solvers. It should be a callable ``profile_scores -> solver_scores``, where profile_scores is a 4D array containing scores for all profiles. The first dimension of profile_scores corresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring function takes the average of the history-based performance profiles under all the tolerances.",
  "type": "callable"
}
```

## profile_options.score_only

```json
{
  "default": "False",
  "description": "Whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default is False.",
  "type": "bool"
}
```

## profile_options.score_weight_fun

```json
{
  "default": "a constant function returning 1",
  "description": "The weight function to calculate the scores of the solvers in the performance and data profiles. It should be a callable representing a nonnegative function in R^+. Default is a constant function returning 1.",
  "type": "callable"
}
```

## profile_options.seed

```json
{
  "default": "0",
  "description": "The seed of the random number generator. Default is 0.",
  "type": "int"
}
```

## profile_options.semilogx

```json
{
  "default": "True",
  "description": "Whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default is True.",
  "type": "bool"
}
```

## profile_options.silent

```json
{
  "default": "False",
  "description": "Whether to show the information of the progress. Default is False.",
  "type": "bool"
}
```

## profile_options.solver_isrand

```json
{
  "default": "a list of bools of the same length as the number of solvers, where the value is True if the solver is randomized, and False otherwise",
  "description": "Whether the solvers are randomized or not. Default is a list of bools of the same length as the number of solvers, where the value is True if the solver is randomized, and False otherwise. Note that if 'n_runs' is not specified, we will set it 5 for the randomized solvers.",
  "type": "list of bool"
}
```

## profile_options.solver_names

```json
{
  "default": "the names of the callables in solvers",
  "description": "The names of the solvers. Default is the names of the callables in solvers.",
  "type": "list of str"
}
```

## profile_options.solver_verbose

```json
{
  "default": "1",
  "description": "The level of the verbosity of the solvers. 0 means no verbosity, 1 means some verbosity, and 2 means full verbosity. Default is 1.",
  "type": "int"
}
```

## profile_options.solvers_to_load

```json
{
  "default": "all the solvers",
  "description": "The indices of the solvers to load when the 'load' option is provided. It can be a list of different integers selected from 0 to the total number of solvers minus 1 of the loading experiment. At least two indices should be provided. Default is all the solvers.",
  "type": "list of int"
}
```

## profile_options.summarize_data_profiles

```json
{
  "default": "True",
  "description": "Whether to add all the data profiles to the summary PDF. Default is True.",
  "type": "bool"
}
```

## profile_options.summarize_log_ratio_profiles

```json
{
  "default": "False",
  "description": "Whether to add all the log-ratio profiles to the summary PDF. Default is False.",
  "type": "bool"
}
```

## profile_options.summarize_output_based_profiles

```json
{
  "default": "True",
  "description": "Whether to add all the output-based profiles of the selected profiles to the summary PDF. Default is True.",
  "type": "bool"
}
```

## profile_options.summarize_performance_profiles

```json
{
  "default": "True",
  "description": "Whether to add all the performance profiles to the summary PDF. Default is True.",
  "type": "bool"
}
```

## profile_options.xlabel_data_profile

```json
{
  "default": "'Number of simplex gradients'",
  "description": "The label of the x-axis of the data profiles. Default is 'Number of simplex gradients'. Note: LaTeX formatting is supported. The same applies to the options 'xlabel_log_ratio_profile', 'xlabel_performance_profile', 'ylabel_data_profile', 'ylabel_log_ratio_profile', and 'ylabel_performance_profile'.",
  "type": "str"
}
```

## profile_options.xlabel_log_ratio_profile

```json
{
  "default": "'Problem'",
  "description": "The label of the x-axis of the log-ratio profiles. Default is 'Problem'.",
  "type": "str"
}
```

## profile_options.xlabel_performance_profile

```json
{
  "default": "'Performance ratio'",
  "description": "The label of the x-axis of the performance profiles. Default is 'Performance ratio'.",
  "type": "str"
}
```

## profile_options.ylabel_data_profile

```json
{
  "default": "'Data profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
  "description": "The label of the y-axis of the data profiles. Default is 'Data profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format. You can also use '%s' in your custom label, and it will be replaced accordingly. The same applies to the options 'ylabel_log_ratio_profile' and 'ylabel_performance_profile'.",
  "type": "str"
}
```

## profile_options.ylabel_log_ratio_profile

```json
{
  "default": "'Log-ratio profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
  "description": "The label of the y-axis of the log-ratio profiles. Default is 'Log-ratio profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format.",
  "type": "str"
}
```

## profile_options.ylabel_performance_profile

```json
{
  "default": "'Performance profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
  "description": "The label of the y-axis of the performance profiles. Default is 'Performance profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format.",
  "type": "str"
}
```

## problem_options

```json
{
  "custom_problem_libs_path": {
    "choices": [
      "s2mpj",
      "pycutest",
      "custom"
    ],
    "default": "None, meaning only built-in libraries are available",
    "description": "The path to a directory containing custom problem libraries. Each subdirectory in this path should be a problem library with the same structure as the built-in libraries (e.g., 's2mpj', 'pycutest', 'custom'). Specifically, each subdirectory should contain a file named '<library_name>_tools.py' with two functions: '<library_name>_load' and '<library_name>_select'. This option allows users to use their own problem libraries without modifying the installed package. Default is None, meaning only built-in libraries are available.",
    "type": "str or Path"
  },
  "excludelist": {
    "default": "not to exclude any problem",
    "description": "The list of problems to be excluded. Default is not to exclude any problem.",
    "type": "list"
  },
  "maxb": {
    "default": "minb + 10",
    "description": "The maximum number of bound constraints of the problems to be selected. Default is minb + 10.",
    "type": "int"
  },
  "maxcon": {
    "default": "max(maxlcon, maxnlcon)",
    "description": "The maximum number of linear and nonlinear constraints of the problems to be selected. Default is max(maxlcon, maxnlcon).",
    "type": "int"
  },
  "maxdim": {
    "default": "mindim + 1",
    "description": "The maximum dimension of the problems to be selected. Default is mindim + 1.",
    "type": "int"
  },
  "maxlcon": {
    "default": "minlcon + 10",
    "description": "The maximum number of linear constraints of the problems to be selected. Default is minlcon + 10.",
    "type": "int"
  },
  "maxnlcon": {
    "default": "minnlcon + 10",
    "description": "The maximum number of nonlinear constraints of the problems to be selected. Default is minnlcon + 10.",
    "type": "int"
  },
  "minb": {
    "default": "0",
    "description": "The minimum number of bound constraints of the problems to be selected. Default is 0.",
    "type": "int"
  },
  "mincon": {
    "default": "min(minlcon, minnlcon)",
    "description": "The minimum number of linear and nonlinear constraints of the problems to be selected. Default is min(minlcon, minnlcon).",
    "type": "int"
  },
  "mindim": {
    "default": "1",
    "description": "The minimum dimension of the problems to be selected. Default is 1.",
    "type": "int"
  },
  "minlcon": {
    "default": "0",
    "description": "The minimum number of linear constraints of the problems to be selected. Default is 0.",
    "type": "int"
  },
  "minnlcon": {
    "default": "0",
    "description": "The minimum number of nonlinear constraints of the problems to be selected. Default is 0.",
    "type": "int"
  },
  "plibs": {
    "default": "``'s2mpj'``",
    "description": "The problem libraries to be used. It should be a list of strs. The built-in choices are ``'s2mpj'``, ``'pycutest'``, and ``'custom'``. Default setting is ``'s2mpj'``. Note that ``'pycutest'`` requires the separate installation of the ``pycutest`` package; see https://jfowkes.github.io/pycutest/ for installation instructions. You can also use your own problem library by specifying its name here together with the ``custom_problem_libs_path`` option.",
    "type": "list of str"
  },
  "problem": {
    "default": "not to set any problem",
    "description": "A problem to be benchmarked. It should be an instance of the class Problem. If it is provided, we will only run the test on this problem with the given feature and draw the history plots. Default is not to set any problem.",
    "type": "Problem"
  },
  "problem_names": {
    "default": "not to select any problem by name but by the options above",
    "description": "The names of the problems to be selected. It should be a list of strs. Default is not to select any problem by name but by the options above.",
    "type": "list of str"
  },
  "ptype": {
    "choices": [
      "u",
      "b",
      "l",
      "n",
      "b",
      "ul",
      "ubn"
    ],
    "default": "'u'",
    "description": "The type of the problems to be selected. It should be a str consisting of any combination of 'u' (unconstrained), 'b' (bound constrained), 'l' (linearly constrained), and 'n' (nonlinearly constrained), such as 'b', 'ul', 'ubn'. Default is 'u'.",
    "type": "str"
  }
}
```

## problem_options.plibs

```json
{
  "default": "``'s2mpj'``",
  "description": "The problem libraries to be used. It should be a list of strs. The built-in choices are ``'s2mpj'``, ``'pycutest'``, and ``'custom'``. Default setting is ``'s2mpj'``. Note that ``'pycutest'`` requires the separate installation of the ``pycutest`` package; see https://jfowkes.github.io/pycutest/ for installation instructions. You can also use your own problem library by specifying its name here together with the ``custom_problem_libs_path`` option.",
  "type": "list of str"
}
```

## problem_options.ptype

```json
{
  "choices": [
    "u",
    "b",
    "l",
    "n",
    "b",
    "ul",
    "ubn"
  ],
  "default": "'u'",
  "description": "The type of the problems to be selected. It should be a str consisting of any combination of 'u' (unconstrained), 'b' (bound constrained), 'l' (linearly constrained), and 'n' (nonlinearly constrained), such as 'b', 'ul', 'ubn'. Default is 'u'.",
  "type": "str"
}
```

## problem_options.mindim

```json
{
  "default": "1",
  "description": "The minimum dimension of the problems to be selected. Default is 1.",
  "type": "int"
}
```

## problem_options.maxdim

```json
{
  "default": "mindim + 1",
  "description": "The maximum dimension of the problems to be selected. Default is mindim + 1.",
  "type": "int"
}
```

## problem_options.minb

```json
{
  "default": "0",
  "description": "The minimum number of bound constraints of the problems to be selected. Default is 0.",
  "type": "int"
}
```

## problem_options.maxb

```json
{
  "default": "minb + 10",
  "description": "The maximum number of bound constraints of the problems to be selected. Default is minb + 10.",
  "type": "int"
}
```

## problem_options.minlcon

```json
{
  "default": "0",
  "description": "The minimum number of linear constraints of the problems to be selected. Default is 0.",
  "type": "int"
}
```

## problem_options.maxlcon

```json
{
  "default": "minlcon + 10",
  "description": "The maximum number of linear constraints of the problems to be selected. Default is minlcon + 10.",
  "type": "int"
}
```

## problem_options.minnlcon

```json
{
  "default": "0",
  "description": "The minimum number of nonlinear constraints of the problems to be selected. Default is 0.",
  "type": "int"
}
```

## problem_options.maxnlcon

```json
{
  "default": "minnlcon + 10",
  "description": "The maximum number of nonlinear constraints of the problems to be selected. Default is minnlcon + 10.",
  "type": "int"
}
```

## problem_options.mincon

```json
{
  "default": "min(minlcon, minnlcon)",
  "description": "The minimum number of linear and nonlinear constraints of the problems to be selected. Default is min(minlcon, minnlcon).",
  "type": "int"
}
```

## problem_options.maxcon

```json
{
  "default": "max(maxlcon, maxnlcon)",
  "description": "The maximum number of linear and nonlinear constraints of the problems to be selected. Default is max(maxlcon, maxnlcon).",
  "type": "int"
}
```

## problem_options.custom_problem_libs_path

```json
{
  "choices": [
    "s2mpj",
    "pycutest",
    "custom"
  ],
  "default": "None, meaning only built-in libraries are available",
  "description": "The path to a directory containing custom problem libraries. Each subdirectory in this path should be a problem library with the same structure as the built-in libraries (e.g., 's2mpj', 'pycutest', 'custom'). Specifically, each subdirectory should contain a file named '<library_name>_tools.py' with two functions: '<library_name>_load' and '<library_name>_select'. This option allows users to use their own problem libraries without modifying the installed package. Default is None, meaning only built-in libraries are available.",
  "type": "str or Path"
}
```

## problem_options.excludelist

```json
{
  "default": "not to exclude any problem",
  "description": "The list of problems to be excluded. Default is not to exclude any problem.",
  "type": "list"
}
```

## problem_options.problem_names

```json
{
  "default": "not to select any problem by name but by the options above",
  "description": "The names of the problems to be selected. It should be a list of strs. Default is not to select any problem by name but by the options above.",
  "type": "list of str"
}
```

## problem_options.problem

```json
{
  "default": "not to set any problem",
  "description": "A problem to be benchmarked. It should be an instance of the class Problem. If it is provided, we will only run the test on this problem with the given feature and draw the history plots. Default is not to set any problem.",
  "type": "Problem"
}
```

## returns

```json
{
  "curves": {
    "description": "A list containing the curves of all the profiles.",
    "type": "list of dict or None"
  },
  "profile_scores": {
    "description": "A 4D array containing scores for all profiles. The first dimension corresponds to the index of the solver, the second to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles.",
    "type": "numpy.ndarray or None"
  },
  "solver_scores": {
    "description": "Scores of the solvers based on the profiles. See 'score_fun' in 'Other Parameters' for more details.",
    "type": "numpy.ndarray"
  }
}
```

## returns.solver_scores

```json
{
  "description": "Scores of the solvers based on the profiles. See 'score_fun' in 'Other Parameters' for more details.",
  "type": "numpy.ndarray"
}
```

## returns.profile_scores

```json
{
  "description": "A 4D array containing scores for all profiles. The first dimension corresponds to the index of the solver, the second to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles.",
  "type": "numpy.ndarray or None"
}
```

## returns.curves

```json
{
  "description": "A list containing the curves of all the profiles.",
  "type": "list of dict or None"
}
```

## output_artifacts

```json
{
  "result_directory": "`<savepath>/<benchmark_id>/<feature_stamp>_<timestamp>/`",
  "summary_pdf": "summary.pdf summarizes performance profiles and data profiles.",
  "test_log_log": "test_log/log.txt records messages printed during the run.",
  "test_log_report": "test_log/report.txt records selected problem names, timing information, merit_init = phi(x_0) = inf cases, abnormal solver terminations, output fallbacks, and solver scores."
}
```

## raises

```json
[
  {
    "description": "If an argument received an invalid value.",
    "exception": "TypeError"
  },
  {
    "description": "If the arguments are inconsistent.",
    "exception": "ValueError"
  }
]
```

## notes

```text
The current version supports benchmarking derivative-free optimization solvers.  .. caution::  The log-ratio profiles are available only when there are exactly two solvers. For more information on performance and data profiles, see [1]_, [2]_, [5]_. For that of log-ratio profiles, see [4]_, [6]_.  .. caution::  All callable arguments (``solvers``, ``distribution``, ``mod_x0``, ``mod_affine``, ``mod_bounds``, ``mod_linear_ub``, ``mod_linear_eq``, ``mod_fun``, ``mod_cub``, ``mod_ceq``, ``merit_fun``, ``score_fun``, ``score_weight_fun``) must be picklable for parallel execution (``n_jobs > 1``). In particular, **lambda functions are not picklable** and will cause the benchmark to fall back to sequential mode automatically. To take advantage of parallel execution, define named functions (using ``def``) instead of lambda expressions.  1. Two problem libraries are available by default: `S2MPJ <https://github.com/GrattonToint/S2MPJ>`_ (see [3]_) and `PyCUTEst <https://jfowkes.github.io/pycutest/>`_ (Linux and macOS only). To use your own problem library, see the ``custom_problem_libs_path`` option or the guide on our `website <https://www.optprof.com>`_.  2. Each problem library has a ``config.txt`` file that controls options such as ``variable_size`` and ``test_feasibility_problems``. You can override these at runtime using `set_plib_config` or by setting the corresponding environment variables (e.g., ``S2MPJ_VARIABLE_SIZE``). See `get_plib_config` and `set_plib_config` for details.  3. When the ``load`` option is provided, the function loads data from a previous experiment and draws profiles using the provided options. Available options in this mode are:  - *Profile and plot options*: ``benchmark_id``, ``solver_names``, ``feature_stamp``, ``errorbar_type``, ``savepath``, ``max_tol_order``, ``merit_fun``, ``run_plain``, ``score_only``, ``summarize_performance_profiles``, ``summarize_data_profiles``, ``summarize_log_ratio_profiles``, ``summarize_output_based_profiles``, ``silent``, ``semilogx``, ``normalized_scores``, ``score_weight_fun``, ``score_fun``, ``solvers_to_load``, ``line_colors``, ``line_styles``, ``line_widths``, ``bar_colors``. - *Feature options*: none. - *Problem options*: ``plibs``, ``ptype``, ``mindim``, ``maxdim``, ``minb``, ``maxb``, ``minlcon``, ``maxlcon``, ``minnlcon``, ``maxnlcon``, ``mincon``, ``maxcon``, ``excludelist``.  4. More information about OptiProfiler can be found at https://www.optprof.com.
```

## see_also

```json
[
  {
    "description": "Representation of optimization problems.",
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
    "description": "Problem equipped with a specific feature.",
    "name": [
      "FeaturedProblem",
      null
    ]
  }
]
```

## name

```text
benchmark
```

## solver_signatures

```json
{
  "bound_constrained": "solver(fun, x0, xl, xu) -> numpy.ndarray",
  "linearly_constrained": "solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray",
  "nonlinearly_constrained": "solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray",
  "unconstrained": "solver(fun, x0) -> numpy.ndarray"
}
```

## solver_signatures.unconstrained

```text
solver(fun, x0) -> numpy.ndarray
```

## solver_signatures.bound_constrained

```text
solver(fun, x0, xl, xu) -> numpy.ndarray
```

## solver_signatures.linearly_constrained

```text
solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray
```

## solver_signatures.nonlinearly_constrained

```text
solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray
```

## solver_notes

```json
[
  "fun(x) -> float: provides ONLY function values — no gradient/Hessian (DFO).",
  "Must return numpy.ndarray of shape (n,).",
  "At least 2 solvers required."
]
```

## Canonical JSON Mirror

```json
{
  "calling_convention": {
    "options": "keyword arguments to benchmark(). Example: benchmark(solvers, ptype='u', mindim=2)",
    "solvers": "list of callables: [solver1, solver2]",
    "syntax": "scores = benchmark([solver1, solver2], ptype='u', mindim=2, maxdim=20)"
  },
  "description": "Benchmark optimization solvers on a set of problems with specified features. This function creates multiple profiles for benchmarking optimization solvers on a set of problems with different features. It generates performance profiles, data profiles, and log-ratio profiles [1]_, [2]_, [4]_, [5]_ for the given solvers on various test suites, returning solver scores based on the profiles.",
  "feature_options": {
    "condition_factor": {
      "default": "0",
      "description": "The scaling factor of the condition number of the linear transformation in the 'linearly_transformed' feature. More specifically, the condition number of the linear transformation will be 2 ** (condition_factor * n / 2), where n is the dimension of the problem. Default is 0.",
      "type": "float"
    },
    "distribution": {
      "default": "'spherical')",
      "description": "The distribution of perturbation in 'perturbed_x0' feature or noise in 'noisy' feature. It should be either a str (or char), or a callable ``(random_stream, dimension) -> random vector``, accepting a random_stream and the dimension of a problem and returning a random vector with the given dimension. In 'perturbed_x0' case, the str should be either 'spherical' or 'gaussian' (default is 'spherical'). In 'noisy' case, the str should be either 'gaussian' or 'uniform' (default is 'gaussian').",
      "type": "str or callable"
    },
    "feature_name": {
      "choices": [
        "plain",
        "perturbed_x0",
        "noisy",
        "truncated",
        "permuted",
        "linearly_transformed",
        "random_nan",
        "unrelaxable_constraints",
        "nonquantifiable_constraints",
        "quantized",
        "custom"
      ],
      "default": "'plain'",
      "description": "Name of the feature to apply to problems. The available features are 'plain', 'perturbed_x0', 'noisy', 'truncated', 'permuted', 'linearly_transformed', 'random_nan', 'unrelaxable_constraints', 'nonquantifiable_constraints', 'quantized', and 'custom'. Default is 'plain'.",
      "type": "str"
    },
    "ground_truth": {
      "default": "True",
      "description": "Whether the featured problem is the ground truth or not in the 'quantized' feature. Default is True.",
      "type": "bool"
    },
    "mesh_size": {
      "default": "1e-3",
      "description": "The size of the mesh in the 'quantized' feature. Default is 1e-3.",
      "type": "float"
    },
    "mesh_type": {
      "choices": [
        "absolute",
        "relative"
      ],
      "default": "'absolute'",
      "description": "The type of the mesh in the 'quantized' feature. It should be either 'absolute' or 'relative'. Default is 'absolute'.",
      "type": "str"
    },
    "mod_affine": {
      "description": "The modifier function to generate the affine transformation applied to the variables in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (A, b, inv)``, where problem is an instance of the class Problem, A is the matrix of the affine transformation, b is the vector of the affine transformation, and inv is the inverse of matrix A. No default.",
      "type": "callable"
    },
    "mod_bounds": {
      "description": "The modifier function to modify the bound constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_xl, modified_xu)``, where problem is an instance of the class Problem, modified_xl is the modified lower bound, and modified_xu is the modified upper bound. No default.",
      "type": "callable"
    },
    "mod_ceq": {
      "description": "The modifier function to modify the nonlinear equality constraints in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_ceq``, where x is the evaluation point, problem is an instance of the class Problem, and modified_ceq is the modified vector of the nonlinear equality constraints. No default.",
      "type": "callable"
    },
    "mod_cub": {
      "description": "The modifier function to modify the nonlinear inequality constraints in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_cub``, where x is the evaluation point, problem is an instance of the class Problem, and modified_cub is the modified vector of the nonlinear inequality constraints. No default.",
      "type": "callable"
    },
    "mod_fun": {
      "description": "The modifier function to modify the objective function in the 'custom' feature. It should be a callable ``(x, random_stream, problem) -> modified_fun``, where x is the evaluation point, problem is an instance of the class Problem, and modified_fun is the modified objective function value. No default.",
      "type": "callable"
    },
    "mod_linear_eq": {
      "description": "The modifier function to modify the linear equality constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_aeq, modified_beq)``, where problem is an instance of the class Problem, modified_aeq is the modified matrix of the linear equality constraints, and modified_beq is the modified vector of the linear equality constraints. No default.",
      "type": "callable"
    },
    "mod_linear_ub": {
      "description": "The modifier function to modify the linear inequality constraints in the 'custom' feature. It should be a callable ``(random_stream, problem) -> (modified_aub, modified_bub)``, where problem is an instance of the class Problem, modified_aub is the modified matrix of the linear inequality constraints, and modified_bub is the modified vector of the linear inequality constraints. No default.",
      "type": "callable"
    },
    "mod_x0": {
      "description": "The modifier function to modify the initial guess in the 'custom' feature. It should be a callable ``(random_stream, problem) -> modified_x0``, where problem is an instance of the class Problem, and modified_x0 is the modified initial guess. No default.",
      "type": "callable"
    },
    "n_runs": {
      "default": "5 for stochastic features and 1 for deterministic features",
      "description": "The number of runs of the experiments with the given feature. Default is 5 for stochastic features and 1 for deterministic features.",
      "type": "int"
    },
    "nan_rate": {
      "default": "0.05",
      "description": "The probability that the evaluation of the objective function will return np.nan in the 'random_nan' feature. Default is 0.05.",
      "type": "float"
    },
    "noise_level": {
      "default": "1e-3",
      "description": "The magnitude of the noise in the 'noisy' feature. Default is 1e-3.",
      "type": "float"
    },
    "noise_type": {
      "choices": [
        "absolute",
        "relative",
        "mixed"
      ],
      "default": "'mixed'",
      "description": "The type of the noise in the 'noisy' features. It should be either 'absolute', 'relative', or 'mixed'. Default is 'mixed'.",
      "type": "str"
    },
    "perturbation_level": {
      "default": "1e-3",
      "description": "The magnitude of the perturbation to the initial guess in the 'perturbed_x0' feature. Default is 1e-3.",
      "type": "float"
    },
    "perturbed_trailing_digits": {
      "default": "False",
      "description": "Whether we will randomize the trailing digits of the objective function value in the 'truncated' feature. Default is False.",
      "type": "bool"
    },
    "rotated": {
      "default": "True",
      "description": "Whether to use a random or given rotation matrix to rotate the coordinates of a problem in the 'linearly_transformed' feature. Default is True.",
      "type": "bool"
    },
    "significant_digits": {
      "default": "6",
      "description": "The number of significant digits in the 'truncated' feature. Default is 6.",
      "type": "int"
    },
    "unrelaxable_bounds": {
      "default": "True",
      "description": "Whether the bound constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is True.",
      "type": "bool"
    },
    "unrelaxable_linear_constraints": {
      "default": "False",
      "description": "Whether the linear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is False.",
      "type": "bool"
    },
    "unrelaxable_nonlinear_constraints": {
      "default": "False",
      "description": "Whether the nonlinear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is False.",
      "type": "bool"
    }
  },
  "name": "benchmark",
  "notes": "The current version supports benchmarking derivative-free optimization solvers.  .. caution::  The log-ratio profiles are available only when there are exactly two solvers. For more information on performance and data profiles, see [1]_, [2]_, [5]_. For that of log-ratio profiles, see [4]_, [6]_.  .. caution::  All callable arguments (``solvers``, ``distribution``, ``mod_x0``, ``mod_affine``, ``mod_bounds``, ``mod_linear_ub``, ``mod_linear_eq``, ``mod_fun``, ``mod_cub``, ``mod_ceq``, ``merit_fun``, ``score_fun``, ``score_weight_fun``) must be picklable for parallel execution (``n_jobs > 1``). In particular, **lambda functions are not picklable** and will cause the benchmark to fall back to sequential mode automatically. To take advantage of parallel execution, define named functions (using ``def``) instead of lambda expressions.  1. Two problem libraries are available by default: `S2MPJ <https://github.com/GrattonToint/S2MPJ>`_ (see [3]_) and `PyCUTEst <https://jfowkes.github.io/pycutest/>`_ (Linux and macOS only). To use your own problem library, see the ``custom_problem_libs_path`` option or the guide on our `website <https://www.optprof.com>`_.  2. Each problem library has a ``config.txt`` file that controls options such as ``variable_size`` and ``test_feasibility_problems``. You can override these at runtime using `set_plib_config` or by setting the corresponding environment variables (e.g., ``S2MPJ_VARIABLE_SIZE``). See `get_plib_config` and `set_plib_config` for details.  3. When the ``load`` option is provided, the function loads data from a previous experiment and draws profiles using the provided options. Available options in this mode are:  - *Profile and plot options*: ``benchmark_id``, ``solver_names``, ``feature_stamp``, ``errorbar_type``, ``savepath``, ``max_tol_order``, ``merit_fun``, ``run_plain``, ``score_only``, ``summarize_performance_profiles``, ``summarize_data_profiles``, ``summarize_log_ratio_profiles``, ``summarize_output_based_profiles``, ``silent``, ``semilogx``, ``normalized_scores``, ``score_weight_fun``, ``score_fun``, ``solvers_to_load``, ``line_colors``, ``line_styles``, ``line_widths``, ``bar_colors``. - *Feature options*: none. - *Problem options*: ``plibs``, ``ptype``, ``mindim``, ``maxdim``, ``minb``, ``maxb``, ``minlcon``, ``maxlcon``, ``minnlcon``, ``maxnlcon``, ``mincon``, ``maxcon``, ``excludelist``.  4. More information about OptiProfiler can be found at https://www.optprof.com.",
  "output_artifacts": {
    "result_directory": "`<savepath>/<benchmark_id>/<feature_stamp>_<timestamp>/`",
    "summary_pdf": "summary.pdf summarizes performance profiles and data profiles.",
    "test_log_log": "test_log/log.txt records messages printed during the run.",
    "test_log_report": "test_log/report.txt records selected problem names, timing information, merit_init = phi(x_0) = inf cases, abnormal solver terminations, output fallbacks, and solver scores."
  },
  "parameters": {
    "solvers": {
      "description": "Solvers to benchmark. Each solver must be a callable accepting corresponding arguments depending on the test suite you choose:  - for an unconstrained problem, ``solver(fun, x0) -> numpy.ndarray, shape (n,)``, where ``fun`` is the objective function accepting a 1-D array and returning a float, and ``x0`` is the initial guess (1-D array); - for a bound-constrained problem, ``solver(fun, x0, xl, xu) -> numpy.ndarray, shape (n,)``, where ``xl`` and ``xu`` are the lower and upper bounds (1-D arrays, may contain ``-numpy.inf`` or ``numpy.inf``); - for a linearly constrained problem, ``solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray, shape (n,)``, where ``aub`` and ``aeq`` are the coefficient matrices of the linear inequality and equality constraints, and ``bub`` and ``beq`` are the right-hand side vectors; - for a nonlinearly constrained problem, ``solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray, shape (n,)``, where ``cub`` and ``ceq`` are the nonlinear inequality and equality constraint functions accepting a 1-D array and returning a 1-D array.  All vectors and matrices mentioned above are `numpy.ndarray`.  If the 'load' option is provided in ``**kwargs``, solvers can be None, in which case data from a previous experiment will be loaded to generate profiles.",
      "type": "list of callable if 'load' in ``**kwargs``"
    }
  },
  "problem_options": {
    "custom_problem_libs_path": {
      "choices": [
        "s2mpj",
        "pycutest",
        "custom"
      ],
      "default": "None, meaning only built-in libraries are available",
      "description": "The path to a directory containing custom problem libraries. Each subdirectory in this path should be a problem library with the same structure as the built-in libraries (e.g., 's2mpj', 'pycutest', 'custom'). Specifically, each subdirectory should contain a file named '<library_name>_tools.py' with two functions: '<library_name>_load' and '<library_name>_select'. This option allows users to use their own problem libraries without modifying the installed package. Default is None, meaning only built-in libraries are available.",
      "type": "str or Path"
    },
    "excludelist": {
      "default": "not to exclude any problem",
      "description": "The list of problems to be excluded. Default is not to exclude any problem.",
      "type": "list"
    },
    "maxb": {
      "default": "minb + 10",
      "description": "The maximum number of bound constraints of the problems to be selected. Default is minb + 10.",
      "type": "int"
    },
    "maxcon": {
      "default": "max(maxlcon, maxnlcon)",
      "description": "The maximum number of linear and nonlinear constraints of the problems to be selected. Default is max(maxlcon, maxnlcon).",
      "type": "int"
    },
    "maxdim": {
      "default": "mindim + 1",
      "description": "The maximum dimension of the problems to be selected. Default is mindim + 1.",
      "type": "int"
    },
    "maxlcon": {
      "default": "minlcon + 10",
      "description": "The maximum number of linear constraints of the problems to be selected. Default is minlcon + 10.",
      "type": "int"
    },
    "maxnlcon": {
      "default": "minnlcon + 10",
      "description": "The maximum number of nonlinear constraints of the problems to be selected. Default is minnlcon + 10.",
      "type": "int"
    },
    "minb": {
      "default": "0",
      "description": "The minimum number of bound constraints of the problems to be selected. Default is 0.",
      "type": "int"
    },
    "mincon": {
      "default": "min(minlcon, minnlcon)",
      "description": "The minimum number of linear and nonlinear constraints of the problems to be selected. Default is min(minlcon, minnlcon).",
      "type": "int"
    },
    "mindim": {
      "default": "1",
      "description": "The minimum dimension of the problems to be selected. Default is 1.",
      "type": "int"
    },
    "minlcon": {
      "default": "0",
      "description": "The minimum number of linear constraints of the problems to be selected. Default is 0.",
      "type": "int"
    },
    "minnlcon": {
      "default": "0",
      "description": "The minimum number of nonlinear constraints of the problems to be selected. Default is 0.",
      "type": "int"
    },
    "plibs": {
      "default": "``'s2mpj'``",
      "description": "The problem libraries to be used. It should be a list of strs. The built-in choices are ``'s2mpj'``, ``'pycutest'``, and ``'custom'``. Default setting is ``'s2mpj'``. Note that ``'pycutest'`` requires the separate installation of the ``pycutest`` package; see https://jfowkes.github.io/pycutest/ for installation instructions. You can also use your own problem library by specifying its name here together with the ``custom_problem_libs_path`` option.",
      "type": "list of str"
    },
    "problem": {
      "default": "not to set any problem",
      "description": "A problem to be benchmarked. It should be an instance of the class Problem. If it is provided, we will only run the test on this problem with the given feature and draw the history plots. Default is not to set any problem.",
      "type": "Problem"
    },
    "problem_names": {
      "default": "not to select any problem by name but by the options above",
      "description": "The names of the problems to be selected. It should be a list of strs. Default is not to select any problem by name but by the options above.",
      "type": "list of str"
    },
    "ptype": {
      "choices": [
        "u",
        "b",
        "l",
        "n",
        "b",
        "ul",
        "ubn"
      ],
      "default": "'u'",
      "description": "The type of the problems to be selected. It should be a str consisting of any combination of 'u' (unconstrained), 'b' (bound constrained), 'l' (linearly constrained), and 'n' (nonlinearly constrained), such as 'b', 'ul', 'ubn'. Default is 'u'.",
      "type": "str"
    }
  },
  "profile_options": {
    "bar_colors": {
      "choices": [
        "r",
        "g",
        "b",
        "c",
        "m",
        "y",
        "k"
      ],
      "default": "set to the first two colors in the 'line_colors' option",
      "description": "Two different colors for the bars of two solvers in the log-ratio profiles. It can be a list of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a 2-by-3 array with each row being a RGB triplet. Default is set to the first two colors in the 'line_colors' option.",
      "type": "list or numpy.ndarray"
    },
    "benchmark_id": {
      "default": "'out' if the option 'load' is not provided, otherwise default is '.'",
      "description": "The identifier of the test. It is used to create the specific directory to store the results. Default is 'out' if the option 'load' is not provided, otherwise default is '.'.",
      "type": "str"
    },
    "draw_hist_plots": {
      "choices": [
        "none",
        "sequential",
        "parallel"
      ],
      "default": "'parallel'",
      "description": "Whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', we will not draw the history plots. If it is 'parallel', we will draw the history plots at the same time when solvers are solving the problems. If it is 'sequential', we will draw the history plots after all the problems are solved. Default is 'parallel'.",
      "type": "str"
    },
    "errorbar_type": {
      "choices": [
        "minmax",
        "meanstd"
      ],
      "default": "'minmax', meaning that we takes the pointwise minimum and maximum of the curves",
      "description": "The type of the uncertainty interval that can be either 'minmax' or 'meanstd'. When 'n_runs' is greater than 1, we run several times of the experiments and get average curves and uncertainty intervals. Default is 'minmax', meaning that we takes the pointwise minimum and maximum of the curves.",
      "type": "str"
    },
    "feature_stamp": {
      "description": "The stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features.",
      "type": "str"
    },
    "hist_aggregation": {
      "choices": [
        "min",
        "mean",
        "max"
      ],
      "default": "'min'",
      "description": "The aggregation method we use to reduce the number of points in the history plots. It can be 'min', 'mean', or 'max'. Default is 'min'.",
      "type": "str"
    },
    "line_colors": {
      "description": "The colors of the lines in the plots. It can be a list of any valid matplotlib colors (short names, hex strings, RGB tuples, etc.). Default line colors are from the matplotlib tab10 color cycle. Note that if the number of solvers is greater than the number of colors, we will cycle through the colors.",
      "type": "list"
    },
    "line_styles": {
      "choices": [
        "-",
        "-.",
        "--",
        ":",
        "o",
        "+",
        "*",
        ".",
        "x",
        "s",
        "d",
        "^",
        "v",
        ">",
        "<",
        "p",
        "h"
      ],
      "description": "The styles of the lines in the plots. It can be a list of strs that are the combinations of line styles ('-', '-.', '--', ':') and markers ('o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'). Default line style order is ['-', '-.', '--', ':']. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles.",
      "type": "list of str"
    },
    "line_widths": {
      "default": "1.5",
      "description": "The widths of the lines in the plots. It should be a positive float or a list. Default is 1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths.",
      "type": "float or list"
    },
    "load": {
      "choices": [
        "latest",
        "yyyyMMdd_HHmmss"
      ],
      "description": "Loading the stored data from a completed experiment and draw profiles. It can be either 'latest' or a time stamp of an experiment in the format of 'yyyyMMdd_HHmmss'. No default. Note that if solvers is None, this key must be provided to load data from a previous experiment and generate profiles.",
      "type": "str"
    },
    "max_eval_factor": {
      "default": "500",
      "description": "The factor multiplied to each problem's dimension to get the maximum number of evaluations for each problem. Default is 500.",
      "type": "int"
    },
    "max_tol_order": {
      "default": "10",
      "description": "The maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of 'tolerances' to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as ``10**(-k)`` for ``k = 1, 2, ..., max_tol_order``. Default is 10.",
      "type": "int"
    },
    "merit_fun": {
      "description": "The merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a callable ``(fun_value, maxcv_value, maxcv_init) -> merit_value``, where fun_value is the objective function value, maxcv_value is the maximum constraint violation, and maxcv_init is the maximum constraint violation at the initial guess. The default merit function varphi(x) is defined by the objective function f(x) and the maximum constraint violation v(x) as::  varphi(x) = f(x)                        if v(x) <= v1 varphi(x) = f(x) + 1e5 * (v(x) - v1)   if v1 < v(x) <= v2 varphi(x) = np.inf                       if v(x) > v2  where v1 = min(0.01, 1e-10 * max(1, v0)), v2 = max(0.1, 2 * v0), and v0 is the maximum constraint violation at the initial guess. If varphi(x_0) is inf for a problem/run, all solvers are declared to pass that degenerate convergence test, and the case is listed in test_log/report.txt.",
      "type": "callable"
    },
    "n_jobs": {
      "default": "about half of available workers, at least 2 when possible",
      "description": "The number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers, with at least 2 when more than one worker is available.",
      "type": "int"
    },
    "normalized_scores": {
      "default": "True",
      "description": "Whether to normalize the scores of the solvers by the maximum score of the solvers. Default is True.",
      "type": "bool"
    },
    "project_x0": {
      "default": "False",
      "description": "Whether to project the initial point to the feasible set. Default is False.",
      "type": "bool"
    },
    "run_plain": {
      "default": "False",
      "description": "Whether to run an extra experiment with the 'plain' feature. Default is False.",
      "type": "bool"
    },
    "savepath": {
      "default": "the current working directory",
      "description": "The path to store the results. Default is the current working directory.",
      "type": "str"
    },
    "score_fun": {
      "description": "The scoring function to calculate the scores of the solvers. It should be a callable ``profile_scores -> solver_scores``, where profile_scores is a 4D array containing scores for all profiles. The first dimension of profile_scores corresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring function takes the average of the history-based performance profiles under all the tolerances.",
      "type": "callable"
    },
    "score_only": {
      "default": "False",
      "description": "Whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default is False.",
      "type": "bool"
    },
    "score_weight_fun": {
      "default": "a constant function returning 1",
      "description": "The weight function to calculate the scores of the solvers in the performance and data profiles. It should be a callable representing a nonnegative function in R^+. Default is a constant function returning 1.",
      "type": "callable"
    },
    "seed": {
      "default": "0",
      "description": "The seed of the random number generator. Default is 0.",
      "type": "int"
    },
    "semilogx": {
      "default": "True",
      "description": "Whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default is True.",
      "type": "bool"
    },
    "silent": {
      "default": "False",
      "description": "Whether to show the information of the progress. Default is False.",
      "type": "bool"
    },
    "solver_isrand": {
      "default": "a list of bools of the same length as the number of solvers, where the value is True if the solver is randomized, and False otherwise",
      "description": "Whether the solvers are randomized or not. Default is a list of bools of the same length as the number of solvers, where the value is True if the solver is randomized, and False otherwise. Note that if 'n_runs' is not specified, we will set it 5 for the randomized solvers.",
      "type": "list of bool"
    },
    "solver_names": {
      "default": "the names of the callables in solvers",
      "description": "The names of the solvers. Default is the names of the callables in solvers.",
      "type": "list of str"
    },
    "solver_verbose": {
      "default": "1",
      "description": "The level of the verbosity of the solvers. 0 means no verbosity, 1 means some verbosity, and 2 means full verbosity. Default is 1.",
      "type": "int"
    },
    "solvers_to_load": {
      "default": "all the solvers",
      "description": "The indices of the solvers to load when the 'load' option is provided. It can be a list of different integers selected from 0 to the total number of solvers minus 1 of the loading experiment. At least two indices should be provided. Default is all the solvers.",
      "type": "list of int"
    },
    "summarize_data_profiles": {
      "default": "True",
      "description": "Whether to add all the data profiles to the summary PDF. Default is True.",
      "type": "bool"
    },
    "summarize_log_ratio_profiles": {
      "default": "False",
      "description": "Whether to add all the log-ratio profiles to the summary PDF. Default is False.",
      "type": "bool"
    },
    "summarize_output_based_profiles": {
      "default": "True",
      "description": "Whether to add all the output-based profiles of the selected profiles to the summary PDF. Default is True.",
      "type": "bool"
    },
    "summarize_performance_profiles": {
      "default": "True",
      "description": "Whether to add all the performance profiles to the summary PDF. Default is True.",
      "type": "bool"
    },
    "xlabel_data_profile": {
      "default": "'Number of simplex gradients'",
      "description": "The label of the x-axis of the data profiles. Default is 'Number of simplex gradients'. Note: LaTeX formatting is supported. The same applies to the options 'xlabel_log_ratio_profile', 'xlabel_performance_profile', 'ylabel_data_profile', 'ylabel_log_ratio_profile', and 'ylabel_performance_profile'.",
      "type": "str"
    },
    "xlabel_log_ratio_profile": {
      "default": "'Problem'",
      "description": "The label of the x-axis of the log-ratio profiles. Default is 'Problem'.",
      "type": "str"
    },
    "xlabel_performance_profile": {
      "default": "'Performance ratio'",
      "description": "The label of the x-axis of the performance profiles. Default is 'Performance ratio'.",
      "type": "str"
    },
    "ylabel_data_profile": {
      "default": "'Data profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
      "description": "The label of the y-axis of the data profiles. Default is 'Data profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format. You can also use '%s' in your custom label, and it will be replaced accordingly. The same applies to the options 'ylabel_log_ratio_profile' and 'ylabel_performance_profile'.",
      "type": "str"
    },
    "ylabel_log_ratio_profile": {
      "default": "'Log-ratio profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
      "description": "The label of the y-axis of the log-ratio profiles. Default is 'Log-ratio profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format.",
      "type": "str"
    },
    "ylabel_performance_profile": {
      "default": "'Performance profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format",
      "description": "The label of the y-axis of the performance profiles. Default is 'Performance profiles ($\\mathrm{tol} = %s$)', where '%s' will be replaced by the current tolerance in LaTeX format.",
      "type": "str"
    }
  },
  "raises": [
    {
      "description": "If an argument received an invalid value.",
      "exception": "TypeError"
    },
    {
      "description": "If the arguments are inconsistent.",
      "exception": "ValueError"
    }
  ],
  "returns": {
    "curves": {
      "description": "A list containing the curves of all the profiles.",
      "type": "list of dict or None"
    },
    "profile_scores": {
      "description": "A 4D array containing scores for all profiles. The first dimension corresponds to the index of the solver, the second to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles.",
      "type": "numpy.ndarray or None"
    },
    "solver_scores": {
      "description": "Scores of the solvers based on the profiles. See 'score_fun' in 'Other Parameters' for more details.",
      "type": "numpy.ndarray"
    }
  },
  "see_also": [
    {
      "description": "Representation of optimization problems.",
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
      "description": "Problem equipped with a specific feature.",
      "name": [
        "FeaturedProblem",
        null
      ]
    }
  ],
  "signature": "(solvers: 'list[callable] | None' = None, /, **kwargs) -> 'tuple[np.ndarray, np.ndarray | None, list[dict] | None]'",
  "solver_notes": [
    "fun(x) -> float: provides ONLY function values — no gradient/Hessian (DFO).",
    "Must return numpy.ndarray of shape (n,).",
    "At least 2 solvers required."
  ],
  "solver_signatures": {
    "bound_constrained": "solver(fun, x0, xl, xu) -> numpy.ndarray",
    "linearly_constrained": "solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray",
    "nonlinearly_constrained": "solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray",
    "unconstrained": "solver(fun, x0) -> numpy.ndarray"
  }
}
```
