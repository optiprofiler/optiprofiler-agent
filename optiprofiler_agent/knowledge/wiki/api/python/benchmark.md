---
tags: [api, python, benchmark, reference]
sources: [_sources/python/benchmark.json, _sources/python/api_notes.json]
related: [concepts/benchmark-function.md, concepts/solver-interface.md, api/python/problem-class.md]
last_updated: 2026-06-05
---

# Python benchmark() API Reference

```python
from optiprofiler import benchmark

scores, profile_scores, curves = benchmark(
    [solver1, solver2], ptype='u', mindim=2, maxdim=20
)
```

## Solver Signatures

| Problem Type | Signature |
|---|---|
| Unconstrained | `solver(fun, x0) -> np.ndarray` |
| Bound-constrained | `solver(fun, x0, xl, xu) -> np.ndarray` |
| Linearly constrained | `solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> np.ndarray` |
| Nonlinearly constrained | `solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> np.ndarray` |

All vectors are 1-D `numpy.ndarray` of shape `(n,)`.

## Problem Selection Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ptype` | str | `'u'` | Problem type: combination of `'u'`, `'b'`, `'l'`, `'n'` |
| `plibs` | list | `['s2mpj']` | Problem libraries: `'s2mpj'`, `'pycutest'`, `'custom'` |
| `mindim` | int | 1 | Minimum problem dimension |
| `maxdim` | int | `mindim+1` | Maximum problem dimension |
| `minb` / `maxb` | int | 0 / `minb+10` | Bound constraint count range |
| `minlcon` / `maxlcon` | int | 0 / `minlcon+10` | Linear constraint count range |
| `minnlcon` / `maxnlcon` | int | 0 / `minnlcon+10` | Nonlinear constraint count range |
| `excludelist` | list | `[]` | Problems to exclude |
| `problem_names` | list | None | Specific problems to select |
| `custom_problem_libs_path` | str/Path | None | Path to custom problem libraries |

## Feature Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `feature_name` | str | `'plain'` | Feature to apply (see [Features](../../concepts/features.md)) |
| `n_runs` | int | 5 (stochastic) / 1 (deterministic) | Number of experiment runs |
| `noise_level` | float | 1e-3 | Noise magnitude for `'noisy'` |
| `noise_type` | str | `'mixed'` | `'absolute'`, `'relative'`, or `'mixed'` |
| `perturbation_level` | float | 1e-3 | Perturbation for `'perturbed_x0'` |
| `distribution` | str/callable | feature-dependent | Distribution for `'perturbed_x0'` perturbation or `'noisy'` noise; see mapping below |
| `significant_digits` | int | 6 | Digits for `'truncated'` |
| `nan_rate` | float | 0.05 | NaN probability for `'random_nan'` |
| `mesh_size` | float | 1e-3 | Mesh size for `'quantized'` |

### `distribution` Option Mapping

`distribution` is valid only for `feature_name='perturbed_x0'` and
`feature_name='noisy'`. Its legal string values, defaults, and callable
contracts depend on the feature:

| `feature_name` | Default | Allowed string values | Callable contract |
|---|---|---|---|
| `'perturbed_x0'` | `'spherical'` | `'spherical'`, `'gaussian'` | `distribution(random_stream, dimension) -> random vector` |
| `'noisy'` | `'gaussian'` | `'gaussian'`, `'uniform'` | Objective noise: `distribution(random_stream) -> scalar`; nonlinear constraint noise: `distribution(random_stream, dimension) -> random vector` |

For `feature_name='noisy'`, the built-in string distributions mean:

- `'gaussian'`: standard normal noise via `random_stream.standard_normal()`
  for objective values, or `standard_normal(size)` for nonlinear
  constraint vectors.
- `'uniform'`: uniform noise on `[-1, 1]`, scalar for objective values
  and vector-valued for nonlinear constraints.

`noise_type` controls how the noise is applied:

| `noise_type` | Objective formula |
|---|---|
| `'absolute'` | `f + noise_level * noise` |
| `'relative'` | `f * (1 + noise_level * noise)` |
| `'mixed'` | `f + max(1, abs(f)) * noise_level * noise` |

For nonlinear constraints `cub` and `ceq`, the same three formulas are
applied elementwise, replacing `f` with the constraint vector and using
`np.maximum(1, np.abs(values))` for `'mixed'`.

Any callable passed through `distribution` must be a module-level `def`
when `n_jobs > 1`; lambdas and nested functions are not reliably
picklable in Python multiprocessing.

## Profile & Plot Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_jobs` | int | conservative auto | Parallel workers; about half of available workers, at least 2 when possible |
| `seed` | int | 0 | Random seed |
| `max_tol_order` | int | 10 | Tolerances: 10^(-1) to 10^(-max_tol_order) |
| `max_eval_factor` | int | 500 | Max evaluations = factor * dimension |
| `savepath` | str | cwd | Output directory |
| `benchmark_id` | str | `'out'` | Subdirectory name for results |
| `solver_names` | list | auto | Display names for solvers |
| `silent` | bool | False | Suppress progress output |
| `score_only` | bool | False | Skip plots, only compute scores |
| `semilogx` | bool | True | Logarithmic x-axis on profiles |
| `normalized_scores` | bool | True | Normalize scores by maximum |
| `draw_hist_plots` | str | `'parallel'` | `'none'`, `'sequential'`, or `'parallel'` |
| `load` | str | None | Load previous results: `'latest'` or timestamp |
| `solvers_to_load` | list | all | 0-indexed solver indices to load |

`score_only=True` forces history plotting off. In `load` mode,
OptiProfiler redraws profiles from stored data; solver indices in
`solvers_to_load` are 0-indexed in Python.

## Return Values

`benchmark()` returns a tuple:

| Return | Type | Meaning |
|---|---|---|
| `solver_scores` | `numpy.ndarray` | Aggregate solver scores computed from the profiles |
| `profile_scores` | `numpy.ndarray | None` | 4D tensor indexed by solver, tolerance, history/output basis, and profile type |
| `curves` | `list[dict] | None` | Raw profile curve data |

Use tuple unpacking when you need all details:

```python
solver_scores, profile_scores, curves = benchmark([solver1, solver2])
```

## Output Artifacts

By default, a normal run writes
`<savepath>/<benchmark_id>/<feature_stamp>_<timestamp>/`. The folder
contains `summary.pdf`, stored result data, history/profile outputs, and
`test_log/`.

`test_log/report.txt` records selected problem names, timing,
`merit_init = phi(x_0) = inf` cases, abnormal solver terminations,
output fallbacks, and solver scores. `test_log/log.txt` records messages
printed during the run.

## Python-Specific Notes

- Solver format: list of callables `[solver1, solver2]`
- Options: keyword arguments to `benchmark()`
- Vectors: 1-D numpy arrays, shape `(n,)`
- **Lambda functions are not picklable** — use named functions for `n_jobs > 1`
- PyCUTEst requires separate installation; Linux and macOS only
- `custom_problem_libs_path` is Python-only (MATLAB uses folder structure)
- `pip install optiprofiler` and `conda install conda-forge::optiprofiler`
  are both supported installation paths.

## See Also

- [Benchmark Function Concept](../../concepts/benchmark-function.md)
- [Problem Class](problem-class.md) — the Problem data structure
- [MATLAB benchmark()](../matlab/benchmark.md) — MATLAB equivalent
- [Python Quickstart](../../guides/quickstart-python.md) — getting started
