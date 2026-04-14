---
tags: [api, python, benchmark, reference]
sources: [_sources/python/benchmark.json, _sources/python/api_notes.json]
related: [concepts/benchmark-function.md, concepts/solver-interface.md, api/python/problem-class.md]
last_updated: 2025-04-13
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
| Bound constrained | `solver(fun, x0, xl, xu) -> np.ndarray` |
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
| `distribution` | str/callable | `'spherical'` | Distribution for perturbation/noise |
| `significant_digits` | int | 6 | Digits for `'truncated'` |
| `nan_rate` | float | 0.05 | NaN probability for `'random_nan'` |
| `mesh_size` | float | 1e-3 | Mesh size for `'quantized'` |

## Profile & Plot Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_jobs` | int | auto | Parallel workers |
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

## Python-Specific Notes

- Solver format: list of callables `[solver1, solver2]`
- Options: keyword arguments to `benchmark()`
- Vectors: 1-D numpy arrays, shape `(n,)`
- **Lambda functions are not picklable** — use named functions for `n_jobs > 1`
- PyCUTEst requires separate installation; Linux and macOS only
- `custom_problem_libs_path` is Python-only (MATLAB uses folder structure)

## See Also

- [Benchmark Function Concept](../../concepts/benchmark-function.md)
- [Problem Class](problem-class.md) — the Problem data structure
- [MATLAB benchmark()](../matlab/benchmark.md) — MATLAB equivalent
- [Python Quickstart](../../guides/quickstart-python.md) — getting started
