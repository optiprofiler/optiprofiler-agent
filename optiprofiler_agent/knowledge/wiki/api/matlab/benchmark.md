---
tags: [api, matlab, benchmark, reference]
sources: [_sources/matlab/benchmark.json, _sources/matlab/api_notes.json]
related: [concepts/benchmark-function.md, concepts/solver-interface.md, api/python/benchmark.md]
last_updated: 2026-06-05
---

# MATLAB benchmark() API Reference

```matlab
scores = benchmark({@solver1, @solver2}, options)
```

## Solver Signatures

| Problem Type | Signature |
|---|---|
| Unconstrained | `solver(fun, x0)` |
| Bound-constrained | `solver(fun, x0, xl, xu)` |
| Linearly constrained | `solver(fun, x0, xl, xu, aub, bub, aeq, beq)` |
| Nonlinearly constrained | `solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)` |

All vectors are column vectors (n x 1 matrices).

## Key Differences from Python

| Parameter | MATLAB Default | Python Default |
|---|---|---|
| `maxdim` | `mindim + 10` | `mindim + 1` |
| `draw_hist_plots` | `'parallel'` in normal runs; `load` mode forces `'sequential'` | `'parallel'` |
| `solvers_to_load` | 1-indexed | 0-indexed |
| `line_colors` | MATLAB 'gem' colororder | matplotlib tab10 |
| Custom plib location | subfolder under `optiprofiler/problem_libs` selected by `options.plibs` | `custom_problem_libs_path` option |

## Options Format

Options are passed as a struct:

```matlab
options.ptype = 'u';
options.mindim = 2;
options.maxdim = 20;
options.feature_name = 'noisy';
scores = benchmark({@solver1, @solver2}, options)
```

`options.n_jobs` defaults to a conservative worker count: about half of
the available workers, with at least 2 when more than one worker is
available. Set `options.n_jobs = 1` for reproducible timing experiments.

## Return Values

MATLAB supports one, two, or three outputs:

```matlab
[solver_scores, profile_scores, curves] = benchmark({@solver1, @solver2}, options)
```

- `solver_scores`: aggregate solver scores.
- `profile_scores`: 4D tensor indexed by solver, tolerance, history/output
  basis, and profile type.
- `curves`: cell array containing profile curves.

## Output Artifacts

Normal runs write `<savepath>/<benchmark_id>/<feature_stamp>_<timestamp>/`.
The output includes `summary.pdf`, stored results, history/profile files,
and `test_log/`.

`test_log/report.txt` records selected problem names, timing,
`merit_init = phi(x_0) = Inf` cases, abnormal solver terminations, output
fallbacks, and solver scores. `test_log/log.txt` records printed run
messages.

## Problem Libraries

- **s2mpj**: Default, bundled with OptiProfiler
- **matcutest**: Requires setup; **Linux only**

## See Also

- [Python benchmark()](../python/benchmark.md) — Python equivalent
- [Benchmark Function Concept](../../concepts/benchmark-function.md)
- [MATLAB Quickstart](../../guides/quickstart-matlab.md)
