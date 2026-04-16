---
tags: [api, matlab, benchmark, reference]
sources: [_sources/matlab/benchmark.json, _sources/matlab/api_notes.json]
related: [concepts/benchmark-function.md, concepts/solver-interface.md, api/python/benchmark.md]
last_updated: 2025-04-13
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
| `draw_hist_plots` | `'sequential'` | `'parallel'` |
| `solvers_to_load` | 1-indexed | 0-indexed |
| `line_colors` | MATLAB 'gem' colororder | matplotlib tab10 |
| Custom plib path | folder structure | `custom_problem_libs_path` option |

## Options Format

Options are passed as a struct:

```matlab
options.ptype = 'u';
options.mindim = 2;
options.maxdim = 20;
options.feature_name = 'noisy';
scores = benchmark({@solver1, @solver2}, options)
```

## Problem Libraries

- **s2mpj**: Default, bundled with OptiProfiler
- **matcutest**: Requires setup; **Linux only**

## See Also

- [Python benchmark()](../python/benchmark.md) — Python equivalent
- [Benchmark Function Concept](../../concepts/benchmark-function.md)
- [MATLAB Quickstart](../../guides/quickstart-matlab.md)
