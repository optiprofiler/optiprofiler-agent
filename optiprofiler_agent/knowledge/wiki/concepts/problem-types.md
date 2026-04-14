---
tags: [concept, problem, ptype]
sources: [_sources/python/benchmark.json, _sources/matlab/benchmark.json]
related: [concepts/solver-interface.md, concepts/benchmark-function.md, api/python/benchmark.md]
last_updated: 2025-04-13
---

# Problem Types (ptype)

OptiProfiler classifies optimization problems into four types, controlled
by the `ptype` parameter in `benchmark()`.

## Types

| Code | Type                      | Solver Arguments                                    |
|------|---------------------------|-----------------------------------------------------|
| `'u'`| Unconstrained             | `fun, x0`                                          |
| `'b'`| Bound constrained         | `fun, x0, xl, xu`                                  |
| `'l'`| Linearly constrained      | `fun, x0, xl, xu, aub, bub, aeq, beq`              |
| `'n'`| Nonlinearly constrained   | `fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq`   |

## Combining Types

`ptype` can combine multiple letters to benchmark across problem categories:

```python
benchmark(solvers, ptype='ub')    # unconstrained + bound
benchmark(solvers, ptype='ubln')  # all four types
```

The default is `'u'` (unconstrained only).

## Problem Selection Parameters

These parameters filter problems from the test library:

| Parameter | Default         | Description                                 |
|-----------|-----------------|---------------------------------------------|
| `mindim`  | 1               | Minimum problem dimension                   |
| `maxdim`  | `mindim + 1`    | Maximum problem dimension (Python default)  |
| `minb`    | 0               | Minimum number of bound constraints          |
| `maxb`    | `minb + 10`     | Maximum number of bound constraints          |
| `minlcon` | 0               | Minimum number of linear constraints         |
| `maxlcon` | `minlcon + 10`  | Maximum number of linear constraints         |
| `minnlcon`| 0               | Minimum number of nonlinear constraints      |
| `maxnlcon`| `minnlcon + 10` | Maximum number of nonlinear constraints      |

Note: MATLAB's default for `maxdim` is `mindim + 10` (differs from Python).

## See Also

- [Solver Interface](solver-interface.md) — how ptype determines solver signature
- [Benchmark Function](benchmark-function.md) — the benchmark() entry point
- [Python benchmark() API](../api/python/benchmark.md) — full parameter list
