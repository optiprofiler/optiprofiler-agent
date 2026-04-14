---
tags: [api, python, class, problem]
sources: [_sources/python/classes.json]
related: [concepts/problem-types.md, api/python/benchmark.md, api/python/plib-tools.md]
last_updated: 2025-04-13
---

# Python Problem Class

The `Problem` class represents an optimization problem in OptiProfiler.
It encapsulates the objective function, constraints, initial point, and
problem metadata.

## Properties

| Property | Type | Description |
|---|---|---|
| `fun` | callable | Objective function `fun(x) -> float` |
| `x0` | np.ndarray | Initial guess, shape `(n,)` |
| `n` | int | Problem dimension |
| `xl` | np.ndarray | Lower bounds (may contain `-np.inf`) |
| `xu` | np.ndarray | Upper bounds (may contain `np.inf`) |
| `aub` | np.ndarray | Linear inequality constraint matrix |
| `bub` | np.ndarray | Linear inequality right-hand side |
| `aeq` | np.ndarray | Linear equality constraint matrix |
| `beq` | np.ndarray | Linear equality right-hand side |
| `cub` | callable | Nonlinear inequality constraints `cub(x) -> array` |
| `ceq` | callable | Nonlinear equality constraints `ceq(x) -> array` |

## Related Classes

- **`Feature`**: Represents a feature applied to problems during
  benchmarking (e.g., noise, perturbation).
- **`FeaturedProblem`**: A `Problem` equipped with a specific `Feature`,
  used internally during benchmark execution.

## See Also

- [Problem Types](../../concepts/problem-types.md) — problem categories
- [Python benchmark()](benchmark.md) — how problems are used
- [Problem Library Tools](plib-tools.md) — loading problems from libraries
