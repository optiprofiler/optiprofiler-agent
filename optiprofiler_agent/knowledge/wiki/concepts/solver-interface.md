---
tags: [concept, solver, interface]
sources: [_sources/python/benchmark.json, _sources/matlab/benchmark.json]
related: [concepts/dfo.md, concepts/benchmark-function.md, concepts/problem-types.md, troubleshooting/solver-compat.md]
last_updated: 2025-04-13
---

# Solver Interface

Every solver passed to `benchmark()` must be a callable that accepts a
specific set of arguments determined by the [problem type](problem-types.md).
The solver must return the solution vector `x`.

## Python Signatures

| Problem Type           | Signature                                                        |
|------------------------|------------------------------------------------------------------|
| Unconstrained (`'u'`)  | `solver(fun, x0) -> np.ndarray`                                 |
| Bound (`'b'`)          | `solver(fun, x0, xl, xu) -> np.ndarray`                         |
| Linear (`'l'`)         | `solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> np.ndarray`    |
| Nonlinear (`'n'`)      | `solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> np.ndarray` |

- All vectors are 1-D `numpy.ndarray` of shape `(n,)`
- `fun(x) -> float`: returns only the function value (no gradient)
- `xl`, `xu` may contain `-np.inf` / `np.inf`
- `cub(x)`, `ceq(x)` return 1-D arrays

## MATLAB Signatures

| Problem Type           | Signature                                                    |
|------------------------|--------------------------------------------------------------|
| Unconstrained (`'u'`)  | `solver(fun, x0)`                                           |
| Bound (`'b'`)          | `solver(fun, x0, xl, xu)`                                   |
| Linear (`'l'`)         | `solver(fun, x0, xl, xu, aub, bub, aeq, beq)`               |
| Nonlinear (`'n'`)      | `solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)`    |

- All vectors are column vectors (n x 1 matrices)
- `fun(x)` returns a scalar

## Important Notes

- `fun` provides **only function values** — no gradient or Hessian (this is DFO)
- The solver must return a vector of shape `(n,)` (Python) or n x 1 (MATLAB)
- Lambda functions are **not picklable** in Python — use `def` for parallel
  execution (`n_jobs > 1`)

## See Also

- [DFO](dfo.md) — why no gradients are available
- [Problem Types](problem-types.md) — how ptype determines the signature
- [Solver Compatibility](../troubleshooting/solver-compat.md) — adapting third-party solvers
- [Custom Solver Guide](../guides/custom-solver.md) — writing solver wrappers
