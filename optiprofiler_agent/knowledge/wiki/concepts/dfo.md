---
tags: [concept, dfo, optimization]
sources: [_sources/python/benchmark.json, _sources/refs/bibliography.md]
related: [concepts/benchmark-function.md, concepts/solver-interface.md, concepts/problem-types.md]
last_updated: 2025-04-13
---

# Derivative-Free Optimization (DFO)

Derivative-Free Optimization (DFO) refers to optimization methods that do
**not** require gradient or Hessian information. The objective function `fun`
provides **only function values** — a scalar `float` for a given input
vector `x`.

This is the fundamental assumption underlying OptiProfiler: all solvers
benchmarked through the platform must operate without derivative
information.

## Why DFO?

DFO solvers are essential when:

- The objective function is a black-box (e.g., simulation output)
- Derivatives are unavailable, unreliable, or too expensive to compute
- The function is noisy, discontinuous, or has limited precision

## DFO in OptiProfiler

In OptiProfiler, the `fun` argument passed to solvers always has the
signature:

```python
fun(x) -> float  # Python
```

```matlab
fun(x) -> scalar  % MATLAB
```

The solver **must not** assume access to gradients. Any solver that requires
derivative information is incompatible with OptiProfiler's benchmarking
framework.

## See Also

- [Benchmark Function](benchmark-function.md) — the central API for running benchmarks
- [Solver Interface](solver-interface.md) — expected solver signatures
- [Problem Types](problem-types.md) — unconstrained, bound, linear, nonlinear
