---
tags: [solver, dfo, direct-search, unconstrained]
sources: [_sources/refs/bibliography.md]
related: [solvers/overview.md, solvers/powell.md, concepts/solver-interface.md]
last_updated: 2025-04-13
---

# Nelder-Mead

**Nelder-Mead** (downhill simplex) is a classic direct-search method that
operates on a simplex of n+1 vertices without building a model.

## Properties

| Property | Value |
|---|---|
| Type | Direct search (simplex) |
| Constraints | Unconstrained |
| Available via | SciPy (`method='Nelder-Mead'`) |

## Strengths

- Simple and widely available
- Robust to moderate noise
- No model building overhead
- Works on non-smooth functions

## Weaknesses

- Slow convergence, especially in high dimensions
- No convergence guarantee in theory
- Unconstrained only (in standard form)

## Expected Profile Behavior

- **Performance profiles**: Slower rise than model-based methods
- **Robustness**: Higher plateau (solves more problems eventually)
- **Under noise**: Maintains performance better than model-based solvers
- **Data profiles**: Requires more simplex gradients but more reliable

## Wrapping for OptiProfiler

```python
def nelder_mead(fun, x0):
    from scipy.optimize import minimize
    return minimize(fun, x0, method='Nelder-Mead').x
```

## See Also

- [Powell](powell.md) — another direct search method
- [NEWUOA](newuoa.md) — model-based alternative
- [Solver Overview](overview.md) — all solvers
