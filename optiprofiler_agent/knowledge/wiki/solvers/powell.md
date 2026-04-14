---
tags: [solver, dfo, direct-search, unconstrained]
sources: [_sources/refs/bibliography.md]
related: [solvers/overview.md, solvers/nelder-mead.md]
last_updated: 2025-04-13
---

# Powell's Method

**Powell's method** (as implemented in SciPy) is a direction-set method
that performs sequential one-dimensional minimizations along conjugate
directions.

## Properties

| Property | Value |
|---|---|
| Type | Direction set / coordinate search |
| Constraints | Unconstrained |
| Available via | SciPy (`method='Powell'`) |

## Strengths

- No gradient computation required
- Can be effective on separable or nearly separable functions
- Available in SciPy

## Weaknesses

- Slow on non-separable problems
- Not rotationally invariant (degrades under `linearly_transformed`)
- Limited to unconstrained problems

## Expected Profile Behavior

- **Performance profiles**: Moderate, between Nelder-Mead and model-based
- **Under linearly_transformed**: Significant degradation due to
  coordinate-aligned search directions
- **Data profiles**: Moderate evaluation count

## Wrapping for OptiProfiler

```python
def powell_solver(fun, x0):
    from scipy.optimize import minimize
    return minimize(fun, x0, method='Powell').x
```

## See Also

- [Nelder-Mead](nelder-mead.md) — another direct search method
- [Solver Overview](overview.md) — all solvers
