---
tags: [solver, dfo, model-based, constrained]
sources: [_sources/refs/bibliography.md]
related: [solvers/overview.md, solvers/newuoa.md, solvers/prima.md, concepts/problem-types.md]
last_updated: 2025-04-13
---

# COBYLA

**COBYLA** (Constrained Optimization BY Linear Approximations) is a DFO
solver by M.J.D. Powell that handles nonlinear constraints using linear
approximations.

## Properties

| Property | Value |
|---|---|
| Type | Model-based (linear approximation) |
| Constraints | All types (u, b, l, n) |
| Author | M.J.D. Powell |
| Available via | SciPy, PRIMA, PDFO |

## Strengths

- Handles all constraint types
- Widely available (in SciPy)
- Works with nonlinear constraints

## Weaknesses

- Uses linear (not quadratic) models — slower convergence than NEWUOA
- Less accurate on smooth unconstrained problems
- May struggle with highly nonlinear constraints

## Expected Profile Behavior

- **Performance profiles**: Moderate rise, slower than NEWUOA on
  unconstrained but competitive on constrained
- **Versatility**: Maintains good performance across all ptype settings
- **Data profiles**: Steady convergence, rarely the fastest but reliable

## See Also

- [NEWUOA](newuoa.md) — unconstrained variant with quadratic models
- [PRIMA](prima.md) — reference implementation
- [Problem Types](../concepts/problem-types.md) — constraint categories
