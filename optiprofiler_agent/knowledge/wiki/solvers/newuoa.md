---
tags: [solver, dfo, model-based, unconstrained]
sources: [_sources/refs/bibliography.md]
related: [solvers/overview.md, solvers/bobyqa.md, solvers/prima.md, concepts/solver-interface.md]
last_updated: 2025-04-13
---

# NEWUOA

**NEWUOA** (NEW Unconstrained Optimization Algorithm) is a model-based DFO
solver by M.J.D. Powell. It builds and maintains a quadratic interpolation
model of the objective function.

## Properties

| Property | Value |
|---|---|
| Type | Model-based (quadratic interpolation) |
| Constraints | Unconstrained only |
| Author | M.J.D. Powell |
| Available via | PRIMA, PDFO |

## Strengths

- Very efficient for smooth, unconstrained problems
- Typically requires fewer function evaluations than direct search
- Good convergence to high accuracy

## Weaknesses

- Unconstrained only — cannot handle bounds or constraints
- Sensitive to noise (model accuracy degrades)
- May fail on non-smooth functions

## Expected Profile Behavior

- **Performance profiles**: Steep rise, often the fastest on smooth problems
- **Under noise**: Significant degradation compared to direct search
- **Data profiles**: Reaches high accuracy quickly but may plateau
  earlier under noise

## See Also

- [BOBYQA](bobyqa.md) — bound-constrained variant
- [PRIMA](prima.md) — reference implementation
- [Solver Overview](overview.md) — all solvers
