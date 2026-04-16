---
tags: [solver, dfo, model-based, bound-constrained]
sources: [_sources/refs/bibliography.md]
related: [solvers/overview.md, solvers/newuoa.md, solvers/prima.md]
last_updated: 2025-04-13
---

# BOBYQA

**BOBYQA** (Bound Optimization BY Quadratic Approximation) is a
bound-constrained DFO solver by M.J.D. Powell. It extends the NEWUOA
approach to handle bound constraints.

## Properties

| Property | Value |
|---|---|
| Type | Model-based (quadratic interpolation) |
| Constraints | Bound-constrained |
| Author | M.J.D. Powell |
| Available via | PRIMA, PDFO, Py-BOBYQA |

## Strengths

- Efficient for smooth, bound-constrained problems
- Quadratic models enable fast convergence
- Respects bounds strictly

## Weaknesses

- Only handles bound constraints (not linear or nonlinear)
- Sensitive to noise
- Performance depends on interpolation point management

## Expected Profile Behavior

- Similar to NEWUOA for bound-constrained problems
- **Performance profiles**: Steep rise on smooth problems
- **Under noise**: Degradation, but less than unconstrained NEWUOA
  since bounds limit the search region

## See Also

- [NEWUOA](newuoa.md) — unconstrained variant
- [COBYLA](cobyla.md) — general constraint support
- [PRIMA](prima.md) — reference implementation
