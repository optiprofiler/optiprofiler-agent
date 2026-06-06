---
tags: [profiles, features, interpretation]
sources: [_sources/python/benchmark.json]
related: [concepts/features.md, profiles/methodology.md, profiles/performance-profile.md]
last_updated: 2026-06-07
---

# Feature Effects on Profiles

Different [features](../concepts/features.md) affect solver performance in
characteristic ways. Understanding these effects is essential for
interpreting benchmark results.

## Plain (Baseline)

No modification. Establishes baseline performance. If a solver performs
poorly here, the issue is fundamental rather than feature-related.

## Perturbed x0

Perturbs the initial guess. Tests **sensitivity to starting point**.
- Solvers that rely heavily on good initialization will degrade
- Robust solvers maintain similar performance
- Expect wider uncertainty bands with multiple runs

## Noisy

Adds noise to function evaluations. Tests **noise resilience**.
- Solvers using finite differences degrade significantly
- Direct-search methods (e.g., Nelder-Mead) may be more robust
- Higher noise levels amplify differences between solvers
- `noise_type='relative'` is harder than `'absolute'` for functions
  with large values
- `distribution='gaussian'` is the default; `distribution='uniform'`
  draws from `[-1, 1]`. For custom noise, pass a callable rather than an
  unsupported string.
- `noise_type='mixed'` applies
  `f + max(1, abs(f)) * noise_level * noise` to objective values and the
  analogous elementwise formula to nonlinear constraints.

`distribution` for `noisy` has a subtle callable contract:

- Objective values use scalar noise: `distribution(random_stream)`.
- Nonlinear constraints use vector noise:
  `distribution(random_stream, dimension)`.

For `perturbed_x0`, the same option has a different contract:
`distribution(random_stream, dimension) -> random vector`, with allowed
strings `'spherical'` (default) and `'gaussian'`.

## Truncated

Limits function value precision. Tests **handling of limited accuracy**.
- Similar to noisy but with deterministic truncation
- Solvers requiring high precision may fail to converge

## Linearly Transformed

Applies a random linear transformation to coordinates. Tests
**rotational invariance**.
- Coordinate-descent methods degrade
- Methods invariant to rotation (e.g., trust-region) are unaffected
- `condition_factor > 0` adds ill-conditioning

## Random NaN

Randomly returns NaN for some evaluations. Tests **fault tolerance**.
- Solvers that don't handle NaN will crash
- Robust solvers retry or skip failed evaluations

## Unrelaxable Constraints

Makes certain constraints strictly enforced. Tests **feasibility handling**.
- Solvers that explore infeasible regions will struggle
- Projection-based methods handle this better

## See Also

- [Features](../concepts/features.md) — feature definitions and parameters
- [Methodology](methodology.md) — how convergence is measured
- [Solver Traits](../solvers/overview.md) — expected solver behaviors
