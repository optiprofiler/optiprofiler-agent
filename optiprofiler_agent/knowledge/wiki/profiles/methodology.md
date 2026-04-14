---
tags: [profiles, methodology, theory]
sources: [_sources/refs/bibliography.md]
related: [profiles/performance-profile.md, profiles/data-profile.md, profiles/log-ratio-profile.md, concepts/benchmark-function.md]
last_updated: 2025-04-13
---

# Profile Methodology

OptiProfiler generates three types of profiles to compare solver
performance. All profiles are based on the concept of **convergence test**:
a solver is considered to have "solved" a problem at tolerance `tau` if
its merit function value reaches within `tau` of the best known value.

## Convergence Test

For a given tolerance `tau = 10^(-k)` (k = 1, ..., `max_tol_order`),
solver `s` is said to converge on problem `p` if:

```
f(x_s) - f_best <= tau * max(1, |f_best|)
```

where `f(x_s)` is the merit function value at the solver's output and
`f_best` is the best value achieved by any solver.

## Merit Function

For constrained problems, the merit function combines objective value
and constraint violation:

```
varphi(x) = f(x)                          if v(x) <= v1
varphi(x) = f(x) + 1e5 * (v(x) - v1)     if v1 < v(x) <= v2
varphi(x) = inf                            if v(x) > v2
```

where `v1 = min(0.01, 1e-10 * max(1, v0))`, `v2 = max(0.1, 2*v0)`,
and `v0` is the constraint violation at the initial point.

## History-Based vs Output-Based

- **History-based**: Tracks the merit function value throughout the
  optimization process (at each function evaluation)
- **Output-based**: Only considers the final output point of the solver

History-based profiles are generally more informative as they capture
convergence speed, not just final quality.

## AUC Scoring

Solver scores are computed as the **Area Under the Curve (AUC)** of
profile curves, averaged across tolerances. This provides a single
scalar metric that summarizes a solver's overall performance.

## Multi-Run Handling

For stochastic features (`n_runs > 1`), profiles show average curves with
uncertainty intervals. The `errorbar_type` parameter controls whether
intervals show min/max (`'minmax'`) or mean ± std (`'meanstd'`).

## See Also

- [Performance Profiles](performance-profile.md) — ratio-based comparison
- [Data Profiles](data-profile.md) — budget-based comparison
- [Log-Ratio Profiles](log-ratio-profile.md) — pairwise comparison
- [Feature Effects](feature-effects.md) — how features affect profiles
