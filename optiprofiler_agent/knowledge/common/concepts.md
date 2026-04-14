# OptiProfiler Core Concepts

## What is OptiProfiler?

OptiProfiler is a platform for benchmarking optimization solvers.
It supports both Python and MATLAB, with nearly identical APIs.
It generates performance profiles, data profiles, and log-ratio profiles
to compare solver effectiveness across standardized test problem sets.

## Derivative-Free Optimization (DFO)

OptiProfiler is designed for **derivative-free optimization** benchmarking.
The objective function `fun` returns **only a scalar function value**.
No gradient, Jacobian, or Hessian information is available.

Every call to `fun` is counted internally and used for performance scoring.
Methods that internally approximate gradients via finite differences
(BFGS, L-BFGS-B, CG, Newton-CG, TNC, fminunc) consume extra `fun` evaluations,
making them generally unsuitable for DFO benchmarking.

**Recommended DFO methods** (see language-specific guides for details).

## Solver Requirements

- `benchmark()` requires **at least 2 solvers** for comparison.
- Each solver must follow a specific signature depending on the problem type.
- Solvers must return the solution vector (not a dict or result object).

## Four Problem Types

| Type | Signature Pattern |
|------|------------------|
| Unconstrained | `solver(fun, x0)` |
| Bound-constrained | `solver(fun, x0, xl, xu)` |
| Linearly constrained | `solver(fun, x0, xl, xu, aub, bub, aeq, beq)` |
| Nonlinearly constrained | `solver(fun, x0, ..., cub, ceq)` |

## Profiles and Scoring

- **Performance profiles**: fraction of problems solved within a factor of the best solver
- **Data profiles**: fraction of problems solved as a function of computational budget
- **Log-ratio profiles**: pairwise comparison (exactly 2 solvers only)
- **Scores**: by default, average of history-based performance profiles across tolerances

## Output Structure

Running `benchmark()` creates:
- `<benchmark_id>/<feature_stamp>/` directory with per-problem results
- `summary.pdf` with all profile plots
- Return values: `solver_scores`, `profile_scores` (4D), `curves`

## Additional Notes

- **Log-ratio profiles** are available only when there are exactly 2 solvers.
- The `load` option allows reloading a previous experiment to redraw profiles with different options.
- More information: https://www.optprof.com
