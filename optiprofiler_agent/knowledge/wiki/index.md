---
tags: [index, navigation]
last_updated: 2025-04-13
---

# OptiProfiler Knowledge Wiki — Index

## Concepts

- [Derivative-Free Optimization](concepts/dfo.md) — DFO fundamentals and why fun provides only values
- [Benchmark Function](concepts/benchmark-function.md) — the central benchmark() API entry point
- [Solver Interface](concepts/solver-interface.md) — required solver signatures for each problem type
- [Problem Types](concepts/problem-types.md) — ptype options (u, b, l, n) and selection parameters
- [Features](concepts/features.md) — feature_name options that modify test problems

## API Reference

### Python
- [Imports and Public API](api/python/imports-and-exports.md) — what you can `from optiprofiler import …`
- [benchmark()](api/python/benchmark.md) — full Python parameter reference
- [Problem Class](api/python/problem-class.md) — Problem, Feature, FeaturedProblem classes
- [Problem Library Tools](api/python/plib-tools.md) — s2mpj_load, pycutest_select, etc.

### MATLAB
- [benchmark()](api/matlab/benchmark.md) — MATLAB API reference and differences from Python

## Guides

- [Python Quickstart](guides/quickstart-python.md) — installation and first benchmark
- [MATLAB Quickstart](guides/quickstart-matlab.md) — setup and first benchmark
- [Custom Solver Guide](guides/custom-solver.md) — writing solver wrappers for OptiProfiler

## Profiles

- [Methodology](profiles/methodology.md) — convergence tests, merit function, AUC scoring
- [Performance Profiles](profiles/performance-profile.md) — ratio-based solver comparison
- [Data Profiles](profiles/data-profile.md) — budget-based solver comparison
- [Log-Ratio Profiles](profiles/log-ratio-profile.md) — pairwise two-solver comparison
- [Feature Effects](profiles/feature-effects.md) — how features affect profile results

## Solvers

- [Solver Overview](solvers/overview.md) — solver categories and general profile expectations
- [NEWUOA](solvers/newuoa.md) — model-based unconstrained solver
- [BOBYQA](solvers/bobyqa.md) — model-based bound-constrained solver
- [COBYLA](solvers/cobyla.md) — constrained solver with linear approximations
- [Nelder-Mead](solvers/nelder-mead.md) — direct-search simplex method
- [Powell](solvers/powell.md) — direction-set coordinate search
- [PRIMA](solvers/prima.md) — reference implementation of Powell's methods

## Troubleshooting

- [Common Errors](troubleshooting/common-errors.md) — error catalog with fixes
- [Solver Compatibility](troubleshooting/solver-compat.md) — adapting third-party solvers
- [Timeout Issues](troubleshooting/timeout-issues.md) — slow benchmarks and solutions
