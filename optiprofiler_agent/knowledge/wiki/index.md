---
tags: [index, navigation]
last_updated: 2026-06-18
---

# OptiProfiler Knowledge Wiki — Index

## Concepts

- [Derivative-Free Optimization](concepts/dfo.md) — DFO fundamentals and why fun provides only values
- [Benchmark Function](concepts/benchmark-function.md) — the central benchmark() API entry point
- [Solver Interface](concepts/solver-interface.md) — required solver signatures for each problem type
- [Problem Types](concepts/problem-types.md) — ptype options (u, b, l, n) and selection parameters
- [Features](concepts/features.md) — feature_name options, including `distribution` mappings
- [Parallel & Pickle Rules](concepts/parallel-and-pickle.md) — when lambdas break parallel mode

## Platform

- [Platform Overview](platform/overview.md) — hosted submission flow, sandbox runners, Agent A/B/C touchpoints, and leaderboard shape
- [Agent Role In The DFO Ecosystem](platform/ecosystem-agent-role.md) — how OPA should support problem libraries, solvers, benchmarking tools, and loop engineering

## API Reference

### Python
- [Imports and Public API](api/python/imports-and-exports.md) — what you can `from optiprofiler import …`
- [benchmark()](api/python/benchmark.md) — full Python parameter reference
- [Problem Class](api/python/problem-class.md) — Problem, Feature, FeaturedProblem classes
- [Problem Library Tools](api/python/plib-tools.md) — adapter-level problem library helpers and config APIs

### MATLAB
- [benchmark()](api/matlab/benchmark.md) — MATLAB API reference and differences from Python

## Source-Backed Reference

- [Python API Notes Source](reference/python-api_notes.md) — lossless mirror of `_sources/python/api_notes.json`
- [Python benchmark.json Source](reference/python-benchmark.md) — source-backed Python benchmark facts, options, defaults, choices, returns, notes
- [Python Classes Source](reference/python-classes.md) — lossless mirror of Python Problem, Feature, and FeaturedProblem metadata
- [Python Problem Library Tools Source](reference/python-plib_tools.md) — source-backed Python plib helper signatures and descriptions
- [MATLAB API Notes Source](reference/matlab-api_notes.md) — lossless mirror of `_sources/matlab/api_notes.json`
- [MATLAB benchmark.json Source](reference/matlab-benchmark.md) — source-backed MATLAB benchmark facts, options, defaults, choices, returns, notes
- [MATLAB Classes Source](reference/matlab-classes.md) — lossless mirror of MATLAB Problem, Feature, and FeaturedProblem metadata
- [MATLAB Problem Library Tools Source](reference/matlab-plib_tools.md) — source-backed MATLAB plib helper signatures and descriptions
- [Platform Manifest Source](reference/platform-manifest.md) — lossless mirror of the local platform source snapshot manifest
- [Platform Docs Source](reference/platform-docs.md) — exact bundled platform documentation snapshot for platform workflow and ecosystem facts
- [Enums Source](reference/enums.md) — lossless mirror of bundled enum constants
- [Legacy Docs and Examples Source](reference/legacy-docs.md) — exact bundled examples, installation notes, problem library notes, profiles, and debugging docs
- [Bibliography Source](reference/bibliography.md) — exact bundled bibliography source

## Guides

- [Python Quickstart](guides/quickstart-python.md) — installation and first benchmark
- [MATLAB Quickstart](guides/quickstart-matlab.md) — setup and first benchmark
- [Custom Solver Guide](guides/custom-solver.md) — writing solver wrappers for OptiProfiler
- [Custom Feature Guide](guides/custom-feature.md) — every `mod_*` callable with examples
- [Custom Problem Library — Python](guides/custom-problem-library-python.md) — write `<name>_load` / `<name>_select` in Python
- [Custom Problem Library — MATLAB](guides/custom-problem-library-matlab.md) — write `<name>_load.m` / `<name>_select.m`
- [Problem-Set Metadata Helper](guides/problem-metadata.md) — generate `probinfo_<lib>.csv` for fast `select()`

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
