---
tags: [guide, python, quickstart]
sources: [_sources/python/api_notes.json]
related: [api/python/benchmark.md, concepts/benchmark-function.md, guides/custom-solver.md]
last_updated: 2026-06-05
---

# Python Quickstart

## Installation

```bash
pip install optiprofiler
```

Conda-forge is also supported:

```bash
conda install conda-forge::optiprofiler
```

S2MPJ is bundled by default. PyCUTEst is optional, requires a separate
installation, and is available only on Linux and macOS.

## Example 1: Basic Benchmark

Compare two solvers on unconstrained problems with dimension 2-5:

```python
from optiprofiler import benchmark

def my_solver(fun, x0):
    # A simple solver wrapper
    from scipy.optimize import minimize
    res = minimize(fun, x0, method='Nelder-Mead')
    return res.x

def another_solver(fun, x0):
    from scipy.optimize import minimize
    res = minimize(fun, x0, method='Powell')
    return res.x

scores = benchmark([my_solver, another_solver], ptype='u', mindim=2, maxdim=5)
```

**Important**: At least **2 solvers** are required.

By default, OptiProfiler creates an `out/<feature_stamp>_<timestamp>/`
folder under the current working directory and writes `summary.pdf`,
per-problem results, and `test_log/`. `test_log/report.txt` records the
selected problem names, timing information, `merit_init = phi(x_0) =
inf` cases, abnormal solver terminations, output fallbacks, and solver
scores; `test_log/log.txt` contains printed run messages.

## Example 2: Noisy Feature

```python
scores = benchmark(
    [solver1, solver2],
    feature_name='noisy',
    noise_level=1e-3,
    noise_type='mixed',
    n_runs=5,
    ptype='u',
    mindim=2,
    maxdim=5,
)
```

## Example 3: Loading Previous Results

```python
scores = benchmark(
    [solver1, solver2],
    load='latest',  # or a timestamp like '20250101_120000'
)
```

## Example 4: Parallel Execution

```python
def my_solver(fun, x0):
    # Must use `def`, not `lambda` — see concepts/parallel-and-pickle.md
    from scipy.optimize import minimize
    return minimize(fun, x0, method='Nelder-Mead').x

scores = benchmark([my_solver, other_solver], n_jobs=4)
```

If `n_jobs` is omitted, OptiProfiler chooses a conservative default:
about half of the available workers, with at least 2 workers when more
than one worker is available. Set `n_jobs=1` for the most reproducible
timing experiments.

The same `def`-not-`lambda` rule applies to any callable that has to
cross a worker boundary (custom `mod_*` features, `profile_options`
callables). See [Parallel & Pickle Rules](../concepts/parallel-and-pickle.md)
for the four exact places this matters.

## Example 5: Custom Problem Library

```python
scores = benchmark(
    solvers,
    plibs=['s2mpj', 'mylib'],
    custom_problem_libs_path='/path/to/my/libraries',
)
```

Each custom library directory must contain `<name>_tools.py` with
`<name>_load` and `<name>_select` functions. For the full template,
typing rules, and a worked end-to-end example see the
[Custom Problem Library — Python](custom-problem-library-python.md)
guide.

## See Also

- [Python benchmark() API](../api/python/benchmark.md) — full parameter reference
- [Custom Solver Guide](custom-solver.md) — writing solver wrappers
- [MATLAB Quickstart](quickstart-matlab.md) — MATLAB equivalent
