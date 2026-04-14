---
tags: [guide, python, quickstart]
sources: [_sources/python/api_notes.json]
related: [api/python/benchmark.md, concepts/benchmark-function.md, guides/custom-solver.md]
last_updated: 2025-04-13
---

# Python Quickstart

## Installation

```bash
pip install optiprofiler
```

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
    # Must use `def`, not `lambda` — lambdas are not picklable
    from scipy.optimize import minimize
    return minimize(fun, x0, method='Nelder-Mead').x

scores = benchmark([my_solver, other_solver], n_jobs=4)
```

## Example 5: Custom Problem Library

```python
scores = benchmark(
    solvers,
    plibs=['s2mpj', 'mylib'],
    custom_problem_libs_path='/path/to/my/libraries',
)
```

Each custom library directory must contain `<name>_tools.py` with
`<name>_load` and `<name>_select` functions.

## See Also

- [Python benchmark() API](../api/python/benchmark.md) — full parameter reference
- [Custom Solver Guide](custom-solver.md) — writing solver wrappers
- [MATLAB Quickstart](quickstart-matlab.md) — MATLAB equivalent
