# Common Fix Patterns

## 1. Interface Wrapper (solver signature mismatch)

When a solver has extra parameters or different ordering:

```python
def my_solver_wrapper(fun, x0):
    """Wrapper to adapt my_solver to OptiProfiler's interface."""
    return my_solver(fun, x0, maxiter=1000)
```

## 2. Import Fix (missing module)

```python
# Before: from scipy.optimize import minimize
# Fix: ensure scipy is installed, or use a fallback
try:
    from scipy.optimize import minimize
except ImportError:
    raise ImportError("scipy is required. Install with: pip install scipy")
```

## 3. Numerical Protection

```python
def safe_solver(fun, x0):
    """Wrapper with numerical safeguards."""
    import numpy as np

    def safe_fun(x):
        val = fun(x)
        if np.isnan(val) or np.isinf(val):
            return 1e30
        return val

    return original_solver(safe_fun, x0)
```

## 4. Minimum 2 Solvers

```python
# Wrong: benchmark([my_solver])
# Fix: always provide at least 2 solvers
from optiprofiler import benchmark
benchmark([solver_a, solver_b])
```

## 5. Return Value Fix

OptiProfiler expects the solver to return `x` (the solution point):

```python
def my_solver(fun, x0):
    # ... optimization logic ...
    return x_best  # must return the best point found
```
