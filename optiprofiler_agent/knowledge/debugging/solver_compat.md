# Solver Compatibility Guide

This document describes how to adapt third-party solvers to work with OptiProfiler.

## OptiProfiler Solver Interface

OptiProfiler calls solvers based on the problem type (ptype):

### Unconstrained (ptype='u')
```python
x = solver(fun, x0)
```
- `fun`: callable, accepts 1D numpy array, returns scalar float (NO gradient)
- `x0`: 1D numpy array, initial guess
- `x`: must return 1D numpy array of same length as x0

### Bound-constrained (ptype='b')
```python
x = solver(fun, x0, xl, xu)
```
- `xl`, `xu`: 1D numpy arrays (lower/upper bounds, may contain -inf/inf)

### Linearly constrained (ptype='l')
```python
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)
```
- `aub`: 2D array (m_ub × n), `bub`: 1D array (m_ub,)
- `aeq`: 2D array (m_eq × n), `beq`: 1D array (m_eq,)

### Nonlinearly constrained (ptype='n')
```python
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
```
- `cub`, `ceq`: callables, accept 1D array, return 1D array

## Common Adaptation Patterns

### scipy.optimize.minimize
```python
from scipy.optimize import minimize

def scipy_nelder_mead(fun, x0):
    result = minimize(fun, x0, method='Nelder-Mead')
    return result.x

def scipy_cobyla(fun, x0):
    result = minimize(fun, x0, method='COBYLA')
    return result.x
```

**Common mistake**: Using `minimize` with `jac=True` or gradient-based methods — OptiProfiler's `fun` does NOT provide gradients.

### NLopt
```python
import nlopt

def nlopt_cobyla(fun, x0):
    n = len(x0)
    opt = nlopt.opt(nlopt.LN_COBYLA, n)
    opt.set_min_objective(lambda x, grad: fun(x))
    opt.set_maxeval(1000 * n)
    x = opt.optimize(x0.tolist())
    return np.array(x)
```

**Note**: NLopt's objective function signature includes a `grad` argument even for derivative-free methods. Always ignore it.

### PDFO
```python
from pdfo import pdfo

def pdfo_solver(fun, x0):
    result = pdfo(fun, x0)
    return result.x
```

### Custom solver with extra parameters
```python
def my_custom_solver(fun, x0, max_iter=1000, tol=1e-8):
    # ... algorithm logic ...
    return x_best

# Wrap for OptiProfiler
def my_solver_for_benchmark(fun, x0):
    return my_custom_solver(fun, x0, max_iter=2000, tol=1e-10)
```

## Handling Edge Cases

### NaN from fun()
If OptiProfiler's fun returns NaN (due to solver exception or random_nan feature):
```python
def robust_solver(fun, x0):
    def safe_fun(x):
        val = fun(x)
        if np.isnan(val) or np.isinf(val):
            return 1e30  # large penalty
        return val
    # ... use safe_fun instead of fun ...
```

### Evaluation budget
OptiProfiler limits evaluations to `ceil(max_eval_factor * n)`. After this budget, further calls to `fun` will raise `StopIteration`. Solvers should either:
1. Have their own termination criteria that stop before the budget, or
2. Catch StopIteration and return the best point found so far.

### Return value shape
The returned `x` MUST be a 1D numpy array with the same length as `x0`. Common errors:
- Returning a scalar instead of array (for n=1 problems)
- Returning a 2D array (n,1) instead of (n,)
- Returning a list instead of numpy array
