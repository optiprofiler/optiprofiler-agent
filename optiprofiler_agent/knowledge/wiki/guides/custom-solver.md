---
tags: [guide, solver, wrapper, adaptation]
sources: [_sources/python/benchmark.json, _sources/matlab/benchmark.json]
related: [concepts/solver-interface.md, troubleshooting/solver-compat.md, concepts/problem-types.md]
last_updated: 2025-04-13
---

# Writing Custom Solver Wrappers

OptiProfiler requires solvers to follow a specific
[signature](../concepts/solver-interface.md) depending on the problem type.
Most third-party solvers need a thin wrapper to adapt their interface.

## Python: Wrapping scipy.optimize.minimize

```python
def scipy_nelder_mead(fun, x0):
    from scipy.optimize import minimize
    res = minimize(fun, x0, method='Nelder-Mead')
    return res.x

def scipy_powell(fun, x0):
    from scipy.optimize import minimize
    res = minimize(fun, x0, method='Powell')
    return res.x

scores = benchmark([scipy_nelder_mead, scipy_powell], ptype='u')
```

## Python: Wrapping a Bound-Constrained Solver

```python
def scipy_lbfgsb(fun, x0, xl, xu):
    from scipy.optimize import minimize
    bounds = list(zip(xl, xu))
    res = minimize(fun, x0, method='L-BFGS-B', bounds=bounds)
    return res.x
```

## Python: Wrapping NLopt

```python
def nlopt_bobyqa(fun, x0, xl, xu):
    import nlopt
    n = len(x0)
    opt = nlopt.opt(nlopt.LN_BOBYQA, n)
    opt.set_min_objective(lambda x, grad: fun(x))
    opt.set_lower_bounds(xl)
    opt.set_upper_bounds(xu)
    return opt.optimize(x0)
```

## Python: Wrapping PDFO

```python
def pdfo_newuoa(fun, x0):
    from pdfo import pdfo
    res = pdfo(fun, x0, method='newuoa')
    return res.x
```

## Key Rules

1. **Return type**: Must return `np.ndarray` of shape `(n,)` (Python) or
   column vector (MATLAB)
2. **No gradients**: The `fun` argument provides only function values
3. **Named functions**: Use `def`, not `lambda`, for parallel execution
   (`n_jobs > 1`) — lambdas are not picklable
4. **Match the signature**: The wrapper must accept exactly the arguments
   for the chosen `ptype`

## See Also

- [Solver Interface](../concepts/solver-interface.md) — full signature table
- [Solver Compatibility](../troubleshooting/solver-compat.md) — common adaptation issues
- [Problem Types](../concepts/problem-types.md) — how ptype determines arguments
