---
tags: [troubleshooting, solver, compatibility, wrapper]
sources: [_sources/python/benchmark.json, _sources/matlab/benchmark.json]
related: [troubleshooting/common-errors.md, concepts/solver-interface.md, guides/custom-solver.md]
last_updated: 2026-06-05
---

# Solver Compatibility

This page covers common issues when adapting third-party optimization
solvers to work with OptiProfiler.

## Python: scipy.optimize.minimize

SciPy's `minimize` returns an `OptimizeResult` object, not an array.

```python
# WRONG
def bad_solver(fun, x0):
    return minimize(fun, x0, method='Nelder-Mead')  # Returns OptimizeResult!

# CORRECT
def good_solver(fun, x0):
    return minimize(fun, x0, method='Nelder-Mead').x  # Returns np.ndarray
```

For bound-constrained problems:

```python
def scipy_lbfgsb(fun, x0, xl, xu):
    bounds = list(zip(xl, xu))
    return minimize(fun, x0, method='L-BFGS-B', bounds=bounds).x
```

For nonlinearly constrained problems, OptiProfiler provides `cub` and
`ceq` separately. SciPy `minimize` expects constraint objects, so wrap
`cub(x) <= 0` as `NonlinearConstraint(cub, -np.inf, zeros)` and
`ceq(x) = 0` as `NonlinearConstraint(ceq, zeros, zeros)`, where `zeros`
matches the vector length returned by the callback. Linear constraints
become `LinearConstraint(aub, -np.inf, bub)` and
`LinearConstraint(aeq, beq, beq)`.

```python
import numpy as np
from scipy.optimize import NonlinearConstraint

c_ub_x0 = np.atleast_1d(cub(x0))
ineq = NonlinearConstraint(cub, -np.inf, np.zeros_like(c_ub_x0))

c_eq_x0 = np.atleast_1d(ceq(x0))
eq = NonlinearConstraint(ceq, np.zeros_like(c_eq_x0), np.zeros_like(c_eq_x0))
```

## Python: NLopt

NLopt uses a different objective function convention:

```python
def nlopt_solver(fun, x0):
    import nlopt
    opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
    opt.set_min_objective(lambda x, grad: fun(x))  # grad is unused in DFO
    return opt.optimize(x0)
```

## Python: PDFO

```python
def pdfo_solver(fun, x0):
    from pdfo import pdfo
    return pdfo(fun, x0, method='newuoa').x
```

## MATLAB: fminsearch

```matlab
function x = my_solver(fun, x0)
    x = fminsearch(fun, x0);
end
```

## MATLAB: fmincon Nonlinear Constraints

`fmincon` expects a single nonlinear constraint function returning two
outputs. Convert OptiProfiler's `cub` and `ceq` callbacks with `deal`:

```matlab
function x = fmincon_wrapper(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
    nonlcon = @(x) deal(cub(x), ceq(x));
    x = fmincon(fun, x0, aub, bub, aeq, beq, xl, xu, nonlcon);
end
```

## Common Pitfalls

1. **Returning the wrong object**: Always return the solution vector,
   not a result object or tuple
2. **Shape mismatch**: Ensure the returned array has shape `(n,)` in
   Python or n x 1 in MATLAB
3. **Extra arguments**: If a solver needs extra parameters, use a closure
   or partial application
4. **Gradient-based solvers**: Solvers requiring gradients will not work
   since `fun` provides only values

## See Also

- [Common Errors](common-errors.md) — error messages and fixes
- [Solver Interface](../concepts/solver-interface.md) — required signatures
- [Custom Solver Guide](../guides/custom-solver.md) — step-by-step guide
