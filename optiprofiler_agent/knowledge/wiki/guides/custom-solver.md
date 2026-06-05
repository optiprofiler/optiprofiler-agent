---
tags: [guide, solver, wrapper, adaptation]
sources: [_sources/python/benchmark.json, _sources/matlab/benchmark.json]
related: [concepts/solver-interface.md, troubleshooting/solver-compat.md, concepts/problem-types.md]
last_updated: 2026-06-05
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

## Python: Wrapping SciPy COBYQA for Nonlinear Constraints

For `ptype='n'`, OptiProfiler calls solvers as
`solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)`, with
`cub(x) <= 0` and `ceq(x) = 0`. SciPy `minimize` represents these with
`Bounds`, `LinearConstraint`, and `NonlinearConstraint`.

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize


def scipy_cobyqa_wrapper(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq, maxfev=200):
    constraints = []

    if bub.size > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if beq.size > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))

    c_ub_x0 = np.atleast_1d(cub(x0))
    if c_ub_x0.size > 0:
        constraints.append(NonlinearConstraint(cub, -np.inf, np.zeros_like(c_ub_x0)))

    c_eq_x0 = np.atleast_1d(ceq(x0))
    if c_eq_x0.size > 0:
        constraints.append(NonlinearConstraint(ceq, np.zeros_like(c_eq_x0), np.zeros_like(c_eq_x0)))

    result = minimize(
        fun,
        x0,
        method="COBYQA",
        bounds=Bounds(xl, xu),
        constraints=constraints,
        options={"maxfev": maxfev},
    )
    return result.x
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

## MATLAB: Wrapping fmincon for Nonlinear Constraints

For `ptype='n'`, OptiProfiler passes nonlinear inequality and equality
callbacks separately. `fmincon` expects one `nonlcon` callback returning
two outputs, so use `deal(cub(x), ceq(x))`.

```matlab
function x = fmincon_wrapper(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq, max_fun_evals)
    nonlcon = @(x) deal(cub(x), ceq(x));
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', max_fun_evals);
    x = fmincon(fun, x0, aub, bub, aeq, beq, xl, xu, nonlcon, options);
end
```

Then benchmark two named wrappers with different budgets:

```matlab
options.ptype = 'n';
options.problem_names = {'HS10', 'HS11', 'HS12'};
options.plibs = {'s2mpj'};
options.n_jobs = 1;
options.solver_names = {'fmincon short', 'fmincon long'};
scores = benchmark({@fmincon_short, @fmincon_long}, options);
```

## Key Rules

1. **Return type**: Must return `np.ndarray` of shape `(n,)` (Python) or
   column vector (MATLAB)
2. **No gradients**: The `fun` argument provides only function values
3. **Named functions**: Use `def`, not `lambda`, for parallel execution
   (`n_jobs > 1`) — lambdas are not picklable
4. **Match the signature**: The wrapper must accept exactly the arguments
   for the chosen `ptype`
5. **Adapt nonlinear constraints explicitly**: SciPy needs
   `NonlinearConstraint`; MATLAB `fmincon` needs one two-output
   `nonlcon` callback.

## See Also

- [Solver Interface](../concepts/solver-interface.md) — full signature table
- [Solver Compatibility](../troubleshooting/solver-compat.md) — common adaptation issues
- [Problem Types](../concepts/problem-types.md) — how ptype determines arguments
