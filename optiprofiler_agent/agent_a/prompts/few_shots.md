# Agent A — Few-shot Examples

## Example 1: Factual Query

**User**: What is the default value of `n_jobs`?

**Assistant**: The default value of `n_jobs` is `os.cpu_count()` (the number of CPU cores on your machine). For example, on an 8-core machine, it defaults to `8`. Set it to `1` to run in sequential mode.

---

## Example 2: Configuration Advice

**User**: I want to test some derivative-free solvers on 50-dimensional unconstrained problems. How should I configure it?

**Assistant**: Here is a suggested configuration using two derivative-free methods from scipy:

```python
from optiprofiler import benchmark
from scipy.optimize import minimize

def cobyla_solver(fun, x0):
    result = minimize(fun, x0, method='COBYLA')
    return result.x

def nelder_mead_solver(fun, x0):
    result = minimize(fun, x0, method='Nelder-Mead')
    return result.x

benchmark(
    [cobyla_solver, nelder_mead_solver],
    ptype="unconstrained",
    mindim=2,
    maxdim=50,
)
```

Note: OptiProfiler is designed for derivative-free optimization (DFO) benchmarking. The `fun` only provides function values — no gradient information. So use DFO methods like `COBYLA`, `Nelder-Mead`, `Powell`, or `COBYQA`.

---

## Example 3: Interface Adaptation — Unconstrained Solver

**User**: My solver has the signature `my_solver(objective, initial_point)`. How do I adapt it for OptiProfiler?

**Assistant**: OptiProfiler requires unconstrained solvers to accept `(fun, x0)` as the first two positional arguments. You need a simple wrapper:

```python
def my_solver_wrapper(fun, x0):
    return my_solver(objective=fun, initial_point=x0)
```

Then pass `my_solver_wrapper` to `benchmark()` along with at least one other solver (benchmark requires at least 2 solvers for comparison).

---

## Example 4: Interface Adaptation — Bound-constrained Solver

**User**: My solver takes `(fun, x0, bounds)` where bounds is a `scipy.optimize.Bounds` object. How do I use it with OptiProfiler?

**Assistant**: For bound-constrained problems, OptiProfiler passes `(fun, x0, xl, xu)` where `xl` and `xu` are 1-D numpy arrays for lower and upper bounds. You need a wrapper that converts to the format your solver expects:

```python
import numpy as np
from scipy.optimize import Bounds

def my_solver_wrapper(fun, x0, xl, xu):
    bounds = Bounds(xl, xu)
    result = my_solver(fun, x0, bounds=bounds)
    return result.x
```

---

## Example 5: Interface Adaptation — Full Constrained Solver

**User**: I have a solver that can handle all problem types (unconstrained, bounds, linear, nonlinear). How do I write a wrapper?

**Assistant**: OptiProfiler uses different signatures for different problem types, but you can write a single wrapper that handles all four by making the extra parameters optional:

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

def my_solver_wrapper(fun, x0, xl=None, xu=None, aub=None, bub=None,
                      aeq=None, beq=None, cub=None, ceq=None):
    bounds = None
    if xl is not None or xu is not None:
        lb = xl if xl is not None else np.full(len(x0), -np.inf)
        ub = xu if xu is not None else np.full(len(x0), np.inf)
        bounds = Bounds(lb, ub)

    constraints = []
    if aub is not None and bub is not None and len(bub) > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if aeq is not None and beq is not None and len(beq) > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    if cub is not None:
        constraints.append(NonlinearConstraint(cub, -np.inf, 0))
    if ceq is not None:
        constraints.append(NonlinearConstraint(ceq, 0, 0))

    result = minimize(fun, x0, method='COBYLA', bounds=bounds,
                      constraints=constraints if constraints else ())
    return result.x
```

OptiProfiler will pass only the parameters relevant to the problem type:
- Unconstrained: `(fun, x0)`
- Bound-constrained: `(fun, x0, xl, xu)`
- Linearly constrained: `(fun, x0, xl, xu, aub, bub, aeq, beq)`
- Nonlinearly constrained: `(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)`

Since the extra parameters have default value `None`, the same wrapper works for all cases.

---

## Example 6: Gradient-based Method Question

**User**: Can you help me benchmark BFGS against L-BFGS?

**Assistant**: I should point out that OptiProfiler is primarily designed for **derivative-free optimization (DFO)** benchmarking. The `fun` passed to solvers provides only function values — no gradient or Hessian information. Methods like BFGS and L-BFGS internally approximate gradients via finite differences, which consumes extra function evaluations counted by OptiProfiler, making the comparison unfair against true DFO methods.

I'd recommend using derivative-free solvers instead, such as `COBYLA`, `Nelder-Mead`, `Powell`, or `COBYQA`. Would you like me to set up a benchmark comparing two of these DFO methods?

---

## Example 7: MATLAB Configuration

**User**: I want to benchmark fminsearch and patternsearch on unconstrained problems in MATLAB.

**Assistant**: Here's how to set it up in MATLAB. Note that options are passed as a **struct**, not as name-value pairs:

```matlab
% Create solver wrappers
function x = fminsearch_wrapper(fun, x0)
    x = fminsearch(fun, x0);
end

function x = patternsearch_wrapper(fun, x0)
    x = patternsearch(fun, x0, [], [], [], [], [], [], [], optimoptions('patternsearch', 'Display', 'off'));
end

% Configure options as a struct
options.ptype = 'u';
options.mindim = 2;
options.maxdim = 20;
options.plibs = {'s2mpj'};
options.solver_names = {'fminsearch', 'patternsearch'};

% Run benchmark
scores = benchmark({@fminsearch_wrapper, @patternsearch_wrapper}, options);
```

Both `fminsearch` (Nelder-Mead) and `patternsearch` are derivative-free methods, making them suitable for OptiProfiler benchmarking.

---

## Example 8: MATLAB Problem Library

**User**: How do I use s2mpj in MATLAB?

**Assistant**: S2MPJ is the default problem library and is bundled with OptiProfiler. After running `setup` in the OptiProfiler root directory, it's ready to use.

To explicitly select it:

```matlab
options.plibs = {'s2mpj'};
options.ptype = 'u';
options.mindim = 2;
options.maxdim = 10;

scores = benchmark({@solver1, @solver2}, options);
```

To filter specific problems by name:

```matlab
options.plibs = {'s2mpj'};
options.problem_names = {'ROSENBR', 'BEALE'};

scores = benchmark({@solver1, @solver2}, options);
```

---

## Example 9: Results Question

**User**: Why is my highest score only 0.85, not 1.0?

**Assistant**: By default, scores are averaged across all tolerance levels (`normalized_scores=True`). If your solver is only the best at some tolerance levels, the averaged score will be below 1.0. You can examine the per-tolerance profile plots for a detailed breakdown.
