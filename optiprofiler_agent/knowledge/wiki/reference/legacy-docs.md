---
tags: [reference, source-backed, legacy-docs, examples]
sources: [common/concepts.md, common/solver_interface.md, python/examples.md, python/installation.md, python/problem_libs.md, matlab/examples.md, matlab/installation.md, matlab/problem_libs.md, profiles/feature_effects.md, profiles/methodology.md, profiles/solver_traits.md, debugging/common_errors.md, debugging/solver_compat.md]
related: []
last_updated: 2026-06-18
generated: true
---

# Source Reference: Legacy Docs And Examples

This page mirrors bundled Markdown knowledge sources exactly.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing a source.

## common/concepts.md

- Source SHA256: `cb73106026a5874144d10a956ef27e5ae92f47053653db08de6eefdbaca23d67`

```markdown
# OptiProfiler Core Concepts

## What is OptiProfiler?

OptiProfiler is a platform for benchmarking optimization solvers.
It supports both Python and MATLAB, with nearly identical APIs.
It generates performance profiles, data profiles, and log-ratio profiles
to compare solver effectiveness across standardized test problem sets.

## Derivative-Free Optimization (DFO)

OptiProfiler is designed for **derivative-free optimization** benchmarking.
The objective function `fun` returns **only a scalar function value**.
No gradient, Jacobian, or Hessian information is available.

Every call to `fun` is counted internally and used for performance scoring.
Methods that internally approximate gradients via finite differences
consume extra `fun` evaluations, making them generally unsuitable
for DFO benchmarking.

**Recommended DFO methods** (see language-specific guides for details).

## Solver Requirements

- `benchmark()` requires **at least 2 solvers** for comparison.
- Each solver must follow a specific signature depending on the problem type.
- Solvers must return the solution vector (not a dict or result object).

## Four Problem Types

| Type | Signature Pattern |
|------|------------------|
| Unconstrained | `solver(fun, x0)` |
| Bound-constrained | `solver(fun, x0, xl, xu)` |
| Linearly constrained | `solver(fun, x0, xl, xu, aub, bub, aeq, beq)` |
| Nonlinearly constrained | `solver(fun, x0, ..., cub, ceq)` |

## Profiles and Scoring

- **Performance profiles**: fraction of problems solved within a factor of the best solver
- **Data profiles**: fraction of problems solved as a function of computational budget
- **Log-ratio profiles**: pairwise comparison (exactly 2 solvers only)
- **Scores**: by default, average of history-based performance profiles across tolerances

## Output Structure

Running `benchmark()` creates:
- `<benchmark_id>/<feature_stamp>/` directory with per-problem results
- `summary.pdf` with all profile plots
- Return values: `solver_scores`, `profile_scores` (4D), `curves`

## Additional Notes

- **Log-ratio profiles** are available only when there are exactly 2 solvers.
- The `load` option allows reloading a previous experiment to redraw profiles with different options.
- More information: https://www.optprof.com

```

## common/solver_interface.md

- Source SHA256: `412bb5995c7001eae95f472ca5ae2534b8e5ff3fa7863b8bd0f54e9109c1b437`

````markdown
# Solver Function Interface Specification

OptiProfiler benchmarks **derivative-free optimization (DFO)** solvers.
`benchmark()` requires **at least 2 solvers**.

## Python Signatures

```python
# Unconstrained
def solver(fun, x0) -> numpy.ndarray: ...

# Bound-constrained
def solver(fun, x0, xl, xu) -> numpy.ndarray: ...

# Linearly constrained
def solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray: ...

# Nonlinearly constrained
def solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray: ...

# Universal wrapper (handles all types)
def solver(fun, x0, xl=None, xu=None, aub=None, bub=None,
           aeq=None, beq=None, cub=None, ceq=None) -> numpy.ndarray: ...
```

## MATLAB Signatures

```matlab
% Unconstrained
x = solver(fun, x0)

% Bound-constrained
x = solver(fun, x0, xl, xu)

% Linearly constrained
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)

% Nonlinearly constrained
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
```

## Key Differences

| Aspect | Python | MATLAB |
|--------|--------|--------|
| Solvers arg | list of callables | cell array of function handles |
| Options | keyword arguments | struct |
| Vectors | numpy 1-D arrays (n,) | column vectors (n×1) |
| Return | numpy.ndarray | column vector |

````

## python/examples.md

- Source SHA256: `e74ffb253650d4c8ef110b3972580146587330a19abe91bd39947f3089e5fcb1`

````markdown
# Python Examples

> **OptiProfiler benchmarks derivative-free optimization (DFO) solvers.**
> `fun` provides ONLY function values — no gradient or Hessian.
> `benchmark()` requires at least 2 solvers.

---

# Usage for Python

OptiProfiler provides a benchmark() function. This is the main entry point to the package. It benchmarks given solvers on the selected test suite.

We provide below simple examples on how to use OptiProfiler in Python. For more details on the signature of the benchmark() function, please refer to the Python API documentation.

## Examples

### Example 1: first example to try out

Let us first try to benchmark two callable optimization solvers solver1 and solver2 on the default test suite. (Note that each solver must accept signatures mentioned in the Cautions part of the benchmark() function according to the type of problems you want to solve.)

To do this, run:

```python
from optiprofiler import benchmark

scores = benchmark([solver1, solver2])
```

This will benchmark the two solvers under the default test setting, which means 'plain' feature (see Feature) and unconstrained problems from the default problem library whose dimension is smaller or equal to 2. It will also return the scores of the two solvers based on the profiles.

There will be a new folder named out in the current working directory, which contains a subfolder named plain_<timestamp> with all the detailed results.

Additionally, a PDF file named summary.pdf is generated, summarizing all the performance profiles and data profiles.

The subfolder test_log contains diagnostic files for the experiment. In particular, test_log/report.txt records selected problem names, timing information, and special cases detected while building the profiles: problems where merit_init = phi(x_0) = inf (all solvers are declared to pass that problem/run), solver runs that terminated abnormally, and solver outputs that were replaced by the initial point as an output-based penalty. The file test_log/log.txt contains the messages printed during the run.

### Example 2: one step further by adding options

You can also add options to the benchmark function. For example, if you want to benchmark three solvers solver1, solver2, and solver3 on the test suite with the 'noisy' feature and all the unconstrained and bound-constrained problems with dimension between 6 and 10 from the default problem set, you can run:

```python
from optiprofiler import benchmark

scores = benchmark(
    [solver1, solver2, solver3],
    ptype='ub',
    mindim=6,
    maxdim=10,
    feature_name='noisy',
)
```

This will create the corresponding folders out/noisy_<timestamp> and files as in Example 1. More details on the options can be found in the benchmark() function documentation.

For the deterministic noisy variant from Moré and Wild’s benchmarking model, set noise_mode='deterministic'. If n_runs is not provided, OptiProfiler uses one run for this deterministic feature unless solver_isrand marks at least one solver as randomized, in which case OptiProfiler uses five runs as usual.

```python
scores = benchmark(
    [solver1, solver2, solver3],
    feature_name='noisy',
    noise_mode='deterministic',
    noise_map='chebyshev',
)
```

By default, n_jobs is set conservatively to about half of the available workers instead of all workers. For the most reproducible timing experiments, set n_jobs explicitly, for example n_jobs=1 for sequential runs.

### Example 3: useful optionload

OptiProfiler provides a practically useful option named load. This option allows you to load the results from a previous benchmarking run (without solving all the problems again) and use them to draw new profiles with different options. For example, if you have just run Example 2 and OptiProfiler has finished the job and successfully created the folder out in the current working directory, you can run:

```python
from optiprofiler import benchmark

scores = benchmark(
    load='latest',
    solvers_to_load=[0, 2],
    ptype='u',
    mindim=7,
    maxdim=9,
)
```

This will directly draw the profiles for the solver1 and solver3 with the 'noisy' feature and all the unconstrained problems with dimension between 7 and 9 selected from the previous run. The results will also be saved under the current directory with a new subfolder named noisy_<timestamp> with the new timestamp.

### Example 4: testing parametrized solvers

If you want to benchmark a solver with one variable parameter, you can define callables by looping over the parameter values. For example, if solver accepts the signature solver(fun, x0, para), and you want to benchmark it with the parameter para taking values from 1 to 3, you can run:

```python
from optiprofiler import benchmark

def make_solver(para):
    def solver_wrapper(fun, x0):
        return solver(fun, x0, para)
    return solver_wrapper

solvers = [make_solver(i) for i in range(1, 4)]
solver_names = [f'solver{i}' for i in range(1, 4)]
scores = benchmark(solvers, solver_names=solver_names)
```

> **Note**
> We use named functions (def) instead of lambda expressions here so that the benchmark can still run in parallel when n_jobs > 1. See Callable arguments must be picklable when running in parallel for the full list of affected callables and the rationale.

### Example 5: customizing the test suite

OptiProfiler allows you to customize the test suite by creating your own feature and loading your own problem library. For example, if you want to create a new feature that adds noise to the objective function and perturbs the initial guess at the same time, you can try the following:

```python
from optiprofiler import benchmark

def mod_fun(x, rand_stream, problem):
    return problem.fun(x) + 1e-3 * rand_stream.standard_normal()

def mod_x0(rand_stream, problem):
    return problem.x0 + 1e-3 * rand_stream.standard_normal(problem.n)

scores = benchmark(
    [solver1, solver2],
    feature_name='custom',
    mod_fun=mod_fun,
    mod_x0=mod_x0,
)
```

> **Note**
> Again, mod_fun and mod_x0 are defined with def rather than lambda so that the benchmark can run in parallel when n_jobs > 1. See Callable arguments must be picklable when running in parallel for details.

If you want to benchmark solvers based on your own problem library, you should do the following three steps:

- Create a directory anywhere on your system (e.g., '/path/to/my_libs'), and create a subfolder inside it for your problem library (e.g., 'myproblems'), so the structure looks like: /path/to/my_libs/ └── myproblems/ └── myproblems_tools.py

- In myproblems_tools.py, implement two functions: myproblems_load: A function that accepts a string representing the optimization problem name and returns a Problem instance. myproblems_select: A function that accepts a dictionary to specify desired problem characteristics and returns a list of problem names that satisfy the requirements. In general, the module should be named <library_name>_tools.py and the two functions should be named <library_name>_load and <library_name>_select. OptiProfiler does not infer the library name from other *_tools.py files; for example, a library named myproblems must provide myproblems_tools.py.

- Use the benchmark function with the custom_problem_libs_path option pointing to your directory. This path can be either the parent directory containing custom libraries or the directory of one custom library. For example, to use both the default S2MPJ library and your custom library 'myproblems', you can run:

```python
scores = benchmark(
    [solver1, solver2],
    plibs=['s2mpj', 'myproblems'],
    custom_problem_libs_path='/path/to/my_libs',
)
```

For a detailed guide on the required structure of a custom problem library, please refer to the guide on our website or the README on GitHub.

### Example 6: wrapping SciPy solvers with nonlinear constraints

(See also the file in the repository: python/examples/scipy_cobyqa_wrapper.py)

For nonlinearly constrained problems, OptiProfiler calls each solver with the signature

```python
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
```

where cub(x) <= 0 contains the nonlinear inequality constraints and ceq(x) = 0 contains the nonlinear equality constraints. SciPy’s minimize interface represents constraints with objects such as Bounds, LinearConstraint, and NonlinearConstraint. The adapter in the wrapper below is the conversion from OptiProfiler’s callback signature to SciPy’s constraint objects: linear constraints become LinearConstraint objects, while cub and ceq are wrapped as NonlinearConstraint objects with bounds (-inf, 0) and (0, 0), respectively. The SciPy documentation for COBYQA and the optimization tutorial show this object-based constraint interface; see also the API references for LinearConstraint and NonlinearConstraint.

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

def scipy_cobyqa_wrapper(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq,
                         maxfev=200):
    constraints = []

    if bub.size > 0:
        # OptiProfiler gives aub @ x <= bub; SciPy stores it as a
        # LinearConstraint with lower bound -inf and upper bound bub.
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if beq.size > 0:
        # Equality constraints use identical lower and upper bounds.
        constraints.append(LinearConstraint(aeq, beq, beq))

    c_ub_x0 = np.atleast_1d(cub(x0))
    if c_ub_x0.size > 0:
        # Convert cub(x) <= 0 to a SciPy NonlinearConstraint.
        constraints.append(NonlinearConstraint(cub, -np.inf, np.zeros_like(c_ub_x0)))
    c_eq_x0 = np.atleast_1d(ceq(x0))
    if c_eq_x0.size > 0:
        # Convert ceq(x) = 0 by using zero lower and upper bounds.
        constraints.append(NonlinearConstraint(ceq, np.zeros_like(c_eq_x0),
                                               np.zeros_like(c_eq_x0)))

    result = minimize(
        fun,
        x0,
        method='COBYQA',
        bounds=Bounds(xl, xu),
        constraints=constraints,
        options={'maxfev': maxfev},
    )
    return result.x
```

Then pass the wrapper to benchmark as an ordinary solver. Since benchmark compares at least two solvers, this example compares two COBYQA wrappers with different function-evaluation budgets:

```python
def scipy_cobyqa_short(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq):
    return scipy_cobyqa_wrapper(
        fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq, maxfev=100
    )

def scipy_cobyqa_long(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq):
    return scipy_cobyqa_wrapper(
        fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq, maxfev=200
    )

scores = benchmark(
    [scipy_cobyqa_short, scipy_cobyqa_long],
    solver_names=['SciPy COBYQA short', 'SciPy COBYQA long'],
    ptype='n',
    problem_names=['HS10', 'HS11', 'HS12'],
    mindim=2,
    maxdim=5,
    max_eval_factor=500,
    plibs=['s2mpj'],
    draw_hist_plots='none',
    n_jobs=1,
)
```

## Cautions

### Callable arguments must be picklable when running in parallel

When n_jobs > 1, OptiProfiler dispatches problems to worker processes via multiprocessing. The following callable arguments are sent across process boundaries and must therefore be picklable:

- the entries of solvers;

- feature options: distribution, noise_map, mod_x0, mod_affine, mod_bounds, mod_linear_ub, mod_linear_eq, mod_fun, mod_cub, mod_ceq;

- profile options: merit_fun, score_fun, score_weight_fun.

Lambda expressions and locally-defined nested functions are not picklable. If any of the callables above is a lambda, OptiProfiler detects the failure when serializing the worker arguments and silently falls back to sequential mode (n_jobs = 1), which can be much slower.

To enable parallel execution, define these callables as module-level functions using def. For parametrized solvers, use a closure factory (see Example 4: testing parametrized solvers) or functools.partial() instead of a lambda.

````

## python/installation.md

- Source SHA256: `63d32f5d225afbedff913b624b7787e3d3dbff5efb7470cde307a23950a84cef`

````markdown
# Python Installation

Install OptiProfiler from PyPI:

```bash
pip install optiprofiler
```

You can also install OptiProfiler from conda-forge:

```bash
conda install conda-forge::optiprofiler
```

> **Note**
> OptiProfiler includes the S2MPJ problem library by default. If you also want to use the PyCUTEst problem library (available on Linux and macOS only), please follow the PyCUTEst installation guide.

````

## python/problem_libs.md

- Source SHA256: `5e8d05a1c6e5396d8ba98e129ebf58df33407ac6f6fb794c90703d4b9e6cadde`

````markdown
# Python Problem Libraries

## Built-in Libraries

- **s2mpj**: Default. Pure Python, no extra installation.
- **pycutest**: Requires separate installation. Linux and macOS only.
  See https://jfowkes.github.io/pycutest/

## Custom Libraries

Use `custom_problem_libs_path` to add your own:

```
/path/to/my_libs/
└── myproblems/
    └── myproblems_tools.py  (implements myproblems_load + myproblems_select)
```

```python
benchmark(
    [solver1, solver2],
    plibs=['s2mpj', 'myproblems'],
    custom_problem_libs_path='/path/to/my_libs',
)
```

````

## matlab/examples.md

- Source SHA256: `f27860de28fb750e06c77bfd44e1c4318c090219d0da006a3fe9211c0cec2fd2`

````markdown
# Matlab Examples

> **OptiProfiler benchmarks derivative-free optimization (DFO) solvers.**
> `fun` provides ONLY function values — no gradient or Hessian.
> `benchmark()` requires at least 2 solvers.

---

# Usage for MATLAB

OptiProfiler provides a benchmark function. This is the main entry point to the package. It benchmarks given solvers on the selected test suite.

We provide below simple examples on how to use OptiProfiler in MATLAB. For more details on the signature of the benchmark function, please refer to the MATLAB API documentation.

## Examples

### Example 1: first example to try out

(See also the file in the repository: matlab/examples/example1.m)

Let us first try to benchmark two callable optimization solvers solver1 and solver2 (e.g., fminsearch and fminunc in MATLAB Optimization Toolbox) on the default test suite. (Note that each solver must accept signatures mentioned in the Cautions part of the benchmark function according to the type of problems you want to solve.)

To do this, run:

```matlab
scores = benchmark({@solver1, @solver2})
```

This will benchmark the two solvers under the default test setting, which means 'plain' feature (see Feature) and unconstrained problems from the default problem library whose dimension is smaller or equal to 2. It will also return the scores of the two solvers based on the profiles.

There will be a new folder named out in the current working directory, which contains a subfolder named plain_<timestamp> with all the detailed results.

Additionally, a PDF file named summary.pdf is generated, summarizing all the performance profiles and data profiles.

The subfolder test_log contains diagnostic files for the experiment. In particular, test_log/report.txt records selected problem names, timing information, and special cases detected while building the profiles: problems where merit_init = phi(x_0) = Inf (all solvers are declared to pass that problem/run), solver runs that terminated abnormally, and solver outputs that were replaced by the initial point as an output-based penalty. The file test_log/log.txt contains the messages printed during the run.

### Example 2: one step further by adding options

(See also the file in the repository: matlab/examples/example2.m)

You can also add options to the benchmark function. For example, if you want to benchmark three solvers solver1, solver2, and solver3 on the test suite with the 'noisy' feature and all the unconstrained and bound-constrained problems with dimension between 6 and 10 from the default problem set, you can run:

```matlab
options.ptype = 'ub';
options.mindim = 6;
options.maxdim = 10;
options.feature_name = 'noisy';
scores = benchmark({@solver1, @solver2, @solver3}, options)
```

This will create the corresponding folders out/noisy_<timestamp> and files as in Example 1. More details on the options can be found in the benchmark function documentation.

For the deterministic noisy variant from Moré and Wild’s benchmarking model, set options.noise_mode = 'deterministic'. If options.n_runs is not provided, OptiProfiler uses one run for this deterministic feature unless options.solver_isrand marks at least one solver as randomized, in which case OptiProfiler uses five runs as usual.

```matlab
options.feature_name = 'noisy';
options.noise_mode = 'deterministic';
options.noise_map = 'chebyshev';
scores = benchmark({@solver1, @solver2, @solver3}, options)
```

By default, n_jobs is set conservatively to about half of the available workers instead of all workers. For the most reproducible timing experiments, set options.n_jobs explicitly, for example options.n_jobs = 1 for sequential runs.

### Example 3: useful optionload

(See also the file in the repository: matlab/examples/example3.m)

OptiProfiler provides a practically useful option named load. This option allows you to load the results from a previous benchmarking run (without solving all the problems again) and use them to draw new profiles with different options. For example, if you have just run Example 2 and OptiProfiler has finished the job and successfully created the folder out in the current working directory, you can run:

```matlab
options.load = 'latest';
options.solvers_to_load = [1, 3];
options.ptype = 'u';
options.mindim = 7;
options.maxdim = 9;
scores = benchmark(options)
```

This will directly draw the profiles for the solver1 and solver3 with the 'noisy' feature and all the unconstrained problems with dimension between 7 and 9 selected from the previous run. The results will also be saved under the current directory with a new subfolder named noisy_<timestamp> with the new timestamp.

### Example 4: testing parametrized solvers

(See also the file in the repository: matlab/examples/example4.m)

If you want to benchmark a solver with one variable parameter, you can define function handles by looping over the parameter values. For example, if solver accepts the signature @(fun, x0, para), and you want to benchmark it with the parameter para taking values from 1 to 3, you can run:

```matlab
solvers = cell(1, 3);
options.solver_names = cell(1, 3);
for i = 1:3
    solvers{i} = @(fun, x0) solver(fun, x0, i);
    options.solver_names{i} = ['solver' num2str(i)];
end
scores = benchmark(solvers, options)
```

### Example 5: customizing the test suite

(See also the file in the repository: matlab/examples/example5.m)

OptiProfiler allows you to customize the test suite by creating your own feature and loading your own problem library. For example, if you want to create a new feature that adds noise to the objective function and perturbs the initial guess at the same time, you can try the following:

```matlab
options.feature_name = 'custom';
options.mod_fun = @(x, rand_stream, problem) problem.fun(x) + 1e-3 * rand_stream.randn(1);
options.mod_x0 = @(rand_stream, problem) problem.x0 + 1e-3 * rand_stream.randn(problem.n, 1);
scores = benchmark({@solver1, @solver2}, options)
```

If you want to benchmark solvers based on your own problem library, you should do the following three steps:

- Create a new subfolder (e.g., 'myproblems') within the 'problems' folder located in the optiprofiler project root directory.

- Implement two MATLAB functions:

- Use the benchmark function as before, but specify your desired problem libraries. For example, to use both the default S2MPJ library and your custom library in the subfolder 'myproblems', you can run:

```matlab
options.plibs = {'s2mpj', 'myproblems'};
scores = benchmark({@solver1, @solver2}, options)
```

You may also refer to the README file in the 'problems' folder for a detailed guide on how to create and use your own problem library via the OptiProfiler package.

### Example 6: wrapping solvers with nonlinear constraints

(See also the file in the repository: matlab/examples/example6.m)

For nonlinearly constrained problems, OptiProfiler calls each solver with the signature

```matlab
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
```

where cub(x) <= 0 contains the nonlinear inequality constraints and ceq(x) = 0 contains the nonlinear equality constraints. MATLAB solvers such as fmincon instead expect one nonlinear constraint callback nonlcon returning both values. The small but important adapter is deal: the expression @(x) deal(cub(x), ceq(x)) evaluates OptiProfiler’s two callbacks and returns them as the two outputs expected by fmincon, i.e., [c, ceq] = nonlcon(x). See the MathWorks documentation for fmincon nonlinear constraints and deal.

```matlab
function x = fmincon_short(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
    x = fmincon_wrapper(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq, 100);
end

function x = fmincon_long(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
    x = fmincon_wrapper(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq, 200);
end

function x = fmincon_wrapper(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq, max_fun_evals)
    % Convert OptiProfiler's separate nonlinear callbacks to fmincon's
    % two-output callback: [c, ceq] = nonlcon(x).
    nonlcon = @(x) deal(cub(x), ceq(x));
    options = optimoptions('fmincon', 'MaxFunctionEvaluations', max_fun_evals);
    x = fmincon(fun, x0, aub, bub, aeq, beq, xl, xu, nonlcon, options);
end
```

Then pass the wrappers to benchmark as ordinary solvers:

```matlab
options.ptype = 'n';
options.problem_names = {'HS10', 'HS11', 'HS12'};
options.plibs = {'s2mpj'};
options.mindim = 2;
options.maxdim = 5;
options.max_eval_factor = 500;
options.draw_hist_plots = 'none';
options.n_jobs = 1;
options.solver_names = {'fmincon short', 'fmincon long'};
scores = benchmark({@fmincon_short, @fmincon_long}, options)
```

````

## matlab/installation.md

- Source SHA256: `eaf332d51e6adc3b572d04c5dfc94056c972c3d9c0b8ba14a0b7822c23fa39cc`

````markdown
# MATLAB Installation

- Clone the repository using the following command:

```bash
git clone https://github.com/optiprofiler/optiprofiler.git
```

- In MATLAB, navigate to the root directory of this repository, where you can see a file named setup.m. Run the following command in the MATLAB command window:

```matlab
setup
```

The setup function performs the following tasks:

- Adds the necessary directories to the MATLAB search path.

- Clones the default problem libraries, including S2MPJ and MatCUTEst.

Note that the installation of MatCUTEst is optional. During the setup process, you will be asked whether you want to install it. Please be aware that MatCUTEst is only supported on Linux systems and is not available on macOS or Windows.

For automated environments (e.g., CI/CD scripts) where interactive input is not possible, you can bypass the prompt by providing an additional option to the setup function:

```matlab
setup(struct('install_matcutest', true))  % Or false if you do not need MatCUTEst
```

- If you want to uninstall the package, you can run:

```matlab
setup uninstall
```

````

## matlab/problem_libs.md

- Source SHA256: `2477cf32562ac113e4fb2b9ccb1c701d4a0ca6071e2d846e50d3632dbdbf6d1b`

````markdown
# MATLAB Problem Libraries

## Built-in Libraries

- **s2mpj**: Default. Bundled with OptiProfiler.
- **matcutest**: Requires setup. **Linux only.**
  See https://github.com/matcutest

## Custom Libraries

Create a subfolder in the `problems` directory:

```
problems/
└── myproblems/
    ├── myproblems_load.m
    └── myproblems_select.m
```

```matlab
options.plibs = {'s2mpj', 'myproblems'};
scores = benchmark({@solver1, @solver2}, options);
```

````

## profiles/feature_effects.md

- Source SHA256: `e332fa2cddda138806233406afbda07d8789d09c8c5cbe98e0a2692afda0fd1f`

```markdown
# Feature Effects on Solver Performance

OptiProfiler provides built-in features that modify test problems to evaluate solver robustness under different conditions. Understanding these effects is crucial for interpreting benchmark results.

## plain
- No modification to the original problem.
- Baseline for all comparisons.
- Results reflect pure algorithmic performance on clean problems.

## noisy
- Adds noise to the objective function and nonlinear constraints.
- Options: `noise_level` (default varies), `noise_type` (absolute/relative/mixed), `distribution` (gaussian/uniform).
- **Expected effect**: solvers using function value differences (like finite-difference methods) are more affected than direct-search methods. Model-based methods (e.g., NEWUOA, COBYQA) that build surrogate models may be more sensitive to noise than simplex-based methods (e.g., Nelder-Mead/fminsearch).
- **Interpretation tip**: if a solver performs well under `plain` but poorly under `noisy`, it likely relies on accurate function values.

## perturbed_x0
- Randomly perturbs the initial guess x0.
- Options: `distribution` (gaussian/spherical).
- **Expected effect**: tests solver robustness to starting point. Solvers with good global search ability are less affected.
- **Interpretation tip**: large performance drops indicate sensitivity to initialization.

## truncated
- Truncates objective function and constraints to a given precision.
- **Expected effect**: simulates limited-precision computation. Solvers that require high-precision function values (e.g., for finite differences) are more affected.

## permuted
- Randomly permutes the order of variables.
- **Expected effect**: tests whether a solver exploits variable ordering. Well-designed solvers should be invariant to permutation.

## linearly_transformed
- Applies a linear transformation using a positive diagonal matrix times a random orthogonal matrix.
- **Expected effect**: changes the coordinate system and scaling. Solvers that are not scale-invariant will be affected. This tests conditioning sensitivity.

## random_nan
- Randomly sets some objective/constraint values to NaN.
- Options: `nan_rate` (fraction of evaluations that return NaN).
- **Expected effect**: simulates solver calls that fail (e.g., simulation crashes). Solvers that handle NaN gracefully are more robust.

## unrelaxable_constraints
- Sets the objective function to infinity outside the feasible region.
- **Expected effect**: solvers that explore infeasible points aggressively will see infinite values. Tests feasibility-preserving behavior.

## nonquantifiable_constraints
- Replaces constraint values with binary: 0 (satisfied) or 1 (violated).
- **Expected effect**: removes quantitative constraint information, leaving only feasible/infeasible signals.

## quantized
- Quantizes the objective function and constraints to discrete levels.
- **Expected effect**: similar to truncated but with discrete steps. Tests solver behavior with discontinuous objectives.

```

## profiles/methodology.md

- Source SHA256: `c4d9b0883573f49c86cd6e49b04a88b6b089a03768e43605455e002f1a65b785`

```markdown
# Profile Methodology Reference

This document summarizes the mathematical foundations of performance profiles, data profiles, and log-ratio profiles as used in OptiProfiler, based on the established literature [Dolan & Moré 2002, Moré & Wild 2009] and the OptiProfiler paper [Huang, Ragonneau & Zhang 2026].

## Performance Profiles (Dolan & Moré, 2002)

Let P be a set of test problems and S a set of solvers. For each problem p and solver s, the **absolute cost** t_{p,s} is the number of function evaluations solver s needs to solve problem p up to a convergence test.

The **relative cost** (performance ratio) is:

    r_{p,s} = t_{p,s} / min{ t_{p,s} : s in S }

with the convention that infinity/infinity = infinity.

The **performance profile** of solver s is:

    rho_s(alpha) = (1/|P|) * |{ p in P : r_{p,s} <= alpha }|   for alpha >= 1

### How to read a performance profile:
- **At alpha=1**: the fraction of problems where this solver was the fastest (or tied for fastest). Higher is better — this measures **efficiency**.
- **As alpha -> infinity**: the fraction of problems this solver eventually solved (regardless of cost). Higher is better — this measures **robustness**.
- **A curve that is higher everywhere dominates** — the solver is both more efficient and more robust.
- **A crossover** (curves crossing) means one solver is more efficient but the other is more robust.

Performance profiles are considered the "gold standard" in optimization benchmarking (Gould & Scott, 2016). However, note the limitation: "we cannot necessarily assess the performance of one solver relative to another that is not the best" (Gould & Scott, 2016).

## Data Profiles (Moré & Wild, 2009)

The **data profile** of solver s is:

    delta_s(alpha) = (1/|P|) * |{ p in P : t_{p,s} / (n_p + 1) <= alpha }|   for alpha >= 0

where n_p is the dimension of problem p. The quantity t_{p,s}/(n_p+1) is the number of **simplex gradients** — function evaluations normalized by dimension plus one.

### How to read a data profile:
- Data profiles normalize by problem dimension, enabling **fair comparison across problems of different sizes**.
- The x-axis represents the **computational budget** in units of simplex gradients.
- At any given budget, the y-value shows the fraction of problems solved.
- Useful for answering: "Given a budget of K simplex gradients, which solver solves the most problems?"

## Log-ratio Profiles (Shi et al., 2023)

When comparing exactly **two solvers** s1 and s2, the log-ratio is:

    l_p = log2(t_{p,s1} / t_{p,s2})   for p in P\E

where E is the set of problems both solvers fail. The log-ratio profile is these values sorted in ascending order.

### How to read a log-ratio profile:
- Each bar represents one problem.
- **Positive bars**: s1 used more evaluations than s2 (s2 is better on that problem).
- **Negative bars**: s2 used more evaluations than s1 (s1 is better on that problem).
- **Light bars at extremes**: problems where both solvers failed (extended definition).
- The **shaded area** corresponds to the AUC of the performance profile (proven equivalence in Huang et al., 2026).

### Equivalence with performance profiles:
OptiProfiler proves that log-ratio profiles are equivalent to performance profiles when comparing two solvers. Specifically, the log-ratio profile is an inverse function of the performance profile.

## History-based vs Output-based Costs

OptiProfiler provides two methods for measuring the absolute cost:

### History-based cost:
The number of function evaluations to **first reach** a point passing the convergence test. This measures **intrinsic search efficiency** regardless of the solver's stopping rule.

### Output-based cost:
If the solver's final output passes the convergence test, the cost is the total number of evaluations used. Otherwise, the cost is infinity. This measures both the solver's ability to find solutions **and** its effectiveness in deciding when to stop.

**Key insight**: History-based and output-based profiles need not look similar. History-based emphasizes solver efficiency; output-based also reflects stopping criteria quality.

## Convergence Test

A point x passes the convergence test on problem p with tolerance tau in [0,1] if:

    phi(x) <= phi* + tau * (phi(x0) - phi*)

where x0 is the initial guess and phi* is the best merit value achieved by any solver.

Tolerance levels range from tau=0.1 (low accuracy) to tau=10^{-10} (high accuracy). Lower tolerance = stricter convergence = fewer problems solved = lower profile curves.

## Scoring (AUC)

OptiProfiler scores solvers by computing the **area under the curve** (AUC) of their profiles. Larger AUC = better performance.

- For performance and data profiles: AUC is computed up to a truncation point (1.1x the last jump).
- For log-scale axes: AUC is computed with respect to log-transformed coordinates.
- For multiple runs (n_runs > 1): score is based on the average profile across runs.
- **Default score**: average of all history-based performance profile AUCs across all tolerances.

## Budget and Stopping

OptiProfiler enforces a budget-based stopping mechanism:
- `maxfun = ceil(max_eval_factor * n)` where n is the problem dimension.
- If a solver exceeds maxfun, objective/constraint values are set to those at the last evaluated point.
- If the solver doesn't stop after 2*maxfun evaluations, the run is terminated.
- History-based profiles use history truncated at maxfun; output-based profiles use the solver's returned output.

## Multiple Runs (n_runs > 1)

When n_runs > 1 (essential for stochastic features like `noisy`):
- Performance and data profiles show the **average curve** across runs, with error bars (min/max by default).
- Log-ratio profiles treat the same problem in different runs as **distinct problems**, enlarging the problem set.
- Scores for perf/data profiles use the average profile's AUC; log-ratio scores are computed on the enlarged set.

```

## profiles/solver_traits.md

- Source SHA256: `4666284ee36e5a7fb467f15c25692c71a375db98ae1acff5a3c83992fd6e30cc`

```markdown
# Known Solver Traits for Benchmark Interpretation

This document provides prior knowledge about commonly benchmarked DFO solvers, useful for interpreting OptiProfiler results.

## NEWUOA (from PRIMA)
- **Type**: Model-based trust-region, quadratic interpolation.
- **Strengths**: Very competitive on smooth unconstrained problems. Generally the best-performing DFO solver on classical test sets (Moré & Wild, 2009).
- **Weaknesses**: May struggle with noisy problems since the quadratic model is sensitive to function value noise. Not designed for constrained problems.

## COBYLA (scipy.optimize or PRIMA)
- **Type**: Model-based trust-region with linear approximation of objective and constraints.
- **Strengths**: Handles nonlinear constraints. Uses linear models, which makes it more **noise-robust** than quadratic-model methods.
- **Weaknesses**: Linear models are less accurate than quadratic models, so convergence can be slower on smooth problems. May struggle with high-precision convergence.
- **Typical profile behavior**: Higher scores at loose tolerances, scores drop at tight tolerances (precision cliff).

## COBYQA
- **Type**: Model-based trust-region with quadratic approximation for both objective and constraints.
- **Strengths**: More accurate models than COBYLA, better convergence on smooth problems.
- **Weaknesses**: Quadratic models are more sensitive to noise than linear models. More computationally expensive per iteration.
- **Typical profile behavior**: May start slower than COBYLA but achieve better final accuracy.

## Nelder-Mead Simplex (fminsearch in MATLAB)
- **Type**: Direct search, simplex-based.
- **Strengths**: Simple, no model building. Relatively robust to noise since it only uses function value comparisons (ordering).
- **Weaknesses**: Known to converge to non-stationary points in dimensions > 2 (McKinnon, 1998). Slow convergence rate. No constraint handling.
- **Typical profile behavior**: Reasonable at low precision, poor at high precision.

## fminunc (MATLAB)
- **Type**: Finite-difference BFGS (when gradient not provided).
- **Strengths**: Fast convergence using quasi-Newton direction with finite-difference gradients.
- **Weaknesses**: Finite differences are very sensitive to noise. Not a true DFO method — uses O(n) extra function evaluations per iteration for gradient approximation.
- **Typical profile behavior**: Competitive on smooth problems, dramatically worse under noisy/truncated features.

## General Interpretation Guidelines

When comparing solvers in OptiProfiler results:

1. **Model-based vs direct-search**: Model-based methods (NEWUOA, COBYLA, COBYQA) generally converge faster but may be more sensitive to noise. Direct-search methods (Nelder-Mead) are more robust but slower.

2. **Linear vs quadratic models**: Linear models (COBYLA) are more noise-robust but converge slower. Quadratic models (COBYQA, NEWUOA) converge faster but need more accurate function values.

3. **Finite-difference methods** (fminunc) should be expected to perform poorly under noisy/truncated/random_nan features, as they rely on accurate function value differences.

4. **Budget sensitivity**: Some solvers use more evaluations per iteration (e.g., finite-difference methods use O(n) evaluations for gradient approximation). Data profiles (normalized by dimension) help compare evaluation efficiency fairly.

```

## debugging/common_errors.md

- Source SHA256: `5d41ae6e2bc980e554865f63593b52f2e4ff8acaf59f9302ad4d055d15893b12`

````markdown
# Common OptiProfiler Errors and Solutions

This document catalogs error patterns from the OptiProfiler source code with user-friendly explanations and fixes.

## 1. Solver Input Errors

### "At least two solvers must be given"
- **Type**: ValueError
- **Trigger**: `len(solvers) < 2` in `benchmark()`
- **Fix**: OptiProfiler requires at least 2 solvers for comparison. Add a second solver:
  ```python
  benchmark([my_solver, reference_solver])
  ```

### "The solvers must be a list of callables"
- **Type**: TypeError
- **Trigger**: `solvers` is not iterable, or contains non-callable items
- **Fix**: Ensure each solver is a function. Common mistake: passing the solver name as a string instead of the function itself:
  ```python
  # Wrong: benchmark(["my_solver", "other_solver"])
  # Right: benchmark([my_solver, other_solver])
  ```

### "Either solvers or the 'load' option must be given"
- **Type**: ValueError
- **Trigger**: No solvers provided and no `load` option
- **Fix**: Provide solvers or use `load` to resume a previous experiment.

## 2. Solver Signature Errors

### TypeError with solver arguments
- **Trigger**: Solver function does not accept the expected arguments
- **Expected signatures by problem type**:
  - Unconstrained (`u`): `solver(fun, x0)` — fun is callable, x0 is 1D numpy array
  - Bound-constrained (`b`): `solver(fun, x0, xl, xu)`
  - Linearly constrained (`l`): `solver(fun, x0, xl, xu, aub, bub, aeq, beq)`
  - Nonlinearly constrained (`n`): `solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)`
- **Key point**: `fun` provides ONLY function values (no gradients). This is DFO.
- **Fix**: Wrap your solver to match the expected signature.

### Solver return value errors
- **Trigger**: Solver returns something other than a 1D numpy array of length n
- **Symptom**: NumPy broadcasting/shape errors during post-processing
- **Fix**: Ensure `return x_best` where `x_best` is a 1D array with the same length as `x0`.

## 3. Feature and Option Errors

### "Unknown feature name: ..."
- **Type**: ValueError
- **Trigger**: `feature_name` not in the valid set
- **Valid features**: plain, noisy, perturbed_x0, truncated, permuted, linearly_transformed, random_nan, unrelaxable_constraints, nonquantifiable_constraints, quantized

### "Unknown option: ..."
- **Type**: ValueError
- **Trigger**: Passing an unrecognized keyword to `benchmark()`
- **Fix**: Check spelling. Only use documented option names (e.g., `n_runs`, `n_jobs`, `ptype`, `mindim`, `maxdim`, `feature_name`, `noise_level`).

### Feature option mismatches
- **Type**: ValueError
- **Trigger**: Passing options that don't belong to the chosen feature (e.g., `noise_level` with `plain`)
- **Fix**: Only pass options relevant to the selected feature.

## 4. Problem Selection Errors

### "ptype" errors
- **Type**: ValueError
- **Trigger**: `ptype` contains characters other than u, b, l, n
- **Valid values**: Any combination of 'u' (unconstrained), 'b' (bound), 'l' (linear), 'n' (nonlinear)
- **Example**: `ptype='ubl'` selects unconstrained, bound, and linearly constrained problems

### Dimension range errors
- **Type**: TypeError/ValueError
- **Trigger**: `mindim` not a positive integer, or `mindim > maxdim`
- **Fix**: Ensure `mindim >= 1` and `mindim <= maxdim`.

### Problem library errors
- **Type**: ValueError
- **Trigger**: Unknown library name in `plibs`, or custom library path doesn't exist
- **Fix**: Use valid library names ('s2mpj', 'pycutest', 'matcutest') or ensure custom path exists and contains `*_tools.py`.

## 5. Profile Option Errors

### n_jobs issues
- Non-integer → TypeError
- Values < 1 are silently changed to 1 with a warning

### benchmark_id / feature_stamp
- Empty string or illegal characters → ValueError
- Only alphanumeric characters, underscores, dots, and hyphens are allowed

### max_tol_order
- Must be in range [1, 16], controls tolerance levels from 10^{-1} to 10^{-max_tol_order}

### max_eval_factor
- Must be positive. Controls per-problem evaluation budget: maxfun = ceil(max_eval_factor * n)

### Custom functions (merit_fun, score_weight_fun, score_fun)
- Must be callable → TypeError if not

## 6. Runtime Behaviors (Not Errors, But Important)

### Solver exceptions are silently caught
- If a solver raises ANY exception during execution, OptiProfiler catches it and logs a warning. The run is recorded as a failure (no successful evaluation).
- **Implication**: Your solver may be crashing silently. Check the log.txt for warnings.

### StopIteration on evaluation budget
- When a solver exceeds 2 * maxfun evaluations, OptiProfiler raises StopIteration.
- If the solver doesn't catch this, the run is treated as failed.
- **Fix**: Well-designed solvers should have their own termination criteria before hitting the budget.

### No built-in timeout
- OptiProfiler does NOT enforce wall-clock time limits.
- A slow solver can run indefinitely.
- **Fix**: Implement timeout in your solver, or use the `n_jobs` option for parallel execution with external timeout.

### NaN handling
- If `fun(x)` raises an exception, OptiProfiler returns NaN for that evaluation.
- With the `random_nan` feature, some evaluations randomly return NaN.
- Solvers should handle NaN gracefully (e.g., reject the point, try another).

### Empty problem selection
- If problem filters are too strict (e.g., `maxdim=1` with `ptype='n'`), no problems may be selected.
- OptiProfiler does NOT raise an error — it returns zero scores silently.
- **Fix**: Check the log.txt to verify how many problems were selected.

## 7. Load/Resume Errors

### load option format
- Must be a valid path to a previous experiment directory
- Common error: pointing to the wrong directory level (should contain test_log/)

### solver_names with load
- If using `load` without `solvers`, do NOT pass `solver_names` — it will cause a TypeError when OptiProfiler tries to check `len(solvers)` on None.

````

## debugging/solver_compat.md

- Source SHA256: `f35a582ffca1faea6c307b2cf29dbd26f1ce4a22f44dd5506a9434f0f65eef3d`

````markdown
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

````
