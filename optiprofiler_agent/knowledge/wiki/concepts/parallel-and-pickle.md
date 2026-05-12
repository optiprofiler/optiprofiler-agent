---
tags: [concept, parallel, pickle, lambda, gotcha]
related: [concepts/features.md, concepts/benchmark-function.md, guides/custom-feature.md, guides/custom-problem-library-python.md]
last_updated: 2026-05-11
---

# Parallel mode and the "harmful lambda" rule

OptiProfiler can run benchmarks across multiple processes when the user
passes `n_jobs > 1`. The Python implementation uses standard
`multiprocessing`-style pools, which **pickle every object that crosses
a process boundary**. Lambdas, closures, and locally-defined functions
are *not* picklable in CPython, so they break the worker spawn step.

This page is the single source of truth for which lambdas are safe and
which are not. Every guide that asks users to write callables (custom
solvers, custom features, custom problem libraries) links back here.

## TL;DR

**A lambda is harmful only if it has to cross a process boundary.**
That happens in exactly four places:

1. Anything in the **`solvers`** list passed to `benchmark()`.
2. Anything attached to a **`Feature`** object (`mod_fun`, `mod_x0`,
   `mod_affine`, `mod_bounds`, `mod_linear_ub`, `mod_linear_eq`,
   `mod_cub`, `mod_ceq`).
3. Anything in **`profile_options`** consumed by the scoring layer
   (currently `merit_fun`, `score_weight_fun`).
4. Module-level objects referenced by any of the above (because pickle
   re-imports them by qualified name on the worker side).

**A lambda is harmless** if it lives entirely inside a worker process
and is not exposed back to the main process. Typical example: the
`fun = lambda x: p.obj(x)` line inside `pycutest_load` — the lambda
captures the freshly-built `pycutest` problem object, both are created
*inside* the worker, and `Problem` only stores the callable for the
worker's own use. Nothing here is shipped back across processes.

## Why `def` is always safer than `lambda`

Even when a lambda *is* technically picklable (e.g. CPython 3.11+ can
pickle module-level lambdas under some conditions), `def` has three
practical advantages:

- Module-level `def`s are pickled by **qualified name**, not by code
  object, so they survive Python version mismatches between main and
  worker.
- `def`s have a `__name__` and a docstring — debuggers, tracebacks, and
  `repr()` are readable.
- They can be patched in tests via `monkeypatch.setattr(module, "name", ...)`.

## Concrete rules by surface area

### Custom solvers (passed in `solvers=[...]`)

Always `def`. Define them at module top level, not inside another
function.

```python
def my_solver(fun, x0):
    from scipy.optimize import minimize
    return minimize(fun, x0, method="Nelder-Mead").x

benchmark([my_solver, other_solver], n_jobs=4)
```

### Custom features (`mod_*` callables)

Same rule as solvers — these are stored on the `Feature` object and
shipped to every worker. See
[guides/custom-feature.md](../guides/custom-feature.md) for the full
list of `mod_*` signatures and worked examples.

```python
def add_gaussian_noise(x, random_stream, problem):
    return problem.fun(x) + 1e-3 * random_stream.standard_normal()

benchmark(solvers, feature_name="custom", mod_fun=add_gaussian_noise, n_jobs=4)
```

### `profile_options` callables

Same rule — anything you put under `profile_options['merit_fun']` or
`profile_options['score_weight_fun']` is consumed in the scoring
process and must be picklable.

### Internal load-time lambdas (safe)

The built-in adapters in
[`problem_libs/pycutest/pycutest_tools.py`](#) and
[`problem_libs/s2mpj_python/s2mpj_tools.py`](#) routinely build
lambdas inside `*_load`:

```python
def pycutest_load(problem_name, **kwargs):
    ...
    p = pycutest.import_problem(problem_name, ...)
    fun  = lambda x: p.obj(x)     # safe: closure stays in worker
    grad = lambda x: p.grad(x)
    hess = lambda x: p.ihess(x)
    return Problem(fun, x0, ..., grad=grad, hess=hess)
```

These are fine. `Problem` and its closures are created **inside the
worker**, used **inside the worker**, and never re-pickled back to the
controller. Custom problem-library authors can do the same.

If you ever store such a `Problem` and try to ship it back to the main
process (e.g. by caching workers' return values), the lambdas will
break pickling. In that case, swap to a module-level `def` or wrap the
problem object in a small adapter class.

## How to tell which case you're in

A quick mental check: **"Does the call site receive my callable from
the user?"** If yes (solver list, `Feature(mod_*=...)`, `profile_options`),
it must be picklable. If the callable is created locally inside an
adapter you're writing and never escapes, it's safe.

When in doubt, run with `n_jobs=1` first to validate logic, then with
`n_jobs=2` to exercise the pickle path:

```python
benchmark(solvers, feature_name="custom", mod_fun=add_gaussian_noise, n_jobs=2)
```

The error you'll see when something is unpicklable is one of:

- `AttributeError: Can't pickle local object '...<lambda>'`
- `TypeError: cannot pickle 'function' object`
- `_pickle.PicklingError: Can't pickle <function <lambda> at 0x...>`

If any of these appear, follow the surface in the traceback back to a
lambda or a function defined inside another function, and lift it to
module level.

## MATLAB

MATLAB's `parfor` workers receive a fresh workspace copy via
serialisation as well, but anonymous functions (`@(x) ...`) are
serialisable in MATLAB, so the Python rule does **not** apply
verbatim. The practical guidance for MATLAB is different:

- Top-level local functions in the same `.m` file are always fine.
- Anonymous functions that close over small numeric data are fine.
- Anonymous functions that close over large objects or handles to
  external processes (e.g. a MATCUTEst session handle) are fragile —
  prefer a script-level function or a class method.

See [guides/custom-problem-library-matlab.md](../guides/custom-problem-library-matlab.md)
for examples that follow the safer pattern.

## See Also

- [Features](features.md) — the `feature_name='custom'` entry point
- [Custom Feature Guide](../guides/custom-feature.md) — every `mod_*` callable
- [Custom Problem Library (Python)](../guides/custom-problem-library-python.md) — `*_load` / `*_select`
- [Python benchmark()](../api/python/benchmark.md) — `n_jobs` parameter
