---
tags: [api, python, class, problem]
sources: [_sources/python/classes.json]
related: [concepts/problem-types.md, api/python/benchmark.md, api/python/plib-tools.md, guides/custom-problem-library-python.md, guides/custom-problem-library-matlab.md]
last_updated: 2026-05-11
---

# `Problem` Class Reference

The `Problem` class is the single common currency between OptiProfiler
and every problem-library adapter. Built-in adapters (S2MPJ, PyCUTEst,
MatCUTEst) construct `Problem` instances and hand them back to the
benchmark engine; custom-library authors do **exactly** the same.

This page is the canonical type & semantics reference. The companion
[Custom Problem Library guide (Python)](../../guides/custom-problem-library-python.md)
and [Custom Problem Library guide (MATLAB)](../../guides/custom-problem-library-matlab.md)
show how to *produce* `Problem` objects step by step.

The mathematical structure encoded by `Problem` is

\[
\min_x \; \mathrm{fun}(x) \;\text{ s.t. }\;
x_l \le x \le x_u,\;
A_{ub} x \le b_{ub},\;
A_{eq} x = b_{eq},\;
c_{ub}(x) \le 0,\;
c_{eq}(x) = 0
\]

with initial point \(x_0\).

## Python constructor

```python
from optiprofiler.opclasses import Problem

problem = Problem(
    fun, x0,
    name=None,
    xl=None, xu=None,
    aub=None, bub=None,
    aeq=None, beq=None,
    cub=None, ceq=None,
    grad=None, hess=None,
    jcub=None, jceq=None,
    hcub=None, hceq=None,
)
```

Only `fun` and `x0` are required. Every other argument defaults to
"absent" — bounds default to `-inf`/`+inf`, constraint matrices to
empty, constraint callables to functions returning empty arrays.

## Required arguments

| Name | Type | Shape | Semantics |
|---|---|---|---|
| `fun` | `callable` | `fun(x: ndarray (n,)) -> float` | Objective. Must return a finite scalar or `NaN`. |
| `x0`  | `array_like` | `(n,)` | Initial guess. Internally converted to `ndarray` of float; `n` is inferred from this. |

## Optional arguments

### Identity

| Name | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `'Unnamed Problem'` | Used in result logs and profile labels |

### Bounds

| Name | Type | Shape | Default | Semantics |
|---|---|---|---|---|
| `xl` | `array_like` of float | `(n,)` | `-numpy.inf` per coord | Lower bound `xl <= x` |
| `xu` | `array_like` of float | `(n,)` | `numpy.inf` per coord | Upper bound `x <= xu` |

Use `-np.inf` / `np.inf` for "no bound on this coordinate". Adapter
authors converting from libraries that encode `±1e20` as infinity
(CUTEst, S2MPJ) must replace those sentinels with `np.inf` before
passing them in — see the built-in pycutest adapter in
[`problem_libs/pycutest/pycutest_tools.py`](#) lines 77–81.

### Linear constraints

| Name | Type | Shape | Semantics |
|---|---|---|---|
| `aub` | `ndarray` | `(m_linear_ub, n)` | Inequality matrix `aub @ x <= bub` |
| `bub` | `ndarray` | `(m_linear_ub,)`   | RHS of linear inequalities |
| `aeq` | `ndarray` | `(m_linear_eq, n)` | Equality matrix `aeq @ x == beq` |
| `beq` | `ndarray` | `(m_linear_eq,)`   | RHS of linear equalities |

If you have no constraints of a kind, pass `None` (or leave the
argument out). Do **not** pass a `(0, n)` zero matrix manually — the
constructor handles the empty case itself.

### Nonlinear constraints (callables)

| Name | Type | Shape of return | Semantics |
|---|---|---|---|
| `cub` | `callable(x)` | `(m_nonlinear_ub,)` | `cub(x) <= 0` |
| `ceq` | `callable(x)` | `(m_nonlinear_eq,)` | `ceq(x) == 0` |

### First- and second-order info (optional)

These are *not* used by derivative-free solvers but matter for
benchmarking derivative-based competitors. Provide them when the
underlying library can deliver them cheaply.

| Name | Type | Shape | Notes |
|---|---|---|---|
| `grad` | `callable(x)` | `(n,)` | Gradient of `fun` |
| `hess` | `callable(x)` | `(n, n)` | Hessian of `fun` |
| `jcub` | `callable(x)` | `(m_nonlinear_ub, n)` | Jacobian of `cub` |
| `jceq` | `callable(x)` | `(m_nonlinear_eq, n)` | Jacobian of `ceq` |
| `hcub` | `callable(x)` | `list[(n, n)]` of length `m_nonlinear_ub` | Per-constraint Hessians of `cub` |
| `hceq` | `callable(x)` | `list[(n, n)]` of length `m_nonlinear_eq` | Per-constraint Hessians of `ceq` |

If you can't compute a particular derivative, return an `np.nan`-filled
array of the right shape (the built-in adapters do exactly this on
failure — see [`problem_libs/s2mpj_python/s2mpj_tools.py`](#) `_getgrad`).

## Computed attributes (read-only)

After construction these are filled in by `Problem.__init__`:

| Attribute | Type | Meaning |
|---|---|---|
| `n` | `int` | Problem dimension |
| `mb` | `int` | Number of finite bound constraints |
| `m_linear_ub`, `m_linear_eq` | `int` | Linear constraint counts |
| `m_nonlinear_ub`, `m_nonlinear_eq` | `int` | Nonlinear constraint counts |
| `mlcon`, `mnlcon`, `mcon` | `int` | Aggregate counts |
| `ptype` | `'u'` / `'b'` / `'l'` / `'n'` | Problem-type tag used by `select()` filtering |

Adapter authors writing `*_select.py` rely on these attribute
computations matching their own metadata — keep your CSV's `ptype`,
`dim`, `mb`, `mlcon`, `mnlcon`, `mcon` fields aligned with how
OptiProfiler tags problems.

## Methods

```python
problem.fun(x)        # -> float
problem.grad(x)       # -> ndarray (n,), or empty array if not provided
problem.hess(x)       # -> ndarray (n, n), or empty
problem.cub(x)        # -> ndarray (m_nonlinear_ub,), or empty
problem.ceq(x)        # -> ndarray (m_nonlinear_eq,), or empty
problem.jcub(x)       # -> ndarray (m_nonlinear_ub, n), or empty
problem.jceq(x)       # -> ndarray (m_nonlinear_eq, n), or empty
problem.hcub(x)       # -> list of (n, n) arrays
problem.hceq(x)       # -> list of (n, n) arrays

problem.maxcv(x)      # -> float, maximum constraint violation
problem.project_x0()  # attempt to project x0 onto feasible region
```

## MATLAB equivalent

The MATLAB `Problem` class lives at
`optiprofiler/matlab/optiprofiler/src/Problem.m` and is constructed
from a struct:

```matlab
problem = Problem(struct( ...
    'name', 'mySinusoidal',  ...
    'fun',  @(x) sum(sin(x)), ...
    'x0',   x0,              ...
    'xl', xl, 'xu', xu,      ...
    'aub', aub, 'bub', bub,  ...
    'aeq', aeq, 'beq', beq,  ...
    'cub', @cub_callback,    ...
    'ceq', @ceq_callback,    ...
    'grad', @grad_callback,  ...
    'hess', @hess_callback,  ...
    'jcub', @jcub_callback,  ...
    'jceq', @jceq_callback   ...
));
```

Differences vs Python:

| Aspect | Python | MATLAB |
|---|---|---|
| Constructor input | positional + keyword args | a single `struct` |
| Infinity sentinel | `numpy.inf`, `-numpy.inf` | `Inf`, `-Inf` |
| Empty constraints | `None` or omit | empty matrix `[]` (default) |
| Hessian list type | `list[ndarray]` | cell array of matrices |
| Function shape | `fun(x) -> float` | `fun(x) -> double` |

Computed attributes (`n`, `mb`, `m_linear_ub`, …, `ptype`, `mcon`)
have identical names in MATLAB.

## Data-type pitfalls when authoring adapters

1. **Coerce arrays to `float`** — many libraries return integer
   `x0`/`xl` for trivial problems; cast with
   `np.asarray(..., dtype=float)` before handing to `Problem`. Solvers
   like Nelder-Mead silently behave differently on integer arrays.

2. **Replace `±1e20` with `±np.inf`** — CUTEst, S2MPJ, and several
   third-party libraries use `1.0e+20` as the "no bound" sentinel.
   OptiProfiler treats `np.inf` as the absence of a bound, *not*
   `1e20`. Without this conversion, `mb` will be wildly overcounted
   and `ptype` will switch to `'b'` for problems that should be `'u'`.

3. **Flatten properly** — S2MPJ returns column vectors; pycutest
   returns row vectors. Always `.flatten()` to `(n,)` before passing
   in.

4. **Constraint matrix shape** — `aub` must have `n` columns, even if
   `m_linear_ub` is zero. If zero, pass `None`, not `np.zeros((0,))`.

5. **Sparse matrices** — `Problem` expects dense `ndarray`. Convert
   with `.toarray()` if your source library produces sparse output.
   See `_getJx` in
   [`problem_libs/s2mpj_python/s2mpj_tools.py`](#) for the pattern.

6. **No I/O inside `fun`** — many CUTEst-style libraries chatter on
   stdout. Wrap their calls in `contextlib.redirect_stdout(io.StringIO())`
   to silence them (the pycutest adapter does this).

## See Also

- [Custom Problem Library — Python](../../guides/custom-problem-library-python.md)
- [Custom Problem Library — MATLAB](../../guides/custom-problem-library-matlab.md)
- [Problem-set metadata helper](../../guides/problem-metadata.md) — how `select()` knows what's available
- [Problem Types](../../concepts/problem-types.md) — `ptype` semantics
- [Problem Library Tools](plib-tools.md) — built-in `*_load` / `*_select`
- [Python benchmark()](benchmark.md) — where problems are consumed
