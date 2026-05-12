---
tags: [guide, feature, custom, python]
related: [concepts/features.md, concepts/parallel-and-pickle.md, api/python/problem-class.md, api/python/benchmark.md]
last_updated: 2026-05-11
---

# Custom Feature Guide (Python)

OptiProfiler ships ten built-in `feature_name` presets (`plain`,
`noisy`, `perturbed_x0`, `truncated`, `permuted`,
`linearly_transformed`, `random_nan`, `unrelaxable_constraints`,
`nonquantifiable_constraints`, `quantized`). When none of them
matches what you want to test, the **`feature_name='custom'`** preset
lets you compose any of eight independent modifiers.

This page is the reference for those eight modifiers: signatures,
worked examples, and the constraints they must respect.

## Mental model

A feature maps the original problem

\[
\min_x f(x) \;\text{ s.t. } x_l \le x \le x_u,\;
A_{ub}x \le b_{ub},\;A_{eq}x = b_{eq},\;c_{ub}(x) \le 0,\;c_{eq}(x) = 0
\]

to a *new* problem in transformed coordinates \(\tilde x = A x + b\).
Each `mod_*` callable replaces one component of the new problem.
Components you leave as `None` (the default) are passed through
unchanged.

## The eight modifiers — exact signatures

All `mod_*` callables receive a `numpy.random.Generator` for stochastic
behaviour and the original `problem: optiprofiler.Problem` so they can
read properties such as `problem.n`, `problem.x0`, `problem.xl`,
`problem.fun`, etc.

| Modifier | Signature | Returns |
|---|---|---|
| `mod_x0`        | `(rng, problem) -> ndarray (n,)`                              | New initial guess |
| `mod_affine`    | `(rng, problem) -> (A, b, inv)`                               | Coordinate map `tilde_x = A x + b`, with `inv` the pseudo-inverse (`(n, n)`, `(n,)`, `(n, n)`) |
| `mod_bounds`    | `(rng, problem) -> (xl, xu)`                                  | New bounds vectors `(n,)`, `(n,)` |
| `mod_linear_ub` | `(rng, problem) -> (aub, bub)`                                | New linear inequality system |
| `mod_linear_eq` | `(rng, problem) -> (aeq, beq)`                                | New linear equality system |
| `mod_fun`       | `(x, rng, problem) -> float`                                  | New objective value at `x` |
| `mod_cub`       | `(x, rng, problem) -> ndarray (m_nonlinear_ub,)`              | New nonlinear inequality values |
| `mod_ceq`       | `(x, rng, problem) -> ndarray (m_nonlinear_eq,)`              | New nonlinear equality values |

Notes:

- `mod_x0`, `mod_affine`, `mod_bounds`, `mod_linear_ub`,
  `mod_linear_eq` are called **once per problem instance** at
  setup time.
- `mod_fun`, `mod_cub`, `mod_ceq` are called **once per objective
  evaluation** — they must be cheap.
- The `rng` argument is a `numpy.random.Generator`, **not** the legacy
  `numpy.random.RandomState`. Use `rng.standard_normal(...)`,
  `rng.uniform(...)`, etc. Each problem run gets a deterministic seed
  derived from the run index, so reproducibility is preserved.

## Picklability requirement (read this first)

Every `mod_*` callable is attached to a `Feature` object that is
shipped to every worker process when `n_jobs > 1`. **Always define
them as module-level `def`s, never as `lambda`s or nested functions.**
See [Parallel mode and the "harmful lambda" rule](../concepts/parallel-and-pickle.md)
for the full explanation and the four exact places lambdas break.

```python
def add_gaussian_noise(x, rng, problem):
    return problem.fun(x) + 1e-3 * rng.standard_normal()
```

## Worked example 1 — Multiplicative noise on the objective

A noisy objective where the noise scales with the function magnitude
(useful for testing solvers' robustness to relative noise):

```python
import numpy as np

def relative_noise(x, rng, problem):
    f = problem.fun(x)
    if not np.isfinite(f):
        return f
    return f * (1.0 + 1e-3 * rng.standard_normal())

scores = benchmark(
    [solver_a, solver_b],
    feature_name="custom",
    mod_fun=relative_noise,
    n_runs=10,           # average over 10 stochastic replays
    ptype="u",
)
```

## Worked example 2 — Random restart with bounded perturbation

```python
import numpy as np

def perturb_x0_bounded(rng, problem):
    # 1% perturbation, but stay inside [xl, xu]
    step = 1e-2 * rng.standard_normal(problem.n)
    x_new = problem.x0 + step
    return np.clip(x_new, problem.xl, problem.xu)

benchmark(solvers, feature_name="custom", mod_x0=perturb_x0_bounded, n_runs=5)
```

## Worked example 3 — Affine reparameterisation (coordinate rotation)

Test how solvers respond to a random orthogonal rotation of the
coordinate system. The modifier returns `(A, b, A_inv)` where
`A_inv = A^{-1}` is supplied to spare OptiProfiler the cost of
recomputing it.

```python
import numpy as np
from scipy.linalg import qr

def random_rotation(rng, problem):
    n = problem.n
    # Random orthogonal matrix via QR of a Gaussian sample.
    G = rng.standard_normal((n, n))
    Q, _ = qr(G)
    b = np.zeros(n)
    return Q, b, Q.T   # Q is orthogonal, so inv(Q) == Q.T

benchmark(solvers, feature_name="custom", mod_affine=random_rotation, n_runs=5)
```

## Worked example 4 — Tightened bounds

```python
def tighter_bounds(rng, problem):
    # Shrink the feasible region by 10% on both sides.
    width = problem.xu - problem.xl
    return problem.xl + 0.05 * width, problem.xu - 0.05 * width

benchmark(solvers, feature_name="custom", mod_bounds=tighter_bounds, ptype="b")
```

## Worked example 5 — Replacing the nonlinear inequality constraint

```python
import numpy as np

def softened_cub(x, rng, problem):
    # Original cub(x) <= 0 becomes cub(x) - eps <= 0.
    base = problem.cub(x)
    return base - 1e-4

benchmark(solvers, feature_name="custom", mod_cub=softened_cub, ptype="n")
```

## Combining modifiers

You can pass several `mod_*` callables at once — they compose
independently. A common pattern is "noisy + perturbed start":

```python
def add_noise(x, rng, problem):
    return problem.fun(x) + 1e-3 * rng.standard_normal()

def perturb_x0(rng, problem):
    return problem.x0 + 1e-3 * rng.standard_normal(problem.n)

benchmark(
    solvers,
    feature_name="custom",
    mod_fun=add_noise,
    mod_x0=perturb_x0,
    n_runs=10,
)
```

## Reproducibility

Each `(problem_index, run_index)` pair maps to a deterministic
sub-seed, so running the same custom feature twice gives identical
results. Do **not** call `np.random.*` inside the modifiers — that
defeats reproducibility. Always use the `rng` argument.

## Stochastic vs deterministic detection

OptiProfiler decides how many `n_runs` to default-pick based on
whether the active feature is "stochastic". For `custom`, OptiProfiler
assumes stochastic and defaults `n_runs=5`. If your modifiers are
purely deterministic, set `n_runs=1` explicitly to skip redundant
replays.

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `PicklingError: ... <lambda>`  | Lambda or nested `def` used as `mod_*` | Move to module top level with `def` |
| `n_runs` is 5 but results identical | Forgot to use `rng` | Replace `np.random.randn()` with `rng.standard_normal()` |
| `mod_affine` raises shape error | Wrong return tuple length | Must return three arrays: `(A, b, inv)` |
| `mod_fun` returns array instead of scalar | Returning `problem.fun(x) * something` where shape sneaks in | Coerce with `float(value)` |

## See Also

- [Features](../concepts/features.md) — overview of all preset features
- [Parallel & Pickle Rules](../concepts/parallel-and-pickle.md) — why these must be `def`
- [Problem Class](../api/python/problem-class.md) — properties available via `problem.*`
- [benchmark()](../api/python/benchmark.md) — full parameter reference
