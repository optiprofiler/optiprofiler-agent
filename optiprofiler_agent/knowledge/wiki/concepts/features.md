---
tags: [concept, feature, benchmark]
sources: [_sources/python/benchmark.json]
related: [concepts/benchmark-function.md, concepts/parallel-and-pickle.md, profiles/feature-effects.md, api/python/benchmark.md, guides/custom-feature.md]
last_updated: 2026-05-11
---

# Features

Features modify the test problems to simulate real-world conditions. They
are set via the `feature_name` parameter in `benchmark()`.

## Available Features

| Feature Name                    | Effect                                              | Stochastic? |
|---------------------------------|-----------------------------------------------------|-------------|
| `plain`                         | No modification (baseline)                          | No          |
| `perturbed_x0`                  | Perturb the initial guess                           | Yes         |
| `noisy`                         | Add noise to function evaluations                   | Yes         |
| `truncated`                     | Truncate function values to limited precision        | No          |
| `permuted`                      | Permute variable ordering                           | Yes         |
| `linearly_transformed`          | Apply linear transformation to coordinates          | Yes         |
| `random_nan`                    | Randomly return NaN for some evaluations            | Yes         |
| `unrelaxable_constraints`       | Make certain constraints unrelaxable                | No          |
| `nonquantifiable_constraints`   | Make constraint violations non-measurable           | No          |
| `quantized`                     | Restrict variables to a discrete mesh               | No          |
| `custom`                        | User-defined modifier functions                     | Varies      |

## Key Feature Parameters

- **`n_runs`**: Number of experiment repetitions. Default is 5 for stochastic
  features, 1 for deterministic.
- **`noise_level`**: Magnitude of noise for `noisy` feature (default: 1e-3).
- **`noise_type`**: `'absolute'`, `'relative'`, or `'mixed'` (default: `'mixed'`).
- **`perturbation_level`**: Magnitude for `perturbed_x0` (default: 1e-3).
- **`nan_rate`**: Probability of NaN for `random_nan` (default: 0.05).

## Custom Features

Use `feature_name='custom'` with module-level **`def`** modifier
functions. **Do not use `lambda`** for any `mod_*` callable — they are
shipped to worker processes when `n_jobs > 1` and lambdas are not
reliably picklable; see [Parallel mode and the "harmful lambda" rule](parallel-and-pickle.md).

```python
def add_noise(x, random_stream, problem):
    return problem.fun(x) + 1e-3 * random_stream.standard_normal()

def perturb_x0(random_stream, problem):
    return problem.x0 + 1e-3 * random_stream.standard_normal(problem.n)

benchmark(solvers,
    feature_name='custom',
    mod_fun=add_noise,
    mod_x0=perturb_x0,
    n_runs=5,
)
```

The full list of `mod_*` modifiers, their signatures, and worked
examples for each kind of transformation (noise, perturbation, bound
relaxation, linear/nonlinear constraint surgery) live in the
dedicated [Custom Feature Guide](../guides/custom-feature.md).

Available modifiers: `mod_x0`, `mod_affine`, `mod_bounds`, `mod_linear_ub`,
`mod_linear_eq`, `mod_fun`, `mod_cub`, `mod_ceq`.

## See Also

- [Custom Feature Guide](../guides/custom-feature.md) — every `mod_*` signature with examples
- [Parallel & Pickle Rules](parallel-and-pickle.md) — why `mod_*` must be `def`, not `lambda`
- [Feature Effects on Profiles](../profiles/feature-effects.md) — how features affect results
- [Benchmark Function](benchmark-function.md) — where features are specified
- [Python API](../api/python/benchmark.md) — full feature parameter documentation
