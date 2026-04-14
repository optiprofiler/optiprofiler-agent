---
tags: [concept, feature, benchmark]
sources: [_sources/python/benchmark.json]
related: [concepts/benchmark-function.md, profiles/feature-effects.md, api/python/benchmark.md]
last_updated: 2025-04-13
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

Use `feature_name='custom'` with modifier functions:

```python
benchmark(solvers,
    feature_name='custom',
    mod_fun=lambda x, rs, prob: prob.fun(x) + 1e-3 * rs.randn(),
    mod_x0=lambda rs, prob: prob.x0 + 1e-3 * rs.randn(prob.n))
```

Available modifiers: `mod_x0`, `mod_affine`, `mod_bounds`, `mod_linear_ub`,
`mod_linear_eq`, `mod_fun`, `mod_cub`, `mod_ceq`.

## See Also

- [Feature Effects on Profiles](../profiles/feature-effects.md) — how features affect results
- [Benchmark Function](benchmark-function.md) — where features are specified
- [Python API](../api/python/benchmark.md) — full feature parameter documentation
