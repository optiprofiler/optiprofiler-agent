---
tags: [troubleshooting, timeout, performance]
sources: [_sources/python/benchmark.json]
related: [troubleshooting/common-errors.md, concepts/benchmark-function.md]
last_updated: 2025-04-13
---

# Timeout and Performance Issues

Long benchmark runs can timeout or consume excessive resources. This page
covers common causes and solutions.

## Common Causes of Slow Benchmarks

1. **Large problem dimensions**: High `maxdim` means more function
   evaluations per problem
2. **Many problems selected**: Broad `ptype` + large dimension range
   selects many problems
3. **High `max_eval_factor`**: Default is 500, meaning up to
   500 * dimension evaluations per problem
4. **High `n_runs`**: Multiple runs for stochastic features multiply
   total runtime
5. **Slow solvers**: Some solvers have per-evaluation overhead
6. **History plots**: `draw_hist_plots='parallel'` adds plotting overhead

## Solutions

### Reduce Problem Count

```python
benchmark(solvers, ptype='u', mindim=2, maxdim=5)  # Small range
```

### Lower Evaluation Budget

```python
benchmark(solvers, max_eval_factor=100)  # Default is 500
```

### Disable History Plots

```python
benchmark(solvers, draw_hist_plots='none')
```

### Use Score-Only Mode

```python
benchmark(solvers, score_only=True)  # No plots, just scores
```

### Increase Parallelism

```python
benchmark(solvers, n_jobs=8)  # Use more cores
```

### Silence Output

```python
benchmark(solvers, silent=True)
```

## Agent Debug Timeout

When using the OptiProfiler Agent's `debug --run` mode, scripts are
executed with a timeout (default 120 seconds). If your benchmark runs
longer:

```bash
opagent debug script.py --run --timeout 600
```

## See Also

- [Common Errors](common-errors.md) — other error types
- [Benchmark Function](../concepts/benchmark-function.md) — parameter overview
