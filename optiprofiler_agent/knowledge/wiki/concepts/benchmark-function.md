---
tags: [concept, api, benchmark]
sources: [_sources/python/benchmark.json, _sources/matlab/benchmark.json]
related: [concepts/dfo.md, concepts/solver-interface.md, concepts/problem-types.md, concepts/features.md, api/python/benchmark.md, api/matlab/benchmark.md]
last_updated: 2025-04-13
---

# The benchmark() Function

`benchmark()` is the central entry point of OptiProfiler. It runs a set of
optimization solvers on a collection of test problems, generates performance
profiles, data profiles, and log-ratio profiles, and returns solver scores.

## Key Rules

1. **At least 2 solvers are required.** A single solver cannot produce
   meaningful comparative profiles.
2. Solvers must be callable with the correct
   [signature](solver-interface.md) for the chosen
   [problem type](problem-types.md).
3. The function generates output in a timestamped directory including PDF
   profiles, log files, and history plots.

## Python Calling Convention

```python
from optiprofiler import benchmark

scores = benchmark([solver1, solver2], ptype='u', mindim=2, maxdim=20)
```

- `solvers`: a list of callables
- Options are passed as keyword arguments

## MATLAB Calling Convention

```matlab
scores = benchmark({@solver1, @solver2}, options)
```

- `solvers`: a cell array of function handles
- Options are passed as a struct

## Return Values

| Return          | Type                | Description                                  |
|-----------------|---------------------|----------------------------------------------|
| `solver_scores` | array               | Aggregate scores based on profile performance |
| `profile_scores`| 4D array or None    | Per-tolerance, per-profile-type scores        |
| `curves`        | list of dict / None | Raw profile curve data                        |

## See Also

- [DFO](dfo.md) — derivative-free optimization context
- [Solver Interface](solver-interface.md) — how solvers must be defined
- [Problem Types](problem-types.md) — ptype options
- [Features](features.md) — feature_name options
- [Python benchmark() API](../api/python/benchmark.md) — full parameter reference
- [MATLAB benchmark() API](../api/matlab/benchmark.md) — full parameter reference
