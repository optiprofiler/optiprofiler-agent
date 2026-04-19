---
tags: [api, python, imports, exports]
last_updated: 2026-04-19
related: [api/python/benchmark.md, api/python/problem-class.md, api/python/plib-tools.md]
---

# Python Imports and Public API

This page is the **single source of truth** for what you can import from
the `optiprofiler` Python package. The post-generation validator
(`validators/api_checker.py`) auto-derives the same whitelist from
`knowledge/_sources/python/*.json`, so any drift between this page and
the validator is a bug.

## Package Name

The package is named **`optiprofiler`** — one word, lowercase, no hyphen,
no underscore.

Common typos to **avoid**:

| Wrong | Correct |
|---|---|
| `optiprobe` | `optiprofiler` |
| `opti_profiler` | `optiprofiler` |
| `opti-profiler` (would not even parse) | `optiprofiler` |
| `optiprofile` | `optiprofiler` |

## Public Top-Level Exports

The API is **flat**. Everything is imported directly from the package
root, never from a submodule path:

```python
from optiprofiler import (
    benchmark,
    Problem,
    Feature,
    FeaturedProblem,
    s2mpj_load,
    s2mpj_select,
    pycutest_load,
    pycutest_select,
    get_plib_config,
    set_plib_config,
)
```

| Symbol | Kind | One-line description | Detailed page |
|---|---|---|---|
| `benchmark` | function | Run solvers against problem libraries with features. | [benchmark()](benchmark.md) |
| `Problem` | class | One optimization problem (`fun`, `x0`, bounds, constraints). | [Problem class](problem-class.md) |
| `Feature` | class | A perturbation applied to problems during a benchmark run. | [Problem class](problem-class.md) |
| `FeaturedProblem` | class | `Problem` already wrapped with a `Feature`. | [Problem class](problem-class.md) |
| `s2mpj_load` | function | Load a single S2MPJ problem by name. | [Problem libraries](plib-tools.md) |
| `s2mpj_select` | function | List S2MPJ problem names matching criteria. | [Problem libraries](plib-tools.md) |
| `pycutest_load` | function | Load a single PyCUTEst problem by name (Linux/macOS). | [Problem libraries](plib-tools.md) |
| `pycutest_select` | function | List PyCUTEst problem names matching criteria. | [Problem libraries](plib-tools.md) |
| `get_plib_config` | function | Read the effective config of a problem library. | [Problem libraries](plib-tools.md) |
| `set_plib_config` | function | Override a problem library's config at runtime. | [Problem libraries](plib-tools.md) |

## Submodules That Do **Not** Exist

These look plausible but are **not** part of the public API. The
validator will warn on every one of them:

- `optiprofiler.solvers` — solvers come from third-party packages, not
  from `optiprofiler` itself.
- `optiprofiler.algorithms` — same as above.
- `optiprofiler.utils` — there is no public utils namespace.
- `optiprofiler.problems` — use `s2mpj_load` / `pycutest_load` instead.
- `optiprofiler.features` — `Feature` is a top-level class, not a module.

## Where Solvers Come From

`optiprofiler` benchmarks solvers but does **not** ship solvers. You
bring your own callable:

```python
from optiprofiler import benchmark
from prima import bobyqa, cobyla        # PRIMA package — separate install
from scipy.optimize import minimize     # any callable that matches the
                                        # solver signature works

def my_bobyqa(fun, x0, xl, xu):
    return bobyqa(fun, x0, xl=xl, xu=xu).x

def my_cobyla(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq):
    return cobyla(fun, x0, xl=xl, xu=xu,
                  aub=aub, bub=bub, aeq=aeq, beq=beq,
                  cub=cub, ceq=ceq).x

scores, *_ = benchmark([my_bobyqa, my_cobyla], ptype='b', mindim=2, maxdim=20)
```

See [Solver Interface](../../concepts/solver-interface.md) for the
required signatures of each problem type.

## Quick Reference: Allowed Import Forms

```python
import optiprofiler                                # OK
from optiprofiler import benchmark                 # OK
from optiprofiler import benchmark, Problem        # OK
from optiprofiler import (                         # OK
    benchmark, Problem, Feature, FeaturedProblem,
)
import optiprofiler.solvers                        # WARNING — no submodule
from optiprofiler.solvers import bobyqa            # WARNING — no submodule
from optiprobe import benchmark                    # ERROR — package typo
```
