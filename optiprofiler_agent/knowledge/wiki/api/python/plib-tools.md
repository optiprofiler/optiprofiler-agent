---
tags: [api, python, problem-library, tools]
sources: [_sources/python/plib_tools.json]
related: [api/python/benchmark.md, api/python/problem-class.md, guides/quickstart-python.md]
last_updated: 2025-04-13
---

# Python Problem Library Tools

OptiProfiler provides utility functions for loading and selecting problems
from built-in test libraries.

## s2mpj_load

```python
from optiprofiler import s2mpj_load
problem = s2mpj_load("ROSENBR")
```

Converts an S2MPJ problem name to a `Problem` instance. Problem names may
include dimension suffixes like `'PROBLEMNAME_n_m'`.

## s2mpj_select

```python
from optiprofiler import s2mpj_select
names = s2mpj_select({"ptype": "u", "mindim": 2, "maxdim": 10})
```

Select problems matching criteria. Returns a list of problem name strings.

**Selection criteria**: `ptype`, `mindim`, `maxdim`, `minb`, `maxb`,
`minlcon`, `maxlcon`, `minnlcon`, `maxnlcon`, `mincon`, `maxcon`,
`oracle`, `excludelist`.

## pycutest_load / pycutest_select

Same interface as s2mpj equivalents, but for the PyCUTEst library.
PyCUTEst is **only available on Linux and macOS** and requires separate
installation.

## get_plib_config / set_plib_config

```python
from optiprofiler import get_plib_config, set_plib_config

config = get_plib_config("s2mpj")
set_plib_config("s2mpj", variable_size=True)
```

Read or override problem library configuration at runtime. Overrides
persist for the current Python process via environment variables.

## See Also

- [Python benchmark()](benchmark.md) — where `plibs` is configured
- [Problem Class](problem-class.md) — the returned Problem objects
- [Python Quickstart](../../guides/quickstart-python.md) — usage examples
