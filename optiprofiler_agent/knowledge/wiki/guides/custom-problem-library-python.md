---
tags: [guide, python, problem-library, custom, adapter]
related: [api/python/problem-class.md, api/python/plib-tools.md, guides/problem-metadata.md, concepts/parallel-and-pickle.md]
last_updated: 2026-05-11
---

# Custom Problem Library — Python

This guide explains how to plug a **custom problem source** (any
collection of optimisation test problems you can call from Python)
into OptiProfiler so that
[`benchmark()`](../api/python/benchmark.md) can select problems from
it just like the built-in `s2mpj` and `pycutest` adapters.

The two real-world references this page leans on:

- [`problem_libs/s2mpj_python/s2mpj_tools.py`](#) — wraps the S2MPJ
  Python repository (each problem is a generated Python class).
- [`problem_libs/pycutest/pycutest_tools.py`](#) — wraps the PyCUTEst
  binding to the Fortran CUTEst library.

When you write your own adapter, model it on these two — they cover
the two common cases (pure-Python problem classes vs an external
binding with its own life-cycle).

## Step 0 — Reuse what the upstream library already provides

Before writing any adapter code, **read the upstream library's own
source** and figure out which of these primitives it already exposes:

| Upstream primitive | What it lets you reuse | Conversion still required |
|---|---|---|
| A loader for a single problem by name (e.g. MatCUTEst's `macup`, PyCUTEst's `pycutest.import_problem`, a class-per-problem repo like S2MPJ Python) | The whole problem build — geometry, derivatives, constraints | Field renaming + closure wrapping into our `Problem` (see [Problem class](../api/python/problem-class.md)) |
| A selection / discovery function (e.g. MatCUTEst's `secup`, libraries that ship a `find_problems` or `filter`) | The whole filtering layer — no per-problem load needed | Option-key renaming + return-shape conversion (cell / list / ndarray → `list[str]`) |

Two messages:

1. **Reuse trumps re-implementing.** If upstream has a discovery
   function, calling it is almost always the right move — you
   inherit its coverage, edge cases, and performance, and you skip
   the offline metadata-collection pass entirely.
2. **Reuse never means "no conversion".** Field names, option keys,
   infinity sentinels, scalar-vs-vector conventions, and constraint
   sign conventions differ between every upstream API and
   OptiProfiler's `Problem` / shared option vocabulary. The adapter
   exists precisely to normalise them. Write the per-field mapping
   down explicitly; undocumented "passthrough" is how silent bugs
   are born.

The MATLAB side of OptiProfiler is the cleanest illustration of this
pattern in action — see
[Step 0 of the MATLAB guide](custom-problem-library-matlab.md#step-0--reuse-what-the-upstream-library-already-provides)
for the side-by-side `macup → Problem` and `secup → list of names`
conversion tables that `matcutest_load.m` and `matcutest_select.m`
implement.

In Python, the two built-in adapters illustrate **partial** reuse:

- `pycutest_load` reuses **`pycutest.import_problem`** (load side) and
  then does substantial conversion: replaces the `±1e20` infinity
  sentinels, separates linear from nonlinear constraints, builds
  closures, and so on.
- `pycutest_select` does **not** call any upstream selector — it
  reads its own pre-computed `probinfo_pycutest.csv` because the
  per-problem metadata it needs (variable-size suffixes, SIF
  parameters, …) isn't reliably available through any upstream call.
  This is the fallback pattern; see *Implementing `mylib_select`*
  below.
- `s2mpj_load` reuses each problem's auto-generated Python class as
  the loader; `s2mpj_select` reads its own `probinfo_python.csv`.

If your library has both primitives, you get the MatCUTEst-style
shim on both sides. If only one (typical), reuse that half and fall
back to the CSV pattern for the other. If neither, both halves go
through CSV + a `mylib_load` written from scratch.

## Anatomy

OptiProfiler discovers a custom library purely by **directory
convention**. There is no registry, no decorator, no entrypoint —
just files in the right places.

```
my_libs/                       <- you point custom_problem_libs_path here
└── mylib/                     <- this folder name is what users type as plib
    ├── mylib_tools.py         <- MUST exist; contains two functions
    ├── probinfo_mylib.csv     <- optional: metadata for fast filtering
    ├── config.txt             <- optional: per-library overrides
    └── (whatever else mylib_tools.py needs)
```

Then:

```python
benchmark(
    solvers,
    plibs=["s2mpj", "mylib"],
    custom_problem_libs_path="/abs/path/to/my_libs",
)
```

OptiProfiler will `importlib`-load `mylib.mylib_tools`, call
`mylib_load(name)` for each selected problem name, and call
`mylib_select(options)` to pick problem names in the first place.

## Required functions

Every `<name>_tools.py` must expose **exactly two** module-level
functions:

```python
def mylib_load(problem_name, *args, **kwargs) -> "optiprofiler.opclasses.Problem":
    ...

def mylib_select(options: dict) -> list[str]:
    ...
```

Naming matters: the function names must match the *directory name*
(folder `mylib/` → `mylib_load` + `mylib_select`). OptiProfiler
imports them by name; mismatch silently disables your library.

## Building the `Problem` object

`mylib_load` does three things:

1. Resolve `problem_name` to whatever native handle your source
   library uses (load an instance, call a factory, read a file …).
2. Convert raw bound / constraint data into the shapes the
   `Problem` constructor expects — see the
   [Problem class reference](../api/python/problem-class.md) for the
   exact types.
3. Build closures (or top-level helper functions) for `fun`,
   `grad`, `hess`, `cub`, `ceq`, and their Jacobians/Hessians,
   then hand everything to `Problem(...)`.

### Skeleton

```python
import io
import os
from contextlib import redirect_stdout

import numpy as np

from optiprofiler.opclasses import Problem


def mylib_load(problem_name, *args):
    """Convert a `mylib` problem name to a `Problem` instance."""
    # 1. Resolve the native problem.
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    native = _load_native(src_dir, problem_name, args)

    # 2. Pull out coordinates and bounds, sanitising types.
    x0 = np.asarray(native.x0, dtype=float).flatten()
    xl = np.where(np.asarray(native.xl, dtype=float) <= -1e20,
                  -np.inf, np.asarray(native.xl, dtype=float)).flatten()
    xu = np.where(np.asarray(native.xu, dtype=float) >= 1e20,
                  np.inf, np.asarray(native.xu, dtype=float)).flatten()

    # 3. Build callables. These lambdas are SAFE: they live inside the
    #    worker process. See concepts/parallel-and-pickle.md.
    fun  = lambda x: _silent_call(native.eval, x)
    grad = lambda x: _silent_call(native.grad, x)
    hess = lambda x: _silent_call(native.hess, x)

    # 4. Linear / nonlinear constraints — pass None when absent.
    aub, bub, aeq, beq = _linear_blocks(native)
    cub, ceq, jcub, jceq, hcub, hceq = _nonlinear_blocks(native)

    return Problem(
        fun, x0,
        name=problem_name,
        xl=xl, xu=xu,
        aub=aub, bub=bub, aeq=aeq, beq=beq,
        cub=cub, ceq=ceq,
        grad=grad, hess=hess,
        jcub=jcub, jceq=jceq,
        hcub=hcub, hceq=hceq,
    )


def _silent_call(fn, x):
    """Helper: suppress stdout chatter from a native library."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            return fn(x)
        except Exception:
            return np.full(np.asarray(x).shape, np.nan)
```

The pattern of (a) flattening to `(n,)`, (b) replacing `±1e20`
sentinels with `±np.inf`, (c) swallowing native-library stdout,
(d) returning `NaN` arrays of the right shape on failure — is exactly
what the built-in
[`problem_libs/pycutest/pycutest_tools.py`](#)
and
[`problem_libs/s2mpj_python/s2mpj_tools.py`](#)
do. Reuse them.

### Lambda safety

Closures like `fun = lambda x: native.eval(x)` are **safe** here because
the closure is built and consumed entirely inside a worker process —
no pickling happens. The wider lambda rule is documented in
[Parallel mode and the "harmful lambda" rule](../concepts/parallel-and-pickle.md);
the short version: keep callables created inside `mylib_load` private
to the worker and you're fine. The four places where lambdas would
break (`solvers=`, `Feature` modifiers, `profile_options`, module-level
public functions referenced by those) are all outside the adapter.

## Implementing `mylib_select`

`mylib_select(options)` is called *before* any problem is loaded,
typically once per `benchmark()` invocation. Its job is to enumerate
candidate problem names that satisfy a uniform set of filter
criteria. It must accept the **shared option vocabulary** below; if a
key your library doesn't understand is missing, supply a sensible
default.

### Shared option keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ptype` | `str` made of any of `'u'`/`'b'`/`'l'`/`'n'` | `'ubln'` | Allowed problem types |
| `mindim`, `maxdim` | `int` / `inf` | `1` / `inf` | Dimension range |
| `minb`, `maxb` | `int` / `inf` | `0` / `inf` | Bound-constraint count range |
| `minlcon`, `maxlcon` | `int` / `inf` | `0` / `inf` | Linear constraint count range |
| `minnlcon`, `maxnlcon` | `int` / `inf` | `0` / `inf` | Nonlinear constraint count range |
| `mincon`, `maxcon` | `int` / `inf` | `0` / `inf` | Total constraint count range |
| `excludelist` | `list[str]` | `[]` | Problem names to drop |

(`oracle` is also accepted by S2MPJ; you can ignore it if irrelevant.)

The cleanest implementation is the CSV-driven pattern below: store
problem metadata in `probinfo_mylib.csv`, load it once, filter in
Python. This matches the built-in libraries verbatim.

### CSV-driven `_select`

```python
import os
import numpy as np
import pandas as pd


def mylib_select(options: dict) -> list[str]:
    defaults = {
        "ptype": "ubln",
        "mindim": 1,    "maxdim": np.inf,
        "minb":   0,    "maxb":   np.inf,
        "minlcon": 0,   "maxlcon": np.inf,
        "minnlcon": 0,  "maxnlcon": np.inf,
        "mincon": 0,    "maxcon": np.inf,
        "excludelist": [],
    }
    for k, v in defaults.items():
        options.setdefault(k, v)

    here = os.path.dirname(os.path.abspath(__file__))
    info = pd.read_csv(os.path.join(here, "probinfo_mylib.csv"))

    keep = []
    for _, row in info.iterrows():
        if row["ptype"] not in options["ptype"]:
            continue
        if not (options["mindim"]  <= row["dim"]  <= options["maxdim"]):  continue
        if not (options["minb"]    <= row["mb"]   <= options["maxb"]):    continue
        if not (options["minlcon"] <= row["mlcon"] <= options["maxlcon"]):continue
        if not (options["minnlcon"]<= row["mnlcon"]<= options["maxnlcon"]):continue
        if not (options["mincon"]  <= row["mcon"] <= options["maxcon"]): continue
        if row["problem_name"] in options["excludelist"]:                continue
        keep.append(row["problem_name"])
    return keep
```

### Required CSV columns

| Column | Type | Meaning |
|---|---|---|
| `problem_name` | str | Name handed to `mylib_load` |
| `ptype` | one of `u`, `b`, `l`, `n` | Matches `Problem.ptype` |
| `dim` | int | Default dimension |
| `mb` | int | Number of finite bound constraints |
| `mlcon` | int | Number of linear constraints |
| `mnlcon` | int | Number of nonlinear constraints |
| `mcon` | int | `mlcon + mnlcon` |
| `isfeasibility` | 0/1 (optional) | Whether the problem is a pure feasibility test |

The fields `dims`/`mbs`/`mlcons`/`mnlcons`/`mcons`/`argins` are needed
only if your library exposes the same problem with multiple
parameterisations (S2MPJ-style variable-size suffixes). See
[`problem_libs/s2mpj_python/probinfo_python.csv`](#) for a full example.

**Where does the CSV come from?** You generate it once at build time
by introspecting your library. See the companion guide:
[Problem-set metadata helper](problem-metadata.md).

### Without a CSV

If your library is small (< 50 problems), you can hard-code the
metadata in a Python dict literal and skip the CSV. Filtering logic
stays the same.

If your library is enormous and metadata is expensive to compute
upfront (e.g. dimension depends on a load step), `mylib_select` may
need to actually call `mylib_load` on each candidate. That works but
is slow — caching the metadata in a CSV is strongly preferred.

## Optional helpers

The built-in adapters expose more than the two required functions —
none are strictly required by OptiProfiler but they're useful:

- **`mylib_clear_cache(name)` / `mylib_clear_all_cache()`** — for
  libraries that compile or unpack problems on disk
  (PyCUTEst does this).
- **`mylib_get_params(name)`** — return tunable parameters available
  for a problem (PyCUTEst's `_get_sif_params`).
- **`config.txt`** — a plain file in the library directory that
  `mylib_select` reads for runtime overrides like `variable_size` or
  `test_feasibility_problems`. Pattern verbatim:
  read in `mylib_select`, allow environment variables to take
  precedence.

## Worked end-to-end example

A minimal library with two problems hardcoded:

```python
# my_libs/toy/toy_tools.py
import numpy as np
from optiprofiler.opclasses import Problem


_REGISTRY = {
    "rosenbrock_2d": {
        "ptype": "u", "dim": 2, "mb": 0,
        "mlcon": 0, "mnlcon": 0, "mcon": 0,
        "x0": np.array([-1.2, 1.0]),
    },
    "quadratic_5d": {
        "ptype": "u", "dim": 5, "mb": 0,
        "mlcon": 0, "mnlcon": 0, "mcon": 0,
        "x0": np.ones(5),
    },
}


def _rosenbrock(x):
    return float(100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2)


def _quadratic(x):
    return float(np.dot(x, x))


_FUN_BY_NAME = {"rosenbrock_2d": _rosenbrock, "quadratic_5d": _quadratic}


def toy_load(problem_name):
    meta = _REGISTRY[problem_name]
    return Problem(_FUN_BY_NAME[problem_name], meta["x0"], name=problem_name)


def toy_select(options):
    defaults = {
        "ptype": "ubln", "mindim": 1, "maxdim": np.inf,
        "minb": 0, "maxb": np.inf,
        "minlcon": 0, "maxlcon": np.inf,
        "minnlcon": 0, "maxnlcon": np.inf,
        "mincon": 0, "maxcon": np.inf,
        "excludelist": [],
    }
    for k, v in defaults.items():
        options.setdefault(k, v)

    keep = []
    for name, m in _REGISTRY.items():
        if m["ptype"] not in options["ptype"]: continue
        if not (options["mindim"] <= m["dim"] <= options["maxdim"]): continue
        if name in options["excludelist"]: continue
        keep.append(name)
    return keep
```

Usage:

```python
benchmark(
    [solver_a, solver_b],
    plibs=["toy"],
    custom_problem_libs_path="/abs/path/to/my_libs",
    ptype="u", maxdim=3,            # selects only rosenbrock_2d
)
```

## Validation checklist

Before declaring a custom library "done", verify each item below.
Running on `n_jobs=2` is the single best smoke test — most adapter
bugs surface during pickling.

- [ ] `mylib_load(name)` returns an `optiprofiler.Problem` instance
      for every name returned by `mylib_select({})`.
- [ ] `problem.n` matches the dimension you advertise in the CSV.
- [ ] `problem.fun(problem.x0)` returns a finite scalar.
- [ ] No `1e20` sentinels survive in `problem.xl` / `problem.xu`.
- [ ] `problem.maxcv(problem.x0)` returns a finite scalar (sanity-checks
      every constraint callable).
- [ ] `benchmark(..., plibs=["mylib"], n_jobs=2)` completes without a
      `PicklingError` — see [parallel-and-pickle.md](../concepts/parallel-and-pickle.md).
- [ ] `mylib_select({"ptype": "u"})` is a strict subset of `mylib_select({})`.

## See Also

- [Problem Class](../api/python/problem-class.md) — type/shape reference
- [Custom Problem Library — MATLAB](custom-problem-library-matlab.md) — MATLAB side
- [Problem-set metadata helper](problem-metadata.md) — generate the CSV
- [Parallel & Pickle Rules](../concepts/parallel-and-pickle.md)
- [Built-in adapters](../api/python/plib-tools.md) — for orientation
