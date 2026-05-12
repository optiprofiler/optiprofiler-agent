---
tags: [guide, problem-library, metadata, csv]
related: [guides/custom-problem-library-python.md, guides/custom-problem-library-matlab.md, api/python/problem-class.md]
last_updated: 2026-05-11
---

# Problem-Set Metadata Helper

Both [`mylib_select` (Python)](custom-problem-library-python.md#implementing-mylib_select)
and [`mylib_select` (MATLAB)](custom-problem-library-matlab.md#implementing-mylib_select)
filter problems on metadata — dimension, problem type, constraint
counts — *before* loading them. To keep `select()` fast, the
recommended pattern is to precompute that metadata once into a
`probinfo_<lib>.csv` file that lives alongside the adapter.

This page is the canonical schema and the reference recipe for
producing that file. The built-in libraries do exactly this: see
[`problem_libs/s2mpj_python/probinfo_python.csv`](#),
[`problem_libs/pycutest/probinfo_pycutest.csv`](#),
and the MATLAB collector
[`problem_libs/s2mpj_matlab/.github/actions/collect_info/s_getInfo.m`](#).

## When you do *not* need this page

A `probinfo_<lib>.csv` is only one of two legitimate ways to make
`mylib_select` work. Check before you start writing a collector:

1. **Upstream already exposes a selection function** — e.g.
   MatCUTEst's `secup`. In that case `mylib_select` becomes a
   **conversion shim**: rename our option keys to whatever the
   upstream API expects (for MatCUTEst: `ptype → type`,
   `excludelist → blacklist`, `test_feasibility_problems →
   is_feasibility`, …), call upstream, return the result. There is
   no `probinfo_matcutest.csv` because the metadata never needs to
   leave the upstream library.

   Note that "reuse" still requires careful conversion in both
   directions; see Step 0 in the
   [Python](custom-problem-library-python.md#step-0--reuse-what-the-upstream-library-already-provides)
   and [MATLAB](custom-problem-library-matlab.md#step-0--reuse-what-the-upstream-library-already-provides)
   guides for the exact MatCUTEst tables.

2. **No upstream selector exists** (most pure-Python / pure-MATLAB
   problem sets, including S2MPJ and PyCUTEst). Then the CSV pattern
   below is the path of least resistance — and is what every
   in-repo library other than MatCUTEst follows.

Each built-in library that needs a CSV ships its **own** offline
collector (e.g. `p_getInfo.py` for PyCUTEst, `s_getInfo.py` for
S2MPJ-Python, `s_getInfo.m` for S2MPJ-MATLAB). They are *not* a single
shared script: each library's collector knows that library's
quirks (variable-size suffixes, SIF parameters, feasibility lists,
…). The schema below is what the resulting CSVs **agree on**, and
the script later in this page is the **generic baseline** you can
copy when you write the collector for a new library that needs one.

## CSV schema (canonical)

These are the columns used by the built-in libraries. Custom
libraries should use the same names so that any future tooling
(`opagent` lints, RAG indexing, automated metadata audits) can read
them uniformly. **Bold** columns are required by `mylib_select`'s
default filter; the rest are useful but optional.

| Column | Type | Required | Meaning |
|---|---|---|---|
| **`problem_name`** | str | ✓ | Name passed to `mylib_load` |
| **`ptype`** | one of `'u'`/`'b'`/`'l'`/`'n'` | ✓ | Matches `Problem.ptype` |
| `xtype` | one of `'r'` (real), `'i'` (integer), `'m'` (mixed) | – | Variable type tag |
| **`dim`** | int | ✓ | Default dimension `n` |
| **`mb`** | int | ✓ | Number of finite bound constraints |
| `ml` | int | – | Number of finite lower bounds (subset of `mb`) |
| `mu` | int | – | Number of finite upper bounds (subset of `mb`) |
| **`mcon`** | int | ✓ | Total constraints (`mlcon + mnlcon`) |
| **`mlcon`** | int | ✓ | Linear constraint count (`m_linear_ub + m_linear_eq`) |
| **`mnlcon`** | int | ✓ | Nonlinear constraint count (`m_nonlinear_ub + m_nonlinear_eq`) |
| `m_ub` | int | – | All inequality constraints (linear + nonlinear) |
| `m_eq` | int | – | All equality constraints (linear + nonlinear) |
| `m_linear_ub` | int | – | Linear inequality constraints |
| `m_linear_eq` | int | – | Linear equality constraints |
| `m_nonlinear_ub` | int | – | Nonlinear inequality constraints |
| `m_nonlinear_eq` | int | – | Nonlinear equality constraints |
| `f0` | float | – | `fun(x0)` — sanity-check value at the initial guess |
| `isfeasibility` | 0 / 1 | – | 1 ⇒ pure-feasibility test (no real objective) |
| `isgrad`, `ishess`, `isjcub`, `isjceq`, `ishcub`, `ishceq` | 0 / 1 | – | Availability of each derivative |
| `argins` | str | – | Variable-size parameter blob (S2MPJ style) |
| `dims`, `mbs`, `mls`, `mus`, `mcons`, `mlcons`, `mnlcons`, `m_ubs`, `m_eqs`, `m_linear_ubs`, `m_linear_eqs`, `m_nonlinear_ubs`, `m_nonlinear_eqs`, `f0s` | space-separated str | – | Per-parametrisation columns (when a problem has multiple sizes) |

Empty cells are fine — store missing values as blank, not `NaN` or
`-1`. The built-in selectors treat blank as "this parametrisation is
absent".

## Python recipe — `collect_info.py`

A near-complete script you can drop into your custom library
directory:

```python
"""Generate probinfo_mylib.csv for the mylib problem library.

Run once after the library is installed:
    python collect_info.py
"""

import csv
import os
import signal
import sys
from contextlib import contextmanager

import numpy as np

# Import the adapter you are writing.
from mylib_tools import mylib_load


PROBLEM_NAMES_FILE = "list_of_problems.txt"   # one name per line
TIMEOUT_SECONDS    = 50


@contextmanager
def _timeout(seconds):
    """POSIX-only: abort if a load hangs for too long (CUTEst-style)."""
    def _handler(signum, frame):
        raise TimeoutError(f"load timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _classify(problem) -> str:
    """Return the canonical 'u' / 'b' / 'l' / 'n' ptype tag.

    Mirrors the logic in optiprofiler.opclasses.Problem.__init__.
    """
    if problem.m_nonlinear_ub + problem.m_nonlinear_eq > 0:
        return "n"
    if problem.m_linear_ub + problem.m_linear_eq > 0:
        return "l"
    if problem.mb > 0:
        return "b"
    return "u"


def collect_one(name: str) -> dict:
    """Load one problem and return a metadata row."""
    with _timeout(TIMEOUT_SECONDS):
        p = mylib_load(name)
    try:
        f0 = float(p.fun(p.x0))
    except Exception:
        f0 = ""
    return {
        "problem_name":      name,
        "ptype":             _classify(p),
        "xtype":             "r",                       # adjust if you support integer vars
        "dim":               p.n,
        "mb":                int(p.mb),
        "mcon":              int(p.mcon),
        "mlcon":             int(p.mlcon),
        "mnlcon":            int(p.mnlcon),
        "m_linear_ub":       int(p.m_linear_ub),
        "m_linear_eq":       int(p.m_linear_eq),
        "m_nonlinear_ub":    int(p.m_nonlinear_ub),
        "m_nonlinear_eq":    int(p.m_nonlinear_eq),
        "f0":                f0,
        "isfeasibility":     0,
        "isgrad":  int(callable(p.grad)),
        "ishess":  int(callable(p.hess)),
        "isjcub":  int(callable(p.jcub)),
        "isjceq":  int(callable(p.jceq)),
        "ishcub":  int(callable(p.hcub)),
        "ishceq":  int(callable(p.hceq)),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    names_file = os.path.join(here, PROBLEM_NAMES_FILE)
    with open(names_file) as f:
        names = [line.strip() for line in f if line.strip()]

    cols = [
        "problem_name", "ptype", "xtype", "dim", "mb", "mcon", "mlcon",
        "mnlcon", "m_linear_ub", "m_linear_eq", "m_nonlinear_ub",
        "m_nonlinear_eq", "f0", "isfeasibility", "isgrad", "ishess",
        "isjcub", "isjceq", "ishcub", "ishceq",
    ]

    out_path = os.path.join(here, "probinfo_mylib.csv")
    with open(out_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=cols)
        writer.writeheader()
        for name in names:
            try:
                row = collect_one(name)
                writer.writerow(row)
                print(f"  ok   {name}  (n={row['dim']}, ptype={row['ptype']})")
            except TimeoutError as e:
                print(f"  skip {name}  ({e})", file=sys.stderr)
            except Exception as e:
                print(f"  fail {name}  ({type(e).__name__}: {e})", file=sys.stderr)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
```

Run once:

```bash
cd /abs/path/to/my_libs/mylib
python collect_info.py
```

The CSV produced is exactly what `mylib_select`'s default filter
reads — your library is now searchable by `ptype` / `mindim` /
`maxdim` / etc. without loading any problem.

### Variable-size problems

If a single named problem can be loaded at multiple sizes (S2MPJ
style: `ROSENBR_10`, `ROSENBR_100`, …), extend the script with the
extra columns `dims`, `mcons`, `mbs`, `argins` etc. — these are
**space-separated strings** of per-parametrisation values. Pattern:

```python
row["dims"]   = "10 100 1000"     # dim of each known parametrisation
row["mcons"]  = "0 0 0"
row["argins"] = "{N_10} {N_100} {N_1000}"  # opaque blob your _load decodes
```

The selectors built into the project use these to enumerate
parametrisations on the fly — see the
`variable_size in ('default', 'min', 'max', 'all')` branch in
[`problem_libs/s2mpj_python/s2mpj_tools.py:s2mpj_select`](#).

## MATLAB recipe — `collect_info.m`

The MATLAB counterpart mirrors the Python one. The reference
implementation in
[`problem_libs/s2mpj_matlab/.github/actions/collect_info/s_getInfo.m`](#)
is long because it handles S2MPJ-specific quirks; here is the
slimmed-down version you'd start from:

```matlab
function collect_info()
%COLLECT_INFO Generate probinfo_mylib.csv for the mylib library.

    here = fileparts(mfilename('fullpath'));
    list_file = fullfile(here, 'list_of_problems.txt');

    fid = fopen(list_file, 'r');
    names = textscan(fid, '%s');
    fclose(fid);
    names = names{1};

    header = {'problem_name', 'ptype', 'xtype', 'dim', 'mb', 'mcon', ...
              'mlcon', 'mnlcon', 'm_linear_ub', 'm_linear_eq', ...
              'm_nonlinear_ub', 'm_nonlinear_eq', 'f0', 'isfeasibility'};

    rows = cell(0, numel(header));
    for i = 1:numel(names)
        name = names{i};
        try
            p = mylib_load(name);
        catch ME
            fprintf('  fail %s (%s)\n', name, ME.message);
            continue
        end
        try
            f0 = p.fun(p.x0);
        catch
            f0 = NaN;
        end
        rows(end+1, :) = { name, classify(p), 'r', p.n, p.mb, p.mcon, ...
            p.mlcon, p.mnlcon, p.m_linear_ub, p.m_linear_eq, ...
            p.m_nonlinear_ub, p.m_nonlinear_eq, f0, 0 };  %#ok<AGROW>
        fprintf('  ok   %s (n=%d, ptype=%s)\n', name, p.n, classify(p));
    end

    T = cell2table(rows, 'VariableNames', header);
    writetable(T, fullfile(here, 'probinfo_mylib.csv'));
end

function t = classify(p)
    if p.m_nonlinear_ub + p.m_nonlinear_eq > 0
        t = 'n';
    elseif p.m_linear_ub + p.m_linear_eq > 0
        t = 'l';
    elseif p.mb > 0
        t = 'b';
    else
        t = 'u';
    end
end
```

## Re-running

Treat `probinfo_*.csv` as a generated artefact: regenerate it after

- adding / removing problems from `list_of_problems.txt`,
- changing the dimension defaults inside `mylib_load`,
- bumping a dependency that changes how constraints are reported.

Commit the regenerated CSV. The selectors *never* try to "fall back"
when a name in the CSV no longer loads; the agent expects the CSV and
the adapter to be in sync.

## Validation

After regenerating, a quick sanity pass:

```python
from mylib_tools import mylib_load, mylib_select

names = mylib_select({})              # all problems
assert names, "selector returned empty"
for n in names[:5]:
    p = mylib_load(n)
    print(n, p.n, p.fun(p.x0))         # all should print without exception
```

If `mylib_select({"ptype": "u"})` returns problems that load with
`p.mb > 0` or constraints present, your `classify()` (or CSV) and the
runtime `Problem.ptype` disagree — fix the CSV before shipping.

## See Also

- [Custom Problem Library — Python](custom-problem-library-python.md) — where this CSV is consumed
- [Custom Problem Library — MATLAB](custom-problem-library-matlab.md) — MATLAB consumer
- [Problem Class](../api/python/problem-class.md) — `ptype`/`mb`/`mcon` definitions
- [Built-in adapters](../api/python/plib-tools.md) — `s2mpj` / `pycutest` reference
