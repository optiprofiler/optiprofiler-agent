---
tags: [guide, matlab, problem-library, custom, adapter]
related: [api/python/problem-class.md, guides/custom-problem-library-python.md, guides/problem-metadata.md]
last_updated: 2026-06-05
---

# Custom Problem Library — MATLAB

The MATLAB version of OptiProfiler discovers user-supplied problem
libraries through the package `problem_libs/<library>/` folder
convention, not through Python's `custom_problem_libs_path` option.
Each MATLAB library is a subfolder under `optiprofiler/problem_libs`,
with `.m` files instead of `_tools.py`.

The real-world references this guide leans on:

- [`problem_libs/matcutest/matcutest_load.m`](#) — wraps MatCUTEst
  via `macup` / `secup`. A small, clean adapter.
- [`problem_libs/s2mpj_matlab/s2mpj_load.m`](#) — wraps the S2MPJ
  MATLAB problem repository (each problem is a `.m` class file).

## Step 0 — Reuse what the upstream library already provides

The cleanest adapter is a **two-sided conversion shim** over the
upstream library's own load and select primitives. The MatCUTEst
adapter is the canonical illustration because MatCUTEst happens to
expose **both** primitives, and our `*_load.m` / `*_select.m` reuse
each one — but with a non-trivial conversion in both directions.

| Upstream MatCUTEst API | OptiProfiler wrapper | What the wrapper has to convert |
|---|---|---|
| `macup(name)` → MatCUTEst struct `pb` | `matcutest_load.m` → `Problem` instance | Rename `pb.lb/ub → xl/xu`, `pb.Aineq/bineq → aub/bub`, `pb.Aeq/beq → aeq/beq`; wrap `pb.objective` into `fun`/`grad`/`hess` closures; wrap `pb.nonlcon` into `cub`/`ceq`/`jcub`/`jceq`. |
| `secup(options)` → cell array of names | `matcutest_select.m` → list of names | Rename `options.ptype → options.type`, `options.excludelist → options.blacklist`; map `config.txt`'s `test_feasibility_problems` flag → `options.is_feasibility`; pass the dimension / constraint-count fields through. |

Two messages:

1. **Reuse trumps re-implementing.** If the upstream library already
   has a discovery / selection function, calling it is almost always
   the right move — you inherit its problem coverage, edge cases,
   and performance, and you skip the offline metadata-collection
   pass entirely.
2. **Reuse never means "no conversion".** Field names, option keys,
   feasibility flags, infinity sentinels, scalar-vs-vector
   conventions, and constraint sign conventions differ between
   every upstream API and OptiProfiler's `Problem` / shared option
   vocabulary. **The adapter exists precisely to normalise them.**
   Always read the upstream's docs / source carefully and write the
   per-field mapping down explicitly — undocumented "passthrough" is
   how silent bugs are born.

If your library only has one of the two primitives (only a
`macup`-style loader, or only a `secup`-style selector), reuse what
exists and fall back to the [CSV pattern](#csv-driven-pattern-matches-the-python-guide)
for the missing half. **S2MPJ-MATLAB** is exactly this case: it has
no upstream selector, so `s2mpj_select.m` reads
`probinfo_matlab.csv`, while `s2mpj_load.m` reuses S2MPJ's per-problem
class files. If your library has neither, both halves go through the
CSV pattern + an `mylib_load.m` written from scratch.

## Anatomy

```
optiprofiler/problem_libs/
└── mylib/                     <- this folder name is what users put in options.plibs
    ├── mylib_load.m
    ├── mylib_select.m
    ├── probinfo_mylib.csv     <- optional, used by mylib_select for fast filtering
    ├── config.txt             <- optional, runtime overrides
    └── src/                   <- whatever your problems need
```

OptiProfiler validates `options.plibs` against the subfolder names under
`optiprofiler/problem_libs`. It then adds
`optiprofiler/problem_libs/mylib/` to the MATLAB path and calls
`mylib_load` and `mylib_select` by name.

## Required functions

```matlab
function problem = mylib_load(problem_name)
% Returns an optiprofiler `Problem` instance.

function problem_names = mylib_select(options)
% Returns a cell array of problem name char arrays.
```

The function names must match the directory name (folder `mylib/` →
`mylib_load.m` + `mylib_select.m`).

## Building a `Problem` (struct constructor)

The MATLAB `Problem` class takes a struct (see
[Problem class reference](../api/python/problem-class.md#matlab-equivalent)).
Minimal:

```matlab
problem = Problem(struct( ...
    'name', problem_name, ...
    'fun',  @(x) toy_fun(x), ...
    'x0',   x0));
```

Full template with all optional fields (mirrors the matcutest
reference):

```matlab
function problem = mylib_load(problem_name)
%MYLIB_LOAD Converts mylib problem name to a Problem instance.

    problem_name = char(problem_name);

    % 1. Resolve native handle.
    try
        pb = mylib_native_load(problem_name);
    catch
        error("MATLAB:mylib_load:errorLoad", ...
              "Could not load %s. Check mylib installation.", problem_name);
    end

    % 2. Function handles bound to the native problem `pb`.
    %    These anonymous functions are SAFE in MATLAB parfor — see
    %    concepts/parallel-and-pickle.md.
    fun  = @(x) getfun(pb, x);
    grad = @(x) getgrad(pb, x);
    hess = @(x) gethess(pb, x);

    x0 = pb.x0;
    xl = pb.lb;
    xu = pb.ub;

    % 3. Linear constraints (omit fields if absent).
    aub = pb.Aineq;  bub = pb.bineq;
    aeq = pb.Aeq;    beq = pb.beq;

    % 4. Nonlinear constraints.
    cub  = @(x) getcub(pb, x);
    ceq  = @(x) getceq(pb, x);
    jcub = @(x) getjcub(pb, x);
    jceq = @(x) getjceq(pb, x);

    problem = Problem(struct( ...
        'name', problem_name, 'fun', fun, 'grad', grad, 'hess', hess, ...
        'x0', x0, 'xl', xl, 'xu', xu, ...
        'aub', aub, 'bub', bub, 'aeq', aeq, 'beq', beq, ...
        'cub', cub, 'ceq', ceq, 'jcub', jcub, 'jceq', jceq));
end

function fx = getfun(pb, x)
    try
        evalc('fx = pb.objective(x);');   % suppress library stdout
    catch
        fx = NaN;
    end
end

% ...similar local functions for grad/hess/cub/ceq/jcub/jceq
```

### Why `evalc(...)`?

Many MATLAB problem libraries print diagnostics. `evalc` captures
that output silently so it doesn't drown the benchmark log. Use
`try/catch` to convert library exceptions into `NaN` values — never
let an adapter crash propagate; OptiProfiler treats `NaN` as
"evaluation failed" and continues.

## Implementing `mylib_select`

The options struct received by `mylib_select` follows the same shared
vocabulary as the Python side. Keys you must understand and either
honour or default:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `ptype` | char (subset of `'ubln'`) | `'ubln'` | Allowed problem types |
| `mindim`, `maxdim` | double | `1`, `Inf` | Dimension range |
| `minb`, `maxb` | double | `0`, `Inf` | Bound-constraint count range |
| `minlcon`, `maxlcon` | double | `0`, `Inf` | Linear constraint count range |
| `minnlcon`, `maxnlcon` | double | `0`, `Inf` | Nonlinear constraint count range |
| `mincon`, `maxcon` | double | `0`, `Inf` | Total constraint count range |
| `excludelist` | cell array of char | `{}` | Names to drop |

### Forwarding pattern (matcutest style)

If your native library already supports a similar selection function,
the cleanest adapter is a thin forwarder — see
[`problem_libs/matcutest/matcutest_select.m`](#) which delegates to
MatCUTEst's `secup`:

```matlab
function problem_names = mylib_select(options)
    options = fillDefaults(options);            % shared option block
    options.type      = options.ptype;          options = rmfield(options, 'ptype');
    options.blacklist = options.excludelist;    options = rmfield(options, 'excludelist');

    try
        problem_names = mylib_native_select(options);
    catch
        error("MATLAB:mylib_select:errorSelect", "Native selection failed.");
    end
end
```

### CSV-driven pattern (matches the Python guide)

For libraries without a native selection function, generate
`probinfo_mylib.csv` (see [problem-metadata.md](problem-metadata.md))
and filter in MATLAB. Columns are identical to the Python side
(`problem_name`, `ptype`, `dim`, `mb`, `mlcon`, `mnlcon`, `mcon`,
optionally `isfeasibility`).

```matlab
function problem_names = mylib_select(options)
    options = fillDefaults(options);

    here = fileparts(mfilename('fullpath'));
    info = readtable(fullfile(here, 'probinfo_mylib.csv'));

    keep = false(height(info), 1);
    for i = 1:height(info)
        row = info(i, :);
        ok = contains(options.ptype, row.ptype{1}) ...
          && row.dim   >= options.mindim   && row.dim   <= options.maxdim   ...
          && row.mb    >= options.minb     && row.mb    <= options.maxb     ...
          && row.mlcon >= options.minlcon  && row.mlcon <= options.maxlcon  ...
          && row.mnlcon>= options.minnlcon && row.mnlcon<= options.maxnlcon ...
          && row.mcon  >= options.mincon   && row.mcon  <= options.maxcon   ...
          && ~ismember(row.problem_name{1}, options.excludelist);
        keep(i) = ok;
    end
    problem_names = info.problem_name(keep);
end
```

The helper `fillDefaults` is straightforward — copy it verbatim from
[`problem_libs/matcutest/matcutest_select.m`](#) lines 54–90.

## Anonymous-function safety in MATLAB

The Python `lambda` rule does **not** apply verbatim to MATLAB.
MATLAB's anonymous functions are picklable across parfor workers, so
adapter-internal `@(x) getfun(pb, x)` style closures work even with
parallel benchmarks. The thing that still hurts:

- Anonymous functions closing over **large** captured state (e.g. a
  loaded problem instance whose serialisation is expensive).
- Anonymous functions closing over **handles to OS processes**
  (database connections, GPU contexts) which can't be serialised at
  all.

When in doubt, replace the anonymous function with a top-level
script-local function or a class method.

## Cross-language behaviour parity

Custom libraries are expected to behave identically across Python
and MATLAB whenever both are available. The shared option keys,
shared `probinfo_*.csv` columns, shared `ptype` semantics, and the
identical name → `Problem` mapping make this practical. When you ship
both a `mylib_tools.py` and `mylib_load.m`/`mylib_select.m`,
keep them in lockstep so OptiProfiler's results are comparable.

## Validation checklist

- [ ] The library folder lives under `optiprofiler/problem_libs/mylib/`;
      MATLAB has no `custom_problem_libs_path` option.
- [ ] `mylib_load(name)` returns a `Problem` instance for every name
      from `mylib_select(struct())`.
- [ ] `problem.fun(problem.x0)` returns a finite scalar.
- [ ] `problem.maxcv(problem.x0)` is finite.
- [ ] Calling `benchmark` with `options.n_jobs = 2` succeeds (parfor
      smoke test).
- [ ] `mylib_select(struct('ptype', 'u'))` is a strict subset of
      `mylib_select(struct())`.

## See Also

- [Custom Problem Library — Python](custom-problem-library-python.md) — Python counterpart
- [Problem Class — MATLAB section](../api/python/problem-class.md#matlab-equivalent) — type reference
- [Problem-set metadata helper](problem-metadata.md) — building `probinfo_*.csv`
- [Parallel & Pickle Rules](../concepts/parallel-and-pickle.md)
