---
tags: [troubleshooting, errors, debugging]
sources: [_sources/python/benchmark.json]
related: [troubleshooting/solver-compat.md, troubleshooting/timeout-issues.md, concepts/solver-interface.md]
last_updated: 2026-06-05
---

# Common Errors

This page catalogs frequently encountered errors when using OptiProfiler,
with explanations and fixes.

## E1: Wrong Number of Solver Arguments

**Error**: `TypeError: solver() takes N positional arguments but M were given`

**Cause**: The solver's signature doesn't match the chosen `ptype`. For
example, using a 2-argument solver (`fun, x0`) with `ptype='b'` which
passes 4 arguments.

**Fix**: Match the solver signature to the [problem type](../concepts/problem-types.md).

## E2: Solver Returns Wrong Type

**Error**: `TypeError: solver must return a numpy.ndarray` or similar

**Cause**: The solver returns `None`, a tuple (like `OptimizeResult`), or
a non-array type instead of a plain `np.ndarray`.

**Fix**: Ensure the wrapper returns `res.x` (not `res`):

```python
def my_solver(fun, x0):
    res = minimize(fun, x0, method='Nelder-Mead')
    return res.x  # Not `res`!
```

## E3: Only One Solver Provided

**Error**: `ValueError: at least 2 solvers are required`

**Cause**: `benchmark()` requires **at least 2 solvers** for comparative
profiling.

**Fix**: Add a second solver.

## E4: Invalid Feature Name

**Error**: `ValueError: unknown feature_name 'xxx'`

**Cause**: Using a feature name not in the supported list.

**Fix**: Use one of: `plain`, `perturbed_x0`, `noisy`, `truncated`,
`permuted`, `linearly_transformed`, `random_nan`,
`unrelaxable_constraints`, `nonquantifiable_constraints`, `quantized`,
`custom`.

## E5: Lambda with Parallel Execution

**Error**: `PicklingError: Can't pickle <function <lambda>>`

**Cause**: Lambda functions cannot be serialized for parallel execution.

**Fix**: Use named functions (`def`) instead of `lambda`, especially when
`n_jobs > 1`.

## E6: Invalid ptype

**Error**: `ValueError: invalid ptype`

**Cause**: `ptype` contains characters other than `u`, `b`, `l`, `n`.

**Fix**: Use only valid characters: `'u'`, `'b'`, `'l'`, `'n'`, or
combinations like `'ub'`, `'ubln'`.

## E7: PyCUTEst Not Available

**Error**: `ImportError: pycutest is not installed`

**Cause**: Using `plibs=['pycutest']` without PyCUTEst installed.

**Fix**: Install PyCUTEst (Linux/macOS only):
`pip install pycutest`

## E8: No Problems Selected

**Error**: `ValueError: no problems found matching the criteria`

**Cause**: The combination of `ptype`, `mindim`, `maxdim`, and constraint
range filters selects zero problems.

**Fix**: Broaden the selection criteria (increase `maxdim`, adjust
constraint ranges).

## E9: Importing Problem-Library Helpers from the Package Root

**Error**: `ImportError: cannot import name 's2mpj_load' from 'optiprofiler'`

**Cause**: Current OptiProfiler exports a flat public API containing
`benchmark`, `Problem`, `Feature`, `FeaturedProblem`, `show_versions`,
`get_plib_config`, and `set_plib_config`. The library-specific
`s2mpj_*` and `pycutest_*` helper functions are adapter internals, not
top-level exports.

**Fix**: In normal user code, use `benchmark(..., plibs=['s2mpj'])` or
`benchmark(..., plibs=['pycutest'])`. For custom libraries, use
`custom_problem_libs_path` rather than importing built-in adapter
helpers from the package root.

## E10: Reading `test_log/report.txt`

**Symptom**: A run completes, but some problems are listed under
`merit_init = phi(x_0) = inf`, abnormal solver termination, or output
fallback sections.

**Meaning**: `report.txt` is diagnostic, not just a problem list. It
records selected problem names, timing information, degenerate initial
merit cases where all solvers pass by convention, solver calls that
raised exceptions, solver outputs replaced by `x_0` as a penalty, and
final solver scores.

## See Also

- [Solver Compatibility](solver-compat.md) — adapting third-party solvers
- [Timeout Issues](timeout-issues.md) — handling long-running benchmarks
- [Solver Interface](../concepts/solver-interface.md) — correct signatures
