# Common OptiProfiler Errors and Solutions

This document catalogs error patterns from the OptiProfiler source code with user-friendly explanations and fixes.

## 1. Solver Input Errors

### "At least two solvers must be given"
- **Type**: ValueError
- **Trigger**: `len(solvers) < 2` in `benchmark()`
- **Fix**: OptiProfiler requires at least 2 solvers for comparison. Add a second solver:
  ```python
  benchmark([my_solver, reference_solver])
  ```

### "The solvers must be a list of callables"
- **Type**: TypeError
- **Trigger**: `solvers` is not iterable, or contains non-callable items
- **Fix**: Ensure each solver is a function. Common mistake: passing the solver name as a string instead of the function itself:
  ```python
  # Wrong: benchmark(["my_solver", "other_solver"])
  # Right: benchmark([my_solver, other_solver])
  ```

### "Either solvers or the 'load' option must be given"
- **Type**: ValueError
- **Trigger**: No solvers provided and no `load` option
- **Fix**: Provide solvers or use `load` to resume a previous experiment.

## 2. Solver Signature Errors

### TypeError with solver arguments
- **Trigger**: Solver function does not accept the expected arguments
- **Expected signatures by problem type**:
  - Unconstrained (`u`): `solver(fun, x0)` — fun is callable, x0 is 1D numpy array
  - Bound-constrained (`b`): `solver(fun, x0, xl, xu)`
  - Linearly constrained (`l`): `solver(fun, x0, xl, xu, aub, bub, aeq, beq)`
  - Nonlinearly constrained (`n`): `solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)`
- **Key point**: `fun` provides ONLY function values (no gradients). This is DFO.
- **Fix**: Wrap your solver to match the expected signature.

### Solver return value errors
- **Trigger**: Solver returns something other than a 1D numpy array of length n
- **Symptom**: NumPy broadcasting/shape errors during post-processing
- **Fix**: Ensure `return x_best` where `x_best` is a 1D array with the same length as `x0`.

## 3. Feature and Option Errors

### "Unknown feature name: ..."
- **Type**: ValueError
- **Trigger**: `feature_name` not in the valid set
- **Valid features**: plain, noisy, perturbed_x0, truncated, permuted, linearly_transformed, random_nan, unrelaxable_constraints, nonquantifiable_constraints, quantized

### "Unknown option: ..."
- **Type**: ValueError
- **Trigger**: Passing an unrecognized keyword to `benchmark()`
- **Fix**: Check spelling. Only use documented option names (e.g., `n_runs`, `n_jobs`, `ptype`, `mindim`, `maxdim`, `feature_name`, `noise_level`).

### Feature option mismatches
- **Type**: ValueError
- **Trigger**: Passing options that don't belong to the chosen feature (e.g., `noise_level` with `plain`)
- **Fix**: Only pass options relevant to the selected feature.

## 4. Problem Selection Errors

### "ptype" errors
- **Type**: ValueError
- **Trigger**: `ptype` contains characters other than u, b, l, n
- **Valid values**: Any combination of 'u' (unconstrained), 'b' (bound), 'l' (linear), 'n' (nonlinear)
- **Example**: `ptype='ubl'` selects unconstrained, bound, and linearly constrained problems

### Dimension range errors
- **Type**: TypeError/ValueError
- **Trigger**: `mindim` not a positive integer, or `mindim > maxdim`
- **Fix**: Ensure `mindim >= 1` and `mindim <= maxdim`.

### Problem library errors
- **Type**: ValueError
- **Trigger**: Unknown library name in `plibs`, or custom library path doesn't exist
- **Fix**: Use valid library names ('s2mpj', 'pycutest', 'matcutest') or ensure custom path exists and contains `*_tools.py`.

## 5. Profile Option Errors

### n_jobs issues
- Non-integer → TypeError
- Values < 1 are silently changed to 1 with a warning

### benchmark_id / feature_stamp
- Empty string or illegal characters → ValueError
- Only alphanumeric characters, underscores, dots, and hyphens are allowed

### max_tol_order
- Must be in range [1, 16], controls tolerance levels from 10^{-1} to 10^{-max_tol_order}

### max_eval_factor
- Must be positive. Controls per-problem evaluation budget: maxfun = ceil(max_eval_factor * n)

### Custom functions (merit_fun, score_weight_fun, score_fun)
- Must be callable → TypeError if not

## 6. Runtime Behaviors (Not Errors, But Important)

### Solver exceptions are silently caught
- If a solver raises ANY exception during execution, OptiProfiler catches it and logs a warning. The run is recorded as a failure (no successful evaluation).
- **Implication**: Your solver may be crashing silently. Check the log.txt for warnings.

### StopIteration on evaluation budget
- When a solver exceeds 2 * maxfun evaluations, OptiProfiler raises StopIteration.
- If the solver doesn't catch this, the run is treated as failed.
- **Fix**: Well-designed solvers should have their own termination criteria before hitting the budget.

### No built-in timeout
- OptiProfiler does NOT enforce wall-clock time limits.
- A slow solver can run indefinitely.
- **Fix**: Implement timeout in your solver, or use the `n_jobs` option for parallel execution with external timeout.

### NaN handling
- If `fun(x)` raises an exception, OptiProfiler returns NaN for that evaluation.
- With the `random_nan` feature, some evaluations randomly return NaN.
- Solvers should handle NaN gracefully (e.g., reject the point, try another).

### Empty problem selection
- If problem filters are too strict (e.g., `maxdim=1` with `ptype='n'`), no problems may be selected.
- OptiProfiler does NOT raise an error — it returns zero scores silently.
- **Fix**: Check the log.txt to verify how many problems were selected.

## 7. Load/Resume Errors

### load option format
- Must be a valid path to a previous experiment directory
- Common error: pointing to the wrong directory level (should contain test_log/)

### solver_names with load
- If using `load` without `solvers`, do NOT pass `solver_names` — it will cause a TypeError when OptiProfiler tries to check `len(solvers)` on None.
