# Feature Effects on Solver Performance

OptiProfiler provides built-in features that modify test problems to evaluate solver robustness under different conditions. Understanding these effects is crucial for interpreting benchmark results.

## plain
- No modification to the original problem.
- Baseline for all comparisons.
- Results reflect pure algorithmic performance on clean problems.

## noisy
- Adds noise to the objective function and nonlinear constraints.
- Options: `noise_level` (default varies), `noise_type` (absolute/relative/mixed), `distribution` (gaussian/uniform).
- **Expected effect**: solvers using function value differences (like finite-difference methods) are more affected than direct-search methods. Model-based methods (e.g., NEWUOA, COBYQA) that build surrogate models may be more sensitive to noise than simplex-based methods (e.g., Nelder-Mead/fminsearch).
- **Interpretation tip**: if a solver performs well under `plain` but poorly under `noisy`, it likely relies on accurate function values.

## perturbed_x0
- Randomly perturbs the initial guess x0.
- Options: `distribution` (gaussian/spherical).
- **Expected effect**: tests solver robustness to starting point. Solvers with good global search ability are less affected.
- **Interpretation tip**: large performance drops indicate sensitivity to initialization.

## truncated
- Truncates objective function and constraints to a given precision.
- **Expected effect**: simulates limited-precision computation. Solvers that require high-precision function values (e.g., for finite differences) are more affected.

## permuted
- Randomly permutes the order of variables.
- **Expected effect**: tests whether a solver exploits variable ordering. Well-designed solvers should be invariant to permutation.

## linearly_transformed
- Applies a linear transformation using a positive diagonal matrix times a random orthogonal matrix.
- **Expected effect**: changes the coordinate system and scaling. Solvers that are not scale-invariant will be affected. This tests conditioning sensitivity.

## random_nan
- Randomly sets some objective/constraint values to NaN.
- Options: `nan_rate` (fraction of evaluations that return NaN).
- **Expected effect**: simulates solver calls that fail (e.g., simulation crashes). Solvers that handle NaN gracefully are more robust.

## unrelaxable_constraints
- Sets the objective function to infinity outside the feasible region.
- **Expected effect**: solvers that explore infeasible points aggressively will see infinite values. Tests feasibility-preserving behavior.

## nonquantifiable_constraints
- Replaces constraint values with binary: 0 (satisfied) or 1 (violated).
- **Expected effect**: removes quantitative constraint information, leaving only feasible/infeasible signals.

## quantized
- Quantizes the objective function and constraints to discrete levels.
- **Expected effect**: similar to truncated but with discrete steps. Tests solver behavior with discontinuous objectives.
