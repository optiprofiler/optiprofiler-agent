# Known Solver Traits for Benchmark Interpretation

This document provides prior knowledge about commonly benchmarked DFO solvers, useful for interpreting OptiProfiler results.

## NEWUOA (from PRIMA)
- **Type**: Model-based trust-region, quadratic interpolation.
- **Strengths**: Very competitive on smooth unconstrained problems. Generally the best-performing DFO solver on classical test sets (Moré & Wild, 2009).
- **Weaknesses**: May struggle with noisy problems since the quadratic model is sensitive to function value noise. Not designed for constrained problems.

## COBYLA (scipy.optimize or PRIMA)
- **Type**: Model-based trust-region with linear approximation of objective and constraints.
- **Strengths**: Handles nonlinear constraints. Uses linear models, which makes it more **noise-robust** than quadratic-model methods.
- **Weaknesses**: Linear models are less accurate than quadratic models, so convergence can be slower on smooth problems. May struggle with high-precision convergence.
- **Typical profile behavior**: Higher scores at loose tolerances, scores drop at tight tolerances (precision cliff).

## COBYQA
- **Type**: Model-based trust-region with quadratic approximation for both objective and constraints.
- **Strengths**: More accurate models than COBYLA, better convergence on smooth problems.
- **Weaknesses**: Quadratic models are more sensitive to noise than linear models. More computationally expensive per iteration.
- **Typical profile behavior**: May start slower than COBYLA but achieve better final accuracy.

## Nelder-Mead Simplex (fminsearch in MATLAB)
- **Type**: Direct search, simplex-based.
- **Strengths**: Simple, no model building. Relatively robust to noise since it only uses function value comparisons (ordering).
- **Weaknesses**: Known to converge to non-stationary points in dimensions > 2 (McKinnon, 1998). Slow convergence rate. No constraint handling.
- **Typical profile behavior**: Reasonable at low precision, poor at high precision.

## fminunc (MATLAB)
- **Type**: Finite-difference BFGS (when gradient not provided).
- **Strengths**: Fast convergence using quasi-Newton direction with finite-difference gradients.
- **Weaknesses**: Finite differences are very sensitive to noise. Not a true DFO method — uses O(n) extra function evaluations per iteration for gradient approximation.
- **Typical profile behavior**: Competitive on smooth problems, dramatically worse under noisy/truncated features.

## General Interpretation Guidelines

When comparing solvers in OptiProfiler results:

1. **Model-based vs direct-search**: Model-based methods (NEWUOA, COBYLA, COBYQA) generally converge faster but may be more sensitive to noise. Direct-search methods (Nelder-Mead) are more robust but slower.

2. **Linear vs quadratic models**: Linear models (COBYLA) are more noise-robust but converge slower. Quadratic models (COBYQA, NEWUOA) converge faster but need more accurate function values.

3. **Finite-difference methods** (fminunc) should be expected to perform poorly under noisy/truncated/random_nan features, as they rely on accurate function value differences.

4. **Budget sensitivity**: Some solvers use more evaluations per iteration (e.g., finite-difference methods use O(n) evaluations for gradient approximation). Data profiles (normalized by dimension) help compare evaluation efficiency fairly.
