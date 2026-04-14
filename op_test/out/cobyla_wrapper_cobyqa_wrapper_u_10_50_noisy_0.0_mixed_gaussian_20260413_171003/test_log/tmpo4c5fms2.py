import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

# OptiProfiler main API
from optiprofiler import benchmark, get_plib_config

# OptiProfiler classes
from optiprofiler.opclasses import Feature, Problem, FeaturedProblem

# OptiProfiler s2mpj load
from optiprofiler.problem_libs.s2mpj import s2mpj_load

# OptiProfiler utilities
from optiprofiler.utils import (
    FeatureName,
    FeatureOption,
    ProfileOption,
    ProblemError,
)

# Problem libraries (s2mpj is always available)
from optiprofiler.problem_libs.s2mpj import s2mpj_load, s2mpj_select


# =============================================================================
# Solvers
# =============================================================================

def scipy_cobyla(fun, x0, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, **kwargs):
    """COBYLA solver (supports all constraint types via reformulation)."""
    constraints = []
    if xl is not None:
        for i in range(len(xl)):
            if np.isfinite(xl[i]):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, lb=xl[i]: x[i] - lb})
    if xu is not None:
        for i in range(len(xu)):
            if np.isfinite(xu[i]):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, ub=xu[i]: ub - x[i]})
    if aub is not None and bub is not None and len(bub) > 0:
        for i in range(len(bub)):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: bub[i] - aub[i] @ x})
    if aeq is not None and beq is not None and len(beq) > 0:
        for i in range(len(beq)):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: aeq[i] @ x - beq[i]})
    if cub is not None:
        constraints.append({'type': 'ineq', 'fun': lambda x: -np.atleast_1d(cub(x))})
    if ceq is not None:
        constraints.append({'type': 'eq', 'fun': ceq})
    result = minimize(fun, x0, method='COBYLA', constraints=constraints if constraints else (), **kwargs)
    return result.x


def scipy_cobyqa_solver(fun, x0, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, **kwargs):
    """COBYQA solver (native support for bounds and constraints)."""
    bounds = None
    if xl is not None or xu is not None:
        lb = xl if xl is not None else np.full(len(x0), -np.inf)
        ub = xu if xu is not None else np.full(len(x0), np.inf)
        bounds = Bounds(lb, ub)
    constraints = []
    if aub is not None and bub is not None and len(bub) > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if aeq is not None and beq is not None and len(beq) > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    if cub is not None:
        constraints.append(NonlinearConstraint(cub, -np.inf, 0))
    if ceq is not None:
        constraints.append(NonlinearConstraint(ceq, 0, 0))
    result = minimize(fun, x0, method='COBYQA', bounds=bounds, constraints=constraints if constraints else (), **kwargs)
    return result.x


def make_cobyla_solver(maxiter_factor=100):
    """Create a COBYLA solver wrapper with specified iteration limit."""
    def solver(fun, x0):
        return scipy_cobyla(fun, x0, maxiter=maxiter_factor)
    solver.__name__ = f'scipy_cobyla_x{maxiter_factor}'
    return solver


def make_cobyqa_solver(maxiter_factor=100):
    """Create a COBYQA solver wrapper with specified iteration limit."""
    def solver(fun, x0):
        return scipy_cobyqa_solver(fun, x0, maxiter=maxiter_factor)
    solver.__name__ = f'scipy_cobyqa_x{maxiter_factor}'
    return solver


# Default solver wrappers (simple interface for OptiProfiler)
def cobyla_wrapper(fun, x0):
    """COBYLA wrapper for OptiProfiler - accepts only fun and x0."""
    return scipy_cobyla(fun, x0, maxiter=1000)


def cobyqa_wrapper(fun, x0):
    """COBYQA wrapper for OptiProfiler - accepts only fun and x0."""
    return scipy_cobyqa_solver(fun, x0, maxiter=1000)


if __name__ == "__main__":
    # Define solver list - need at least 2 solvers
    solvers = [cobyla_wrapper, cobyqa_wrapper]

    # Benchmark options
    options = {
        'n_runs': 3,
        'feature_name': 'noisy',
        'mindim': 10,
        'maxdim': 50,
        'noise_level': 0.0
    }

    # Run benchmark
    scores = benchmark(
        solvers,
        plibs=['s2mpj'],
        **options
    )

    print("Benchmark completed successfully!")
    print(f"Scores: {scores}")