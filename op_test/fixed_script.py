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

# # PyCUTEst (only on Linux/macOS)
# try:
#     from optiprofiler.problem_libs.pycutest import pycutest_load, pycutest_select
#     PYCUTEST_AVAILABLE = True
# except ImportError:
#     PYCUTEST_AVAILABLE = False
#     print("Note: pycutest not available (Windows or not installed)")


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


def scipy_cobyqa(fun, x0, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, **kwargs):
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

def mod_fun(x, rand_stream, problem):
    return problem.fun(x) + 1e-3 * rand_stream.standard_normal()

def mod_x0(rand_stream, problem):
    return problem.x0 + 1e-3 * rand_stream.standard_normal(problem.n)

def make_solver(para):
    def solver_wrapper(fun, x0):
        return scipy_cobyla(fun, x0, maxiter=100*para)
    return solver_wrapper



if __name__ == "__main__":
    options = {
        'n_runs': 1,
        'feature_name': 'noisy',
    }
    # Fix 1: 'scipy_coby' was undefined — corrected to 'scipy_cobyqa'
    # Fix 2: 'plibs' and options must be passed as keyword arguments, not *options
    # Fix 3: benchmark() does not return 6 values; removed erroneous unpacking
    benchmark([scipy_cobyla, scipy_cobyqa], plibs=['s2mpj'], **options)

    # benchmark(load='latest')