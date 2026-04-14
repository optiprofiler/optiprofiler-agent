# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from optiprofiler import benchmark

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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    options = {
        'n_jobs': 5,
        'n_runs': 10,
        'mindim': 2,
        'maxdim': 50,
        'feature_name': 'noisy'
    }
    scores = benchmark([scipy_cobyla, scipy_cobyqa], plibs=['s2mpj'], **options)