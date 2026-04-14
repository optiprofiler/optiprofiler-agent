import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def scipy_cobyla(fun, x0, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, maxiter=1000, **kwargs):
    """COBYLA solver wrapper."""
    bounds = []
    if xl is not None:
        for i, lb in enumerate(xl):
            lower = lb if np.isfinite(lb) else -np.inf
            upper = xu[i] if xu is not None and np.isfinite(xu[i]) else np.inf
            bounds.append((lower, upper))
    elif xu is not None:
        for ub in xu:
            bounds.append((-np.inf, ub if np.isfinite(ub) else np.inf))
    
    constraints = []
    if aub is not None and bub is not None and len(bub) > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if aeq is not None and beq is not None and len(beq) > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    if cub is not None:
        constraints.append(NonlinearConstraint(cub, -np.inf, 0))
    if ceq is not None:
        constraints.append(NonlinearConstraint(ceq, 0, 0))
    
    result = minimize(fun, x0, method='COBYLA', bounds=bounds, constraints=constraints if constraints else (), maxiter=maxiter, **kwargs)
    return result.x


def scipy_cobyqa(fun, x0, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, maxiter=1000, **kwargs):
    """COBYQA solver wrapper."""
    bounds = []
    if xl is not None:
        for i, lb in enumerate(xl):
            lower = lb if np.isfinite(lb) else -np.inf
            upper = xu[i] if xu is not None and np.isfinite(xu[i]) else np.inf
            bounds.append((lower, upper))
    elif xu is not None:
        for ub in xu:
            bounds.append((-np.inf, ub if np.isfinite(ub) else np.inf))
    
    constraints = []
    if aub is not None and bub is not None and len(bub) > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if aeq is not None and beq is not None and len(beq) > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    if cub is not None:
        constraints.append(NonlinearConstraint(cub, -np.inf, 0))
    if ceq is not None:
        constraints.append(NonlinearConstraint(ceq, 0, 0))
    result = minimize(fun, x0, method='COBYQA', bounds=bounds, constraints=constraints if constraints else (), maxiter=maxiter, **kwargs)
    return result.x


def scipy_slsqp(fun, x0, xl=None, xu=None, aub=None, bub=None, aeq=None, beq=None, cub=None, ceq=None, maxiter=1000, **kwargs):
    """SLSQP solver for comparison."""
    bounds = []
    if xl is not None:
        for i, lb in enumerate(xl):
            lower = lb if np.isfinite(lb) else -np.inf
            upper = xu[i] if xu is not None and np.isfinite(xu[i]) else np.inf
            bounds.append((lower, upper))
    elif xu is not None:
        for ub in xu:
            bounds.append((-np.inf, ub if np.isfinite(ub) else np.inf))
    
    constraints = []
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
    
    result = minimize(fun, x0, method='SLSQP', bounds=bounds if bounds else None, 
                      constraints=constraints if constraints else (), maxiter=maxiter, **kwargs)
    return result.x


# =============================================================================
# OptiProfiler Interface Wrappers
# =============================================================================

def cobyla_wrapper(fun, x0):
    """Wrapper for scipy_cobyla matching OptiProfiler solver(fun, x0) interface."""
    return scipy_cobyla(fun, x0, maxiter=1000)


def cobyqa_wrapper(fun, x0):
    """Wrapper for scipy_cobyqa matching OptiProfiler solver(fun, x0) interface."""
    return scipy_cobyqa(fun, x0, maxiter=1000)


def slsqp_wrapper(fun, x0):
    """Wrapper for scipy_slsqp matching OptiProfiler solver(fun, x0) interface."""
    return scipy_slsqp(fun, x0, maxiter=1000)


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == "__main__":
    from optiprofiler import benchmark
    
    options = {
        'n_runs': 10,
        'ptype': 'un'  # 'u' = unconstrained, 'n' = noisy (use 'n' not 'noisy')
    }
    
    # Require at least 2 solvers; using 3 for a comprehensive comparison
    scores = benchmark([cobyla_wrapper, cobyqa_wrapper, slsqp_wrapper], **options)