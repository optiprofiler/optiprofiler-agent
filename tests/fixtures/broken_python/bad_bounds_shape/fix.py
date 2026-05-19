def validate_bounds(x0, lb, ub):
    if len(lb) != len(x0) or len(ub) != len(x0):
        raise ValueError("bounds shape mismatch")


validate_bounds([0.0, 1.0], [0.0, 0.0], [2.0, 3.0])
