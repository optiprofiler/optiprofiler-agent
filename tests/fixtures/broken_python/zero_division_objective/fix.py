def objective(x):
    denom = x[0] if abs(x[0]) > 1e-12 else 1e-12
    return 1.0 / denom


objective([0.0, 1.0])
