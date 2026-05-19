import math


def objective(x):
    return sum(v * v for v in x)


value = objective([0.0, 0.0])
assert math.isfinite(value)
