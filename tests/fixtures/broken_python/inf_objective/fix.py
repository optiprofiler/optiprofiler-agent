import math


def objective(x):
    return sum(v * v for v in x)


value = objective([1.0, 2.0])
assert math.isfinite(value)
