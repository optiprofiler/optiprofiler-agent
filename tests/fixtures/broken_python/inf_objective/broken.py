import math


def objective(x):
    return float("inf")


value = objective([1.0, 2.0])
if not math.isfinite(value):
    raise OverflowError("objective returned Inf at x0")
