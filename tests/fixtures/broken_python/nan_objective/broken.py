import math


def objective(x):
    return float("nan")


value = objective([0.0, 0.0])
if math.isnan(value):
    raise ValueError("objective returned NaN at x0")
