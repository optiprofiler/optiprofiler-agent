def validate_x0(x0):
    if not all(isinstance(v, (int, float)) for v in x0):
        raise TypeError("x0 must contain numeric values")


validate_x0([0.0, 1.0])
