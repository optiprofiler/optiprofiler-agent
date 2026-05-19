def solver(fun, x0):
    return fun(x0)


def run_solver(candidate):
    return candidate(lambda x: sum(v * v for v in x), [1.0, 2.0])


run_solver(solver)
