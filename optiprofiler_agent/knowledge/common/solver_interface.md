# Solver Function Interface Specification

OptiProfiler benchmarks **derivative-free optimization (DFO)** solvers.
`benchmark()` requires **at least 2 solvers**.

## Python Signatures

```python
# Unconstrained
def solver(fun, x0) -> numpy.ndarray: ...

# Bound-constrained
def solver(fun, x0, xl, xu) -> numpy.ndarray: ...

# Linearly constrained
def solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray: ...

# Nonlinearly constrained
def solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray: ...

# Universal wrapper (handles all types)
def solver(fun, x0, xl=None, xu=None, aub=None, bub=None,
           aeq=None, beq=None, cub=None, ceq=None) -> numpy.ndarray: ...
```

## MATLAB Signatures

```matlab
% Unconstrained
x = solver(fun, x0)

% Bound-constrained
x = solver(fun, x0, xl, xu)

% Linearly constrained
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)

% Nonlinearly constrained
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
```

## Key Differences

| Aspect | Python | MATLAB |
|--------|--------|--------|
| Solvers arg | list of callables | cell array of function handles |
| Options | keyword arguments | struct |
| Vectors | numpy 1-D arrays (n,) | column vectors (n×1) |
| Return | numpy.ndarray | column vector |
