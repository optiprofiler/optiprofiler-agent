---
tags: [solver, dfo, model-based, framework]
sources: [_sources/refs/bibliography.md]
related: [solvers/overview.md, solvers/newuoa.md, solvers/bobyqa.md, solvers/cobyla.md]
last_updated: 2025-04-13
---

# PRIMA

**PRIMA** (Reference Implementation for Powell's Methods with Modernization
and Amelioration) provides modern, robust implementations of Powell's
classical DFO solvers.

## Included Solvers

| Solver | Problem Type | Model |
|---|---|---|
| NEWUOA | Unconstrained | Quadratic |
| BOBYQA | Bound constrained | Quadratic |
| LINCOA | Linearly constrained | Quadratic |
| COBYLA | All constraints | Linear |
| UOBYQA | Unconstrained | Quadratic (full) |

## Properties

| Property | Value |
|---|---|
| Type | Meta-framework (multiple solvers) |
| Constraints | All types |
| Languages | Fortran, Python, MATLAB |
| Source | https://github.com/libprima/prima |

## Strengths

- Reference implementation with improved numerical stability
- Covers all problem types through different solvers
- Actively maintained

## Usage

```python
from prima import minimize as prima_minimize

def prima_newuoa(fun, x0):
    res = prima_minimize(fun, x0, method='newuoa')
    return res.x
```

## See Also

- [NEWUOA](newuoa.md), [BOBYQA](bobyqa.md), [COBYLA](cobyla.md) — individual solver pages
- [Solver Overview](overview.md) — all solvers
