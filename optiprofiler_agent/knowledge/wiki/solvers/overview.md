---
tags: [solvers, overview, dfo]
sources: [_sources/refs/bibliography.md]
related: [concepts/dfo.md, concepts/solver-interface.md, solvers/newuoa.md, solvers/cobyla.md, solvers/nelder-mead.md, solvers/bobyqa.md, solvers/powell.md, solvers/prima.md]
last_updated: 2025-04-13
---

# DFO Solver Overview

This section provides entity pages for commonly benchmarked DFO solvers.
Each page describes the solver's algorithm type, strengths, weaknesses,
and expected profile behavior.

## Solver Categories

| Category | Solvers | Key Trait |
|---|---|---|
| Model-based (quadratic) | [NEWUOA](newuoa.md), [BOBYQA](bobyqa.md), [COBYLA](cobyla.md) | Fast convergence, good for smooth functions |
| Direct search | [Nelder-Mead](nelder-mead.md) | No model, robust to noise |
| Pattern/coordinate | [Powell](powell.md) | Coordinate-wise search |
| Meta-framework | [PRIMA](prima.md) | Reference implementation of Powell's methods |

## General Profile Expectations

- **Model-based solvers** typically show steep performance profile curves
  (fast convergence) but may be less robust to noise
- **Direct search methods** show flatter curves but higher plateau values
  (more problems eventually solved)
- **Under noisy features**, direct search methods often outperform
  model-based methods at high tolerances

## See Also

- [DFO](../concepts/dfo.md) — derivative-free optimization context
- [Solver Interface](../concepts/solver-interface.md) — required signatures
- [Feature Effects](../profiles/feature-effects.md) — how features affect results
