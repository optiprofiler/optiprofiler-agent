---
tags: [profiles, data-profile]
sources: [_sources/refs/bibliography.md]
related: [profiles/methodology.md, profiles/performance-profile.md, profiles/log-ratio-profile.md]
last_updated: 2025-04-13
---

# Data Profiles

Data profiles (Moré & Wild, 2009) compare solvers by their
**computational budget** — how many function evaluations (measured in
"simplex gradients") are needed to solve each problem.

## How to Read

- **X-axis**: Number of simplex gradients (function evaluations / (n+1))
- **Y-axis**: Fraction of problems solved within the given budget
- A curve that rises faster means the solver needs **fewer evaluations**

## Simplex Gradients

The x-axis is normalized by problem dimension: one simplex gradient
equals `n+1` function evaluations, where `n` is the problem dimension.
This normalization makes comparisons fair across different problem sizes.

## Interpretation Tips

- **Higher is better** at every budget level
- Steeper initial rise indicates faster convergence on easy problems
- The plateau height indicates overall reliability
- Data profiles are especially useful for DFO where function evaluations
  are expensive

## In OptiProfiler Output

Data profile PDFs are named:
- `data_*.pdf` — history-based data profiles
- `data_*_output.pdf` — output-based data profiles

## See Also

- [Methodology](methodology.md) — convergence test and scoring
- [Performance Profiles](performance-profile.md) — ratio-based alternative
- [Log-Ratio Profiles](log-ratio-profile.md) — pairwise comparison
