---
tags: [profiles, log-ratio-profile]
sources: [_sources/refs/bibliography.md]
related: [profiles/methodology.md, profiles/performance-profile.md, profiles/data-profile.md]
last_updated: 2025-04-13
---

# Log-Ratio Profiles

Log-ratio profiles provide a **pairwise comparison** between exactly two
solvers. They show, for each problem, the log-ratio of function evaluations
needed by the two solvers.

## Requirement

Log-ratio profiles are **only available when there are exactly 2 solvers.**
With more solvers, only performance and data profiles are generated.

## How to Read

- **X-axis**: Problems (sorted by log-ratio)
- **Y-axis**: Log-ratio `log2(t1/t2)` where `t1`, `t2` are evaluation
  counts for solvers 1 and 2
- **Positive bars**: Solver 2 is faster (solver 1 used more evaluations)
- **Negative bars**: Solver 1 is faster
- **Taller bars**: Larger performance difference

## Interpretation Tips

- A profile skewed to one side indicates one solver dominates
- Roughly symmetric profiles suggest comparable performance
- Problems where one solver fails appear as extreme values
- The two bar colors correspond to the two solvers (configurable via
  `bar_colors`)

## In OptiProfiler Output

Log-ratio profile PDFs are named:
- `log_ratio_*.pdf` — history-based log-ratio profiles
- `log_ratio_*_output.pdf` — output-based log-ratio profiles

## See Also

- [Methodology](methodology.md) — convergence test and scoring
- [Performance Profiles](performance-profile.md) — multi-solver comparison
- [Data Profiles](data-profile.md) — budget-based comparison
