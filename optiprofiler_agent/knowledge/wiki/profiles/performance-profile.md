---
tags: [profiles, performance-profile]
sources: [_sources/refs/bibliography.md]
related: [profiles/methodology.md, profiles/data-profile.md, profiles/log-ratio-profile.md]
last_updated: 2025-04-13
---

# Performance Profiles

Performance profiles (Dolan & Moré, 2002) compare solvers by their
**performance ratio** — how much slower each solver is relative to the
fastest solver on each problem.

## How to Read

- **X-axis**: Performance ratio `alpha` (log scale by default)
- **Y-axis**: Fraction of problems solved within ratio `alpha`
- A curve that reaches higher values at the left is **faster**
- The value at `alpha = 1` shows the fraction of problems where the
  solver is the fastest
- The rightmost value shows the **robustness** (fraction of all problems
  solved, regardless of speed)

## Interpretation Tips

- **Higher is better** at every point on the x-axis
- A solver that dominates everywhere is both fastest and most robust
- If curves cross, one solver is faster but less robust, or vice versa
- At very large `alpha`, the y-values reflect overall reliability

## In OptiProfiler Output

Performance profile PDFs are named:
- `perf_*.pdf` — history-based performance profiles
- `perf_*_output.pdf` — output-based performance profiles

Multiple PDFs are generated, one per tolerance level.

## See Also

- [Methodology](methodology.md) — convergence test and scoring
- [Data Profiles](data-profile.md) — budget-based alternative
- [Log-Ratio Profiles](log-ratio-profile.md) — pairwise comparison
