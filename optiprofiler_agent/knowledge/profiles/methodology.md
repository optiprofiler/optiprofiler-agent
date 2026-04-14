# Profile Methodology Reference

This document summarizes the mathematical foundations of performance profiles, data profiles, and log-ratio profiles as used in OptiProfiler, based on the established literature [Dolan & Moré 2002, Moré & Wild 2009] and the OptiProfiler paper [Huang, Ragonneau & Zhang 2026].

## Performance Profiles (Dolan & Moré, 2002)

Let P be a set of test problems and S a set of solvers. For each problem p and solver s, the **absolute cost** t_{p,s} is the number of function evaluations solver s needs to solve problem p up to a convergence test.

The **relative cost** (performance ratio) is:

    r_{p,s} = t_{p,s} / min{ t_{p,s} : s in S }

with the convention that infinity/infinity = infinity.

The **performance profile** of solver s is:

    rho_s(alpha) = (1/|P|) * |{ p in P : r_{p,s} <= alpha }|   for alpha >= 1

### How to read a performance profile:
- **At alpha=1**: the fraction of problems where this solver was the fastest (or tied for fastest). Higher is better — this measures **efficiency**.
- **As alpha -> infinity**: the fraction of problems this solver eventually solved (regardless of cost). Higher is better — this measures **robustness**.
- **A curve that is higher everywhere dominates** — the solver is both more efficient and more robust.
- **A crossover** (curves crossing) means one solver is more efficient but the other is more robust.

Performance profiles are considered the "gold standard" in optimization benchmarking (Gould & Scott, 2016). However, note the limitation: "we cannot necessarily assess the performance of one solver relative to another that is not the best" (Gould & Scott, 2016).

## Data Profiles (Moré & Wild, 2009)

The **data profile** of solver s is:

    delta_s(alpha) = (1/|P|) * |{ p in P : t_{p,s} / (n_p + 1) <= alpha }|   for alpha >= 0

where n_p is the dimension of problem p. The quantity t_{p,s}/(n_p+1) is the number of **simplex gradients** — function evaluations normalized by dimension plus one.

### How to read a data profile:
- Data profiles normalize by problem dimension, enabling **fair comparison across problems of different sizes**.
- The x-axis represents the **computational budget** in units of simplex gradients.
- At any given budget, the y-value shows the fraction of problems solved.
- Useful for answering: "Given a budget of K simplex gradients, which solver solves the most problems?"

## Log-ratio Profiles (Shi et al., 2023)

When comparing exactly **two solvers** s1 and s2, the log-ratio is:

    l_p = log2(t_{p,s1} / t_{p,s2})   for p in P\E

where E is the set of problems both solvers fail. The log-ratio profile is these values sorted in ascending order.

### How to read a log-ratio profile:
- Each bar represents one problem.
- **Positive bars**: s1 used more evaluations than s2 (s2 is better on that problem).
- **Negative bars**: s2 used more evaluations than s1 (s1 is better on that problem).
- **Light bars at extremes**: problems where both solvers failed (extended definition).
- The **shaded area** corresponds to the AUC of the performance profile (proven equivalence in Huang et al., 2026).

### Equivalence with performance profiles:
OptiProfiler proves that log-ratio profiles are equivalent to performance profiles when comparing two solvers. Specifically, the log-ratio profile is an inverse function of the performance profile.

## History-based vs Output-based Costs

OptiProfiler provides two methods for measuring the absolute cost:

### History-based cost:
The number of function evaluations to **first reach** a point passing the convergence test. This measures **intrinsic search efficiency** regardless of the solver's stopping rule.

### Output-based cost:
If the solver's final output passes the convergence test, the cost is the total number of evaluations used. Otherwise, the cost is infinity. This measures both the solver's ability to find solutions **and** its effectiveness in deciding when to stop.

**Key insight**: History-based and output-based profiles need not look similar. History-based emphasizes solver efficiency; output-based also reflects stopping criteria quality.

## Convergence Test

A point x passes the convergence test on problem p with tolerance tau in [0,1] if:

    phi(x) <= phi* + tau * (phi(x0) - phi*)

where x0 is the initial guess and phi* is the best merit value achieved by any solver.

Tolerance levels range from tau=0.1 (low accuracy) to tau=10^{-10} (high accuracy). Lower tolerance = stricter convergence = fewer problems solved = lower profile curves.

## Scoring (AUC)

OptiProfiler scores solvers by computing the **area under the curve** (AUC) of their profiles. Larger AUC = better performance.

- For performance and data profiles: AUC is computed up to a truncation point (1.1x the last jump).
- For log-scale axes: AUC is computed with respect to log-transformed coordinates.
- For multiple runs (n_runs > 1): score is based on the average profile across runs.
- **Default score**: average of all history-based performance profile AUCs across all tolerances.

## Budget and Stopping

OptiProfiler enforces a budget-based stopping mechanism:
- `maxfun = ceil(max_eval_factor * n)` where n is the problem dimension.
- If a solver exceeds maxfun, objective/constraint values are set to those at the last evaluated point.
- If the solver doesn't stop after 2*maxfun evaluations, the run is terminated.
- History-based profiles use history truncated at maxfun; output-based profiles use the solver's returned output.

## Multiple Runs (n_runs > 1)

When n_runs > 1 (essential for stochastic features like `noisy`):
- Performance and data profiles show the **average curve** across runs, with error bars (min/max by default).
- Log-ratio profiles treat the same problem in different runs as **distinct problems**, enlarging the problem set.
- Scores for perf/data profiles use the average profile's AUC; log-ratio scores are computed on the enlarged set.
