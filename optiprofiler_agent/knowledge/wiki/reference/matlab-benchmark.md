---
tags: [reference, source-backed, matlab, benchmark]
sources: [_sources/matlab/benchmark.json]
related: [../api/matlab/benchmark.md]
last_updated: 2026-06-18
generated: true
---

# Source Reference: Matlab benchmark.json

This page is auto-generated from `_sources/matlab/benchmark.json`. It is the lossless wiki mirror for this source.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.

## Source Metadata

- Source path: `_sources/matlab/benchmark.json`
- Canonical SHA256: `cd161bfa8902a6585f8280b08ddb1c7c296820143b2d0238d366f1a9d0e447d7`
- Top-level keys: `name`, `description`, `calling_convention`, `solver_signatures`, `solver_notes`, `output_artifacts`, `profile_options`, `feature_options`, `problem_options`, `returns`

## Path Index

| Path | Kind |
|---|---|
| `name` | str |
| `description` | str |
| `calling_convention` | dict[3] |
| `solver_signatures` | dict[4] |
| `solver_signatures.unconstrained` | str |
| `solver_signatures.bound_constrained` | str |
| `solver_signatures.linearly_constrained` | str |
| `solver_signatures.nonlinearly_constrained` | str |
| `solver_notes` | list[5] |
| `output_artifacts` | dict[5] |
| `profile_options` | dict[38] |
| `profile_options.bar_colors` | dict[3] |
| `profile_options.benchmark_id` | dict[2] |
| `profile_options.draw_hist_plots` | dict[4] |
| `profile_options.errorbar_type` | dict[3] |
| `profile_options.feature_stamp` | dict[1] |
| `profile_options.hist_aggregation` | dict[3] |
| `profile_options.line_colors` | dict[2] |
| `profile_options.line_styles` | dict[2] |
| `profile_options.line_widths` | dict[2] |
| `profile_options.load` | dict[1] |
| `profile_options.max_eval_factor` | dict[2] |
| `profile_options.max_tol_order` | dict[2] |
| `profile_options.merit_fun` | dict[1] |
| `profile_options.n_jobs` | dict[2] |
| `profile_options.normalized_scores` | dict[2] |
| `profile_options.project_x0` | dict[2] |
| `profile_options.run_plain` | dict[2] |
| `profile_options.savepath` | dict[2] |
| `profile_options.score_fun` | dict[1] |
| `profile_options.score_only` | dict[2] |
| `profile_options.score_weight_fun` | dict[2] |
| `profile_options.seed` | dict[2] |
| `profile_options.semilogx` | dict[2] |
| `profile_options.silent` | dict[2] |
| `profile_options.solver_isrand` | dict[2] |
| `profile_options.solver_names` | dict[2] |
| `profile_options.solver_verbose` | dict[2] |
| `profile_options.solvers_to_load` | dict[2] |
| `profile_options.summarize_data_profiles` | dict[2] |
| `profile_options.summarize_log_ratio_profiles` | dict[2] |
| `profile_options.summarize_output_based_profiles` | dict[2] |
| `profile_options.summarize_performance_profiles` | dict[2] |
| `profile_options.xlabel_data_profile` | dict[2] |
| `profile_options.xlabel_log_ratio_profile` | dict[2] |
| `profile_options.xlabel_performance_profile` | dict[2] |
| `profile_options.ylabel_data_profile` | dict[2] |
| `profile_options.ylabel_log_ratio_profile` | dict[2] |
| `profile_options.ylabel_performance_profile` | dict[2] |
| `feature_options` | dict[27] |
| `feature_options.feature_name` | dict[3] |
| `feature_options.n_runs` | dict[2] |
| `feature_options.distribution` | dict[2] |
| `feature_options.perturbation_level` | dict[2] |
| `feature_options.noise_level` | dict[2] |
| `feature_options.noise_type` | dict[3] |
| `feature_options.noise_mode` | dict[3] |
| `feature_options.noise_map` | dict[2] |
| `feature_options.significant_digits` | dict[2] |
| `feature_options.perturbed_trailing_digits` | dict[2] |
| `feature_options.rotated` | dict[2] |
| `feature_options.condition_factor` | dict[2] |
| `feature_options.nan_rate` | dict[2] |
| `feature_options.unrelaxable_bounds` | dict[2] |
| `feature_options.unrelaxable_linear_constraints` | dict[2] |
| `feature_options.unrelaxable_nonlinear_constraints` | dict[2] |
| `feature_options.mesh_size` | dict[2] |
| `feature_options.mesh_type` | dict[3] |
| `feature_options.ground_truth` | dict[2] |
| `feature_options.mod_x0` | dict[1] |
| `feature_options.mod_affine` | dict[1] |
| `feature_options.mod_bounds` | dict[1] |
| `feature_options.mod_linear_ub` | dict[1] |
| `feature_options.mod_linear_eq` | dict[1] |
| `feature_options.mod_fun` | dict[1] |
| `feature_options.mod_cub` | dict[1] |
| `feature_options.mod_ceq` | dict[1] |
| `problem_options` | dict[15] |
| `problem_options.plibs` | dict[2] |
| `problem_options.ptype` | dict[3] |
| `problem_options.mindim` | dict[2] |
| `problem_options.maxdim` | dict[2] |
| `problem_options.minb` | dict[2] |
| `problem_options.maxb` | dict[2] |
| `problem_options.minlcon` | dict[2] |
| `problem_options.maxlcon` | dict[2] |
| `problem_options.minnlcon` | dict[2] |
| `problem_options.maxnlcon` | dict[2] |
| `problem_options.mincon` | dict[2] |
| `problem_options.maxcon` | dict[2] |
| `problem_options.excludelist` | dict[2] |
| `problem_options.problem_names` | dict[2] |
| `problem_options.problem` | dict[2] |
| `returns` | dict[3] |
| `returns.solver_scores` | dict[2] |
| `returns.profile_scores` | dict[2] |
| `returns.curves` | dict[2] |

## name

```text
benchmark
```

## description

```text
Benchmark optimization solvers on a set of problems with specified features.
```

## calling_convention

```json
{
  "options": "struct with fields (NOT name-value pairs). Example: options.ptype = 'u'; options.mindim = 2; benchmark(solvers, options);",
  "solvers": "cell array of function handles: {@solver1, @solver2}",
  "syntax": "[solver_scores, profile_scores, curves] = benchmark(solvers, options)"
}
```

## solver_signatures

```json
{
  "bound_constrained": "x = solver(fun, x0, xl, xu)",
  "linearly_constrained": "x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)",
  "nonlinearly_constrained": "x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)",
  "unconstrained": "x = solver(fun, x0)"
}
```

## solver_signatures.unconstrained

```text
x = solver(fun, x0)
```

## solver_signatures.bound_constrained

```text
x = solver(fun, x0, xl, xu)
```

## solver_signatures.linearly_constrained

```text
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)
```

## solver_signatures.nonlinearly_constrained

```text
x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
```

## solver_notes

```json
[
  "fun is a function handle: fun(x) -> scalar. Provides ONLY function values (DFO).",
  "x0 is a column vector.",
  "All constraint vectors are column vectors.",
  "Must return column vector x.",
  "At least 2 solvers required (cell array of function handles)."
]
```

## output_artifacts

```json
{
  "detailed_profiles": "detailed_profiles/ contains high-quality single profile PDFs.",
  "history_plots": "history_plots/ contains per-problem history plots when draw_hist_plots is not 'none'.",
  "summary_pdf": "summary_<stamp>.pdf contains the merged summary profiles for the run.",
  "test_log": "test_log/ stores log files, report.txt, option snapshots, curves, and profile scores.",
  "test_log_report": "test_log/report.txt records selected problems, timing, merit_init = phi(x_0) = Inf cases, abnormal solver terminations, output fallbacks, and solver scores."
}
```

## profile_options

```json
{
  "bar_colors": {
    "choices": [
      "r",
      "g",
      "b",
      "c",
      "m",
      "y",
      "k"
    ],
    "default": "set to the first two colors in the line_colors option",
    "description": "two different colors for the bars of two solvers in the log-ratio profiles. It can be a cell array of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a 2-by-3 matrix with each row being a RGB triplet. Default is set to the first two colors in the line_colors option."
  },
  "benchmark_id": {
    "default": "'out' if the option load is not provided, otherwise default is '.'",
    "description": "the identifier of the test. It is used to create the specific directory to store the results. Default is 'out' if the option load is not provided, otherwise default is '.'."
  },
  "draw_hist_plots": {
    "choices": [
      "none",
      "sequential",
      "parallel"
    ],
    "default": "'parallel'",
    "description": "whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', we will not draw the history plots. If it is 'parallel', we will draw the history plots in the same time when solvers are solving the problems. If it is 'sequential', we will draw the history plots after all the problems are solved. Default is 'sequential'.",
    "source_note": "MATLAB getDefaultProfileOptions.m sets draw_hist_plots to 'parallel' in normal runs; load mode forces it to 'sequential'."
  },
  "errorbar_type": {
    "choices": [
      "minmax",
      "meanstd"
    ],
    "default": "'minmax', meaning that we takes the pointwise minimum and maximum of the curves",
    "description": "the type of the uncertainty interval that can be either 'minmax' or 'meanstd'. When n_runs is greater than 1, we run several times of the experiments and get average curves and get average curves and uncertainty intervals. Default is 'minmax', meaning that we takes the pointwise minimum and maximum of the curves."
  },
  "feature_stamp": {
    "description": "the stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features."
  },
  "hist_aggregation": {
    "choices": [
      "min",
      "mean",
      "max"
    ],
    "default": "'min'",
    "description": "the aggregation method we use to reduce the number of points in the history plots. It can be 'min', 'mean', or 'max'. Default is 'min'."
  },
  "line_colors": {
    "choices": [
      "r",
      "g",
      "b",
      "c",
      "m",
      "y",
      "k"
    ],
    "description": "the colors of the lines in the plots. It can be a cell array of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a matrix with each row being a RGB triplet. Default line colors are those in the palettename named “gem” (see MATLAB documentation for ‘colororder’). Note that if the number of solvers is greater than the number of colors, we will cycle through the colors."
  },
  "line_styles": {
    "choices": [
      "-",
      "-.",
      "--",
      ":",
      "o",
      "+",
      "*",
      ".",
      "x",
      "s",
      "d",
      "^",
      "v",
      ">",
      "<",
      "p",
      "h"
    ],
    "description": "the styles of the lines in the plots. It can be a cell array of chars that are the combinations of line styles ('-', '-.', '--', ':') and markers ('o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'). Default line style order is {'-', '-.', '--', ':'}. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles."
  },
  "line_widths": {
    "default": "1.5",
    "description": "the widths of the lines in the plots. It should be a positive scalar or a vector. Default is 1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths."
  },
  "load": {
    "description": "loading the stored data from a completed experiment and draw profiles. It can be either 'latest' or a time stamp of an experiment in the format of ‘yyyyMMdd_HHmmss’. No default."
  },
  "max_eval_factor": {
    "default": "500",
    "description": "the factor multiplied to each problem’s dimension to get the maximum number of evaluations for each problem. Default is 500."
  },
  "max_tol_order": {
    "default": "10",
    "description": "the maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of ‘tolerances’ to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as 10^(-1:-1:-max_tol_order). Default is 10."
  },
  "merit_fun": {
    "description": "the merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a function handle (fun_value, maxcv_value, maxcv_init) -> merit_value, where fun_value is the objective function value, maxcv_value is the maximum constraint violation, and maxcv_init is the maximum constraint violation at the initial guess. The size of fun_values and maxcv_values is the same, and the size of maxcv_init is the same as the second to last dimensions of fun_values. The default merit function varphi(x) is defined by the objective function f(x) and the maximum constraint violation v(x) as \\[\\begin{split}\\varphi(x) = \\begin{cases} f(x), & \\text{if } v(x) \\le v_1, \\\\ f(x) + 10^5 \\cdot (v(x) - v_1), & \\text{if } v_1 < v(x) \\le v_2, \\\\ +\\infty, & \\text{if } v(x) > v_2, \\end{cases}\\end{split}\\] where \\(v_1 = \\min(0.01,\\; 10^{-10} \\max(1, v_0))\\), \\(v_2 = \\max(0.1,\\; 2v_0)\\), and \\(v_0\\) is the initial maximum constraint violation. If \\(\\varphi(x_0) = +\\infty\\) for a problem/run, the convergence test is degenerate; by convention, all solvers are declared to pass that problem/run. These cases are listed in test_log/report.txt."
  },
  "n_jobs": {
    "default": "a conservative number of workers, chosen as about half of the available workers (at least 2 when more than one worker is available)",
    "description": "the number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers (at least 2 when more than one worker is available)."
  },
  "normalized_scores": {
    "default": "true",
    "description": "whether to normalize the scores of the solvers by the maximum score of the solvers. Default is true."
  },
  "project_x0": {
    "default": "false",
    "description": "whether to project the initial point to the feasible set. Default is false."
  },
  "run_plain": {
    "default": "false",
    "description": "whether to run an extra experiment with the 'plain' feature. Default is false."
  },
  "savepath": {
    "default": "'pwd', the current working directory",
    "description": "the path to store the results. Default is 'pwd', the current working directory."
  },
  "score_fun": {
    "description": "the scoring function to calculate the scores of the solvers. It should be a function handle profile_scores -> solver_scores, where profile_scores is a 4D tensor containing scores for all profiles. The first dimension of profile_scores corresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring function takes the average of the history-based performance profiles under all the tolerances."
  },
  "score_only": {
    "default": "false",
    "description": "whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default is false."
  },
  "score_weight_fun": {
    "default": "1",
    "description": "the weight function to calculate the scores of the solvers in the performance and data profiles. It should be a function handle representing a nonnegative function in R^+. Default is 1."
  },
  "seed": {
    "default": "0",
    "description": "the seed of the random number generator. Default is 0."
  },
  "semilogx": {
    "default": "true",
    "description": "whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default is true."
  },
  "silent": {
    "default": "false",
    "description": "whether to show the information of the progress. Default is false."
  },
  "solver_isrand": {
    "default": "all false",
    "description": "whether the solvers are randomized or not. It is a logical array of the same length as the number of solvers, where the value is true if the solver is randomized, and false otherwise. Default is all false. Note that if n_runs is not specified, we will set it 5 for the randomized solvers."
  },
  "solver_names": {
    "default": "the names of the function handles in solvers",
    "description": "the names of the solvers. Default is the names of the function handles in solvers."
  },
  "solver_verbose": {
    "default": "1",
    "description": "the level of the verbosity of the solvers. 0 means no verbosity, 1 means some verbosity, and 2 means full verbosity. Default is 1."
  },
  "solvers_to_load": {
    "default": "all the solvers",
    "description": "the indices of the solvers to load when the load option is provided. It can be a vector of different integers selected from 1 to the total number of solvers of the loading experiment. At least two indices should be provided. Default is all the solvers."
  },
  "summarize_data_profiles": {
    "default": "true",
    "description": "whether to add all the data profiles to the summary PDF. Default is true."
  },
  "summarize_log_ratio_profiles": {
    "default": "false",
    "description": "whether to add all the log-ratio profiles to the summary PDF. Default is false."
  },
  "summarize_output_based_profiles": {
    "default": "true",
    "description": "whether to add all the output-based profiles of the selected profiles to the summary PDF. Default is true."
  },
  "summarize_performance_profiles": {
    "default": "true",
    "description": "whether to add all the performance profiles to the summary PDF. Default is true."
  },
  "xlabel_data_profile": {
    "default": "'Number of simplex gradients'",
    "description": "the label of the x-axis of the data profiles. Default is 'Number of simplex gradients'. Note: the 'Interpreter' property is set to 'latex', so LaTeX formatting is supported. The same applies to the options xlabel_log_ratio_profile, xlabel_performance_profile, ylabel_data_profile, ylabel_log_ratio_profile, and ylabel_performance_profile."
  },
  "xlabel_log_ratio_profile": {
    "default": "'Problem'",
    "description": "the label of the x-axis of the log-ratio profiles. Default is 'Problem'."
  },
  "xlabel_performance_profile": {
    "default": "'Performance ratio'",
    "description": "the label of the x-axis of the performance profiles. Default is 'Performance ratio'."
  },
  "ylabel_data_profile": {
    "default": "'Data profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
    "description": "the label of the y-axis of the data profiles. Default is 'Data profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format. You can also use %s in your custom label, and it will be replaced accordingly. The same applies to the options ylabel_log_ratio_profile and ylabel_performance_profile."
  },
  "ylabel_log_ratio_profile": {
    "default": "'Log-ratio profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
    "description": "the label of the y-axis of the log-ratio profiles. Default is 'Log-ratio profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format."
  },
  "ylabel_performance_profile": {
    "default": "'Performance profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
    "description": "the label of the y-axis of the performance profiles. Default is 'Performance profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format."
  }
}
```

## profile_options.bar_colors

```json
{
  "choices": [
    "r",
    "g",
    "b",
    "c",
    "m",
    "y",
    "k"
  ],
  "default": "set to the first two colors in the line_colors option",
  "description": "two different colors for the bars of two solvers in the log-ratio profiles. It can be a cell array of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a 2-by-3 matrix with each row being a RGB triplet. Default is set to the first two colors in the line_colors option."
}
```

## profile_options.benchmark_id

```json
{
  "default": "'out' if the option load is not provided, otherwise default is '.'",
  "description": "the identifier of the test. It is used to create the specific directory to store the results. Default is 'out' if the option load is not provided, otherwise default is '.'."
}
```

## profile_options.draw_hist_plots

```json
{
  "choices": [
    "none",
    "sequential",
    "parallel"
  ],
  "default": "'parallel'",
  "description": "whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', we will not draw the history plots. If it is 'parallel', we will draw the history plots in the same time when solvers are solving the problems. If it is 'sequential', we will draw the history plots after all the problems are solved. Default is 'sequential'.",
  "source_note": "MATLAB getDefaultProfileOptions.m sets draw_hist_plots to 'parallel' in normal runs; load mode forces it to 'sequential'."
}
```

## profile_options.errorbar_type

```json
{
  "choices": [
    "minmax",
    "meanstd"
  ],
  "default": "'minmax', meaning that we takes the pointwise minimum and maximum of the curves",
  "description": "the type of the uncertainty interval that can be either 'minmax' or 'meanstd'. When n_runs is greater than 1, we run several times of the experiments and get average curves and get average curves and uncertainty intervals. Default is 'minmax', meaning that we takes the pointwise minimum and maximum of the curves."
}
```

## profile_options.feature_stamp

```json
{
  "description": "the stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features."
}
```

## profile_options.hist_aggregation

```json
{
  "choices": [
    "min",
    "mean",
    "max"
  ],
  "default": "'min'",
  "description": "the aggregation method we use to reduce the number of points in the history plots. It can be 'min', 'mean', or 'max'. Default is 'min'."
}
```

## profile_options.line_colors

```json
{
  "choices": [
    "r",
    "g",
    "b",
    "c",
    "m",
    "y",
    "k"
  ],
  "description": "the colors of the lines in the plots. It can be a cell array of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a matrix with each row being a RGB triplet. Default line colors are those in the palettename named “gem” (see MATLAB documentation for ‘colororder’). Note that if the number of solvers is greater than the number of colors, we will cycle through the colors."
}
```

## profile_options.line_styles

```json
{
  "choices": [
    "-",
    "-.",
    "--",
    ":",
    "o",
    "+",
    "*",
    ".",
    "x",
    "s",
    "d",
    "^",
    "v",
    ">",
    "<",
    "p",
    "h"
  ],
  "description": "the styles of the lines in the plots. It can be a cell array of chars that are the combinations of line styles ('-', '-.', '--', ':') and markers ('o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'). Default line style order is {'-', '-.', '--', ':'}. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles."
}
```

## profile_options.line_widths

```json
{
  "default": "1.5",
  "description": "the widths of the lines in the plots. It should be a positive scalar or a vector. Default is 1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths."
}
```

## profile_options.load

```json
{
  "description": "loading the stored data from a completed experiment and draw profiles. It can be either 'latest' or a time stamp of an experiment in the format of ‘yyyyMMdd_HHmmss’. No default."
}
```

## profile_options.max_eval_factor

```json
{
  "default": "500",
  "description": "the factor multiplied to each problem’s dimension to get the maximum number of evaluations for each problem. Default is 500."
}
```

## profile_options.max_tol_order

```json
{
  "default": "10",
  "description": "the maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of ‘tolerances’ to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as 10^(-1:-1:-max_tol_order). Default is 10."
}
```

## profile_options.merit_fun

```json
{
  "description": "the merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a function handle (fun_value, maxcv_value, maxcv_init) -> merit_value, where fun_value is the objective function value, maxcv_value is the maximum constraint violation, and maxcv_init is the maximum constraint violation at the initial guess. The size of fun_values and maxcv_values is the same, and the size of maxcv_init is the same as the second to last dimensions of fun_values. The default merit function varphi(x) is defined by the objective function f(x) and the maximum constraint violation v(x) as \\[\\begin{split}\\varphi(x) = \\begin{cases} f(x), & \\text{if } v(x) \\le v_1, \\\\ f(x) + 10^5 \\cdot (v(x) - v_1), & \\text{if } v_1 < v(x) \\le v_2, \\\\ +\\infty, & \\text{if } v(x) > v_2, \\end{cases}\\end{split}\\] where \\(v_1 = \\min(0.01,\\; 10^{-10} \\max(1, v_0))\\), \\(v_2 = \\max(0.1,\\; 2v_0)\\), and \\(v_0\\) is the initial maximum constraint violation. If \\(\\varphi(x_0) = +\\infty\\) for a problem/run, the convergence test is degenerate; by convention, all solvers are declared to pass that problem/run. These cases are listed in test_log/report.txt."
}
```

## profile_options.n_jobs

```json
{
  "default": "a conservative number of workers, chosen as about half of the available workers (at least 2 when more than one worker is available)",
  "description": "the number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers (at least 2 when more than one worker is available)."
}
```

## profile_options.normalized_scores

```json
{
  "default": "true",
  "description": "whether to normalize the scores of the solvers by the maximum score of the solvers. Default is true."
}
```

## profile_options.project_x0

```json
{
  "default": "false",
  "description": "whether to project the initial point to the feasible set. Default is false."
}
```

## profile_options.run_plain

```json
{
  "default": "false",
  "description": "whether to run an extra experiment with the 'plain' feature. Default is false."
}
```

## profile_options.savepath

```json
{
  "default": "'pwd', the current working directory",
  "description": "the path to store the results. Default is 'pwd', the current working directory."
}
```

## profile_options.score_fun

```json
{
  "description": "the scoring function to calculate the scores of the solvers. It should be a function handle profile_scores -> solver_scores, where profile_scores is a 4D tensor containing scores for all profiles. The first dimension of profile_scores corresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring function takes the average of the history-based performance profiles under all the tolerances."
}
```

## profile_options.score_only

```json
{
  "default": "false",
  "description": "whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default is false."
}
```

## profile_options.score_weight_fun

```json
{
  "default": "1",
  "description": "the weight function to calculate the scores of the solvers in the performance and data profiles. It should be a function handle representing a nonnegative function in R^+. Default is 1."
}
```

## profile_options.seed

```json
{
  "default": "0",
  "description": "the seed of the random number generator. Default is 0."
}
```

## profile_options.semilogx

```json
{
  "default": "true",
  "description": "whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default is true."
}
```

## profile_options.silent

```json
{
  "default": "false",
  "description": "whether to show the information of the progress. Default is false."
}
```

## profile_options.solver_isrand

```json
{
  "default": "all false",
  "description": "whether the solvers are randomized or not. It is a logical array of the same length as the number of solvers, where the value is true if the solver is randomized, and false otherwise. Default is all false. Note that if n_runs is not specified, we will set it 5 for the randomized solvers."
}
```

## profile_options.solver_names

```json
{
  "default": "the names of the function handles in solvers",
  "description": "the names of the solvers. Default is the names of the function handles in solvers."
}
```

## profile_options.solver_verbose

```json
{
  "default": "1",
  "description": "the level of the verbosity of the solvers. 0 means no verbosity, 1 means some verbosity, and 2 means full verbosity. Default is 1."
}
```

## profile_options.solvers_to_load

```json
{
  "default": "all the solvers",
  "description": "the indices of the solvers to load when the load option is provided. It can be a vector of different integers selected from 1 to the total number of solvers of the loading experiment. At least two indices should be provided. Default is all the solvers."
}
```

## profile_options.summarize_data_profiles

```json
{
  "default": "true",
  "description": "whether to add all the data profiles to the summary PDF. Default is true."
}
```

## profile_options.summarize_log_ratio_profiles

```json
{
  "default": "false",
  "description": "whether to add all the log-ratio profiles to the summary PDF. Default is false."
}
```

## profile_options.summarize_output_based_profiles

```json
{
  "default": "true",
  "description": "whether to add all the output-based profiles of the selected profiles to the summary PDF. Default is true."
}
```

## profile_options.summarize_performance_profiles

```json
{
  "default": "true",
  "description": "whether to add all the performance profiles to the summary PDF. Default is true."
}
```

## profile_options.xlabel_data_profile

```json
{
  "default": "'Number of simplex gradients'",
  "description": "the label of the x-axis of the data profiles. Default is 'Number of simplex gradients'. Note: the 'Interpreter' property is set to 'latex', so LaTeX formatting is supported. The same applies to the options xlabel_log_ratio_profile, xlabel_performance_profile, ylabel_data_profile, ylabel_log_ratio_profile, and ylabel_performance_profile."
}
```

## profile_options.xlabel_log_ratio_profile

```json
{
  "default": "'Problem'",
  "description": "the label of the x-axis of the log-ratio profiles. Default is 'Problem'."
}
```

## profile_options.xlabel_performance_profile

```json
{
  "default": "'Performance ratio'",
  "description": "the label of the x-axis of the performance profiles. Default is 'Performance ratio'."
}
```

## profile_options.ylabel_data_profile

```json
{
  "default": "'Data profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
  "description": "the label of the y-axis of the data profiles. Default is 'Data profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format. You can also use %s in your custom label, and it will be replaced accordingly. The same applies to the options ylabel_log_ratio_profile and ylabel_performance_profile."
}
```

## profile_options.ylabel_log_ratio_profile

```json
{
  "default": "'Log-ratio profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
  "description": "the label of the y-axis of the log-ratio profiles. Default is 'Log-ratio profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format."
}
```

## profile_options.ylabel_performance_profile

```json
{
  "default": "'Performance profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
  "description": "the label of the y-axis of the performance profiles. Default is 'Performance profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format."
}
```

## feature_options

```json
{
  "condition_factor": {
    "default": "0",
    "description": "the scaling factor of the condition number of the linear transformation in the 'linearly_transformed' feature. More specifically, the condition number of the linear transformation will be 2 ^ (condition_factor * n / 2), where n is the dimension of the problem. Default is 0."
  },
  "distribution": {
    "default": "'spherical'",
    "description": "the distribution of perturbation in 'perturbed_x0' feature or random noise in 'noisy' feature. It should be either a string (or char), or a function handle (random_stream, dimension) -> random vector that accepts a random_stream and the dimension of a problem and returning a random vector with the given dimension. In 'perturbed_x0' case, the char should be either 'spherical' or 'gaussian' (default is 'spherical'). In 'noisy' case, the char should be either 'gaussian' or 'uniform' (default is 'gaussian'), and the function handle should accept a random stream and output size."
  },
  "feature_name": {
    "choices": [
      "plain",
      "perturbed_x0",
      "noisy",
      "truncated",
      "permuted",
      "linearly_transformed",
      "random_nan",
      "unrelaxable_constraints",
      "nonquantifiable_constraints",
      "quantized",
      "custom"
    ],
    "default": "'plain'",
    "description": "the name of the feature. The available features are 'plain', 'perturbed_x0', 'noisy', 'truncated', 'permuted', 'linearly_transformed', 'random_nan', 'unrelaxable_constraints', 'nonquantifiable_constraints', 'quantized', and 'custom'. Default is 'plain'."
  },
  "ground_truth": {
    "default": "true",
    "description": "whether the featured problem is the ground truth or not in the 'quantized' feature. Default is true."
  },
  "mesh_size": {
    "default": "1e-3",
    "description": "the size of the mesh in the 'quantized' feature. Default is 1e-3."
  },
  "mesh_type": {
    "choices": [
      "absolute",
      "relative"
    ],
    "default": "'absolute'",
    "description": "the type of the mesh in the 'quantized' feature. It should be either 'absolute' or 'relative'. Default is 'absolute'."
  },
  "mod_affine": {
    "description": "the modifier function to generate the affine transformation applied to the variables in the 'custom' feature. It should be a function handle (random_stream, problem) -> (A, b, inv), where problem is an instance of the class Problem, A is the matrix of the affine transformation, b is the vector of the affine transformation, and inv is the inverse of matrix A. No default."
  },
  "mod_bounds": {
    "description": "the modifier function to modify the bound constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_xl, modified_xu), where problem is an instance of the class Problem, modified_xl is the modified lower bound, and modified_xu is the modified upper bound. No default."
  },
  "mod_ceq": {
    "description": "the modifier function to modify the nonlinear equality constraints in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_ceq, where x is the evaluation point, problem is an instance of the class Problem, and modified_ceq is the modified vector of the nonlinear equality constraints. No default."
  },
  "mod_cub": {
    "description": "the modifier function to modify the nonlinear inequality constraints in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_cub, where x is the evaluation point, problem is an instance of the class Problem, and modified_cub is the modified vector of the nonlinear inequality constraints. No default."
  },
  "mod_fun": {
    "description": "the modifier function to modify the objective function in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_fun, where x is the evaluation point, problem is an instance of the class Problem, and modified_fun is the modified objective function value. No default."
  },
  "mod_linear_eq": {
    "description": "the modifier function to modify the linear equality constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_aeq, modified_beq), where problem is an instance of the class Problem, modified_aeq is the modified matrix of the linear equality constraints, and modified_beq is the modified vector of the linear equality constraints. No default."
  },
  "mod_linear_ub": {
    "description": "the modifier function to modify the linear inequality constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_aub, modified_bub), where problem is an instance of the class Problem, modified_aub is the modified matrix of the linear inequality constraints, and modified_bub is the modified vector of the linear inequality constraints. No default."
  },
  "mod_x0": {
    "description": "the modifier function to modify the inital guess in the 'custom' feature. It should be a function handle (random_stream, problem) -> modified_x0, where problem is an instance of the class Problem, and modified_x0 is the modified initial guess. No default."
  },
  "n_runs": {
    "default": "5 for stochastic features and 1 for deterministic features",
    "description": "the number of runs of the experiments with the given feature. Default is 5 for stochastic features and 1 for deterministic features."
  },
  "nan_rate": {
    "default": "0.05",
    "description": "the probability that the evaluation of the objective function will return NaN in the 'random_nan' feature. Default is 0.05."
  },
  "noise_level": {
    "default": "1e-3",
    "description": "the magnitude of the noise in the 'noisy' feature. Default is 1e-3."
  },
  "noise_map": {
    "default": "'chebyshev'",
    "description": "the deterministic scalar noise map in the 'noisy' feature. It should be either 'chebyshev' or a function handle x -> noise that accepts the evaluation point and returns a real scalar. It is used only when noise_mode is 'deterministic'. Default is 'chebyshev'. The built-in 'chebyshev' map follows the deterministic noise model in Moré and Wild [5]."
  },
  "noise_mode": {
    "choices": [
      "random",
      "deterministic"
    ],
    "default": "'random'",
    "description": "the mode of the noise in the 'noisy' feature. It should be either 'random' or 'deterministic'. Default is 'random'. When it is 'deterministic' and n_runs is not specified, n_runs defaults to 1."
  },
  "noise_type": {
    "choices": [
      "absolute",
      "relative",
      "mixed"
    ],
    "default": "'mixed'",
    "description": "the type of the noise in the 'noisy' features. It should be either 'absolute', 'relative', or 'mixed'. Default is 'mixed'."
  },
  "perturbation_level": {
    "default": "1e-3",
    "description": "the magnitude of the perturbation to the initial guess in the 'perturbed_x0' feature. Default is 1e-3."
  },
  "perturbed_trailing_digits": {
    "default": "false",
    "description": "whether we will randomize the trailing digits of the objective function value in the 'truncated' feature. Default is false."
  },
  "rotated": {
    "default": "true",
    "description": "whether to use a random or given rotation matrix to rotate the coordinates of a problem in the 'linearly_transformed' feature. Default is true."
  },
  "significant_digits": {
    "default": "6",
    "description": "the number of significant digits in the 'truncated' feature. Default is 6."
  },
  "unrelaxable_bounds": {
    "default": "true",
    "description": "whether the bound constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is true."
  },
  "unrelaxable_linear_constraints": {
    "default": "false",
    "description": "whether the linear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is false."
  },
  "unrelaxable_nonlinear_constraints": {
    "default": "false",
    "description": "whether the nonlinear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is false."
  }
}
```

## feature_options.feature_name

```json
{
  "choices": [
    "plain",
    "perturbed_x0",
    "noisy",
    "truncated",
    "permuted",
    "linearly_transformed",
    "random_nan",
    "unrelaxable_constraints",
    "nonquantifiable_constraints",
    "quantized",
    "custom"
  ],
  "default": "'plain'",
  "description": "the name of the feature. The available features are 'plain', 'perturbed_x0', 'noisy', 'truncated', 'permuted', 'linearly_transformed', 'random_nan', 'unrelaxable_constraints', 'nonquantifiable_constraints', 'quantized', and 'custom'. Default is 'plain'."
}
```

## feature_options.n_runs

```json
{
  "default": "5 for stochastic features and 1 for deterministic features",
  "description": "the number of runs of the experiments with the given feature. Default is 5 for stochastic features and 1 for deterministic features."
}
```

## feature_options.distribution

```json
{
  "default": "'spherical'",
  "description": "the distribution of perturbation in 'perturbed_x0' feature or random noise in 'noisy' feature. It should be either a string (or char), or a function handle (random_stream, dimension) -> random vector that accepts a random_stream and the dimension of a problem and returning a random vector with the given dimension. In 'perturbed_x0' case, the char should be either 'spherical' or 'gaussian' (default is 'spherical'). In 'noisy' case, the char should be either 'gaussian' or 'uniform' (default is 'gaussian'), and the function handle should accept a random stream and output size."
}
```

## feature_options.perturbation_level

```json
{
  "default": "1e-3",
  "description": "the magnitude of the perturbation to the initial guess in the 'perturbed_x0' feature. Default is 1e-3."
}
```

## feature_options.noise_level

```json
{
  "default": "1e-3",
  "description": "the magnitude of the noise in the 'noisy' feature. Default is 1e-3."
}
```

## feature_options.noise_type

```json
{
  "choices": [
    "absolute",
    "relative",
    "mixed"
  ],
  "default": "'mixed'",
  "description": "the type of the noise in the 'noisy' features. It should be either 'absolute', 'relative', or 'mixed'. Default is 'mixed'."
}
```

## feature_options.noise_mode

```json
{
  "choices": [
    "random",
    "deterministic"
  ],
  "default": "'random'",
  "description": "the mode of the noise in the 'noisy' feature. It should be either 'random' or 'deterministic'. Default is 'random'. When it is 'deterministic' and n_runs is not specified, n_runs defaults to 1."
}
```

## feature_options.noise_map

```json
{
  "default": "'chebyshev'",
  "description": "the deterministic scalar noise map in the 'noisy' feature. It should be either 'chebyshev' or a function handle x -> noise that accepts the evaluation point and returns a real scalar. It is used only when noise_mode is 'deterministic'. Default is 'chebyshev'. The built-in 'chebyshev' map follows the deterministic noise model in Moré and Wild [5]."
}
```

## feature_options.significant_digits

```json
{
  "default": "6",
  "description": "the number of significant digits in the 'truncated' feature. Default is 6."
}
```

## feature_options.perturbed_trailing_digits

```json
{
  "default": "false",
  "description": "whether we will randomize the trailing digits of the objective function value in the 'truncated' feature. Default is false."
}
```

## feature_options.rotated

```json
{
  "default": "true",
  "description": "whether to use a random or given rotation matrix to rotate the coordinates of a problem in the 'linearly_transformed' feature. Default is true."
}
```

## feature_options.condition_factor

```json
{
  "default": "0",
  "description": "the scaling factor of the condition number of the linear transformation in the 'linearly_transformed' feature. More specifically, the condition number of the linear transformation will be 2 ^ (condition_factor * n / 2), where n is the dimension of the problem. Default is 0."
}
```

## feature_options.nan_rate

```json
{
  "default": "0.05",
  "description": "the probability that the evaluation of the objective function will return NaN in the 'random_nan' feature. Default is 0.05."
}
```

## feature_options.unrelaxable_bounds

```json
{
  "default": "true",
  "description": "whether the bound constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is true."
}
```

## feature_options.unrelaxable_linear_constraints

```json
{
  "default": "false",
  "description": "whether the linear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is false."
}
```

## feature_options.unrelaxable_nonlinear_constraints

```json
{
  "default": "false",
  "description": "whether the nonlinear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is false."
}
```

## feature_options.mesh_size

```json
{
  "default": "1e-3",
  "description": "the size of the mesh in the 'quantized' feature. Default is 1e-3."
}
```

## feature_options.mesh_type

```json
{
  "choices": [
    "absolute",
    "relative"
  ],
  "default": "'absolute'",
  "description": "the type of the mesh in the 'quantized' feature. It should be either 'absolute' or 'relative'. Default is 'absolute'."
}
```

## feature_options.ground_truth

```json
{
  "default": "true",
  "description": "whether the featured problem is the ground truth or not in the 'quantized' feature. Default is true."
}
```

## feature_options.mod_x0

```json
{
  "description": "the modifier function to modify the inital guess in the 'custom' feature. It should be a function handle (random_stream, problem) -> modified_x0, where problem is an instance of the class Problem, and modified_x0 is the modified initial guess. No default."
}
```

## feature_options.mod_affine

```json
{
  "description": "the modifier function to generate the affine transformation applied to the variables in the 'custom' feature. It should be a function handle (random_stream, problem) -> (A, b, inv), where problem is an instance of the class Problem, A is the matrix of the affine transformation, b is the vector of the affine transformation, and inv is the inverse of matrix A. No default."
}
```

## feature_options.mod_bounds

```json
{
  "description": "the modifier function to modify the bound constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_xl, modified_xu), where problem is an instance of the class Problem, modified_xl is the modified lower bound, and modified_xu is the modified upper bound. No default."
}
```

## feature_options.mod_linear_ub

```json
{
  "description": "the modifier function to modify the linear inequality constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_aub, modified_bub), where problem is an instance of the class Problem, modified_aub is the modified matrix of the linear inequality constraints, and modified_bub is the modified vector of the linear inequality constraints. No default."
}
```

## feature_options.mod_linear_eq

```json
{
  "description": "the modifier function to modify the linear equality constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_aeq, modified_beq), where problem is an instance of the class Problem, modified_aeq is the modified matrix of the linear equality constraints, and modified_beq is the modified vector of the linear equality constraints. No default."
}
```

## feature_options.mod_fun

```json
{
  "description": "the modifier function to modify the objective function in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_fun, where x is the evaluation point, problem is an instance of the class Problem, and modified_fun is the modified objective function value. No default."
}
```

## feature_options.mod_cub

```json
{
  "description": "the modifier function to modify the nonlinear inequality constraints in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_cub, where x is the evaluation point, problem is an instance of the class Problem, and modified_cub is the modified vector of the nonlinear inequality constraints. No default."
}
```

## feature_options.mod_ceq

```json
{
  "description": "the modifier function to modify the nonlinear equality constraints in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_ceq, where x is the evaluation point, problem is an instance of the class Problem, and modified_ceq is the modified vector of the nonlinear equality constraints. No default."
}
```

## problem_options

```json
{
  "excludelist": {
    "default": "not to exclude any problem",
    "description": "the list of problems to be excluded. Default is not to exclude any problem."
  },
  "maxb": {
    "default": "minb + 10",
    "description": "the maximum number of bound constraints of the problems to be selected. Default is minb + 10."
  },
  "maxcon": {
    "default": "max(maxlcon, maxnlcon)",
    "description": "the maximum number of linear and nonlinear constraints of the problems to be selected. Default is max(maxlcon, maxnlcon)."
  },
  "maxdim": {
    "default": "mindim + 10",
    "description": "the maximum dimension of the problems to be selected. Default is mindim + 10."
  },
  "maxlcon": {
    "default": "minlcon + 10",
    "description": "the maximum number of linear constraints of the problems to be selected. Default is minlcon + 10."
  },
  "maxnlcon": {
    "default": "minnlcon + 10",
    "description": "the maximum number of nonlinear constraints of the problems to be selected. Default is minnlcon + 10."
  },
  "minb": {
    "default": "0",
    "description": "the minimum number of bound constraints of the problems to be selected. Default is 0."
  },
  "mincon": {
    "default": "min(minlcon, minnlcon)",
    "description": "the minimum number of linear and nonlinear constraints of the problems to be selected. Default is min(minlcon, minnlcon)."
  },
  "mindim": {
    "default": "1",
    "description": "the minimum dimension of the problems to be selected. Default is 1."
  },
  "minlcon": {
    "default": "0",
    "description": "the minimum number of linear constraints of the problems to be selected. Default is 0."
  },
  "minnlcon": {
    "default": "0",
    "description": "the minimum number of nonlinear constraints of the problems to be selected. Default is 0."
  },
  "plibs": {
    "default": "'s2mpj'",
    "description": "the problem libraries to be used. It should be a cell array of strings or chars. The built-in choices are s2mpj, matcutest, and custom. Default setting is 's2mpj'."
  },
  "problem": {
    "default": "not to set any problem",
    "description": "a problem to be benchmarked. It should be an instance of the class Problem. If it is provided, we will only run the test on this problem with the given feature and draw the history plots. Default is not to set any problem."
  },
  "problem_names": {
    "default": "not to select any problem by name but by the options above",
    "description": "the names of the problems to be selected. It should be a cell array of strings or chars. Default is not to select any problem by name but by the options above."
  },
  "ptype": {
    "choices": [
      "u",
      "b",
      "l",
      "n",
      "b",
      "ul",
      "ubn"
    ],
    "default": "'u'",
    "description": "the type of the problems to be selected. It should be a string or char consisting of any combination of 'u' (unconstrained), 'b' (bound constrained), 'l' (linearly constrained), and 'n' (nonlinearly constrained), such as 'b', 'ul', 'ubn'. Default is 'u'."
  }
}
```

## problem_options.plibs

```json
{
  "default": "'s2mpj'",
  "description": "the problem libraries to be used. It should be a cell array of strings or chars. The built-in choices are s2mpj, matcutest, and custom. Default setting is 's2mpj'."
}
```

## problem_options.ptype

```json
{
  "choices": [
    "u",
    "b",
    "l",
    "n",
    "b",
    "ul",
    "ubn"
  ],
  "default": "'u'",
  "description": "the type of the problems to be selected. It should be a string or char consisting of any combination of 'u' (unconstrained), 'b' (bound constrained), 'l' (linearly constrained), and 'n' (nonlinearly constrained), such as 'b', 'ul', 'ubn'. Default is 'u'."
}
```

## problem_options.mindim

```json
{
  "default": "1",
  "description": "the minimum dimension of the problems to be selected. Default is 1."
}
```

## problem_options.maxdim

```json
{
  "default": "mindim + 10",
  "description": "the maximum dimension of the problems to be selected. Default is mindim + 10."
}
```

## problem_options.minb

```json
{
  "default": "0",
  "description": "the minimum number of bound constraints of the problems to be selected. Default is 0."
}
```

## problem_options.maxb

```json
{
  "default": "minb + 10",
  "description": "the maximum number of bound constraints of the problems to be selected. Default is minb + 10."
}
```

## problem_options.minlcon

```json
{
  "default": "0",
  "description": "the minimum number of linear constraints of the problems to be selected. Default is 0."
}
```

## problem_options.maxlcon

```json
{
  "default": "minlcon + 10",
  "description": "the maximum number of linear constraints of the problems to be selected. Default is minlcon + 10."
}
```

## problem_options.minnlcon

```json
{
  "default": "0",
  "description": "the minimum number of nonlinear constraints of the problems to be selected. Default is 0."
}
```

## problem_options.maxnlcon

```json
{
  "default": "minnlcon + 10",
  "description": "the maximum number of nonlinear constraints of the problems to be selected. Default is minnlcon + 10."
}
```

## problem_options.mincon

```json
{
  "default": "min(minlcon, minnlcon)",
  "description": "the minimum number of linear and nonlinear constraints of the problems to be selected. Default is min(minlcon, minnlcon)."
}
```

## problem_options.maxcon

```json
{
  "default": "max(maxlcon, maxnlcon)",
  "description": "the maximum number of linear and nonlinear constraints of the problems to be selected. Default is max(maxlcon, maxnlcon)."
}
```

## problem_options.excludelist

```json
{
  "default": "not to exclude any problem",
  "description": "the list of problems to be excluded. Default is not to exclude any problem."
}
```

## problem_options.problem_names

```json
{
  "default": "not to select any problem by name but by the options above",
  "description": "the names of the problems to be selected. It should be a cell array of strings or chars. Default is not to select any problem by name but by the options above."
}
```

## problem_options.problem

```json
{
  "default": "not to set any problem",
  "description": "a problem to be benchmarked. It should be an instance of the class Problem. If it is provided, we will only run the test on this problem with the given feature and draw the history plots. Default is not to set any problem."
}
```

## returns

```json
{
  "curves": {
    "description": "Curves of all the profiles.",
    "type": "cell array"
  },
  "profile_scores": {
    "description": "Scores for all profiles (solver × tolerance × hist/output × profile_type).",
    "type": "4D tensor"
  },
  "solver_scores": {
    "description": "Scores of the solvers based on the profiles.",
    "type": "vector"
  }
}
```

## returns.solver_scores

```json
{
  "description": "Scores of the solvers based on the profiles.",
  "type": "vector"
}
```

## returns.profile_scores

```json
{
  "description": "Scores for all profiles (solver × tolerance × hist/output × profile_type).",
  "type": "4D tensor"
}
```

## returns.curves

```json
{
  "description": "Curves of all the profiles.",
  "type": "cell array"
}
```

## Canonical JSON Mirror

```json
{
  "calling_convention": {
    "options": "struct with fields (NOT name-value pairs). Example: options.ptype = 'u'; options.mindim = 2; benchmark(solvers, options);",
    "solvers": "cell array of function handles: {@solver1, @solver2}",
    "syntax": "[solver_scores, profile_scores, curves] = benchmark(solvers, options)"
  },
  "description": "Benchmark optimization solvers on a set of problems with specified features.",
  "feature_options": {
    "condition_factor": {
      "default": "0",
      "description": "the scaling factor of the condition number of the linear transformation in the 'linearly_transformed' feature. More specifically, the condition number of the linear transformation will be 2 ^ (condition_factor * n / 2), where n is the dimension of the problem. Default is 0."
    },
    "distribution": {
      "default": "'spherical'",
      "description": "the distribution of perturbation in 'perturbed_x0' feature or random noise in 'noisy' feature. It should be either a string (or char), or a function handle (random_stream, dimension) -> random vector that accepts a random_stream and the dimension of a problem and returning a random vector with the given dimension. In 'perturbed_x0' case, the char should be either 'spherical' or 'gaussian' (default is 'spherical'). In 'noisy' case, the char should be either 'gaussian' or 'uniform' (default is 'gaussian'), and the function handle should accept a random stream and output size."
    },
    "feature_name": {
      "choices": [
        "plain",
        "perturbed_x0",
        "noisy",
        "truncated",
        "permuted",
        "linearly_transformed",
        "random_nan",
        "unrelaxable_constraints",
        "nonquantifiable_constraints",
        "quantized",
        "custom"
      ],
      "default": "'plain'",
      "description": "the name of the feature. The available features are 'plain', 'perturbed_x0', 'noisy', 'truncated', 'permuted', 'linearly_transformed', 'random_nan', 'unrelaxable_constraints', 'nonquantifiable_constraints', 'quantized', and 'custom'. Default is 'plain'."
    },
    "ground_truth": {
      "default": "true",
      "description": "whether the featured problem is the ground truth or not in the 'quantized' feature. Default is true."
    },
    "mesh_size": {
      "default": "1e-3",
      "description": "the size of the mesh in the 'quantized' feature. Default is 1e-3."
    },
    "mesh_type": {
      "choices": [
        "absolute",
        "relative"
      ],
      "default": "'absolute'",
      "description": "the type of the mesh in the 'quantized' feature. It should be either 'absolute' or 'relative'. Default is 'absolute'."
    },
    "mod_affine": {
      "description": "the modifier function to generate the affine transformation applied to the variables in the 'custom' feature. It should be a function handle (random_stream, problem) -> (A, b, inv), where problem is an instance of the class Problem, A is the matrix of the affine transformation, b is the vector of the affine transformation, and inv is the inverse of matrix A. No default."
    },
    "mod_bounds": {
      "description": "the modifier function to modify the bound constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_xl, modified_xu), where problem is an instance of the class Problem, modified_xl is the modified lower bound, and modified_xu is the modified upper bound. No default."
    },
    "mod_ceq": {
      "description": "the modifier function to modify the nonlinear equality constraints in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_ceq, where x is the evaluation point, problem is an instance of the class Problem, and modified_ceq is the modified vector of the nonlinear equality constraints. No default."
    },
    "mod_cub": {
      "description": "the modifier function to modify the nonlinear inequality constraints in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_cub, where x is the evaluation point, problem is an instance of the class Problem, and modified_cub is the modified vector of the nonlinear inequality constraints. No default."
    },
    "mod_fun": {
      "description": "the modifier function to modify the objective function in the 'custom' feature. It should be a function handle (x, random_stream, problem) -> modified_fun, where x is the evaluation point, problem is an instance of the class Problem, and modified_fun is the modified objective function value. No default."
    },
    "mod_linear_eq": {
      "description": "the modifier function to modify the linear equality constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_aeq, modified_beq), where problem is an instance of the class Problem, modified_aeq is the modified matrix of the linear equality constraints, and modified_beq is the modified vector of the linear equality constraints. No default."
    },
    "mod_linear_ub": {
      "description": "the modifier function to modify the linear inequality constraints in the 'custom' feature. It should be a function handle (random_stream, problem) -> (modified_aub, modified_bub), where problem is an instance of the class Problem, modified_aub is the modified matrix of the linear inequality constraints, and modified_bub is the modified vector of the linear inequality constraints. No default."
    },
    "mod_x0": {
      "description": "the modifier function to modify the inital guess in the 'custom' feature. It should be a function handle (random_stream, problem) -> modified_x0, where problem is an instance of the class Problem, and modified_x0 is the modified initial guess. No default."
    },
    "n_runs": {
      "default": "5 for stochastic features and 1 for deterministic features",
      "description": "the number of runs of the experiments with the given feature. Default is 5 for stochastic features and 1 for deterministic features."
    },
    "nan_rate": {
      "default": "0.05",
      "description": "the probability that the evaluation of the objective function will return NaN in the 'random_nan' feature. Default is 0.05."
    },
    "noise_level": {
      "default": "1e-3",
      "description": "the magnitude of the noise in the 'noisy' feature. Default is 1e-3."
    },
    "noise_map": {
      "default": "'chebyshev'",
      "description": "the deterministic scalar noise map in the 'noisy' feature. It should be either 'chebyshev' or a function handle x -> noise that accepts the evaluation point and returns a real scalar. It is used only when noise_mode is 'deterministic'. Default is 'chebyshev'. The built-in 'chebyshev' map follows the deterministic noise model in Moré and Wild [5]."
    },
    "noise_mode": {
      "choices": [
        "random",
        "deterministic"
      ],
      "default": "'random'",
      "description": "the mode of the noise in the 'noisy' feature. It should be either 'random' or 'deterministic'. Default is 'random'. When it is 'deterministic' and n_runs is not specified, n_runs defaults to 1."
    },
    "noise_type": {
      "choices": [
        "absolute",
        "relative",
        "mixed"
      ],
      "default": "'mixed'",
      "description": "the type of the noise in the 'noisy' features. It should be either 'absolute', 'relative', or 'mixed'. Default is 'mixed'."
    },
    "perturbation_level": {
      "default": "1e-3",
      "description": "the magnitude of the perturbation to the initial guess in the 'perturbed_x0' feature. Default is 1e-3."
    },
    "perturbed_trailing_digits": {
      "default": "false",
      "description": "whether we will randomize the trailing digits of the objective function value in the 'truncated' feature. Default is false."
    },
    "rotated": {
      "default": "true",
      "description": "whether to use a random or given rotation matrix to rotate the coordinates of a problem in the 'linearly_transformed' feature. Default is true."
    },
    "significant_digits": {
      "default": "6",
      "description": "the number of significant digits in the 'truncated' feature. Default is 6."
    },
    "unrelaxable_bounds": {
      "default": "true",
      "description": "whether the bound constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is true."
    },
    "unrelaxable_linear_constraints": {
      "default": "false",
      "description": "whether the linear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is false."
    },
    "unrelaxable_nonlinear_constraints": {
      "default": "false",
      "description": "whether the nonlinear constraints are unrelaxable or not in the 'unrelaxable_constraints' feature. Default is false."
    }
  },
  "name": "benchmark",
  "output_artifacts": {
    "detailed_profiles": "detailed_profiles/ contains high-quality single profile PDFs.",
    "history_plots": "history_plots/ contains per-problem history plots when draw_hist_plots is not 'none'.",
    "summary_pdf": "summary_<stamp>.pdf contains the merged summary profiles for the run.",
    "test_log": "test_log/ stores log files, report.txt, option snapshots, curves, and profile scores.",
    "test_log_report": "test_log/report.txt records selected problems, timing, merit_init = phi(x_0) = Inf cases, abnormal solver terminations, output fallbacks, and solver scores."
  },
  "problem_options": {
    "excludelist": {
      "default": "not to exclude any problem",
      "description": "the list of problems to be excluded. Default is not to exclude any problem."
    },
    "maxb": {
      "default": "minb + 10",
      "description": "the maximum number of bound constraints of the problems to be selected. Default is minb + 10."
    },
    "maxcon": {
      "default": "max(maxlcon, maxnlcon)",
      "description": "the maximum number of linear and nonlinear constraints of the problems to be selected. Default is max(maxlcon, maxnlcon)."
    },
    "maxdim": {
      "default": "mindim + 10",
      "description": "the maximum dimension of the problems to be selected. Default is mindim + 10."
    },
    "maxlcon": {
      "default": "minlcon + 10",
      "description": "the maximum number of linear constraints of the problems to be selected. Default is minlcon + 10."
    },
    "maxnlcon": {
      "default": "minnlcon + 10",
      "description": "the maximum number of nonlinear constraints of the problems to be selected. Default is minnlcon + 10."
    },
    "minb": {
      "default": "0",
      "description": "the minimum number of bound constraints of the problems to be selected. Default is 0."
    },
    "mincon": {
      "default": "min(minlcon, minnlcon)",
      "description": "the minimum number of linear and nonlinear constraints of the problems to be selected. Default is min(minlcon, minnlcon)."
    },
    "mindim": {
      "default": "1",
      "description": "the minimum dimension of the problems to be selected. Default is 1."
    },
    "minlcon": {
      "default": "0",
      "description": "the minimum number of linear constraints of the problems to be selected. Default is 0."
    },
    "minnlcon": {
      "default": "0",
      "description": "the minimum number of nonlinear constraints of the problems to be selected. Default is 0."
    },
    "plibs": {
      "default": "'s2mpj'",
      "description": "the problem libraries to be used. It should be a cell array of strings or chars. The built-in choices are s2mpj, matcutest, and custom. Default setting is 's2mpj'."
    },
    "problem": {
      "default": "not to set any problem",
      "description": "a problem to be benchmarked. It should be an instance of the class Problem. If it is provided, we will only run the test on this problem with the given feature and draw the history plots. Default is not to set any problem."
    },
    "problem_names": {
      "default": "not to select any problem by name but by the options above",
      "description": "the names of the problems to be selected. It should be a cell array of strings or chars. Default is not to select any problem by name but by the options above."
    },
    "ptype": {
      "choices": [
        "u",
        "b",
        "l",
        "n",
        "b",
        "ul",
        "ubn"
      ],
      "default": "'u'",
      "description": "the type of the problems to be selected. It should be a string or char consisting of any combination of 'u' (unconstrained), 'b' (bound constrained), 'l' (linearly constrained), and 'n' (nonlinearly constrained), such as 'b', 'ul', 'ubn'. Default is 'u'."
    }
  },
  "profile_options": {
    "bar_colors": {
      "choices": [
        "r",
        "g",
        "b",
        "c",
        "m",
        "y",
        "k"
      ],
      "default": "set to the first two colors in the line_colors option",
      "description": "two different colors for the bars of two solvers in the log-ratio profiles. It can be a cell array of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a 2-by-3 matrix with each row being a RGB triplet. Default is set to the first two colors in the line_colors option."
    },
    "benchmark_id": {
      "default": "'out' if the option load is not provided, otherwise default is '.'",
      "description": "the identifier of the test. It is used to create the specific directory to store the results. Default is 'out' if the option load is not provided, otherwise default is '.'."
    },
    "draw_hist_plots": {
      "choices": [
        "none",
        "sequential",
        "parallel"
      ],
      "default": "'parallel'",
      "description": "whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', we will not draw the history plots. If it is 'parallel', we will draw the history plots in the same time when solvers are solving the problems. If it is 'sequential', we will draw the history plots after all the problems are solved. Default is 'sequential'.",
      "source_note": "MATLAB getDefaultProfileOptions.m sets draw_hist_plots to 'parallel' in normal runs; load mode forces it to 'sequential'."
    },
    "errorbar_type": {
      "choices": [
        "minmax",
        "meanstd"
      ],
      "default": "'minmax', meaning that we takes the pointwise minimum and maximum of the curves",
      "description": "the type of the uncertainty interval that can be either 'minmax' or 'meanstd'. When n_runs is greater than 1, we run several times of the experiments and get average curves and get average curves and uncertainty intervals. Default is 'minmax', meaning that we takes the pointwise minimum and maximum of the curves."
    },
    "feature_stamp": {
      "description": "the stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features."
    },
    "hist_aggregation": {
      "choices": [
        "min",
        "mean",
        "max"
      ],
      "default": "'min'",
      "description": "the aggregation method we use to reduce the number of points in the history plots. It can be 'min', 'mean', or 'max'. Default is 'min'."
    },
    "line_colors": {
      "choices": [
        "r",
        "g",
        "b",
        "c",
        "m",
        "y",
        "k"
      ],
      "description": "the colors of the lines in the plots. It can be a cell array of short names of colors ('r', 'g', 'b', 'c', 'm', 'y', 'k') or a matrix with each row being a RGB triplet. Default line colors are those in the palettename named “gem” (see MATLAB documentation for ‘colororder’). Note that if the number of solvers is greater than the number of colors, we will cycle through the colors."
    },
    "line_styles": {
      "choices": [
        "-",
        "-.",
        "--",
        ":",
        "o",
        "+",
        "*",
        ".",
        "x",
        "s",
        "d",
        "^",
        "v",
        ">",
        "<",
        "p",
        "h"
      ],
      "description": "the styles of the lines in the plots. It can be a cell array of chars that are the combinations of line styles ('-', '-.', '--', ':') and markers ('o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'). Default line style order is {'-', '-.', '--', ':'}. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles."
    },
    "line_widths": {
      "default": "1.5",
      "description": "the widths of the lines in the plots. It should be a positive scalar or a vector. Default is 1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths."
    },
    "load": {
      "description": "loading the stored data from a completed experiment and draw profiles. It can be either 'latest' or a time stamp of an experiment in the format of ‘yyyyMMdd_HHmmss’. No default."
    },
    "max_eval_factor": {
      "default": "500",
      "description": "the factor multiplied to each problem’s dimension to get the maximum number of evaluations for each problem. Default is 500."
    },
    "max_tol_order": {
      "default": "10",
      "description": "the maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of ‘tolerances’ to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as 10^(-1:-1:-max_tol_order). Default is 10."
    },
    "merit_fun": {
      "description": "the merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a function handle (fun_value, maxcv_value, maxcv_init) -> merit_value, where fun_value is the objective function value, maxcv_value is the maximum constraint violation, and maxcv_init is the maximum constraint violation at the initial guess. The size of fun_values and maxcv_values is the same, and the size of maxcv_init is the same as the second to last dimensions of fun_values. The default merit function varphi(x) is defined by the objective function f(x) and the maximum constraint violation v(x) as \\[\\begin{split}\\varphi(x) = \\begin{cases} f(x), & \\text{if } v(x) \\le v_1, \\\\ f(x) + 10^5 \\cdot (v(x) - v_1), & \\text{if } v_1 < v(x) \\le v_2, \\\\ +\\infty, & \\text{if } v(x) > v_2, \\end{cases}\\end{split}\\] where \\(v_1 = \\min(0.01,\\; 10^{-10} \\max(1, v_0))\\), \\(v_2 = \\max(0.1,\\; 2v_0)\\), and \\(v_0\\) is the initial maximum constraint violation. If \\(\\varphi(x_0) = +\\infty\\) for a problem/run, the convergence test is degenerate; by convention, all solvers are declared to pass that problem/run. These cases are listed in test_log/report.txt."
    },
    "n_jobs": {
      "default": "a conservative number of workers, chosen as about half of the available workers (at least 2 when more than one worker is available)",
      "description": "the number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers (at least 2 when more than one worker is available)."
    },
    "normalized_scores": {
      "default": "true",
      "description": "whether to normalize the scores of the solvers by the maximum score of the solvers. Default is true."
    },
    "project_x0": {
      "default": "false",
      "description": "whether to project the initial point to the feasible set. Default is false."
    },
    "run_plain": {
      "default": "false",
      "description": "whether to run an extra experiment with the 'plain' feature. Default is false."
    },
    "savepath": {
      "default": "'pwd', the current working directory",
      "description": "the path to store the results. Default is 'pwd', the current working directory."
    },
    "score_fun": {
      "description": "the scoring function to calculate the scores of the solvers. It should be a function handle profile_scores -> solver_scores, where profile_scores is a 4D tensor containing scores for all profiles. The first dimension of profile_scores corresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring function takes the average of the history-based performance profiles under all the tolerances."
    },
    "score_only": {
      "default": "false",
      "description": "whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default is false."
    },
    "score_weight_fun": {
      "default": "1",
      "description": "the weight function to calculate the scores of the solvers in the performance and data profiles. It should be a function handle representing a nonnegative function in R^+. Default is 1."
    },
    "seed": {
      "default": "0",
      "description": "the seed of the random number generator. Default is 0."
    },
    "semilogx": {
      "default": "true",
      "description": "whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default is true."
    },
    "silent": {
      "default": "false",
      "description": "whether to show the information of the progress. Default is false."
    },
    "solver_isrand": {
      "default": "all false",
      "description": "whether the solvers are randomized or not. It is a logical array of the same length as the number of solvers, where the value is true if the solver is randomized, and false otherwise. Default is all false. Note that if n_runs is not specified, we will set it 5 for the randomized solvers."
    },
    "solver_names": {
      "default": "the names of the function handles in solvers",
      "description": "the names of the solvers. Default is the names of the function handles in solvers."
    },
    "solver_verbose": {
      "default": "1",
      "description": "the level of the verbosity of the solvers. 0 means no verbosity, 1 means some verbosity, and 2 means full verbosity. Default is 1."
    },
    "solvers_to_load": {
      "default": "all the solvers",
      "description": "the indices of the solvers to load when the load option is provided. It can be a vector of different integers selected from 1 to the total number of solvers of the loading experiment. At least two indices should be provided. Default is all the solvers."
    },
    "summarize_data_profiles": {
      "default": "true",
      "description": "whether to add all the data profiles to the summary PDF. Default is true."
    },
    "summarize_log_ratio_profiles": {
      "default": "false",
      "description": "whether to add all the log-ratio profiles to the summary PDF. Default is false."
    },
    "summarize_output_based_profiles": {
      "default": "true",
      "description": "whether to add all the output-based profiles of the selected profiles to the summary PDF. Default is true."
    },
    "summarize_performance_profiles": {
      "default": "true",
      "description": "whether to add all the performance profiles to the summary PDF. Default is true."
    },
    "xlabel_data_profile": {
      "default": "'Number of simplex gradients'",
      "description": "the label of the x-axis of the data profiles. Default is 'Number of simplex gradients'. Note: the 'Interpreter' property is set to 'latex', so LaTeX formatting is supported. The same applies to the options xlabel_log_ratio_profile, xlabel_performance_profile, ylabel_data_profile, ylabel_log_ratio_profile, and ylabel_performance_profile."
    },
    "xlabel_log_ratio_profile": {
      "default": "'Problem'",
      "description": "the label of the x-axis of the log-ratio profiles. Default is 'Problem'."
    },
    "xlabel_performance_profile": {
      "default": "'Performance ratio'",
      "description": "the label of the x-axis of the performance profiles. Default is 'Performance ratio'."
    },
    "ylabel_data_profile": {
      "default": "'Data profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
      "description": "the label of the y-axis of the data profiles. Default is 'Data profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format. You can also use %s in your custom label, and it will be replaced accordingly. The same applies to the options ylabel_log_ratio_profile and ylabel_performance_profile."
    },
    "ylabel_log_ratio_profile": {
      "default": "'Log-ratio profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
      "description": "the label of the y-axis of the log-ratio profiles. Default is 'Log-ratio profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format."
    },
    "ylabel_performance_profile": {
      "default": "'Performance profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format",
      "description": "the label of the y-axis of the performance profiles. Default is 'Performance profiles ($\\\\mathrm{tol} = %s$)', where %s will be replaced by the current tolerance in LaTeX format."
    }
  },
  "returns": {
    "curves": {
      "description": "Curves of all the profiles.",
      "type": "cell array"
    },
    "profile_scores": {
      "description": "Scores for all profiles (solver × tolerance × hist/output × profile_type).",
      "type": "4D tensor"
    },
    "solver_scores": {
      "description": "Scores of the solvers based on the profiles.",
      "type": "vector"
    }
  },
  "solver_notes": [
    "fun is a function handle: fun(x) -> scalar. Provides ONLY function values (DFO).",
    "x0 is a column vector.",
    "All constraint vectors are column vectors.",
    "Must return column vector x.",
    "At least 2 solvers required (cell array of function handles)."
  ],
  "solver_signatures": {
    "bound_constrained": "x = solver(fun, x0, xl, xu)",
    "linearly_constrained": "x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)",
    "nonlinearly_constrained": "x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)",
    "unconstrained": "x = solver(fun, x0)"
  }
}
```
