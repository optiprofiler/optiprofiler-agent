---
tags: [reference, source-backed, matlab, benchmark]
sources: [_sources/matlab/benchmark.json]
related: [../api/matlab/benchmark.md]
last_updated: 2026-06-07
generated: true
---

# Source Reference: Matlab benchmark.json

This page is auto-generated from `_sources/matlab/benchmark.json`. It is the lossless wiki mirror for this source.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.

## Source Metadata

- Source path: `_sources/matlab/benchmark.json`
- Canonical SHA256: `ceb5f28c2e3ccd497c8af4fc12e2778d8a8260cf0d4894f3ece9d47dffe60119`
- Top-level keys: `name`, `description`, `calling_convention`, `solver_signatures`, `solver_notes`, `profile_options`, `feature_options`, `problem_options`, `returns`, `output_artifacts`

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
| `profile_options` | dict[38] |
| `profile_options.bar_colors` | dict[3] |
| `profile_options.benchmark_id` | dict[1] |
| `profile_options.draw_hist_plots` | dict[3] |
| `profile_options.errorbar_type` | dict[2] |
| `profile_options.feature_stamp` | dict[1] |
| `profile_options.hist_aggregation` | dict[1] |
| `profile_options.line_colors` | dict[2] |
| `profile_options.line_styles` | dict[2] |
| `profile_options.line_widths` | dict[1] |
| `profile_options.load` | dict[1] |
| `profile_options.max_eval_factor` | dict[1] |
| `profile_options.max_tol_order` | dict[1] |
| `profile_options.merit_fun` | dict[1] |
| `profile_options.n_jobs` | dict[2] |
| `profile_options.normalized_scores` | dict[1] |
| `profile_options.project_x0` | dict[1] |
| `profile_options.run_plain` | dict[1] |
| `profile_options.savepath` | dict[1] |
| `profile_options.score_fun` | dict[1] |
| `profile_options.score_only` | dict[1] |
| `profile_options.score_weight_fun` | dict[1] |
| `profile_options.seed` | dict[1] |
| `profile_options.semilogx` | dict[1] |
| `profile_options.silent` | dict[1] |
| `profile_options.solver_isrand` | dict[2] |
| `profile_options.solver_names` | dict[2] |
| `profile_options.solver_verbose` | dict[1] |
| `profile_options.solvers_to_load` | dict[2] |
| `profile_options.summarize_data_profiles` | dict[1] |
| `profile_options.summarize_log_ratio_profiles` | dict[1] |
| `profile_options.summarize_output_based_profiles` | dict[1] |
| `profile_options.summarize_performance_profiles` | dict[1] |
| `profile_options.xlabel_data_profile` | dict[1] |
| `profile_options.xlabel_log_ratio_profile` | dict[1] |
| `profile_options.xlabel_performance_profile` | dict[1] |
| `profile_options.ylabel_data_profile` | dict[1] |
| `profile_options.ylabel_log_ratio_profile` | dict[1] |
| `profile_options.ylabel_performance_profile` | dict[1] |
| `feature_options` | dict[25] |
| `feature_options.feature_name` | dict[1] |
| `feature_options.n_runs` | dict[1] |
| `feature_options.distribution` | dict[1] |
| `feature_options.perturbation_level` | dict[1] |
| `feature_options.noise_level` | dict[1] |
| `feature_options.noise_type` | dict[2] |
| `feature_options.significant_digits` | dict[1] |
| `feature_options.perturbed_trailing_digits` | dict[1] |
| `feature_options.rotated` | dict[1] |
| `feature_options.condition_factor` | dict[1] |
| `feature_options.nan_rate` | dict[1] |
| `feature_options.unrelaxable_bounds` | dict[1] |
| `feature_options.unrelaxable_linear_constraints` | dict[1] |
| `feature_options.unrelaxable_nonlinear_constraints` | dict[1] |
| `feature_options.mesh_size` | dict[1] |
| `feature_options.mesh_type` | dict[2] |
| `feature_options.ground_truth` | dict[1] |
| `feature_options.mod_x0` | dict[1] |
| `feature_options.mod_affine` | dict[1] |
| `feature_options.mod_bounds` | dict[1] |
| `feature_options.mod_linear_ub` | dict[1] |
| `feature_options.mod_linear_eq` | dict[1] |
| `feature_options.mod_fun` | dict[1] |
| `feature_options.mod_cub` | dict[1] |
| `feature_options.mod_ceq` | dict[1] |
| `problem_options` | dict[15] |
| `problem_options.plibs` | dict[1] |
| `problem_options.ptype` | dict[2] |
| `problem_options.mindim` | dict[1] |
| `problem_options.maxdim` | dict[1] |
| `problem_options.minb` | dict[1] |
| `problem_options.maxb` | dict[1] |
| `problem_options.minlcon` | dict[1] |
| `problem_options.maxlcon` | dict[1] |
| `problem_options.minnlcon` | dict[1] |
| `problem_options.maxnlcon` | dict[1] |
| `problem_options.mincon` | dict[1] |
| `problem_options.maxcon` | dict[1] |
| `problem_options.excludelist` | dict[2] |
| `problem_options.problem_names` | dict[2] |
| `problem_options.problem` | dict[2] |
| `returns` | dict[3] |
| `returns.solver_scores` | dict[2] |
| `returns.profile_scores` | dict[2] |
| `returns.curves` | dict[2] |
| `output_artifacts` | dict[4] |

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
    "default": "set to the first two colors in theline_colorsoption",
    "description": "two different colors for the bars of two solvers in the log-ratio profiles. It can be a cell array of short names of colors('r','g','b','c','m','y','k')or a 2-by-3 matrix with each row being a RGB triplet. Default is set to the first two colors in theline_colorsoption."
  },
  "benchmark_id": {
    "description": "the identifier of the test. It is used to create the specific directory to store the results. Default is'out'if the optionloadis not provided, otherwise default is'.'."
  },
  "draw_hist_plots": {
    "choices": [
      "none",
      "sequential",
      "parallel"
    ],
    "default": "'parallel' in normal runs; 'sequential' in load mode",
    "description": "whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', no history plots are drawn. If it is 'parallel', history plots are drawn while solvers are solving problems. If it is 'sequential', history plots are drawn after all problems are solved. Default is 'parallel' in normal runs; when the load option is provided, OptiProfiler forces 'sequential'."
  },
  "errorbar_type": {
    "choices": [
      "minmax",
      "meanstd"
    ],
    "description": "the type of the uncertainty interval that can be either'minmax'or'meanstd'. Whenn_runsis greater than 1, we run several times of the experiments and get average curves and get average curves and uncertainty intervals. Default is'minmax', meaning that we takes the pointwise minimum and maximum of the curves."
  },
  "feature_stamp": {
    "description": "the stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features."
  },
  "hist_aggregation": {
    "description": "the aggregation method we use to reduce the number of points in the history plots. It can be'min','mean', or'max'. Default is'min'."
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
    "description": "the colors of the lines in the plots. It can be a cell array of short names of colors('r','g','b','c','m','y','k')or a matrix with each row being a RGB triplet. Default line colors are those in the palettename named “gem” (see MATLAB documentation for ‘colororder’). Note that if the number of solvers is greater than the number of colors, we will cycle through the colors."
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
    "description": "the styles of the lines in the plots. It can be a cell array of chars that are the combinations of line styles('-','-.','--',':')and markers('o','+','*','.','x','s','d','^','v','>','<','p','h'). Default line style order is{'-','-.','--',':'}. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles."
  },
  "line_widths": {
    "description": "the widths of the lines in the plots. It should be a positive scalar or a vector. Default is1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths."
  },
  "load": {
    "description": "loading the stored data from a completed experiment and draw profiles. It can be either'latest'or a time stamp of an experiment in the format of ‘yyyyMMdd_HHmmss’. No default."
  },
  "max_eval_factor": {
    "description": "the factor multiplied to each problem’s dimension to get the maximum number of evaluations for each problem. Default is500."
  },
  "max_tol_order": {
    "description": "the maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of ‘tolerances’ to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as10^(-1:-1:-max_tol_order). Default is10."
  },
  "merit_fun": {
    "description": "the merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a function handle(fun_value,maxcv_value,maxcv_init)->merit_value,wherefun_valueis the objective function value,maxcv_valueis the maximum constraint violation, andmaxcv_initis the maximum constraint violation at the initial guess. The size offun_valuesandmaxcv_valuesis the same, and the size ofmaxcv_initis the same as the second to last dimensions offun_values"
  },
  "n_jobs": {
    "default": "about half of available workers, at least 2 when possible",
    "description": "the number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers, with at least 2 when more than one worker is available."
  },
  "normalized_scores": {
    "description": "whether to normalize the scores of the solvers by the maximum score of the solvers. Default istrue."
  },
  "project_x0": {
    "description": "whether to project the initial point to the feasible set. Default isfalse."
  },
  "run_plain": {
    "description": "whether to run an extra experiment with the'plain'feature. Default isfalse."
  },
  "savepath": {
    "description": "the path to store the results. Default is'pwd', the current working directory."
  },
  "score_fun": {
    "description": "the scoring function to calculate the scores of the solvers. It should be a function handleprofile_scores->solver_scores,whereprofile_scoresis a 4D tensor containing scores for all profiles. The first dimension ofprofile_scorescorresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring "
  },
  "score_only": {
    "description": "whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default isfalse."
  },
  "score_weight_fun": {
    "description": "the weight function to calculate the scores of the solvers in the performance and data profiles. It should be a function handle representing a nonnegative function in R^+. Default is1."
  },
  "seed": {
    "description": "the seed of the random number generator. Default is0."
  },
  "semilogx": {
    "description": "whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default istrue."
  },
  "silent": {
    "description": "whether to show the information of the progress. Default isfalse."
  },
  "solver_isrand": {
    "default": "all false",
    "description": "whether the solvers are randomized or not. It is a logical array of the same length as the number of solvers, where the value is true if the solver is randomized, and false otherwise. Default is all false. Note that ifn_runsis not specified, we will set it 5 for the randomized solvers."
  },
  "solver_names": {
    "default": "the names of the function handles insolvers",
    "description": "the names of the solvers. Default is the names of the function handles insolvers."
  },
  "solver_verbose": {
    "description": "the level of the verbosity of the solvers.0means no verbosity,1means some verbosity, and2means full verbosity. Default is1."
  },
  "solvers_to_load": {
    "default": "all the solvers",
    "description": "the indices of the solvers to load when theloadoption is provided. It can be a vector of different integers selected from 1 to the total number of solvers of the loading experiment. At least two indices should be provided. Default is all the solvers."
  },
  "summarize_data_profiles": {
    "description": "whether to add all the data profiles to the summary PDF. Default istrue."
  },
  "summarize_log_ratio_profiles": {
    "description": "whether to add all the log-ratio profiles to the summary PDF. Default isfalse."
  },
  "summarize_output_based_profiles": {
    "description": "whether to add all the output-based profiles of the selected profiles to the summary PDF. Default istrue."
  },
  "summarize_performance_profiles": {
    "description": "whether to add all the performance profiles to the summary PDF. Default istrue."
  },
  "xlabel_data_profile": {
    "description": "the label of the x-axis of the data profiles. Default is'Numberofsimplexgradients'. Note: the'Interpreter'property is set to'latex', so LaTeX formatting is supported. The same applies to the optionsxlabel_log_ratio_profile,xlabel_performance_profile,ylabel_data_profile,ylabel_log_ratio_profile, andylabel_performance_profile."
  },
  "xlabel_log_ratio_profile": {
    "description": "the label of the x-axis of the log-ratio profiles. Default is'Problem'."
  },
  "xlabel_performance_profile": {
    "description": "the label of the x-axis of the performance profiles. Default is'Performanceratio'."
  },
  "ylabel_data_profile": {
    "description": "the label of the y-axis of the data profiles. Default is'Dataprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format. You can also use%sin your custom label, and it will be replaced accordingly. The same applies to the optionsylabel_log_ratio_profileandylabel_performance_profile."
  },
  "ylabel_log_ratio_profile": {
    "description": "the label of the y-axis of the log-ratio profiles. Default is'Log-ratioprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format."
  },
  "ylabel_performance_profile": {
    "description": "ylabel_performance_profile: the label of the y-axis of the performance profiles. Default is'Performanceprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format."
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
  "default": "set to the first two colors in theline_colorsoption",
  "description": "two different colors for the bars of two solvers in the log-ratio profiles. It can be a cell array of short names of colors('r','g','b','c','m','y','k')or a 2-by-3 matrix with each row being a RGB triplet. Default is set to the first two colors in theline_colorsoption."
}
```

## profile_options.benchmark_id

```json
{
  "description": "the identifier of the test. It is used to create the specific directory to store the results. Default is'out'if the optionloadis not provided, otherwise default is'.'."
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
  "default": "'parallel' in normal runs; 'sequential' in load mode",
  "description": "whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', no history plots are drawn. If it is 'parallel', history plots are drawn while solvers are solving problems. If it is 'sequential', history plots are drawn after all problems are solved. Default is 'parallel' in normal runs; when the load option is provided, OptiProfiler forces 'sequential'."
}
```

## profile_options.errorbar_type

```json
{
  "choices": [
    "minmax",
    "meanstd"
  ],
  "description": "the type of the uncertainty interval that can be either'minmax'or'meanstd'. Whenn_runsis greater than 1, we run several times of the experiments and get average curves and get average curves and uncertainty intervals. Default is'minmax', meaning that we takes the pointwise minimum and maximum of the curves."
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
  "description": "the aggregation method we use to reduce the number of points in the history plots. It can be'min','mean', or'max'. Default is'min'."
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
  "description": "the colors of the lines in the plots. It can be a cell array of short names of colors('r','g','b','c','m','y','k')or a matrix with each row being a RGB triplet. Default line colors are those in the palettename named “gem” (see MATLAB documentation for ‘colororder’). Note that if the number of solvers is greater than the number of colors, we will cycle through the colors."
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
  "description": "the styles of the lines in the plots. It can be a cell array of chars that are the combinations of line styles('-','-.','--',':')and markers('o','+','*','.','x','s','d','^','v','>','<','p','h'). Default line style order is{'-','-.','--',':'}. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles."
}
```

## profile_options.line_widths

```json
{
  "description": "the widths of the lines in the plots. It should be a positive scalar or a vector. Default is1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths."
}
```

## profile_options.load

```json
{
  "description": "loading the stored data from a completed experiment and draw profiles. It can be either'latest'or a time stamp of an experiment in the format of ‘yyyyMMdd_HHmmss’. No default."
}
```

## profile_options.max_eval_factor

```json
{
  "description": "the factor multiplied to each problem’s dimension to get the maximum number of evaluations for each problem. Default is500."
}
```

## profile_options.max_tol_order

```json
{
  "description": "the maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of ‘tolerances’ to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as10^(-1:-1:-max_tol_order). Default is10."
}
```

## profile_options.merit_fun

```json
{
  "description": "the merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a function handle(fun_value,maxcv_value,maxcv_init)->merit_value,wherefun_valueis the objective function value,maxcv_valueis the maximum constraint violation, andmaxcv_initis the maximum constraint violation at the initial guess. The size offun_valuesandmaxcv_valuesis the same, and the size ofmaxcv_initis the same as the second to last dimensions offun_values"
}
```

## profile_options.n_jobs

```json
{
  "default": "about half of available workers, at least 2 when possible",
  "description": "the number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers, with at least 2 when more than one worker is available."
}
```

## profile_options.normalized_scores

```json
{
  "description": "whether to normalize the scores of the solvers by the maximum score of the solvers. Default istrue."
}
```

## profile_options.project_x0

```json
{
  "description": "whether to project the initial point to the feasible set. Default isfalse."
}
```

## profile_options.run_plain

```json
{
  "description": "whether to run an extra experiment with the'plain'feature. Default isfalse."
}
```

## profile_options.savepath

```json
{
  "description": "the path to store the results. Default is'pwd', the current working directory."
}
```

## profile_options.score_fun

```json
{
  "description": "the scoring function to calculate the scores of the solvers. It should be a function handleprofile_scores->solver_scores,whereprofile_scoresis a 4D tensor containing scores for all profiles. The first dimension ofprofile_scorescorresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring "
}
```

## profile_options.score_only

```json
{
  "description": "whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default isfalse."
}
```

## profile_options.score_weight_fun

```json
{
  "description": "the weight function to calculate the scores of the solvers in the performance and data profiles. It should be a function handle representing a nonnegative function in R^+. Default is1."
}
```

## profile_options.seed

```json
{
  "description": "the seed of the random number generator. Default is0."
}
```

## profile_options.semilogx

```json
{
  "description": "whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default istrue."
}
```

## profile_options.silent

```json
{
  "description": "whether to show the information of the progress. Default isfalse."
}
```

## profile_options.solver_isrand

```json
{
  "default": "all false",
  "description": "whether the solvers are randomized or not. It is a logical array of the same length as the number of solvers, where the value is true if the solver is randomized, and false otherwise. Default is all false. Note that ifn_runsis not specified, we will set it 5 for the randomized solvers."
}
```

## profile_options.solver_names

```json
{
  "default": "the names of the function handles insolvers",
  "description": "the names of the solvers. Default is the names of the function handles insolvers."
}
```

## profile_options.solver_verbose

```json
{
  "description": "the level of the verbosity of the solvers.0means no verbosity,1means some verbosity, and2means full verbosity. Default is1."
}
```

## profile_options.solvers_to_load

```json
{
  "default": "all the solvers",
  "description": "the indices of the solvers to load when theloadoption is provided. It can be a vector of different integers selected from 1 to the total number of solvers of the loading experiment. At least two indices should be provided. Default is all the solvers."
}
```

## profile_options.summarize_data_profiles

```json
{
  "description": "whether to add all the data profiles to the summary PDF. Default istrue."
}
```

## profile_options.summarize_log_ratio_profiles

```json
{
  "description": "whether to add all the log-ratio profiles to the summary PDF. Default isfalse."
}
```

## profile_options.summarize_output_based_profiles

```json
{
  "description": "whether to add all the output-based profiles of the selected profiles to the summary PDF. Default istrue."
}
```

## profile_options.summarize_performance_profiles

```json
{
  "description": "whether to add all the performance profiles to the summary PDF. Default istrue."
}
```

## profile_options.xlabel_data_profile

```json
{
  "description": "the label of the x-axis of the data profiles. Default is'Numberofsimplexgradients'. Note: the'Interpreter'property is set to'latex', so LaTeX formatting is supported. The same applies to the optionsxlabel_log_ratio_profile,xlabel_performance_profile,ylabel_data_profile,ylabel_log_ratio_profile, andylabel_performance_profile."
}
```

## profile_options.xlabel_log_ratio_profile

```json
{
  "description": "the label of the x-axis of the log-ratio profiles. Default is'Problem'."
}
```

## profile_options.xlabel_performance_profile

```json
{
  "description": "the label of the x-axis of the performance profiles. Default is'Performanceratio'."
}
```

## profile_options.ylabel_data_profile

```json
{
  "description": "the label of the y-axis of the data profiles. Default is'Dataprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format. You can also use%sin your custom label, and it will be replaced accordingly. The same applies to the optionsylabel_log_ratio_profileandylabel_performance_profile."
}
```

## profile_options.ylabel_log_ratio_profile

```json
{
  "description": "the label of the y-axis of the log-ratio profiles. Default is'Log-ratioprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format."
}
```

## profile_options.ylabel_performance_profile

```json
{
  "description": "ylabel_performance_profile: the label of the y-axis of the performance profiles. Default is'Performanceprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format."
}
```

## feature_options

```json
{
  "condition_factor": {
    "description": "the scaling factor of the condition number of the linear transformation in the'linearly_transformed'feature. More specifically, the condition number of the linear transformation will be2^(condition_factor*n/2), wherenis the dimension of the problem. Default is0."
  },
  "distribution": {
    "description": "the distribution of perturbation in'perturbed_x0'feature or noise in'noisy'feature. It should be either a string (or char), or a function handle(random_stream,dimension)->randomvectorthat accepts arandom_streamand thedimensionof a problem and returning arandomvectorwith the givendimension. In'perturbed_x0'case, the char should be either'spherical'or'gaussian'(default is'spherical'). In'noisy'case, the char should be either'gaussian'or'uniform'(default is'gaussian')."
  },
  "feature_name": {
    "description": "the name of the feature. The available features are'plain','perturbed_x0','noisy','truncated','permuted','linearly_transformed','random_nan','unrelaxable_constraints','nonquantifiable_constraints','quantized', and'custom'. Default is'plain'."
  },
  "ground_truth": {
    "description": "whether the featured problem is the ground truth or not in the'quantized'feature. Default istrue."
  },
  "mesh_size": {
    "description": "the size of the mesh in the'quantized'feature. Default is1e-3."
  },
  "mesh_type": {
    "choices": [
      "absolute",
      "relative"
    ],
    "description": "the type of the mesh in the'quantized'feature. It should be either'absolute'or'relative'. Default is'absolute'."
  },
  "mod_affine": {
    "description": "the modifier function to generate the affine transformation applied to the variables in the'custom'feature. It should be a function handle(random_stream,problem)->(A,b,inv),whereproblemis an instance of the class Problem,Ais the matrix of the affine transformation,bis the vector of the affine transformation, andinvis the inverse of matrixA. No default."
  },
  "mod_bounds": {
    "description": "the modifier function to modify the bound constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_xl,modified_xu),whereproblemis an instance of the class Problem,modified_xlis the modified lower bound, andmodified_xuis the modified upper bound. No default."
  },
  "mod_ceq": {
    "description": "the modifier function to modify the nonlinear equality constraints in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_ceq,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_ceqis the modified vector of the nonlinear equality constraints. No default."
  },
  "mod_cub": {
    "description": "the modifier function to modify the nonlinear inequality constraints in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_cub,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_cubis the modified vector of the nonlinear inequality constraints. No default."
  },
  "mod_fun": {
    "description": "the modifier function to modify the objective function in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_fun,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_funis the modified objective function value. No default."
  },
  "mod_linear_eq": {
    "description": "the modifier function to modify the linear equality constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_aeq,modified_beq),whereproblemis an instance of the class Problem,modified_aeqis the modified matrix of the linear equality constraints, andmodified_beqis the modified vector of the linear equality constraints. No default."
  },
  "mod_linear_ub": {
    "description": "the modifier function to modify the linear inequality constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_aub,modified_bub),whereproblemis an instance of the class Problem,modified_aubis the modified matrix of the linear inequality constraints, andmodified_bubis the modified vector of the linear inequality constraints. No default."
  },
  "mod_x0": {
    "description": "the modifier function to modify the initial guess in the'custom'feature. It should be a function handle(random_stream,problem)->modified_x0,whereproblemis an instance of the class Problem, andmodified_x0is the modified initial guess. No default."
  },
  "n_runs": {
    "description": "the number of runs of the experiments with the given feature. Default is5for stochastic features and1for deterministic features."
  },
  "nan_rate": {
    "description": "the probability that the evaluation of the objective function will return NaN in the'random_nan'feature. Default is0.05."
  },
  "noise_level": {
    "description": "the magnitude of the noise in the'noisy'feature. Default is1e-3."
  },
  "noise_type": {
    "choices": [
      "absolute",
      "relative",
      "mixed"
    ],
    "description": "the type of the noise in the'noisy'features. It should be either'absolute','relative', or'mixed'. Default is'mixed'."
  },
  "perturbation_level": {
    "description": "the magnitude of the perturbation to the initial guess in the'perturbed_x0'feature. Default is1e-3."
  },
  "perturbed_trailing_digits": {
    "description": "whether we will randomize the trailing digits of the objective function value in the'truncated'feature. Default isfalse."
  },
  "rotated": {
    "description": "whether to use a random or given rotation matrix to rotate the coordinates of a problem in the'linearly_transformed'feature. Default istrue."
  },
  "significant_digits": {
    "description": "the number of significant digits in the'truncated'feature. Default is6."
  },
  "unrelaxable_bounds": {
    "description": "whether the bound constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default istrue."
  },
  "unrelaxable_linear_constraints": {
    "description": "whether the linear constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default isfalse."
  },
  "unrelaxable_nonlinear_constraints": {
    "description": "whether the nonlinear constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default isfalse."
  }
}
```

## feature_options.feature_name

```json
{
  "description": "the name of the feature. The available features are'plain','perturbed_x0','noisy','truncated','permuted','linearly_transformed','random_nan','unrelaxable_constraints','nonquantifiable_constraints','quantized', and'custom'. Default is'plain'."
}
```

## feature_options.n_runs

```json
{
  "description": "the number of runs of the experiments with the given feature. Default is5for stochastic features and1for deterministic features."
}
```

## feature_options.distribution

```json
{
  "description": "the distribution of perturbation in'perturbed_x0'feature or noise in'noisy'feature. It should be either a string (or char), or a function handle(random_stream,dimension)->randomvectorthat accepts arandom_streamand thedimensionof a problem and returning arandomvectorwith the givendimension. In'perturbed_x0'case, the char should be either'spherical'or'gaussian'(default is'spherical'). In'noisy'case, the char should be either'gaussian'or'uniform'(default is'gaussian')."
}
```

## feature_options.perturbation_level

```json
{
  "description": "the magnitude of the perturbation to the initial guess in the'perturbed_x0'feature. Default is1e-3."
}
```

## feature_options.noise_level

```json
{
  "description": "the magnitude of the noise in the'noisy'feature. Default is1e-3."
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
  "description": "the type of the noise in the'noisy'features. It should be either'absolute','relative', or'mixed'. Default is'mixed'."
}
```

## feature_options.significant_digits

```json
{
  "description": "the number of significant digits in the'truncated'feature. Default is6."
}
```

## feature_options.perturbed_trailing_digits

```json
{
  "description": "whether we will randomize the trailing digits of the objective function value in the'truncated'feature. Default isfalse."
}
```

## feature_options.rotated

```json
{
  "description": "whether to use a random or given rotation matrix to rotate the coordinates of a problem in the'linearly_transformed'feature. Default istrue."
}
```

## feature_options.condition_factor

```json
{
  "description": "the scaling factor of the condition number of the linear transformation in the'linearly_transformed'feature. More specifically, the condition number of the linear transformation will be2^(condition_factor*n/2), wherenis the dimension of the problem. Default is0."
}
```

## feature_options.nan_rate

```json
{
  "description": "the probability that the evaluation of the objective function will return NaN in the'random_nan'feature. Default is0.05."
}
```

## feature_options.unrelaxable_bounds

```json
{
  "description": "whether the bound constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default istrue."
}
```

## feature_options.unrelaxable_linear_constraints

```json
{
  "description": "whether the linear constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default isfalse."
}
```

## feature_options.unrelaxable_nonlinear_constraints

```json
{
  "description": "whether the nonlinear constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default isfalse."
}
```

## feature_options.mesh_size

```json
{
  "description": "the size of the mesh in the'quantized'feature. Default is1e-3."
}
```

## feature_options.mesh_type

```json
{
  "choices": [
    "absolute",
    "relative"
  ],
  "description": "the type of the mesh in the'quantized'feature. It should be either'absolute'or'relative'. Default is'absolute'."
}
```

## feature_options.ground_truth

```json
{
  "description": "whether the featured problem is the ground truth or not in the'quantized'feature. Default istrue."
}
```

## feature_options.mod_x0

```json
{
  "description": "the modifier function to modify the initial guess in the'custom'feature. It should be a function handle(random_stream,problem)->modified_x0,whereproblemis an instance of the class Problem, andmodified_x0is the modified initial guess. No default."
}
```

## feature_options.mod_affine

```json
{
  "description": "the modifier function to generate the affine transformation applied to the variables in the'custom'feature. It should be a function handle(random_stream,problem)->(A,b,inv),whereproblemis an instance of the class Problem,Ais the matrix of the affine transformation,bis the vector of the affine transformation, andinvis the inverse of matrixA. No default."
}
```

## feature_options.mod_bounds

```json
{
  "description": "the modifier function to modify the bound constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_xl,modified_xu),whereproblemis an instance of the class Problem,modified_xlis the modified lower bound, andmodified_xuis the modified upper bound. No default."
}
```

## feature_options.mod_linear_ub

```json
{
  "description": "the modifier function to modify the linear inequality constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_aub,modified_bub),whereproblemis an instance of the class Problem,modified_aubis the modified matrix of the linear inequality constraints, andmodified_bubis the modified vector of the linear inequality constraints. No default."
}
```

## feature_options.mod_linear_eq

```json
{
  "description": "the modifier function to modify the linear equality constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_aeq,modified_beq),whereproblemis an instance of the class Problem,modified_aeqis the modified matrix of the linear equality constraints, andmodified_beqis the modified vector of the linear equality constraints. No default."
}
```

## feature_options.mod_fun

```json
{
  "description": "the modifier function to modify the objective function in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_fun,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_funis the modified objective function value. No default."
}
```

## feature_options.mod_cub

```json
{
  "description": "the modifier function to modify the nonlinear inequality constraints in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_cub,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_cubis the modified vector of the nonlinear inequality constraints. No default."
}
```

## feature_options.mod_ceq

```json
{
  "description": "the modifier function to modify the nonlinear equality constraints in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_ceq,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_ceqis the modified vector of the nonlinear equality constraints. No default."
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
    "description": "the maximum number of bound constraints of the problems to be selected. Default isminb+10."
  },
  "maxcon": {
    "description": "the maximum number of linear and nonlinear constraints of the problems to be selected. Default ismax(maxlcon,maxnlcon)."
  },
  "maxdim": {
    "description": "the maximum dimension of the problems to be selected. Default ismindim+10."
  },
  "maxlcon": {
    "description": "the maximum number of linear constraints of the problems to be selected. Default isminlcon+10."
  },
  "maxnlcon": {
    "description": "the maximum number of nonlinear constraints of the problems to be selected. Default isminnlcon+10."
  },
  "minb": {
    "description": "the minimum number of bound constraints of the problems to be selected. Default is0."
  },
  "mincon": {
    "description": "the minimum number of linear and nonlinear constraints of the problems to be selected. Default ismin(minlcon,minnlcon)."
  },
  "mindim": {
    "description": "the minimum dimension of the problems to be selected. Default is1."
  },
  "minlcon": {
    "description": "the minimum number of linear constraints of the problems to be selected. Default is0."
  },
  "minnlcon": {
    "description": "the minimum number of nonlinear constraints of the problems to be selected. Default is0."
  },
  "plibs": {
    "description": "the problem libraries to be used. It should be a cell array of strings or chars. The built-in choices ares2mpj,matcutest, andcustom. Default setting is's2mpj'."
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
    "description": "the type of the problems to be selected. It should be a string or char consisting of any combination of'u'(unconstrained),'b'(bound constrained),'l'(linearly constrained), and'n'(nonlinearly constrained), such as'b','ul','ubn'. Default is'u'."
  }
}
```

## problem_options.plibs

```json
{
  "description": "the problem libraries to be used. It should be a cell array of strings or chars. The built-in choices ares2mpj,matcutest, andcustom. Default setting is's2mpj'."
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
  "description": "the type of the problems to be selected. It should be a string or char consisting of any combination of'u'(unconstrained),'b'(bound constrained),'l'(linearly constrained), and'n'(nonlinearly constrained), such as'b','ul','ubn'. Default is'u'."
}
```

## problem_options.mindim

```json
{
  "description": "the minimum dimension of the problems to be selected. Default is1."
}
```

## problem_options.maxdim

```json
{
  "description": "the maximum dimension of the problems to be selected. Default ismindim+10."
}
```

## problem_options.minb

```json
{
  "description": "the minimum number of bound constraints of the problems to be selected. Default is0."
}
```

## problem_options.maxb

```json
{
  "description": "the maximum number of bound constraints of the problems to be selected. Default isminb+10."
}
```

## problem_options.minlcon

```json
{
  "description": "the minimum number of linear constraints of the problems to be selected. Default is0."
}
```

## problem_options.maxlcon

```json
{
  "description": "the maximum number of linear constraints of the problems to be selected. Default isminlcon+10."
}
```

## problem_options.minnlcon

```json
{
  "description": "the minimum number of nonlinear constraints of the problems to be selected. Default is0."
}
```

## problem_options.maxnlcon

```json
{
  "description": "the maximum number of nonlinear constraints of the problems to be selected. Default isminnlcon+10."
}
```

## problem_options.mincon

```json
{
  "description": "the minimum number of linear and nonlinear constraints of the problems to be selected. Default ismin(minlcon,minnlcon)."
}
```

## problem_options.maxcon

```json
{
  "description": "the maximum number of linear and nonlinear constraints of the problems to be selected. Default ismax(maxlcon,maxnlcon)."
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

## output_artifacts

```json
{
  "result_directory": "`<savepath>/<benchmark_id>/<feature_stamp>_<timestamp>/`",
  "summary_pdf": "summary.pdf summarizes performance profiles and data profiles.",
  "test_log_log": "test_log/log.txt records messages printed during the run.",
  "test_log_report": "test_log/report.txt records selected problem names, timing information, merit_init = phi(x_0) = Inf cases, abnormal solver terminations, output fallbacks, and solver scores."
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
      "description": "the scaling factor of the condition number of the linear transformation in the'linearly_transformed'feature. More specifically, the condition number of the linear transformation will be2^(condition_factor*n/2), wherenis the dimension of the problem. Default is0."
    },
    "distribution": {
      "description": "the distribution of perturbation in'perturbed_x0'feature or noise in'noisy'feature. It should be either a string (or char), or a function handle(random_stream,dimension)->randomvectorthat accepts arandom_streamand thedimensionof a problem and returning arandomvectorwith the givendimension. In'perturbed_x0'case, the char should be either'spherical'or'gaussian'(default is'spherical'). In'noisy'case, the char should be either'gaussian'or'uniform'(default is'gaussian')."
    },
    "feature_name": {
      "description": "the name of the feature. The available features are'plain','perturbed_x0','noisy','truncated','permuted','linearly_transformed','random_nan','unrelaxable_constraints','nonquantifiable_constraints','quantized', and'custom'. Default is'plain'."
    },
    "ground_truth": {
      "description": "whether the featured problem is the ground truth or not in the'quantized'feature. Default istrue."
    },
    "mesh_size": {
      "description": "the size of the mesh in the'quantized'feature. Default is1e-3."
    },
    "mesh_type": {
      "choices": [
        "absolute",
        "relative"
      ],
      "description": "the type of the mesh in the'quantized'feature. It should be either'absolute'or'relative'. Default is'absolute'."
    },
    "mod_affine": {
      "description": "the modifier function to generate the affine transformation applied to the variables in the'custom'feature. It should be a function handle(random_stream,problem)->(A,b,inv),whereproblemis an instance of the class Problem,Ais the matrix of the affine transformation,bis the vector of the affine transformation, andinvis the inverse of matrixA. No default."
    },
    "mod_bounds": {
      "description": "the modifier function to modify the bound constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_xl,modified_xu),whereproblemis an instance of the class Problem,modified_xlis the modified lower bound, andmodified_xuis the modified upper bound. No default."
    },
    "mod_ceq": {
      "description": "the modifier function to modify the nonlinear equality constraints in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_ceq,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_ceqis the modified vector of the nonlinear equality constraints. No default."
    },
    "mod_cub": {
      "description": "the modifier function to modify the nonlinear inequality constraints in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_cub,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_cubis the modified vector of the nonlinear inequality constraints. No default."
    },
    "mod_fun": {
      "description": "the modifier function to modify the objective function in the'custom'feature. It should be a function handle(x,random_stream,problem)->modified_fun,wherexis the evaluation point,problemis an instance of the class Problem, andmodified_funis the modified objective function value. No default."
    },
    "mod_linear_eq": {
      "description": "the modifier function to modify the linear equality constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_aeq,modified_beq),whereproblemis an instance of the class Problem,modified_aeqis the modified matrix of the linear equality constraints, andmodified_beqis the modified vector of the linear equality constraints. No default."
    },
    "mod_linear_ub": {
      "description": "the modifier function to modify the linear inequality constraints in the'custom'feature. It should be a function handle(random_stream,problem)->(modified_aub,modified_bub),whereproblemis an instance of the class Problem,modified_aubis the modified matrix of the linear inequality constraints, andmodified_bubis the modified vector of the linear inequality constraints. No default."
    },
    "mod_x0": {
      "description": "the modifier function to modify the initial guess in the'custom'feature. It should be a function handle(random_stream,problem)->modified_x0,whereproblemis an instance of the class Problem, andmodified_x0is the modified initial guess. No default."
    },
    "n_runs": {
      "description": "the number of runs of the experiments with the given feature. Default is5for stochastic features and1for deterministic features."
    },
    "nan_rate": {
      "description": "the probability that the evaluation of the objective function will return NaN in the'random_nan'feature. Default is0.05."
    },
    "noise_level": {
      "description": "the magnitude of the noise in the'noisy'feature. Default is1e-3."
    },
    "noise_type": {
      "choices": [
        "absolute",
        "relative",
        "mixed"
      ],
      "description": "the type of the noise in the'noisy'features. It should be either'absolute','relative', or'mixed'. Default is'mixed'."
    },
    "perturbation_level": {
      "description": "the magnitude of the perturbation to the initial guess in the'perturbed_x0'feature. Default is1e-3."
    },
    "perturbed_trailing_digits": {
      "description": "whether we will randomize the trailing digits of the objective function value in the'truncated'feature. Default isfalse."
    },
    "rotated": {
      "description": "whether to use a random or given rotation matrix to rotate the coordinates of a problem in the'linearly_transformed'feature. Default istrue."
    },
    "significant_digits": {
      "description": "the number of significant digits in the'truncated'feature. Default is6."
    },
    "unrelaxable_bounds": {
      "description": "whether the bound constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default istrue."
    },
    "unrelaxable_linear_constraints": {
      "description": "whether the linear constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default isfalse."
    },
    "unrelaxable_nonlinear_constraints": {
      "description": "whether the nonlinear constraints are unrelaxable or not in the'unrelaxable_constraints'feature. Default isfalse."
    }
  },
  "name": "benchmark",
  "output_artifacts": {
    "result_directory": "`<savepath>/<benchmark_id>/<feature_stamp>_<timestamp>/`",
    "summary_pdf": "summary.pdf summarizes performance profiles and data profiles.",
    "test_log_log": "test_log/log.txt records messages printed during the run.",
    "test_log_report": "test_log/report.txt records selected problem names, timing information, merit_init = phi(x_0) = Inf cases, abnormal solver terminations, output fallbacks, and solver scores."
  },
  "problem_options": {
    "excludelist": {
      "default": "not to exclude any problem",
      "description": "the list of problems to be excluded. Default is not to exclude any problem."
    },
    "maxb": {
      "description": "the maximum number of bound constraints of the problems to be selected. Default isminb+10."
    },
    "maxcon": {
      "description": "the maximum number of linear and nonlinear constraints of the problems to be selected. Default ismax(maxlcon,maxnlcon)."
    },
    "maxdim": {
      "description": "the maximum dimension of the problems to be selected. Default ismindim+10."
    },
    "maxlcon": {
      "description": "the maximum number of linear constraints of the problems to be selected. Default isminlcon+10."
    },
    "maxnlcon": {
      "description": "the maximum number of nonlinear constraints of the problems to be selected. Default isminnlcon+10."
    },
    "minb": {
      "description": "the minimum number of bound constraints of the problems to be selected. Default is0."
    },
    "mincon": {
      "description": "the minimum number of linear and nonlinear constraints of the problems to be selected. Default ismin(minlcon,minnlcon)."
    },
    "mindim": {
      "description": "the minimum dimension of the problems to be selected. Default is1."
    },
    "minlcon": {
      "description": "the minimum number of linear constraints of the problems to be selected. Default is0."
    },
    "minnlcon": {
      "description": "the minimum number of nonlinear constraints of the problems to be selected. Default is0."
    },
    "plibs": {
      "description": "the problem libraries to be used. It should be a cell array of strings or chars. The built-in choices ares2mpj,matcutest, andcustom. Default setting is's2mpj'."
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
      "description": "the type of the problems to be selected. It should be a string or char consisting of any combination of'u'(unconstrained),'b'(bound constrained),'l'(linearly constrained), and'n'(nonlinearly constrained), such as'b','ul','ubn'. Default is'u'."
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
      "default": "set to the first two colors in theline_colorsoption",
      "description": "two different colors for the bars of two solvers in the log-ratio profiles. It can be a cell array of short names of colors('r','g','b','c','m','y','k')or a 2-by-3 matrix with each row being a RGB triplet. Default is set to the first two colors in theline_colorsoption."
    },
    "benchmark_id": {
      "description": "the identifier of the test. It is used to create the specific directory to store the results. Default is'out'if the optionloadis not provided, otherwise default is'.'."
    },
    "draw_hist_plots": {
      "choices": [
        "none",
        "sequential",
        "parallel"
      ],
      "default": "'parallel' in normal runs; 'sequential' in load mode",
      "description": "whether or how to draw the history plots of all the problems. It can be either 'none', 'sequential', or 'parallel'. If it is 'none', no history plots are drawn. If it is 'parallel', history plots are drawn while solvers are solving problems. If it is 'sequential', history plots are drawn after all problems are solved. Default is 'parallel' in normal runs; when the load option is provided, OptiProfiler forces 'sequential'."
    },
    "errorbar_type": {
      "choices": [
        "minmax",
        "meanstd"
      ],
      "description": "the type of the uncertainty interval that can be either'minmax'or'meanstd'. Whenn_runsis greater than 1, we run several times of the experiments and get average curves and get average curves and uncertainty intervals. Default is'minmax', meaning that we takes the pointwise minimum and maximum of the curves."
    },
    "feature_stamp": {
      "description": "the stamp of the feature with the given options. It is used to create the specific directory to store the results. Default depends on features."
    },
    "hist_aggregation": {
      "description": "the aggregation method we use to reduce the number of points in the history plots. It can be'min','mean', or'max'. Default is'min'."
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
      "description": "the colors of the lines in the plots. It can be a cell array of short names of colors('r','g','b','c','m','y','k')or a matrix with each row being a RGB triplet. Default line colors are those in the palettename named “gem” (see MATLAB documentation for ‘colororder’). Note that if the number of solvers is greater than the number of colors, we will cycle through the colors."
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
      "description": "the styles of the lines in the plots. It can be a cell array of chars that are the combinations of line styles('-','-.','--',':')and markers('o','+','*','.','x','s','d','^','v','>','<','p','h'). Default line style order is{'-','-.','--',':'}. Note that if the number of solvers is greater than the number of line styles, we will cycle through the styles."
    },
    "line_widths": {
      "description": "the widths of the lines in the plots. It should be a positive scalar or a vector. Default is1.5. Note that if the number of solvers is greater than the number of line widths, we will cycle through the widths."
    },
    "load": {
      "description": "loading the stored data from a completed experiment and draw profiles. It can be either'latest'or a time stamp of an experiment in the format of ‘yyyyMMdd_HHmmss’. No default."
    },
    "max_eval_factor": {
      "description": "the factor multiplied to each problem’s dimension to get the maximum number of evaluations for each problem. Default is500."
    },
    "max_tol_order": {
      "description": "the maximum order of the tolerance. In any profile (performance profiles, data profiles, and log-ratio profiles), we need to set a group of ‘tolerances’ to define the convergence test of the solvers. (Details can be found in the references.) We will set the tolerances as10^(-1:-1:-max_tol_order). Default is10."
    },
    "merit_fun": {
      "description": "the merit function to measure the quality of a point using the objective function value and the maximum constraint violation. It should be a function handle(fun_value,maxcv_value,maxcv_init)->merit_value,wherefun_valueis the objective function value,maxcv_valueis the maximum constraint violation, andmaxcv_initis the maximum constraint violation at the initial guess. The size offun_valuesandmaxcv_valuesis the same, and the size ofmaxcv_initis the same as the second to last dimensions offun_values"
    },
    "n_jobs": {
      "default": "about half of available workers, at least 2 when possible",
      "description": "the number of parallel jobs to run the test. Default is a conservative number of workers, chosen as about half of the available workers, with at least 2 when more than one worker is available."
    },
    "normalized_scores": {
      "description": "whether to normalize the scores of the solvers by the maximum score of the solvers. Default istrue."
    },
    "project_x0": {
      "description": "whether to project the initial point to the feasible set. Default isfalse."
    },
    "run_plain": {
      "description": "whether to run an extra experiment with the'plain'feature. Default isfalse."
    },
    "savepath": {
      "description": "the path to store the results. Default is'pwd', the current working directory."
    },
    "score_fun": {
      "description": "the scoring function to calculate the scores of the solvers. It should be a function handleprofile_scores->solver_scores,whereprofile_scoresis a 4D tensor containing scores for all profiles. The first dimension ofprofile_scorescorresponds to the index of the solver, the second corresponds to the index of tolerance starting from 1, the third represents history-based or output-based profiles, and the fourth represents performance profiles, data profiles, or log-ratio profiles. The default scoring "
    },
    "score_only": {
      "description": "whether to only calculate the scores of the solvers without drawing the profiles and saving the data. Default isfalse."
    },
    "score_weight_fun": {
      "description": "the weight function to calculate the scores of the solvers in the performance and data profiles. It should be a function handle representing a nonnegative function in R^+. Default is1."
    },
    "seed": {
      "description": "the seed of the random number generator. Default is0."
    },
    "semilogx": {
      "description": "whether to use the semilogx scale during plotting profiles (performance profiles and data profiles). Default istrue."
    },
    "silent": {
      "description": "whether to show the information of the progress. Default isfalse."
    },
    "solver_isrand": {
      "default": "all false",
      "description": "whether the solvers are randomized or not. It is a logical array of the same length as the number of solvers, where the value is true if the solver is randomized, and false otherwise. Default is all false. Note that ifn_runsis not specified, we will set it 5 for the randomized solvers."
    },
    "solver_names": {
      "default": "the names of the function handles insolvers",
      "description": "the names of the solvers. Default is the names of the function handles insolvers."
    },
    "solver_verbose": {
      "description": "the level of the verbosity of the solvers.0means no verbosity,1means some verbosity, and2means full verbosity. Default is1."
    },
    "solvers_to_load": {
      "default": "all the solvers",
      "description": "the indices of the solvers to load when theloadoption is provided. It can be a vector of different integers selected from 1 to the total number of solvers of the loading experiment. At least two indices should be provided. Default is all the solvers."
    },
    "summarize_data_profiles": {
      "description": "whether to add all the data profiles to the summary PDF. Default istrue."
    },
    "summarize_log_ratio_profiles": {
      "description": "whether to add all the log-ratio profiles to the summary PDF. Default isfalse."
    },
    "summarize_output_based_profiles": {
      "description": "whether to add all the output-based profiles of the selected profiles to the summary PDF. Default istrue."
    },
    "summarize_performance_profiles": {
      "description": "whether to add all the performance profiles to the summary PDF. Default istrue."
    },
    "xlabel_data_profile": {
      "description": "the label of the x-axis of the data profiles. Default is'Numberofsimplexgradients'. Note: the'Interpreter'property is set to'latex', so LaTeX formatting is supported. The same applies to the optionsxlabel_log_ratio_profile,xlabel_performance_profile,ylabel_data_profile,ylabel_log_ratio_profile, andylabel_performance_profile."
    },
    "xlabel_log_ratio_profile": {
      "description": "the label of the x-axis of the log-ratio profiles. Default is'Problem'."
    },
    "xlabel_performance_profile": {
      "description": "the label of the x-axis of the performance profiles. Default is'Performanceratio'."
    },
    "ylabel_data_profile": {
      "description": "the label of the y-axis of the data profiles. Default is'Dataprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format. You can also use%sin your custom label, and it will be replaced accordingly. The same applies to the optionsylabel_log_ratio_profileandylabel_performance_profile."
    },
    "ylabel_log_ratio_profile": {
      "description": "the label of the y-axis of the log-ratio profiles. Default is'Log-ratioprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format."
    },
    "ylabel_performance_profile": {
      "description": "ylabel_performance_profile: the label of the y-axis of the performance profiles. Default is'Performanceprofiles($\\\\mathrm{tol}=%s$)', where%swill be replaced by the current tolerance in LaTeX format."
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
