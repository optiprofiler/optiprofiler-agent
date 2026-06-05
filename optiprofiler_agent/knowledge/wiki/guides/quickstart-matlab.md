---
tags: [guide, matlab, quickstart]
sources: [_sources/matlab/api_notes.json]
related: [api/matlab/benchmark.md, concepts/benchmark-function.md, guides/quickstart-python.md]
last_updated: 2026-06-05
---

# MATLAB Quickstart

## Installation

```bash
git clone https://github.com/optiprofiler/optiprofiler.git
```

In MATLAB, navigate to the root directory and run:

```matlab
setup
```

The `setup` function adds necessary directories to the MATLAB path and
clones default problem libraries (S2MPJ and optionally MatCUTEst).

**MatCUTEst** is optional and only supported on **Linux**. During setup you
will be asked whether to install it. For automated environments:

```matlab
setup(struct('install_matcutest', true))  % Or false
```

To uninstall: `setup uninstall`

MatCUTEst is not available on macOS or Windows. On those platforms,
run `setup(struct('install_matcutest', false))` in automated scripts.

## Example 1: Basic Benchmark

```matlab
scores = benchmark({@solver1, @solver2})
```

This benchmarks two solvers on unconstrained problems (default `ptype='u'`).
By default, OptiProfiler creates an `out/<feature_stamp>_<timestamp>/`
folder and writes `summary.pdf`, per-problem results, and `test_log/`.
`test_log/report.txt` records selected problem names, timing,
`merit_init = phi(x_0) = Inf` cases, abnormal solver terminations,
output fallbacks, and solver scores.

## Example 2: With Options

```matlab
options.ptype = 'u';
options.mindim = 2;
options.maxdim = 10;
options.feature_name = 'noisy';
options.noise_level = 1e-3;
scores = benchmark({@solver1, @solver2}, options)
```

## Example 3: Loading Previous Results

```matlab
options.load = 'latest';
scores = benchmark({@solver1, @solver3}, options)
```

## Example 4: Parametrized Solvers

```matlab
solvers = cell(1, 3);
options.solver_names = cell(1, 3);
for i = 1:3
    solvers{i} = @(fun, x0) solver(fun, x0, i);
    options.solver_names{i} = ['solver' num2str(i)];
end
scores = benchmark(solvers, options)
```

If `options.n_jobs` is omitted, OptiProfiler uses a conservative worker
count: about half of the available workers, with at least 2 when more
than one worker is available. Set `options.n_jobs = 1` for reproducible
timing experiments.

## Example 5: Custom Problem Library

Create a subfolder in the `problems` directory with `<name>_load.m`
and `<name>_select.m`, then:

```matlab
options.plibs = {'s2mpj', 'myproblems'};
scores = benchmark({@solver1, @solver2}, options)
```

For the full step-by-step template (struct constructor, `evalc`
stdout-suppression, anonymous-function safety) see the
[Custom Problem Library — MATLAB](custom-problem-library-matlab.md)
guide.

## See Also

- [MATLAB benchmark() API](../api/matlab/benchmark.md) — full parameter reference
- [Python Quickstart](quickstart-python.md) — Python equivalent
- [Custom Solver Guide](custom-solver.md) — writing solver wrappers
