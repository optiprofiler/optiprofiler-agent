---
tags: [guide, matlab, quickstart]
sources: [_sources/matlab/api_notes.json]
related: [api/matlab/benchmark.md, concepts/benchmark-function.md, guides/quickstart-python.md]
last_updated: 2025-04-13
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

## Example 1: Basic Benchmark

```matlab
scores = benchmark({@solver1, @solver2})
```

This benchmarks two solvers on unconstrained problems (default `ptype='u'`).

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

## Example 5: Custom Problem Library

Create a subfolder in the `problems` directory with `_load.m` and
`_select.m` functions, then:

```matlab
options.plibs = {'s2mpj', 'myproblems'};
scores = benchmark({@solver1, @solver2}, options)
```

## See Also

- [MATLAB benchmark() API](../api/matlab/benchmark.md) — full parameter reference
- [Python Quickstart](quickstart-python.md) — Python equivalent
- [Custom Solver Guide](custom-solver.md) — writing solver wrappers
