# Matlab Examples

> **OptiProfiler benchmarks derivative-free optimization (DFO) solvers.**
> `fun` provides ONLY function values — no gradient or Hessian.
> `benchmark()` requires at least 2 solvers.

---

# Usage for MATLAB

OptiProfiler provides a benchmark function. This is the main entry point to the package. It benchmarks given solvers on the selected test suite.

We provide below simple examples on how to use OptiProfiler in MATLAB. For more details on the signature of the benchmark function, please refer to the MATLAB API documentation.

## Examples

### Example 1: first example to try out

(See also the file in the repository: matlab/examples/example1.m)

Let us first try to benchmark two callable optimization solvers solver1 and solver2 (e.g., fminsearch and fminunc in MATLAB Optimization Toolbox) on the default test suite. (Note that each solver must accept signatures mentioned in the Cautions part of the benchmark function according to the type of problems you want to solve.)

To do this, run:

```matlab
scores = benchmark({@solver1, @solver2})
```

This will benchmark the two solvers under the default test setting, which means 'plain' feature (see Feature) and unconstrained problems from the default problem library whose dimension is smaller or equal to 2. It will also return the scores of the two solvers based on the profiles.

There will be a new folder named out in the current working directory, which contains a subfolder named plain_<timestamp> with all the detailed results.

Additionally, a PDF file named summary.pdf is generated, summarizing all the performance profiles and data profiles.

### Example 2: one step further by adding options

(See also the file in the repository: matlab/examples/example2.m)

You can also add options to the benchmark function. For example, if you want to benchmark three solvers solver1, solver2, and solver3 on the test suite with the 'noisy' feature and all the unconstrained and bound-constrained problems with dimension between 6 and 10 from the default problem set, you can run:

```matlab
options.ptype = 'ub';
options.mindim = 6;
options.maxdim = 10;
options.feature_name = 'noisy';
scores = benchmark({@solver1, @solver2, @solver3}, options)
```

This will create the corresponding folders out/noisy_<timestamp> and files as in Example 1. More details on the options can be found in the benchmark function documentation.

### Example 3: useful optionload

(See also the file in the repository: matlab/examples/example3.m)

OptiProfiler provides a practically useful option named load. This option allows you to load the results from a previous benchmarking run (without solving all the problems again) and use them to draw new profiles with different options. For example, if you have just run Example 2 and OptiProfiler has finished the job and successfully created the folder out in the current working directory, you can run:

```matlab
options.load = 'latest';
options.solvers_to_load = [1, 3];
options.ptype = 'u';
options.mindim = 7;
options.maxdim = 9;
scores = benchmark(options)
```

This will directly draw the profiles for the solver1 and solver3 with the 'noisy' feature and all the unconstrained problems with dimension between 7 and 9 selected from the previous run. The results will also be saved under the current directory with a new subfolder named noisy_<timestamp> with the new timestamp.

### Example 4: testing parametrized solvers

(See also the file in the repository: matlab/examples/example4.m)

If you want to benchmark a solver with one variable parameter, you can define function handles by looping over the parameter values. For example, if solver accepts the signature @(fun, x0, para), and you want to benchmark it with the parameter para taking values from 1 to 3, you can run:

```matlab
solvers = cell(1, 3);
options.solver_names = cell(1, 3);
for i = 1:3
    solvers{i} = @(fun, x0) solver(fun, x0, i);
    options.solver_names{i} = ['solver' num2str(i)];
end
scores = benchmark(solvers, options)
```

### Example 5: customizing the test suite

(See also the file in the repository: matlab/examples/example5.m)

OptiProfiler allows you to customize the test suite by creating your own feature and loading your own problem library. For example, if you want to create a new feature that adds noise to the objective function and perturbs the initial guess at the same time, you can try the following:

```matlab
options.feature_name = 'custom';
options.mod_fun = @(x, rand_stream, problem) problem.fun(x) + 1e-3 * rand_stream.randn(1);
options.mod_x0 = @(rand_stream, problem) problem.x0 + 1e-3 * rand_stream.randn(problem.n, 1);
scores = benchmark({@solver1, @solver2}, options)
```

If you want to benchmark solvers based on your own problem library, you should do the following three steps:

- Create a new subfolder (e.g., 'myproblems') within the 'problems' folder located in the optiprofiler project root directory.

- Implement two MATLAB functions:

- Use the benchmark function as before, but specify your desired problem libraries. For example, to use both the default S2MPJ library and your custom library in the subfolder 'myproblems', you can run:

```matlab
options.plibs = {'s2mpj', 'myproblems'};
scores = benchmark({@solver1, @solver2}, options)
```

You may also refer to the README file in the 'problems' folder for a detailed guide on how to create and use your own problem library via the OptiProfiler package.
