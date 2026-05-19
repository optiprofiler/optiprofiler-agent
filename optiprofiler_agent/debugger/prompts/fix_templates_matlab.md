# Common Fix Patterns (MATLAB)

## 1. Interface Wrapper (solver signature mismatch)

When a solver accepts different arguments:

```matlab
function x = solver_wrapper(fun, x0, xl, xu)
    % Wrapper to adapt my_solver to OptiProfiler's bound-constrained interface
    x = my_solver(fun, x0, 'LowerBound', xl, 'UpperBound', xu);
end
```

## 2. Missing Path

```matlab
% If helper functions are in subdirectories:
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));
```

## 3. Numerical Protection

```matlab
function x = safe_solver(fun, x0)
    safe_fun = @(x) guard_nan(fun(x));
    x = original_solver(safe_fun, x0);
end

function v = guard_nan(v)
    if ~isfinite(v), v = 1e30; end
end
```

## 4. Return Value Fix

OptiProfiler expects the solver to return x (column vector):

```matlab
function x = solver(fun, x0)
    result = fminsearch(fun, x0);
    x = result(:);  % ensure column vector
end
```
