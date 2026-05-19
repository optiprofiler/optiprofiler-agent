% BUG: objective is non-finite at the starting point (log(0)=-Inf), and the
% guard explicitly errors. This mimics OptiProfiler raising a NaN-objective
% warning that surfaces as an Inf/NaN diagnostic.
fun = @(x) log(x(1)) + 1./x(1);
x0 = 0;
y0 = fun(x0);
if ~isfinite(y0)
    error('Objective returned Inf/NaN at the starting point x0 = %g', x0);
end
result = fminsearch(fun, x0);
disp(result);
