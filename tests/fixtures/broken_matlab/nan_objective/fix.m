% FIX: start away from the singularity at x=0 so the objective is finite.
fun = @(x) log(x(1)) + 1./x(1);
x0 = 1;
y0 = fun(x0);
if ~isfinite(y0)
    error('Objective returned Inf/NaN at the starting point x0 = %g', x0);
end
result = fminsearch(fun, x0);
disp(result);
