% FIX: start away from the singularity.
fun = @(x) 1 ./ x(1);
x0 = 1;
y0 = fun(x0);
if ~isfinite(y0)
    error('Objective returned Inf because of division by zero at x0');
end
disp(y0);
