% FIX: start in the real-valued domain.
fun = @(x) sqrt(x(1));
x0 = 1;
y0 = fun(x0);
if ~isreal(y0) || ~isfinite(y0)
    error('Objective returned NaN/complex value at x0');
end
disp(y0);
