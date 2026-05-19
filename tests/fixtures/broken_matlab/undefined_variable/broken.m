% BUG: typo in the variable name used for the starting point.
x0 = [1; 2; 3];
fun = @(z) sum(z.^2);
result = fminsearch(fun, x_start);
disp(result);
