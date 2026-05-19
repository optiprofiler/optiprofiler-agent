% FIX: use the defined starting point variable.
x0 = [1; 2; 3];
fun = @(z) sum(z.^2);
result = fminsearch(fun, x0);
disp(result);
