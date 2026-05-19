% BUG: missing closing parenthesis on the fminsearch call.
fun = @(z) sum(z.^2);
x0 = [1; 2; 3];
result = fminsearch(fun, x0;
disp(result);
