% FIX: accept both canonical arguments.
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun, x0)
    x = fminsearch(fun, x0);
end
