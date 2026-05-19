% Driver script — calls my_solver with the canonical (fun, x0) order.
fun = @(z) sum(z.^2);
x0 = [1; 2; 3];
result = my_solver(fun, x0);
disp(result);

% FIX: parameters now match OptiProfiler's expected (fun, x0) order.
function x = my_solver(fun, x0)
    x = fminsearch(fun, x0);
    x = x(:);
end
