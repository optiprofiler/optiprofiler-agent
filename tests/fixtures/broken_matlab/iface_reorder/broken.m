% Driver script — calls my_solver with the canonical (fun, x0) order, but
% the local function below has them swapped, so the inner call fun(x0)
% ends up indexing a numeric vector with a function handle.
fun = @(z) sum(z.^2);
x0 = [1; 2; 3];
result = my_solver(fun, x0);
disp(result);

% BUG: parameters in the wrong order. OptiProfiler calls (fun, x0); this
% function expects (x0, fun), so the call fun(x0) becomes x0_vec(fun_handle)
% which is a MATLAB runtime error.
function x = my_solver(x0, fun)
    x = fun(x0);
end
