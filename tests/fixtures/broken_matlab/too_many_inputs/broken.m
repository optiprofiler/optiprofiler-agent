% BUG: local solver accepts one argument but OptiProfiler-style caller passes
% the canonical (fun, x0) pair.
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun)
    x = fminsearch(fun, [0; 0]);
end
