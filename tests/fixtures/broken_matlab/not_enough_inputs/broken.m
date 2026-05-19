% BUG: local solver needs an extra options argument, but caller provides only
% the canonical (fun, x0) pair.
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun, x0, options)
    x = options.scale * fminsearch(fun, x0);
end
