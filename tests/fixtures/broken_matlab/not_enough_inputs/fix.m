% FIX: make the optional argument optional.
fun = @(z) sum(z.^2);
x0 = [1; 2];
result = my_solver(fun, x0);
disp(result);

function x = my_solver(fun, x0, options)
    if nargin < 3
        options.scale = 1;
    end
    x = options.scale * fminsearch(fun, x0);
end
