% BUG: missing dependency — `cobyqa_mex` is a private mex that isn't on
% the path. OptiProfiler raises ``Undefined function or variable``.
x0 = [1; 2; 3];
fun = @(z) sum(z.^2);
result = cobyqa_mex(fun, x0);
disp(result);
