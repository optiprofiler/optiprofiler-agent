% FIX: swap the unavailable cobyqa_mex for built-in fminsearch.
x0 = [1; 2; 3];
fun = @(z) sum(z.^2);
result = fminsearch(fun, x0);
disp(result);
