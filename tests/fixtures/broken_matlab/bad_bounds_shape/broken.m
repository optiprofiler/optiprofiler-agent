% BUG: lower bound vector does not match x0 length.
x0 = [0; 1];
lb = 0;
ub = [2; 3];
if numel(lb) ~= numel(x0) || numel(ub) ~= numel(x0)
    error('Bounds shape mismatch: expected length 2');
end
disp([lb(:), ub(:)]);
