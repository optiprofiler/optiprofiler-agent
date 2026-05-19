% FIX: bound the index by length(a) before indexing.
a = [1, 2, 3];
idx = min(5, length(a));
disp(a(idx));
