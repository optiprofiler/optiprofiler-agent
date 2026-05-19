% FIX: provide a default ptype field before access.
options.max_eval = 100;
if ~isfield(options, 'ptype')
    options.ptype = 'u';
end
disp(options.ptype);
