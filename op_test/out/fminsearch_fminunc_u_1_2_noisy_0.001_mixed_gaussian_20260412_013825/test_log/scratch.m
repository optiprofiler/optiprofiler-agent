function scratch()

    solvers = {@fminsearch, @fminunc};
    options.n_jobs = 5;
    options.n_runs = 10;
    options.feature_name = 'noisy';

    scores = benchmark(solvers, options)
end
