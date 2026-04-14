# MATLAB Problem Libraries

## Built-in Libraries

- **s2mpj**: Default. Bundled with OptiProfiler.
- **matcutest**: Requires setup. **Linux only.**
  See https://github.com/matcutest

## Custom Libraries

Create a subfolder in the `problems` directory:

```
problems/
└── myproblems/
    ├── myproblems_load.m
    └── myproblems_select.m
```

```matlab
options.plibs = {'s2mpj', 'myproblems'};
scores = benchmark({@solver1, @solver2}, options);
```
