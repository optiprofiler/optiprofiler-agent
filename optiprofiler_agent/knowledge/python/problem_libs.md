# Python Problem Libraries

## Built-in Libraries

- **s2mpj**: Default. Pure Python, no extra installation.
- **pycutest**: Requires separate installation. Linux and macOS only.
  See https://jfowkes.github.io/pycutest/

## Custom Libraries

Use `custom_problem_libs_path` to add your own:

```
/path/to/my_libs/
└── myproblems/
    └── myproblems_tools.py  (implements myproblems_load + myproblems_select)
```

```python
benchmark(
    [solver1, solver2],
    plibs=['s2mpj', 'myproblems'],
    custom_problem_libs_path='/path/to/my_libs',
)
```
