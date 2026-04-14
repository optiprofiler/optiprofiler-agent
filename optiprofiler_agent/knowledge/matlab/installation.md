# MATLAB Installation

Clone the repository:

```bash
git clone https://github.com/optiprofiler/optiprofiler.git
```

In MATLAB, navigate to the root directory and run:

```matlab
setup
```

The `setup` function:
- Adds the necessary directories to the MATLAB search path.
- Clones the default problem libraries, including S2MPJ and MatCUTEst.

**MatCUTEst** is optional and only supported on **Linux**. During setup you will be asked whether to install it. For automated environments:

```matlab
setup(struct('install_matcutest', true))  % Or false
```

To uninstall:

```matlab
setup uninstall
```
