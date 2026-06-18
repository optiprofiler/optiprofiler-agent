---
tags: [reference, source-backed, python, plib-tools]
sources: [_sources/python/plib_tools.json]
related: [../api/python/plib-tools.md]
last_updated: 2026-06-18
generated: true
---

# Source Reference: Python plib_tools.json

This page is auto-generated from `_sources/python/plib_tools.json`. It is the lossless wiki mirror for this source.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.

## Source Metadata

- Source path: `_sources/python/plib_tools.json`
- Canonical SHA256: `ee2090995a5da695e0ce2f2f61b9007b6468071b3135ad2d8f18019fee2d9295`
- Top-level keys: `s2mpj_load`, `s2mpj_select`, `get_plib_config`, `set_plib_config`, `pycutest_load`, `pycutest_select`

## Path Index

| Path | Kind |
|---|---|
| `s2mpj_load` | dict[6] |
| `s2mpj_select` | dict[6] |
| `get_plib_config` | dict[5] |
| `set_plib_config` | dict[4] |
| `pycutest_load` | dict[1] |
| `pycutest_select` | dict[1] |

## s2mpj_load

```json
{
  "description": "Convert an S2MPJ problem name to a `Problem` instance.",
  "notes": "There are two ways to obtain the ``problem_name`` you want:  1. Use `s2mpj_select` to get the problem names that satisfy your criteria. 2. Look for the CSV file ``probinfo_python.csv`` in the same directory as this module. It contains information about all problems in S2MPJ.  The problem name may appear in the form ``'PROBLEMNAME_n_m'`` where ``n`` is the dimension and ``m`` is the number of linear and nonlinear constraints. This happens when a problem accepts extra arguments to change its dimension or number of constraints. This information is stored in the ``probinfo_python.csv`` file.",
  "parameters": {
    "problem_name": {
      "description": "Name of the problem in the S2MPJ collection. More details about S2MPJ can be found at `the official repository <https://github.com/GrattonToint/S2MPJ>`_.",
      "type": "str"
    }
  },
  "returns": {
    "": {
      "description": "A ``Problem`` instance corresponding to the named problem.",
      "type": "optiprofiler.Problem"
    }
  },
  "see_also": [
    {
      "description": "Select problems from S2MPJ by criteria.",
      "name": [
        "s2mpj_select",
        null
      ]
    }
  ],
  "signature": "(problem_name, *args)"
}
```

## s2mpj_select

```json
{
  "description": "Select problems from S2MPJ that satisfy given criteria.",
  "notes": "1. All information about the problems can be found in the CSV file ``probinfo_python.csv`` in the same directory as this module.  2. The problem name may appear in the form ``'PROBLEMNAME_n_m'`` where ``n`` is the dimension and ``m`` is the number of constraints. This happens when a problem accepts extra arguments to change its dimension or number of constraints.  3. There is a file ``config.txt`` in the same directory as this module. It can be used to set the options ``variable_size`` and ``test_feasibility_problems``. See the comments in ``config.txt`` for details. You can also override these options at runtime using `set_plib_config` or by setting environment variables ``S2MPJ_VARIABLE_SIZE`` and ``S2MPJ_TEST_FEASIBILITY_PROBLEMS``. Environment variables take precedence over ``config.txt``.",
  "parameters": {
    "options": {
      "default": "``'ubln'``",
      "description": "A dictionary containing selection criteria. More details about S2MPJ can be found at `the official repository <https://github.com/GrattonToint/S2MPJ>`_. Supported keys:  - **ptype** (*str*) -- Type of problems to select. A string consisting of any combination of ``'u'`` (unconstrained), ``'b'`` (bound constrained), ``'l'`` (linearly constrained), and ``'n'`` (nonlinearly constrained), such as ``'b'``, ``'ul'``, ``'ubn'``. Default is ``'ubln'``. - **mindim** (*int*) -- Minimum dimension. Default is ``1``. - **maxdim** (*int*) -- Maximum dimension. Default is ``inf``. - **minb** (*int*) -- Minimum number of bound constraints. Default is ``0``. - **maxb** (*int*) -- Maximum number of bound constraints. Default is ``inf``. - **minlcon** (*int*) -- Minimum number of linear constraints. Default is ``0``. - **maxlcon** (*int*) -- Maximum number of linear constraints. Default is ``inf``. - **minnlcon** (*int*) -- Minimum number of nonlinear constraints. Default is ``0``. - **maxnlcon** (*int*) -- Maximum number of nonlinear constraints. Default is ``inf``. - **mincon** (*int*) -- Minimum total number of linear and nonlinear constraints. Default is ``0``. - **maxcon** (*int*) -- Maximum total number of linear and nonlinear constraints. Default is ``inf``. - **oracle** (*int*) -- Oracle provided by the problem. ``0`` means zeroth-order, ``1`` means first-order, ``2`` means second-order. Default is ``0``. - **excludelist** (*list of str*) -- List of problem names to exclude. Default is ``[]``.",
      "type": "dict"
    }
  },
  "returns": {
    "": {
      "description": "Problem names that satisfy the given criteria.",
      "type": "list of str"
    }
  },
  "see_also": [
    {
      "description": "Load a problem from S2MPJ.",
      "name": [
        "s2mpj_load",
        null
      ]
    },
    {
      "description": "Read the current configuration.",
      "name": [
        "optiprofiler.get_plib_config",
        null
      ]
    },
    {
      "description": "Override configuration at runtime.",
      "name": [
        "optiprofiler.set_plib_config",
        null
      ]
    }
  ],
  "signature": "(options)"
}
```

## get_plib_config

```json
{
  "description": "Read the current configuration of a problem library. The returned dictionary reflects the effective values: if an environment variable ``<PLIB>_<VARIABLE>`` (all upper-case) has been set (e.g.  via `set_plib_config`), it takes precedence over the value in the library's ``config.txt``.",
  "parameters": {
    "plib": {
      "description": "Name of the problem library (e.g. ``'s2mpj'``, ``'pycutest'``).",
      "type": "str"
    },
    "verbose": {
      "default": "``False``",
      "description": "If ``True``, the full contents of ``config.txt`` (including comments) are printed so that the user can see all available options and their descriptions. Default is ``False``.",
      "type": "bool"
    }
  },
  "raises": [
    {
      "description": "If the problem library does not have a ``config.txt`` file.",
      "exception": "FileNotFoundError"
    }
  ],
  "returns": {
    "": {
      "description": "Effective configuration as ``{variable_name: value}``.",
      "type": "dict"
    }
  },
  "signature": "(plib, verbose=False)"
}
```

## set_plib_config

```json
{
  "description": "Override configuration variables for a problem library. Each keyword argument is translated to the environment variable ``<PLIB>_<VARIABLE>`` (all upper-case) so that subsequent calls to the library's ``select`` function (directly or through `benchmark`) will pick up the new value.  The override persists for the lifetime of the current Python process.",
  "parameters": {
    "plib": {
      "description": "Name of the problem library (e.g. ``'s2mpj'``, ``'pycutest'``).",
      "type": "str"
    }
  },
  "raises": [
    {
      "description": "If the problem library does not have a ``config.txt`` file.",
      "exception": "FileNotFoundError"
    },
    {
      "description": "If a variable name is not recognised.",
      "exception": "ValueError"
    }
  ],
  "signature": "(plib, **kwargs)"
}
```

## pycutest_load

```json
{
  "description": "Load a PyCUTEst problem. Requires pycutest package (Linux/macOS)."
}
```

## pycutest_select

```json
{
  "description": "Select PyCUTEst problems matching criteria. Requires pycutest package."
}
```

## Canonical JSON Mirror

```json
{
  "get_plib_config": {
    "description": "Read the current configuration of a problem library. The returned dictionary reflects the effective values: if an environment variable ``<PLIB>_<VARIABLE>`` (all upper-case) has been set (e.g.  via `set_plib_config`), it takes precedence over the value in the library's ``config.txt``.",
    "parameters": {
      "plib": {
        "description": "Name of the problem library (e.g. ``'s2mpj'``, ``'pycutest'``).",
        "type": "str"
      },
      "verbose": {
        "default": "``False``",
        "description": "If ``True``, the full contents of ``config.txt`` (including comments) are printed so that the user can see all available options and their descriptions. Default is ``False``.",
        "type": "bool"
      }
    },
    "raises": [
      {
        "description": "If the problem library does not have a ``config.txt`` file.",
        "exception": "FileNotFoundError"
      }
    ],
    "returns": {
      "": {
        "description": "Effective configuration as ``{variable_name: value}``.",
        "type": "dict"
      }
    },
    "signature": "(plib, verbose=False)"
  },
  "pycutest_load": {
    "description": "Load a PyCUTEst problem. Requires pycutest package (Linux/macOS)."
  },
  "pycutest_select": {
    "description": "Select PyCUTEst problems matching criteria. Requires pycutest package."
  },
  "s2mpj_load": {
    "description": "Convert an S2MPJ problem name to a `Problem` instance.",
    "notes": "There are two ways to obtain the ``problem_name`` you want:  1. Use `s2mpj_select` to get the problem names that satisfy your criteria. 2. Look for the CSV file ``probinfo_python.csv`` in the same directory as this module. It contains information about all problems in S2MPJ.  The problem name may appear in the form ``'PROBLEMNAME_n_m'`` where ``n`` is the dimension and ``m`` is the number of linear and nonlinear constraints. This happens when a problem accepts extra arguments to change its dimension or number of constraints. This information is stored in the ``probinfo_python.csv`` file.",
    "parameters": {
      "problem_name": {
        "description": "Name of the problem in the S2MPJ collection. More details about S2MPJ can be found at `the official repository <https://github.com/GrattonToint/S2MPJ>`_.",
        "type": "str"
      }
    },
    "returns": {
      "": {
        "description": "A ``Problem`` instance corresponding to the named problem.",
        "type": "optiprofiler.Problem"
      }
    },
    "see_also": [
      {
        "description": "Select problems from S2MPJ by criteria.",
        "name": [
          "s2mpj_select",
          null
        ]
      }
    ],
    "signature": "(problem_name, *args)"
  },
  "s2mpj_select": {
    "description": "Select problems from S2MPJ that satisfy given criteria.",
    "notes": "1. All information about the problems can be found in the CSV file ``probinfo_python.csv`` in the same directory as this module.  2. The problem name may appear in the form ``'PROBLEMNAME_n_m'`` where ``n`` is the dimension and ``m`` is the number of constraints. This happens when a problem accepts extra arguments to change its dimension or number of constraints.  3. There is a file ``config.txt`` in the same directory as this module. It can be used to set the options ``variable_size`` and ``test_feasibility_problems``. See the comments in ``config.txt`` for details. You can also override these options at runtime using `set_plib_config` or by setting environment variables ``S2MPJ_VARIABLE_SIZE`` and ``S2MPJ_TEST_FEASIBILITY_PROBLEMS``. Environment variables take precedence over ``config.txt``.",
    "parameters": {
      "options": {
        "default": "``'ubln'``",
        "description": "A dictionary containing selection criteria. More details about S2MPJ can be found at `the official repository <https://github.com/GrattonToint/S2MPJ>`_. Supported keys:  - **ptype** (*str*) -- Type of problems to select. A string consisting of any combination of ``'u'`` (unconstrained), ``'b'`` (bound constrained), ``'l'`` (linearly constrained), and ``'n'`` (nonlinearly constrained), such as ``'b'``, ``'ul'``, ``'ubn'``. Default is ``'ubln'``. - **mindim** (*int*) -- Minimum dimension. Default is ``1``. - **maxdim** (*int*) -- Maximum dimension. Default is ``inf``. - **minb** (*int*) -- Minimum number of bound constraints. Default is ``0``. - **maxb** (*int*) -- Maximum number of bound constraints. Default is ``inf``. - **minlcon** (*int*) -- Minimum number of linear constraints. Default is ``0``. - **maxlcon** (*int*) -- Maximum number of linear constraints. Default is ``inf``. - **minnlcon** (*int*) -- Minimum number of nonlinear constraints. Default is ``0``. - **maxnlcon** (*int*) -- Maximum number of nonlinear constraints. Default is ``inf``. - **mincon** (*int*) -- Minimum total number of linear and nonlinear constraints. Default is ``0``. - **maxcon** (*int*) -- Maximum total number of linear and nonlinear constraints. Default is ``inf``. - **oracle** (*int*) -- Oracle provided by the problem. ``0`` means zeroth-order, ``1`` means first-order, ``2`` means second-order. Default is ``0``. - **excludelist** (*list of str*) -- List of problem names to exclude. Default is ``[]``.",
        "type": "dict"
      }
    },
    "returns": {
      "": {
        "description": "Problem names that satisfy the given criteria.",
        "type": "list of str"
      }
    },
    "see_also": [
      {
        "description": "Load a problem from S2MPJ.",
        "name": [
          "s2mpj_load",
          null
        ]
      },
      {
        "description": "Read the current configuration.",
        "name": [
          "optiprofiler.get_plib_config",
          null
        ]
      },
      {
        "description": "Override configuration at runtime.",
        "name": [
          "optiprofiler.set_plib_config",
          null
        ]
      }
    ],
    "signature": "(options)"
  },
  "set_plib_config": {
    "description": "Override configuration variables for a problem library. Each keyword argument is translated to the environment variable ``<PLIB>_<VARIABLE>`` (all upper-case) so that subsequent calls to the library's ``select`` function (directly or through `benchmark`) will pick up the new value.  The override persists for the lifetime of the current Python process.",
    "parameters": {
      "plib": {
        "description": "Name of the problem library (e.g. ``'s2mpj'``, ``'pycutest'``).",
        "type": "str"
      }
    },
    "raises": [
      {
        "description": "If the problem library does not have a ``config.txt`` file.",
        "exception": "FileNotFoundError"
      },
      {
        "description": "If a variable name is not recognised.",
        "exception": "ValueError"
      }
    ],
    "signature": "(plib, **kwargs)"
  }
}
```
