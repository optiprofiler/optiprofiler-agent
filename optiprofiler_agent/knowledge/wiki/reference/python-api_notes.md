---
tags: [reference, source-backed, python, api-notes]
sources: [_sources/python/api_notes.json]
related: []
last_updated: 2026-06-18
generated: true
---

# Source Reference: Python api_notes.json

This page is auto-generated from `_sources/python/api_notes.json`. It is the lossless wiki mirror for this source.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.

## Source Metadata

- Source path: `_sources/python/api_notes.json`
- Canonical SHA256: `36a447405e55a7d96e28b669d50ef2b4cf580e67767cc5e0375f376107bb7002`
- Top-level keys: `language`, `public_exports`, `solver_format`, `options_format`, `vector_convention`, `problem_libs`, `python_only_options`, `installation`, `pycutest_note`, `lambda_warning`

## Path Index

| Path | Kind |
|---|---|
| `language` | str |
| `public_exports` | list[7] |
| `solver_format` | str |
| `options_format` | str |
| `vector_convention` | str |
| `problem_libs` | list[4] |
| `python_only_options` | list[1] |
| `installation` | dict[2] |
| `pycutest_note` | str |
| `lambda_warning` | str |

## language

```text
Python
```

## public_exports

```json
[
  "benchmark",
  "Problem",
  "Feature",
  "FeaturedProblem",
  "show_versions",
  "get_plib_config",
  "set_plib_config"
]
```

## solver_format

```text
list of callables: [solver1, solver2]
```

## options_format

```text
keyword arguments to benchmark()
```

## vector_convention

```text
1-D numpy arrays, shape (n,)
```

## problem_libs

```json
[
  "s2mpj",
  "pycutest",
  "solar",
  "custom"
]
```

## python_only_options

```json
[
  "custom_problem_libs_path"
]
```

## installation

```json
{
  "conda_forge": "conda install conda-forge::optiprofiler",
  "pip": "pip install optiprofiler"
}
```

## pycutest_note

```text
Requires separate installation; Linux and macOS only
```

## lambda_warning

```text
Lambda functions are not picklable — use named functions (def) for parallel execution (n_jobs > 1)
```

## Canonical JSON Mirror

```json
{
  "installation": {
    "conda_forge": "conda install conda-forge::optiprofiler",
    "pip": "pip install optiprofiler"
  },
  "lambda_warning": "Lambda functions are not picklable — use named functions (def) for parallel execution (n_jobs > 1)",
  "language": "Python",
  "options_format": "keyword arguments to benchmark()",
  "problem_libs": [
    "s2mpj",
    "pycutest",
    "solar",
    "custom"
  ],
  "public_exports": [
    "benchmark",
    "Problem",
    "Feature",
    "FeaturedProblem",
    "show_versions",
    "get_plib_config",
    "set_plib_config"
  ],
  "pycutest_note": "Requires separate installation; Linux and macOS only",
  "python_only_options": [
    "custom_problem_libs_path"
  ],
  "solver_format": "list of callables: [solver1, solver2]",
  "vector_convention": "1-D numpy arrays, shape (n,)"
}
```
