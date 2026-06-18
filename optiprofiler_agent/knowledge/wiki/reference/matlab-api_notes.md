---
tags: [reference, source-backed, matlab, api-notes]
sources: [_sources/matlab/api_notes.json]
related: []
last_updated: 2026-06-18
generated: true
---

# Source Reference: Matlab api_notes.json

This page is auto-generated from `_sources/matlab/api_notes.json`. It is the lossless wiki mirror for this source.
Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.

## Source Metadata

- Source path: `_sources/matlab/api_notes.json`
- Canonical SHA256: `e4a4f13701602e8f41493b3326cdebab4b564e722a7843d30a5b627bbbd91fa7`
- Top-level keys: `language`, `solver_format`, `options_format`, `vector_convention`, `problem_libs`, `matcutest_note`, `setup`, `differences_from_python`

## Path Index

| Path | Kind |
|---|---|
| `language` | str |
| `solver_format` | str |
| `options_format` | str |
| `vector_convention` | str |
| `problem_libs` | list[2] |
| `matcutest_note` | str |
| `setup` | dict[3] |
| `differences_from_python` | dict[5] |

## language

```text
MATLAB
```

## solver_format

```text
cell array of function handles: {@solver1, @solver2}
```

## options_format

```text
struct with fields: options.ptype = 'u'
```

## vector_convention

```text
column vectors (n×1 matrices)
```

## problem_libs

```json
[
  "s2mpj",
  "matcutest"
]
```

## matcutest_note

```text
matcutest is only available on Linux
```

## setup

```json
{
  "interactive": "setup",
  "noninteractive": "setup(struct('install_matcutest', true)) or setup(struct('install_matcutest', false))",
  "uninstall": "setup uninstall"
}
```

## differences_from_python

```json
{
  "draw_hist_plots_default": "'parallel' in normal runs; load mode forces 'sequential'",
  "line_colors_default": "MATLAB 'gem' colororder (Python: matplotlib tab10)",
  "maxdim_default": "mindim + 10 (Python: mindim + 1)",
  "no_custom_problem_libs_path": "MATLAB uses folder structure instead",
  "solvers_to_load": "1-indexed (Python: 0-indexed)"
}
```

## Canonical JSON Mirror

```json
{
  "differences_from_python": {
    "draw_hist_plots_default": "'parallel' in normal runs; load mode forces 'sequential'",
    "line_colors_default": "MATLAB 'gem' colororder (Python: matplotlib tab10)",
    "maxdim_default": "mindim + 10 (Python: mindim + 1)",
    "no_custom_problem_libs_path": "MATLAB uses folder structure instead",
    "solvers_to_load": "1-indexed (Python: 0-indexed)"
  },
  "language": "MATLAB",
  "matcutest_note": "matcutest is only available on Linux",
  "options_format": "struct with fields: options.ptype = 'u'",
  "problem_libs": [
    "s2mpj",
    "matcutest"
  ],
  "setup": {
    "interactive": "setup",
    "noninteractive": "setup(struct('install_matcutest', true)) or setup(struct('install_matcutest', false))",
    "uninstall": "setup uninstall"
  },
  "solver_format": "cell array of function handles: {@solver1, @solver2}",
  "vector_convention": "column vectors (n×1 matrices)"
}
```
