# Wiki Change Log

## [2026-06-07] update | Source-backed reference mirrors

Added generated `wiki/reference/` pages and coverage tooling so the wiki
remains structured without losing raw facts. `scripts/sync_wiki_reference.py`
now mirrors `_sources/**/*.json`, `_sources/refs/*.md`, enum JSON, and
bundled legacy Markdown docs/examples into source-backed wiki pages.
`scripts/audit_wiki_coverage.py` fails when those pages are missing or
stale.

This closes the failure mode where narrative pages compressed a detailed
source fact, such as feature-dependent `distribution` choices and callable
contracts, into an underspecified summary.

## [2026-06-05] sync | OptiProfiler 1.1 docs/API sync

Synced agent knowledge against the local `optiprofiler` repository at
commit `e6fd6f3` (`Modify the docs`), with source checks against
`README.rst`, `doc/source/user/*.rst`, Python `profiles.py`,
`profile_utils.py`, `opclasses.py`, MATLAB `benchmark.m`,
`getDefaultProfileOptions.m`, and wrapper examples.

**Updated facts:**
- Python install now documents both `pip install optiprofiler` and
  `conda install conda-forge::optiprofiler`.
- MATLAB install documents non-interactive `setup(struct('install_matcutest', ...))`
  and notes MatCUTEst is Linux-only.
- `n_jobs` default is conservative auto: about half of available workers,
  with at least 2 when possible.
- MATLAB `draw_hist_plots` default is `parallel` in normal runs; `load`
  mode forces `sequential`. This follows current source even though some
  generated text had stale wording.
- `test_log/report.txt` now records selected problems, timing,
  `merit_init = phi(x_0)` degenerate cases, abnormal solver terminations,
  output fallbacks, and solver scores.
- Public Python exports are limited to `benchmark`, `Problem`, `Feature`,
  `FeaturedProblem`, `show_versions`, `get_plib_config`, and
  `set_plib_config`; `s2mpj_load` / `pycutest_select` are adapter-level
  internals, not package-root imports.
- Custom solver docs now include SciPy `NonlinearConstraint` conversion
  for `ptype='n'` and MATLAB `fmincon` conversion via
  `@(x) deal(cub(x), ceq(x))`.

## [2025-04-13] migrate | Initial Wiki Migration

Migrated knowledge base from flat directory structure to LLM Wiki pattern.

**Created pages:**
- concepts/: dfo, benchmark-function, solver-interface, problem-types, features
- api/python/: benchmark, problem-class, plib-tools
- api/matlab/: benchmark
- guides/: quickstart-python, quickstart-matlab, custom-solver
- profiles/: methodology, performance-profile, data-profile, log-ratio-profile, feature-effects
- solvers/: overview, newuoa, bobyqa, cobyla, nelder-mead, powell, prima
- troubleshooting/: common-errors, solver-compat, timeout-issues

**Sources migrated:**
- common/concepts.md → split into concepts/dfo.md, concepts/benchmark-function.md, concepts/problem-types.md
- common/solver_interface.md → concepts/solver-interface.md
- python/benchmark.json, classes.json, api_notes.json, plib_tools.json → _sources/python/
- matlab/benchmark.json, classes.json, api_notes.json, plib_tools.json → _sources/matlab/
- profiles/*.md → wiki/profiles/ (split and interlinked)
- debugging/*.md → wiki/troubleshooting/ (enriched with cross-references)
- profiles/solver_traits.md → wiki/solvers/ (split into per-solver pages)

**Architecture:**
- Three-layer structure: _sources/ (raw) → wiki/ (compiled) → SCHEMA.md (rules)
- YAML frontmatter with tags, sources, related, last_updated
- Cross-references via relative markdown links
- index.md as master catalog for two-stage RAG retrieval
