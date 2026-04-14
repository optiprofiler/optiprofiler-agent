# Wiki Change Log

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
