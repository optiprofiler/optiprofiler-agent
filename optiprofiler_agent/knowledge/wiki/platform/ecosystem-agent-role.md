---
tags: [platform, ecosystem, agent-roadmap, loop-engineering]
sources: [_sources/platform/platform-docs.md, _sources/platform/manifest.json, ../../../../docs/ROADMAP.md, ../../../../docs/AGENT_WORKFLOWS.md]
related: [platform/overview.md, guides/custom-problem-library-python.md, guides/custom-feature.md, profiles/methodology.md]
last_updated: 2026-06-18
---

# Agent Role In The DFO Ecosystem

The first mission of OPA was narrow: be an expert assistant that understands
the OptiProfiler package well enough to answer user questions and generate
benchmark scripts. The platform broadens that mission. OPA now sits next to
live submissions, failed solver code, generated benchmark artifacts, and a
curated leaderboard. Its long-term role is to become the operating layer for a
DFO benchmarking ecosystem.

## Current Platform Role

Today the agent should support three production workflows:

- Explain OptiProfiler package concepts, API options, solver signatures, and
  profile interpretation.
- Debug uploaded solver code after platform failures, while respecting the
  platform upload gate and sandbox threat model.
- Interpret successful benchmark outputs into grounded reports that cite the
  observed scores, profiles, failures, and output fallbacks.

The agent must separate package facts from platform facts. Package facts come
from OptiProfiler source/docstrings and live under `_sources/python`,
`_sources/matlab`, and generated API reference pages. Platform facts come from
the platform repository and live under `_sources/platform`.

## Ecosystem Expansion

The platform's planned ecosystem has three core asset families:

- Data: problem libraries, problem subsets, metadata, provenance, citations,
  and resource envelopes.
- Solvers: expert-provided solver wrappers, solver families, capability flags,
  language/runtime dependencies, and reproducibility metadata.
- Benchmarking tools: profile methods, score functions, leaderboard combo
  specs, report modules, and future robustness/failure analyses.

OPA should become the assistant that can reason across those assets. A user
should eventually be able to ask why a solver lost on a combo, which problem
library is appropriate for a use case, how to wrap a new solver, or how to add
a new report module without reading every repo.

## Loop Engineering Direction

Once problem libraries, solver wrappers, and scoring modules are structured,
the agent can support loop engineering: propose a solver change, run or request
a benchmark, inspect the result, diagnose failure modes, and propose the next
change. This is not just chat. It requires versioned experiment records,
grounded result interpretation, reproducible solver patches, and clear stop
conditions so the loop does not optimize against noise or overfit a benchmark
subset.

Near-term agent work should therefore keep interfaces boring and auditable:
source-backed knowledge, explicit tool calls, deterministic eval gates, and
reports that distinguish measured facts from recommendations.

## Maintenance Rule

Every ecosystem-facing feature should add or update the knowledge source it
depends on. New platform APIs, sandbox modes, leaderboard formulas, module
registry rules, problem-library intake rules, and solver contribution policies
belong in platform docs first, then in `_sources/platform`, then in narrative
wiki pages. This keeps OPA useful as the platform grows without relying on
stale prompt memory.
