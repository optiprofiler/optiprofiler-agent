---
tags: [platform, workflow, sandbox, agents]
sources: [_sources/platform/platform-docs.md, _sources/platform/manifest.json, ../../../../docs/PLATFORM_INTEGRATION.md]
related: [platform/ecosystem-agent-role.md, troubleshooting/common-errors.md, profiles/methodology.md]
last_updated: 2026-06-18
---

# OptiProfiler Platform Overview

OptiProfiler Platform is the hosted surface around the OptiProfiler package.
Users upload solvers, choose benchmark settings, and the backend runs
`benchmark()` in an isolated worker/sandbox pipeline. The agent system extends
that flow with three roles: Advisor for workflow questions, Debugger for failed
submissions, and Interpreter for successful benchmark results.

## Online Workflow

The main submission path is:

1. The browser posts a solver and benchmark form to `POST /api/submit`.
2. The backend validates the upload and creates a queued task.
3. A Celery worker acquires CPU slots and spawns the language runner.
4. The runner calls OptiProfiler `benchmark()` and writes `result.json` plus
   the OptiProfiler output tree.
5. The task page exposes downloads, profile PDFs, logs, and optional AI
   analysis through Agent C.

Python hosted runs use the platform Python runner and sandbox image. MATLAB
support is a separate runner path; whether it is Docker-isolated or host-direct
depends on the deployment and license configuration. Agent-side knowledge
should therefore describe MATLAB platform behavior as deployment-gated rather
than assuming every public instance has a MATLAB sandbox.

## Agent Touchpoints

Advisor appears in the chat widget through `/api/chat/message`. It uses the
agent wiki/RAG knowledge base to answer OptiProfiler package and platform
workflow questions.

Debugger is invoked after a failed sandbox run when auto-debug is enabled. The
platform sends the original solver source and traceback to Agent B, receives a
diagnostic report plus an optional patched solver, re-runs the platform upload
gate on any patch, and then performs at most one retry.

Interpreter is pull-based and cached. `POST /api/task/{id}/interpret` invokes
Agent C on the task result directory, stores the Markdown report on the task,
and returns cached output on later calls unless refresh is requested.

## Leaderboard

The leaderboard is a curated benchmark mode built from frozen combos. A combo
pins language, problem library, problem subset, feature, dimensions, evaluation
budget, and baseline solvers. Current scoring uses pairwise data-profile runs:
the user solver is benchmarked against each baseline in separate two-solver
OptiProfiler calls, then the platform averages the user score across opponents.

This avoids the invalid single-solver-cache design because OptiProfiler's
convergence threshold depends on the best merit value observed among the
solvers in the current benchmark run.

## Maintenance Implication

Platform knowledge is maintained as its own source domain under
`_sources/platform/`. Do not mix platform deployment facts into package API JSON
unless the fact actually comes from the OptiProfiler package. After platform
docs change, run `python scripts/sync_knowledge.py` so the source snapshot,
generated reference pages, audit, and wiki lint stay in sync.
