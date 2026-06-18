# OptiProfiler Platform Source Snapshot

This file is generated from the local `optiprofiler-platform` repository.
Do not hand-edit it; run `python scripts/extract_platform_knowledge.py`.

## Snapshot Metadata

- Source repository: `optiprofiler-platform`
- Git commit: `827e9e7`
- Worktree status: `M backend/app/services/language_backends/__init__.py;  M backend/app/services/language_backends/base.py;  M backend/app/services/leaderboard.py;  M docs/README.md;  M docs/adr/0010-multi-language-backend.md; ?? docs/problem-libraries-industrial-dfo.md`

## Included Sources

| Path | Status | Bytes |
|---|---:|---:|
| `README.md` | included | 4306 |
| `docs/api.md` | included | 4174 |
| `docs/architecture.md` | included | 7223 |
| `docs/leaderboard.md` | included | 15512 |
| `docs/adr/0005-agent-c-integration.md` | included | 3468 |
| `docs/adr/0006-chat-widget.md` | included | 4556 |
| `docs/adr/0007-auto-debug.md` | included | 5524 |
| `docs/adr/0008-leaderboard.md` | included | 16796 |
| `docs/adr/0009-matlab-solver-upload.md` | included | 8477 |
| `docs/adr/0010-multi-language-backend.md` | included | 10759 |
| `docs/adr/0011-leaderboard-pairwise-scoring.md` | included | 11195 |
| `docs/adr/0012-matlab-cli-sandbox.md` | included | 19088 |
| `docs/adr/0013-dfo-ecosystem-module-registry.md` | included | 5851 |
| `docs/problem-libraries-industrial-dfo.md` | included | 11942 |

## Source: README.md

```markdown
# OptiProfiler Platform

Online sandbox testing platform — users upload solvers and the platform runs `benchmark()` in Docker sandboxes and returns results.

## Architecture

```
Browser (app.optprof.com)
  │
  ├─ Upload solver file (.py / .zip)
  ├─ Select test configuration (problem type, dimension range, feature)
  └─ View task status & results (live log streaming)
      │
      ▼
Backend API (FastAPI)
  ├─ Auth (GitHub OAuth) + rate limiting
  ├─ File validation (AST whitelist)
  └─ Submit to task queue
      │
      ▼
Task Queue (Celery + Redis; hosted users get per-task wall-clock limits,
maintainers in OP_DEV_GITHUB_LOGINS do not — ADR 0001)
  │                    ┌────────────────────────────┐
  ▼                    │ Kernel Allocator (Redis Lua)│
Celery Worker ────────►│ global_n_jobs_cap enforced  │
  │                    │ per-task min/max n_jobs     │
  │                    └────────────────────────────┘
  ▼
Sandbox Execution (subprocess / Docker)
  ├─ Pre-installed optiprofiler + scipy
  ├─ Network isolation + resource limits
  ├─ Run benchmark(n_jobs=granted)
  └─ Collect results → release slots
      │
      ▼
Storage
  ├─ SQLite / PostgreSQL (tasks / users / results)
  └─ Local disk / S3 (PDFs / plots)
```

## Project Structure

```
backend/
├── app/             # FastAPI application (routes, auth, models)
│   └── services/    # kernel_allocator, sandbox, ast_checker, …
├── workers/         # Celery workers (task scheduling, result collection)
└── sandbox/         # Docker sandbox — per-language runners, Dockerfiles,
                     # baselines (e.g. Dockerfile.python + run_task_python.py +
                     # baseline_solvers_python/, mirrored for matlab)

frontend/            # Next.js frontend (app.optprof.com)

docker/              # Docker Compose orchestration
docs/                # Architecture, ADRs, operations & deployment runbooks
```

## Development

```bash
# Backend
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev

# Full stack (Docker Compose)
docker compose up
```

### Local smoke test (chat, guest limits, auth)

1. **Backend**: ensure SQLite can be written (`backend/data/`) and optional LLM key for real replies (`MINIMAX_API_KEY` or provider in `.env` / sourced file; without it `/api/chat/message` returns 503).
2. **Frontend**: set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local` (must match the origin you open in the browser — use the same host as in `OP_CORS_ORIGINS`).
3. Open `http://localhost:3000`, use OPA chat as **guest** (3 lifetime messages), **submit** once as guest (second submit blocked after a successful one).
4. **Sign in** with GitHub (configure `OP_GITHUB_*` and callback URL). Chat should use session `u{user_id}` and **reload history** from `GET /api/chat/history` after refresh. Sign out clears the panel; sign back in to see persisted DB history again.

### Troubleshooting

**CORS / “Load failed” on submit**  
Browsers treat `http://localhost:3000` and `http://127.0.0.1:3000` as different origins. The API allows a fixed list (see `OP_CORS_ORIGINS` in `backend/.env.example`). Add any dev URL you use, comma-separated.

**500 on submit after pulling new code**  
SQLite does not auto-add new columns. The app runs lightweight `ALTER TABLE` for known columns; if issues persist, stop the API, remove `backend/data/optiprofiler.db`, and restart so the schema is recreated.

**Task stuck in `queued` for a long time**  
Tasks are only moved to `running` when a Celery worker dequeues them from Redis. If Redis was cleared (`FLUSHDB`), restarted without persistence, or the worker was off at submit time, the database row can stay `queued` forever while the broker has no message — that is **not** the benchmark “running slowly”. Use **Retry / Re-enqueue** on the Tasks page (`POST /api/task/{id}/retry`) after starting Redis + a single worker. Avoid running many duplicate `celery worker` processes.

## License

BSD-3-Clause
```

## Source: docs/api.md

```markdown
# REST API reference

All routes live at `<host>:8000`. Auth-aware routes accept `Authorization:
Bearer <jwt>`; routes that *require* auth say so explicitly.

## Health

- `GET /health` → `{"status": "ok", "service": "optiprofiler-platform"}`

## Auth (GitHub OAuth)

- `GET  /api/auth/github` — 302 to GitHub consent screen.
- `GET  /api/auth/github/callback?code=…` — exchanges code, redirects
  to frontend `/auth/callback?token=…`.
- `GET  /api/auth/me` — returns the current user (requires JWT).

## Tasks

| Method  | Path                          | Auth     | Description |
|---------|-------------------------------|----------|-------------|
| POST    | `/api/submit`                 | optional | Upload solver and enqueue benchmark. Form fields: `solver_file`, `ptype`, `mindim`, `maxdim`, `feature_name`, `max_eval_factor`, `min/max{b,lcon,nlcon}`, `n_jobs`, `auto_debug` (default `true`). Anonymous allowed but rate-limited only if logged in. |
| POST    | `/api/repro-package`          | none     | Upload solver and benchmark settings, but do not run a task. Returns a ZIP with `repro/solver.py`, baseline files, `config.json`, and `run_repro.py` for local execution. |
| GET     | `/api/tasks?limit&offset`     | **required** | List the current user's tasks. Anonymous → 401 (otherwise we'd leak everybody's submissions). Per-task endpoints below stay accessible by unguessable `task_id`. |
| GET     | `/api/task/{id}`              | none     | Get one task's metadata. |
| DELETE  | `/api/task/{id}`              | optional | Delete row + uploads + results. Owner-only when task has a user; 409 if `running`. |
| POST    | `/api/task/{id}/retry`        | optional | Re-enqueue a `queued` or `failed` task. Owner-only. |
| POST    | `/api/task/{id}/interpret?refresh` | optional | Generate or fetch cached Markdown report from Agent C. Owner-only. Returns `{report, cached}`. |
| GET     | `/api/results/{id}/download`  | none     | Stream the summary PDF. |
| GET     | `/api/results/{id}/download-all` | none  | Stream a ZIP of all output files (includes `solver_autofixed.py` and `debug_report.md` when auto-debug ran). |
| GET     | `/api/task/{id}/report.md`    | optional | Download the AI analysis as standalone Markdown (404 until generated). |
| GET     | `/api/task/{id}/fixed-solver` | optional | Download the AI-patched solver that actually ran. 404 unless `was_debugged`. Owner-only when task has a user. |
| GET     | `/api/task/{id}/debug-report.md` | optional | Download Agent B's diagnostic report. 404 unless one was produced. |

## Stats

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/stats/platform` | none | Coarse aggregate platform metrics from the app database: page views, users, tasks, recent submissions, and country-level reach. |
| POST | `/api/stats/page-view` | none | Increment coarse first-party page-view aggregates for `{path}` and, when the edge provides a country header, `country_code + UTC day`. Stores no raw IP addresses or user agents. |

## Chat (Agent A — Advisor)

| Method | Path                  | Auth     | Description |
|--------|-----------------------|----------|-------------|
| POST   | `/api/chat/message`   | optional | `{session_id, message}` → `{reply}`. Rate-limited 5/h anon, 30/h authed. |
| POST   | `/api/chat/reset`     | none     | Drop the server-side history for a session_id. |

`session_id` is opaque, generated and persisted client-side
(`localStorage`). Server keeps an LRU of `AdvisorAgent` instances
(30 min TTL, max 200 concurrent). See ADR 0006.

### Status enum

`queued | running | success | failed | timeout`

### Common error mappings

| Status | Meaning                                        |
|--------|------------------------------------------------|
| 400    | Validation (file type, ptype chars, AST check) |
| 403    | Wrong owner (delete, retry, interpret)         |
| 404    | Unknown task or missing result dir             |
| 409    | Cannot delete a running task                   |
| 429    | Daily rate limit hit                           |
| 503    | Job queue (Redis) down, or LLM unreachable     |
```

## Source: docs/architecture.md

```markdown
# Architecture

## High level

```
                    ┌──────────────┐
  Browser  ───────► │  Next.js UI  │  3000
                    │  (App Router)│
                    └──────┬───────┘
                           │ REST + JWT
                           ▼
                    ┌──────────────┐         ┌──────────────┐
                    │  FastAPI app │  8000   │   Redis      │  6379
                    │  (uvicorn)   │ ──────► │  (broker +   │
                    └──────┬───────┘         │   results)   │
                           │ SQLModel        └──────┬───────┘
                           ▼                        │ Celery proto
                    ┌──────────────┐                │
                    │   SQLite     │                ▼
                    │  data/*.db   │         ┌──────────────┐
                    └──────────────┘         │ Celery worker│
                                             │  prefork x N │
                                             └──────┬───────┘
                                                    │ subprocess
                                                    ▼
                                             ┌──────────────────┐
                                             │ Sandbox child    │
                                             │ run_task_python.py
                                             │  / run_task_matlab.sh
                                             │ (n_jobs M)       │
                                             └──────────────────┘
```

## Components

### Frontend (`frontend/`)
- Next.js 15, App Router, React 19, Tailwind.
- `src/contexts/AuthContext.tsx` holds the JWT in memory (and survives
  reloads via `localStorage`).
- All API calls go to `NEXT_PUBLIC_API_URL` (defaults to localhost:8000).

### Backend (`backend/app/`)
- FastAPI + SQLModel + SQLite (Postgres-compatible).
- Routers: `auth.py` (GitHub OAuth + JWT) and `tasks.py` (CRUD + run).
- Services:
  - `ast_checker.py` — static analysis of uploaded solvers (ADR 0004).
  - `sandbox.py` — runs `sandbox/run_task_python.py` (or
    `sandbox/run_task_matlab.sh` for MATLAB tasks) as an isolated subprocess.
  - `interpreter.py` — wraps `optiprofiler-agent` (ADR 0005).
  - `storage.py` — safe filesystem helpers (delete + size).
- `lifespan` startup hook: zombie reap → kernel allocator prune → stale
  sandbox tmp sweep → PID‑1 `multiprocessing-fork` orphan kill → TTL purge →
  quota sweep; the first four steps repeat every 10 min in the periodic reaper.
  Maintainer-owned RUNNING tasks skip the long-running zombie reap (no wall-clock
  cap for `OP_DEV_GITHUB_LOGINS`).

### Worker (`backend/workers/`)
- Celery on Redis. **Prefork pool** (ADR 0001).
  Recommended concurrency: local=4, production=6.
- Celery has **no global** task time limits in `celery_app.py`. Hosted tasks
  receive `soft_time_limit` / `time_limit` on `send_task()` from
  `_celery_time_limit_kwargs()`; maintainers in `OP_DEV_GITHUB_LOGINS` get
  no Celery cap (matches unbounded sandbox `subprocess.wait` in the worker).
  The local sandbox runner uses a `finally` block so an interrupt (including
  Celery soft time exceeded) still tears down the sandbox process group.
- Before running a benchmark, the worker acquires CPU slots from the
  **kernel allocator** (`app/services/kernel_allocator.py`) — a Redis-backed
  centralized scheduler that prevents oversubscription (ADR 0002).
- Each task spawns its own Python subprocess (`sandbox/run_task_python.py`
  for Python tasks, `sandbox/run_task_matlab.sh` for MATLAB tasks) so
  `multiprocessing` inside `optiprofiler.benchmark()` works (ADR 0002).

### Sandbox runner (`backend/sandbox/run_task_python.py` / `run_task_matlab.sh`)
- Self-contained scripts that know nothing about FastAPI / Celery.
- The Python runner loads the user solver dynamically, calls
  `optiprofiler.benchmark()`, writes `result.json` + the OptiProfiler
  output tree. The MATLAB wrapper invokes `matlab -batch` against
  `run_task_matlab.m` which produces the same `result.json` shape (see
  ADR 0012). Both follow the naming convention
  `run_task_<lang>.<ext>` ↔ `Dockerfile.<lang>` ↔
  `optiprofiler-sandbox-<lang>:latest` so a third language slots in
  cleanly.

## Data flow: a single submission

1. **POST /api/submit** with solver `.py` + form fields.
2. AST checker scans source — reject on anything dangerous (ADR 0004).
3. Rate-limit check (per-user daily cap).
4. Insert `Task` row (status=`queued`), write upload to disk.
5. Evict the user's oldest tasks if over `max_tasks_per_user` (ADR 0003).
6. Enqueue `run_benchmark_task(task.id)` to Celery.
7. Worker picks it up → status=`running` → acquires CPU slots from the
   kernel allocator (waits if global cap reached) → spawns sandbox subprocess.
8. Subprocess runs `benchmark()` with `n_jobs=M` (ADR 0002), writes results.
   On completion (success or failure), slots are released back to the allocator.
9. Worker copies results to `data/results/<task_id>/`, status=`success`.
10. Frontend polls `GET /api/tasks` every 3 s while any task is queued/running.
11. User can download PDF, ZIP, or click **AI Analysis** → `POST /api/task/{id}/interpret`
    which calls the results interpreter (OptiProfiler Agent; ADR 0005) and caches the Markdown report.

## Storage layout

```
backend/data/
├── optiprofiler.db                ← SQLite (User, Task, chat, stats/location, leaderboard rows)
├── uploads/<task_id>/solver.py    ← user submissions + sandbox.log
└── results/<task_id>/             ← benchmark output
    ├── result.json                ← summary read by the API
    └── out/<bench_run>/           ← OptiProfiler raw output
        ├── test_log/
        ├── history_plots/
        └── detailed_profiles/
```

`backend/data/` is ignored by git and excluded by the deploy helper. It is
allowed to live inside the project checkout, but code deploys must not touch
it. The repo-root `data/leaderboard/` tree is versioned seed material, not a
place for runtime uploads/results.

## Process model summary

| Process       | Count | Lifetime          | Restart trigger              |
|---------------|-------|-------------------|------------------------------|
| uvicorn       | 1     | session           | code edit (`--reload`)       |
| celery worker | 1 master + N children | session | manual restart on code edit |
| sandbox child | 1 per task | minutes      | per task                     |
| redis         | 1     | session / system  | manual                       |

A code change to `app/` is auto-picked up. A change to `workers/` or
`sandbox/` requires a worker restart — see `operations.md`.
```

## Source: docs/leaderboard.md

```markdown
# Leaderboard (Phase 3)

> Status: shipped in Phase 3, scoring overhauled per ADR 0011. See ADR
> 0008 for the high-level rationale, ADR 0010 for how this fits the
> multi-language backend, and **ADR 0011 (the operative spec)** for the
> pairwise-scoring methodology this doc describes below.

The leaderboard turns a one-shot benchmark into a recurring comparison: every
solver runs against the same frozen problem set, scores stay comparable
across submissions, and the user can climb the rank.

## How it works (one paragraph)

A **combo** is a frozen `(plib, ptype, dim range, feature, max_eval_factor,
language)` spec. Every active combo on disk has

```
backend/data/leaderboard/<combo_id>/
├── combo.json                 # spec + lib_versions snapshot
├── problems.json              # resolved {name, dim, type, n_*, problem_kwargs}
└── baselines/
    ├── pairwise.json          # K×K pairwise score matrix (ADR 0011)
    └── <solver>.json          # per-baseline row of the matrix + aggregate
```

When a user opts into a combo at submit time, the worker reads
`problems.json`, runs **K independent 2-solver benchmarks** — one per
opponent — and aggregates the user's pairwise data-profile scores into
a single mean. Baselines' scores were computed the same way at seed
time, so user and baseline rows are on the same scale and rank against
one another in a single sorted list.

## Scoring (ADR 0011, supersedes ADR 0008 §"Ranking metric")

Every leaderboard score is the **mean over opponents of pairwise
data-profile scores**. Concretely, for solver `S` competing in a combo
with opponent set `O`:

```
for each opponent T ∈ O:
    run benchmark([S, T], problems = combo.problems,
                  feature = combo.feature, …, n_runs = combo.n_runs)
    pair_score(S, T) = mean over tolerances of S's
                       history-based data-profile column

leaderboard_score(S) = mean over T of pair_score(S, T)
```

The relevant slice of OptiProfiler's 4-D `profile_scores` is
`[:, :, 0, 1]` (axis 2 = `0 history / 1 output`, axis 3 =
`0 perf / 1 data / 2 log_ratio`). The platform passes this slicing as
`score_fun=pairwise_data_profile_score_fun` so seed-time and submission
runs use identical math.

### Score range — `normalized_scores=False`

The runner explicitly passes `normalized_scores=False` to
`benchmark()` (see `backend/sandbox/run_task_python.py` and
`backend/sandbox/run_task_matlab.m`). With normalisation off, each
entry of `profile_scores` is the **integral of the data-profile curve
over the budget axis**, so individual scores are NOT bounded in
`[0, 1]`. They live in `[0, max_eval_factor]`:

| combo `max_eval_factor` | typical score range |
|--------------------------|---------------------|
| 30  (small / noisy)      | `[0, 30]`           |
| 50  (default submission) | `[0, 50]`           |
| 200 (`ubln-micro-plain`) | `[0, 200]`          |

Higher = better, ordering preserved.

We deliberately leave normalisation off because **per-pair** normalisation
would force the better solver of every pair to exactly `1.0` and the
loser to a fraction in `[0, 1]`, which destroys the magnitude
information we want to average across pairs. With normalisation off
the value retains its meaning: a solver that crushes both opponents
on the same problem set scores higher than one that barely edges past
them, even when both "win every pair".

If you ever need a `[0, 1]`-bounded display value, normalise on the
**read** side (e.g. divide by `max_eval_factor`) — that operation is
monotonic so the rank is preserved. We do this nowhere in the current
UI; the raw integral is what shows on `/leaderboard`.

### Why pairwise (and not "user runs alone, join cached baselines")

OptiProfiler's Moré-Wild convergence test
`phi(x) ≤ tau · phi(x_0) + (1 − tau) · phi_min`
uses **the merit minimum across the current solver set** as `phi_min`.
Two solvers in a 2-solver run see a different `phi_min` than three
solvers in a 3-solver run, so a "user score from a single-solver run"
cannot be compared with a "baseline score from a 4-solver run" — the
convergence threshold is different. The pairwise design fixes this by
making sure every score lives inside an experiment whose opponent set
matches the aggregation it feeds.

ADR 0011 §"Why we are revisiting ADR 0008" has the full derivation and
the failure mode of the v0 single-solver design (a no-op padding solver
that collapsed `phi_min` to `phi(x_0)` and degenerated the test).

### Future direction: lower-bound reference

A truly **absolute** score would require OptiProfiler itself to read an
external `phi_lower_bound(problem, run, k)` vector when computing
convergence thresholds. We track this as a Phase 4 upstream proposal.
Current pairwise design works without any OptiProfiler-internal change
and is good enough for "is this solver competitive with the established
baseline set".

### Sandbox banner per submission

The runner now emits a verbose pre-flight banner so users can verify
exactly what their score compared against:

```
[leaderboard] combo: ubln-micro-plain
[leaderboard] ptype=ubln dim=1-10 feature=plain max_eval_factor=200
[leaderboard] problem set: 30 pinned problems (plibs=['s2mpj'], n_jobs=1)
[leaderboard] scoring: history-based data profile, mean over tolerances
[leaderboard] opponents (2): scipy_cobyla, scipy_cobyqa
[leaderboard] running 2 pairwise 2-solver benchmarks; each pair uses an
              independent OptiProfiler call so the convergence threshold
              is well-defined within each pair
[leaderboard] pair 1/2: user_solver vs scipy_cobyla → vs_scipy_cobyla/
[leaderboard]   user_solver vs scipy_cobyla: score = 0.6212
[leaderboard] pair 2/2: user_solver vs scipy_cobyqa → vs_scipy_cobyqa/
[leaderboard]   user_solver vs scipy_cobyqa: score = 0.4287
[leaderboard] user score per opponent:
              scipy_cobyla = 0.6212
              scipy_cobyqa = 0.4287
[leaderboard] mean = 0.5249  (this is the published score)
```

Per-pair PDFs and intermediate output land under
`output/vs_<opponent>/` so users (and reviewers) can inspect the
individual pair charts after the run.

Higher score = better. Ranking ties break on earliest `submitted_at`.

## v1 combos

| combo_id              | language | plib     | ptype | dim   | feature              | problems |
|-----------------------|----------|----------|-------|-------|----------------------|---------:|
| `u-small-plain`       | python   | s2mpj    | u     | 1–20  | plain                | 50       |
| `u-small-noisy`       | python   | s2mpj    | u     | 1–20  | noisy (seed=42)      | 50       |
| `u-small-perturbed-x0`| python   | s2mpj    | u     | 1–20  | perturbed_x0         | 50       |
| `b-small-plain`       | python   | s2mpj    | b     | 1–20  | plain                | 40       |
| `ubln-micro-plain`    | python   | s2mpj    | ubln  | 1–10  | plain                | 30       |
| `u-cutest-plain`      | python   | pycutest | u     | 1–20  | plain                | 40 (gated)|

Every combo carries an explicit `language` field. Python and MATLAB
leaderboards are intentionally separate; mixing languages would compare
different language bridges and baseline ecosystems. MATLAB combos will use a
parallel `m-*` family that can share the same `(ptype, dim, feature)` axes but
hashes to a different cache slot because `language` is part of the spec.

`u-cutest-plain` is **not active** by default; activating it requires
rebuilding the sandbox image with `--build-arg WITH_PYCUTEST=1` and
re-running the seed script. Until then the API returns
`is_active=false`, the submit-page selector renders it as
"(not yet seeded)", and the public leaderboard hides it.

## Free-mode competitor defaults

The submit form can compare against multiple competitors, but hosted free-mode
starts with exactly one selected baseline so first runs stay quick:

| language | ptype | default competitor |
|----------|-------|--------------------|
| Python   | `u`   | `scipy_nelder_mead` |
| Python   | any constrained or mixed ptype | `scipy_cobyla` |
| MATLAB   | `u`   | `fminsearch` |
| MATLAB   | any constrained or mixed ptype | `fmincon` |

Additional chips are grouped by solver family in the submit form
(`SciPy`, `PDFO`, `PRIMA`, and MATLAB built-ins). Python PDFO wrappers exist for
`pdfo`, `pdfo_uobyqa`, `pdfo_newuoa`, `pdfo_bobyqa`, `pdfo_lincoa`, and
`pdfo_cobyla`, but they are hidden unless the deployment sets
`OP_ENABLE_PDFO_COMPETITORS=1` and the Python sandbox image has `pdfo`
installed. **PDFO is Python-only** — there is no MATLAB `pdfo` competitor,
because PDFO and PRIMA export the same MATLAB function names and collide on one
path; PRIMA supersedes PDFO, so MATLAB ships PRIMA only. This keeps local Mac
development and unprepared sandboxes from advertising buttons that cannot run.

Independent PRIMA is exposed as its own competitor family, separate from SciPy.
Python entries are `prima`, `prima_uobyqa`, `prima_newuoa`, `prima_bobyqa`,
`prima_lincoa`, and `prima_cobyla`; MATLAB exposes `prima`. They are hidden
unless the deployment installs the official PRIMA package, passes the upstream
smoke tests, and sets the matching capability flag
(`OP_ENABLE_PRIMA_COMPETITORS=1` for Python; for MATLAB,
`OP_ENABLE_MATLAB_PRIMA_COMPETITORS=1` plus `OP_MATLAB_PRIMA_ROOT` pointing at a
clean built `libprima/prima` checkout). Python PRIMA uses the
official SciPy-like `prima.minimize(...)` interface; MATLAB PRIMA uses the
official fmincon-compatible `prima(...)` interface. Neither is an alias for
SciPy `COBYLA` or `COBYQA`.

## Seeding

The seed script lives at `scripts/seed_leaderboard.py`. ADR 0011 changed
its content (not its CLI):

1. Discover problems for each combo's spec (via a tiny `optiprofiler`
   "trivial-solver" run).
2. Sample deterministically down to the combo's `problem_limit`
   (alphabetical, so the manifest is stable).
3. **Run K·(K−1)/2 baseline-vs-baseline pairwise benchmarks** to
   populate the baselines/pairwise.json matrix. Each baseline's
   aggregate score is the mean of its row.
4. Write `combo.json`, `problems.json`, `baselines/pairwise.json`,
   plus per-baseline `<solver>.json` files (which carry that baseline's
   row of the matrix + the aggregate).
5. Upsert the `LeaderboardCombo` + `LeaderboardEntry` (baseline) rows;
   each entry's `pairwise_scores_json` carries its row of the matrix
   so the API returns the per-opponent breakdown without re-reading
   disk.

```bash
cd optiprofiler-platform
/path/to/optiprofiler/venv/bin/python3 scripts/seed_leaderboard.py [--combo u-small-plain] [--skip-pycutest] [--max-eval-factor 30]
```

Run with `--skip-pycutest` until the CUTEst-enabled image is built.

Cost projection (typical numbers per combo with K=4 baselines):

* discovery: a few seconds
* K·(K−1)/2 = 6 pairwise benchmarks × 50 problems × 200 max-eval-factor
  ≈ 8–12 minutes per combo

Use `--max-eval-factor 30` to smoke-test pairwise plumbing in ~1 minute
per combo.

## Submission flow (user opt-in)

1. The submit page exposes a "Submit to leaderboard" combo selector.
   When the user picks one, the rest of the benchmark configuration form
   is visually locked — those settings come from the combo and we
   ignore whatever the form posts.
2. The backend overrides `ptype/dim/feature/max_eval_factor/plib` from
   the combo's `ComboSpec`, sets `task.leaderboard_combo_id`, and queues
   the task.
3. The worker reads the manifest, the combo's baselines, and injects
   `opponent_solver_names = baselines_for(spec.ptype)`,
   `problem_names`, and `leaderboard_mode=True` into the runner config.
4. The runner runs **K independent 2-solver benchmarks** (user vs each
   opponent), collects the user's score from each pair, and stores the
   per-opponent dict + the mean in `result.json`.
5. On success, `_promote_to_leaderboard_entry` writes a
   `LeaderboardEntry` row with `is_baseline=False`, `user_id` set to
   the submitter (None for guests), and `public` set to whatever the
   user chose. The aggregate `scores_median` is the mean of the
   pairwise dict; `scores_min` / `scores_max` describe the range over
   opponents (e.g. "you beat baseline A by 0.6 but only 0.2 over
   baseline B"). Re-trying the same task updates the existing row in
   place — no duplicates appear on the leaderboard.

## Tasks page integration

The `/tasks` page calls `GET /api/leaderboard/by-task/<task_id>` for
every successful task whose `leaderboard_combo_id` is set. The response
carries:

* `rank` (this task's place on the public leaderboard, 1-indexed) and
  `n_entries` (current leaderboard size including baselines + public users).
* `pairwise_scores` — the same per-opponent dict the leaderboard page
  shows, so a user can see at a glance which opponent they did
  best/worst against.
* a deep-link to the combo's public leaderboard.

This is the answer to the v0 UX gap "the task page only shows my
solver's score with no comparison": the comparison is now first-class.

## API

* `GET /api/leaderboard` — list all known combos (active + not-yet-seeded).
* `GET /api/leaderboard/<combo_id>?limit=50` — ranked entries
  (descending by `scores_median`). Each row carries
  `pairwise_scores` and `n_opponents`.
* `GET /api/leaderboard/by-task/<task_id>` — combo + rank + pairwise
  breakdown for a specific user task. Returns 404 if the task isn't
  on the leaderboard (failed, withdrawn, or non-leaderboard).

The caller's auth (`Authorization: Bearer <jwt>`) controls visibility
of their own non-public rows. Otherwise endpoints are public; non-public
entries are filtered for callers that don't own them.

## Anti-cheat (v1 stance)

* **Public source by default.** When `leaderboard_public=True`, the
  source archive is downloadable — reviewers and the community can
  inspect for `if problem.name in KNOWN: ...` shortcuts.
* **Hold-out re-rank.** A private hold-out problem set is **not** part
  of any combo's `problems.json`; we re-rank the top 5 against it
  monthly and demote anything that looks overfit. (Manual today, can
  be automated when the cache hit pattern justifies it.)
* No automated overfitting detection in v1.

## Versioning & cache invalidation

The cache key for a combo's pairwise matrix is the JSON-serialised
`ComboSpec` plus the `lib_versions` dict (optiprofiler / scipy / numpy
/ etc. snapshotted at seed time). Bumping any of those — or bumping
`cache_version` manually — invalidates that combo's pairwise cache on
next seed. The on-disk JSON keeps the snapshot, so a deployment that
wakes up with a different optiprofiler version can detect the drift
before serving stale numbers.

## Adding a new combo

1. Append a `ComboSpec(...)` entry to `V1_COMBOS` in
   `backend/app/services/leaderboard.py` (frozen forever once shipped).
2. Run `scripts/seed_leaderboard.py --combo <new_id>` to materialise
   the on-disk + DB artefacts (this includes the K·(K−1)/2 pairwise
   benchmarks).
3. The submit-page selector and `/leaderboard` page pick it up
   automatically.

## Adding a new baseline

1. Drop `mybaseline.py` into `backend/sandbox/baseline_solvers_python/`.
2. Teach `baselines_for()` in `backend/app/services/baseline_registry.py`
   when to include it.
3. Re-seed every combo whose `ptype` triggers the new baseline; their
   `pairwise.json` matrices and DB rows are upserted on each seed.
   Adding a baseline grows the pairwise matrix from K to K+1; the new
   pair count is `K` extra benchmarks.
```

## Source: docs/adr/0005-agent-c-integration.md

```markdown
# ADR 0005 — Agent C (Results Interpreter) integration

**Status:** accepted (2026-04-21)
**Supersedes:** —

## Context

The OptiProfiler agent system (`pip install optiprofiler-agent[all]`)
ships three flagship agents: A (Advisor), B (Debugger), C (Interpreter).
Agent C reads an experiment directory (`test_log/` and friends) and
produces a Markdown report explaining what the numbers mean.

We want this on the platform so users get expert-level analysis of
their results without having to install the agent locally or pay for
their own LLM credits during exploration.

## Decision

**Pull-based, cached, owner-only.**

- API: `POST /api/task/{id}/interpret?language=English[&refresh=true]`.
- The first call invokes Agent C → caches the Markdown on the `Task`
  row (`result_report TEXT`).
- Subsequent calls return the cached report instantly; `refresh=true`
  forces a fresh LLM call.
- Only the task owner can trigger interpretation (avoids strangers
  burning your LLM quota).
- LLM provider/key resolved from Agent C's own config chain
  (`~/.opagent/.env`, `OP_*_API_KEY`, etc.), so the platform doesn't
  need its own LLM secret store.

## Why pull, not push?

- LLM calls cost money. Most users never click "AI Analysis" on most
  tasks. Generating a report for every task would be 90 % waste.
- Pull also gives the user the choice of *language*, which Agent C
  supports as a free-form parameter.

## Why cache?

- Same task → same numbers → same report. Re-running is pointless and
  costs money.
- Storing in the DB row (not a separate table) keeps deletion atomic:
  `DELETE /api/task/{id}` already nukes the cached report.

## How it's wrapped

`backend/app/services/interpreter.py` is a thin facade:

- Lazy-imports `optiprofiler_agent` (heavy: pulls in torch / chromadb).
  This keeps platform startup time low for users who never call it.
- Auto-descends from the platform's `data/results/<task_id>/` to the
  `out/<bench_run>/` that actually contains `test_log/`. The platform's
  storage layout doesn't match Agent C's expected layout 1:1.
- Translates SDK errors into a custom `InterpretError` so the route
  layer can map them to HTTP 503 cleanly.

## Frontend integration

- New button on success rows: **`✦ AI Analysis`** (purple gradient).
- Clicking opens a modal that:
  - Shows a spinner while the LLM runs (5–30 s typical).
  - Renders the Markdown via `react-markdown` + `remark-gfm`.
  - Indicates "cached" if the result came from cache.
  - Has a Regenerate / Retry button (Retry visible on errors).

## Consequences

- Backend now depends on `optiprofiler-agent[all]` (large transitive
  set: torch, chromadb, sentence-transformers). Docker image gets
  ~1 GB heavier. Acceptable for a research platform.
- LLM provider config lives outside this repo. Operators must set
  `MINIMAX_API_KEY` (or equivalent) in the worker's environment.
- A user can flood Agent C requests since each costs LLM tokens.
  Future work: separate per-day rate limit just for `/interpret`.

## Future: Agents A & B

- **A (Advisor)**: integrates pre-submission. UI: "Ask the advisor"
  button on the Submit page. POST a question + optional problem
  description, get back recommended `ptype/feature_name/n_jobs/
  solver` choices.
- **B (Debugger)**: triggers automatically on FAILED tasks with a
  Python traceback. Posts the traceback + solver source to Agent B,
  shows the diagnosis in the failure-detail panel.

Neither is wired up yet.
```

## Source: docs/adr/0006-chat-widget.md

```markdown
# 0006 — Floating chat widget for the Advisor (Agent A)

Date: 2026-04-22
Status: Accepted

## Context

Two-thirds of incoming user friction in the platform is *pre-submission*:
"what does `feature_name=perturbed_x0` mean", "what should my solver
function look like", "is `n_jobs` set automatically". The platform docs
answer all of these but users do not read docs before they hit Submit.

Agent A (`AdvisorAgent` in `optiprofiler-agent`) already implements this
exact Q&A, with knowledge-base injection and language detection. The
question was *how* to surface it.

## Options considered

1. **Right-sidebar permanent chat panel.** Always visible. Eats real
   estate even when the user only wants the form, and clashes with our
   single-column layout on mobile.
2. **`/chat` standalone page.** Loses cross-page context — the user is
   on `/submit` looking at their config, then has to navigate away to
   ask a question. Higher activation cost.
3. **Cmd+K command palette only.** Power-user friendly, invisible to
   newcomers. We need newcomer surface area.
4. **Floating action button (FAB) in the bottom-right corner that
   expands into an overlay panel.** Industry default for SaaS chat
   (Intercom, Crisp, Stripe docs, Linear AI, Vercel chat, ChatGPT
   widget). Always reachable, never blocks the main flow.

We chose **(4) FAB + Cmd+K shortcut as a power-user accelerator.**

## Decision

- Global `<ChatWidget />` mounted in `app/layout.tsx`. Visible on every
  page including auth callback and 404 — the assistant is a property of
  the *platform*, not of any one route.
- Bottom-right 56×56 px launcher; clicking expands to a panel anchored
  to the same corner. The default panel is 460 × 660 px, can be resized
  by dragging any border or corner, and can be expanded to a near-full-
  screen overlay from the header button. The floating panel frame is
  browser-local and persisted in `localStorage` as
  `op_chat_panel_frame`; older `op_chat_panel_size` values are migrated
  on first load.
- `Cmd+K` / `Ctrl+K` toggles. `Esc` closes.
- Assistant markdown is rendered in a constrained bubble: GFM tables and
  code blocks scroll internally instead of widening the panel, while
  paragraphs and inline code wrap inside the bubble. Assistant replies
  have an icon-only copy affordance; fenced code blocks have their own
  icon-only copy affordance.
- Server-side state is per `session_id`, kept in an LRU dict
  (`backend/app/services/advisor.py`). 30 min idle TTL, 200-session
  hard cap. `session_id` is generated client-side and persisted in
  `localStorage` so a page refresh keeps the conversation visible.
- Per-bucket rate limit: anon 5 msg/h (matches the 5/day task cap
  philosophy); authed 30 msg/h. Anon bucket keyed by IP, authed by
  user.id.

## Why in-process LRU and not Redis-backed sessions

`AdvisorAgent` holds Python objects (LangChain message history, an LLM
client with httpx connection pool, optionally a chromadb retriever
handle) that are **expensive to construct (~200ms-1s)** and not pickle-
clean. Redis-backed sessions would force a rebuild per request, which
would dwarf the LLM call latency for short messages.

The trade-off: this design assumes a single API replica. When we go
multi-replica (post-launch), we will pin sessions to one node via
sticky cookies on the load balancer. The session_id is already in
every request body, so adding `Set-Cookie: chat_node=<uuid>` is a
~10-line change.

## Consequences

- LLM costs visible in the same provider (`MINIMAX_API_KEY` etc.)
  used by Agent C interpreter. Single billing surface — easier to
  monitor.
- Anon users can chat without signing in, which is intentional (low
  friction for first-time visitors evaluating the platform). The
  IP-bucket rate limit prevents drive-by abuse.
- A "new chat" button (refresh icon) drops the server session and
  the local history. Reflects the user's mental model of "starting
  fresh" without surprising them with persisted context.

## Out of scope (deferred)

- **Streaming responses (SSE).** `AdvisorAgent.chat()` currently
  returns a complete string. The chat widget shows a typing indicator
  while waiting; token-by-token streaming is a UX nice-to-have that
  requires API surgery in the agent SDK and a `text/event-stream`
  endpoint. Will revisit if average reply latency exceeds 3-4 s.
- **Conversation export / history page.** localStorage already keeps
  the last 50 messages so refresh is non-destructive. A proper history
  archive needs DB schema and is not justified by current usage.
```

## Source: docs/adr/0007-auto-debug.md

```markdown
# 0007 — Auto-debug failed solver runs (Agent B)

Date: 2026-04-22
Status: Accepted

## Context

Most submission failures fall into a small handful of fixable categories:

- **Interface mismatch.** User selected `b` (bound-constrained) but their
  function still has signature `solver(fun, x0)` and crashes the moment
  OptiProfiler tries to pass `xl, xu`. By far the most common 30-day
  failure mode in dogfood logs.
- **Missing import / typo.** `import scipy.optimze` (sic), `from
  scipy.optimize import minimze`, etc.
- **Stale API.** Calls to functions that moved between SciPy versions.

All three are diagnosable from the traceback. Agent B
(`optiprofiler_agent.debugger.debug_script`) already does this, with a
deterministic fast path for interface mismatches and an LLM fallback for
runtime errors. Question: do we surface it?

## Options considered

1. **Show the raw traceback and let the user fix it.** Status quo —
   loses the user. Failure → tab close → never returns.
2. **One-click "Debug with OPA" button on failed tasks.** Better, but the
   user already paid for one slow Celery slot; they have to wait *again*
   for a manual retry. Friction → low conversion.
3. **Auto-debug on failure, opt-in checkbox at submit time, default
   on.** Failure → patch → re-run, all in the same Celery job. The user
   either sees a successful chart on first reload, or sees a clear
   "we tried this fix, here's why it still didn't work" report. Cost:
   one extra LLM call + at most one extra sandbox run *only when* the
   first run already failed.

We chose **(3)**, with a hard one-shot retry budget per task — never
recursive, never automatic on a successful run.

## Decision

- New `auto_debug_enabled` boolean on `Task`, captured from a checkbox
  on `/submit` (default checked, prominent OPA tag so the user knows
  who's touching their code).
- In `task_worker.run_benchmark_task`, when the first sandbox call
  returns non-success and the task opted in:
  1. Read original solver source from disk.
  2. Call `app.services.debugger.debug(code, error)` (thin wrapper
     around `debug_script`, lazy-imports the agent SDK).
  3. If a `fixed_code` is returned, **re-run the AST checker** on the
     suggestion. The agent should never be a way to bypass our static
     forbidden-API gate (`subprocess`, `socket`, `open(write)`, …).
  4. If the patch passes, persist it as `solver_autofixed.py` next to
     the original, and re-invoke `sandbox.run_benchmark` with that
     path.
- Persist three new fields on success *or* failure of the second run:
  - `was_debugged: bool` — UI shows the 🔧 chip and the gold banner.
  - `debug_report: TEXT` — markdown explanation, served verbatim.
  - `fixed_solver_path: TEXT` — for the download endpoint and the
    bundled zip.
- If the second run also fails, surface both errors in the
  `error_message`: original + second-run, prefixed with "Auto-debug
  applied a fix but the second run still failed." So the user knows
  what happened without having to dig.

## Why one shot, never recursive

Each retry costs (a) one LLM round-trip with the full traceback, (b)
one full sandbox spin-up, and (c) — crucially — burns one of the
user's `max_submissions_per_day` budget worth of compute. A naive
"keep trying until it works" loop turns one bad submission into an
unbounded LLM bill *and* lets a malicious user farm the agent. We can
revisit if telemetry shows the second attempt frequently *almost*
works.

## Why the AST re-check is mandatory

`AdvisorAgent` knowledge-base injection mostly produces patches in the
"swap one numpy call for another" style, but nothing in the contract
*forbids* it from suggesting `import socket` or `open("/etc/passwd")`
if the traceback hints at filesystem issues. The platform's threat
model assumes uploaded code is hostile. Treating Agent B's output as
"trusted because we generated it" would punch a hole through the only
defense against sandbox-escape-by-prompt-injection. Patches that fail
the re-check are still surfaced to the user as a *diagnosis* (the
report) — they just don't get auto-applied.

## Consequences

- Failed-task traceback page now ends with a friendly "OPA debug
  report" pane instead of just a wall of red. Reduces support burden:
  the user can read why their `solver(fun, x0)` doesn't accept `xl`
  *without* having to ask in chat.
- Successful auto-fixed runs show a gold banner + a prominent
  "Download fixed solver" button. The user gets the *exact* code that
  produced the chart they're looking at, which is non-negotiable for a
  benchmarking site (charts without the code are not reproducible).
- Results zip now bundles `solver_autofixed.py` and `debug_report.md`
  at the root so the same artifact set is downloadable for offline
  archiving.
- One extra LLM dependency on the failure path — same provider as
  Agent A and Agent C, no new credentials.

## Out of scope (deferred)

- **Streaming the debug report into the failed-task view.** The fix
  loop in `debug_script` can take 5-15 s for runtime-error cases. We
  currently make the user reload to see the report; a websocket push
  would be nicer but isn't worth the infra weight pre-launch.
- **Showing a diff (original vs. fixed) inline.** The user can `diff`
  the two files locally; an inline syntax-highlighted diff is a
  pure-frontend improvement we'll add when we have a code-viewer
  component.
- **Multi-shot retry with budget cap.** Mentioned above. Will be
  revisited based on telemetry.
```

## Source: docs/adr/0008-leaderboard.md

```markdown
# 0008 — Public solver leaderboard

Date: 2026-05-04 (proposed); 2026-05-18 (accepted, v1 shipped in Phase 3);
2026-05-19 (scoring methodology superseded — see ADR 0011).
Status: **Accepted (scoring superseded by ADR 0011)** — the high-level
shape (combos, baselines, opt-in submit, `/leaderboard` page,
`is_baseline`-mixed entries, anti-cheat policy) is still operative; the
**ranking metric and submission flow** have been replaced by ADR 0011's
pairwise-data-profile design. Read this ADR for context and the v0
trail; read **ADR 0011** for what the system does today.

See `docs/leaderboard.md` for the runbook (seed, score interpretation,
how to add combos / baselines).

## Context

The platform today is a *one-shot benchmarking service*: a user
uploads a solver, we run it against a config they choose, and we
return a chart. The chart only has meaning *relative to the
baselines we happen to throw in* (`scipy_cobyla`, optionally
`nelder-mead`). Two consequences:

- **Slow signal.** Every submission re-runs the same baselines from
  scratch. On hosted hardware that's 30–90 s of compute the user
  doesn't actually need — they want to know whether *their* solver
  is competitive, and the baselines are the same as last week's.
- **No ambient pull.** The user comes once for one curve, never
  comes back. There's no scoreboard to climb, no "you dropped two
  ranks" notification, no community comparison.

The most-cited public optimisation benchmarks (CUTEst, COCO/BBOB,
Mittelmann LP, JuMP DiffEq) all share the same shape: a fixed
problem set + fixed metric + a leaderboard with pre-computed entries
for known solvers. We can do the same on top of OptiProfiler with
much less ceremony, because OptiProfiler already produces the
metric (`profile_scores`) and we already host the runtime.

## Options considered

1. **Live re-run model (status quo).** Every submission runs *all*
   solvers (user + baselines). Honest but expensive and slow.
2. **Cached-baseline leaderboard.** Define a small set of
   `(problem set, dim range, feature, max_eval_factor)` *combos*.
   Pre-compute every known solver's `profile_scores` on each combo
   once and store on disk. Submissions only run the user's solver
   on the combo's problems and join in-memory against the cached
   baselines. **This is the proposal.**
3. **Cached-everything (full result freeze).** Same as (2) but also
   freeze every problem's `(x, f)` history per baseline so future
   features (e.g. a new noise model) re-aggregate without re-running.
   Gives the same speed, larger storage cost, and is most useful if
   we expect to add new features faster than new baselines. Worth
   keeping as a follow-on optimisation, not v1.

We choose **(2)** for v1. (3) is a strict superset and we should
revisit when the cache hit pattern is real, not hypothetical.

## Decision

### Combo definition

A leaderboard *combo* is a frozen test specification:

```
combo_id          : human-readable slug, e.g. "u-dim-1-50-noisy-2pct"
ptype             : 'u' | 'b' | 'l' | 'n' or a small string subset
plibs             : ['cutest'] / ['s2mpj'] / ['cutest','s2mpj']
mindim, maxdim    : ints
feature_name      : 'plain' | 'noisy' | 'perturbed_x0' | 'nan' | …
feature_kwargs    : dict — **must include a fixed seed** for any
                    stochastic feature, otherwise the cache is
                    meaningless
max_eval_factor   : int
lib_versions      : { 'optiprofiler': '...', 'scipy': '...',
                      'numpy': '...', 'platform_runner_sha': '...' }
```

`combo_id` and the JSON-serialised tuple above are both stored. The
JSON is the cache invalidation key — change *any* of these fields and
we recompute baselines from scratch.

A v1 ships with **5 public combos**, not 50. The first board should be
easy to explain, cheap enough to maintain, and representative of what
OptiProfiler uniquely tests.

| combo_id                         | ptype | dim    | feature                         | max eval factor | why |
|----------------------------------|-------|--------|----------------------------------|-----------------|-----|
| `u-small-plain`                  | u     | 1–20   | plain                            | 200             | classical derivative-free unconstrained baseline |
| `u-small-noisy`                  | u     | 1–20   | noisy, fixed seed and noise spec | 300             | robustness to noisy objective values |
| `u-small-perturbed-x0`           | u     | 1–20   | perturbed_x0, fixed seed         | 200             | sensitivity to starting points |
| `b-small-plain`                  | b     | 1–20   | plain                            | 250             | bound-constrained practical tier |
| `ubln-micro-plain`               | ubln  | 1–10   | plain                            | 200             | "can handle every signature" showcase; small to keep runtime controlled |

Two extra combos are useful later, but should not ship in the first public
board:

- `u-medium-plain`, dimension 21–100, plain, max_eval_factor 200. This is
  useful for scaling claims but too expensive for the first hosted preview.
- `n-small-plain`, dimension 1–20, plain. This is scientifically valuable but
  should wait until constrained baselines and nonlinear callback behavior are
  stable.

### Problem-set policy

Each combo stores a resolved problem manifest, not just a dimension range. The
manifest is the actual benchmark object:

```
problems.json:
  combo_id
  optiprofiler_version
  resolved_at
  problem_library
  ptype
  mindim/maxdim
  constraint filters
  feature spec
  problems: [{name, dim, type, n_bounds, n_linear, n_nonlinear, hash?}, ...]
```

For v1, cap each combo at a fixed problem count, e.g. **50 problems** for small
unconstrained/bound combos and **20–30 problems** for the all-signature combo.
If the library resolves more, sample deterministically with a fixed manifest
seed and stratify by dimension. This avoids leaderboard churn when the upstream
problem library changes ordering.

The score is computed from the fixed manifest only. Changing the manifest creates
a new `combo_version`, not an in-place update.

### Baseline solver set

Frozen per combo. Suggested v1 list:

- `scipy_cobyla` (always)
- `scipy_nelder_mead` (`u` only)
- `scipy_cobyqa` (constrained-friendly; covers `b/l/n`)
- `scipy_powell` (`u` and `b` only, where the wrapper is reliable)

Each entry pinned by version in `lib_versions`.

Leaderboard combos remain on the SciPy-only baseline set above so existing
seeded cache files and ranks do not churn. Free-mode comparison, however, also
exposes PDFO (`pdfo==2.2.0`) as optional user-selected competitors:
`pdfo` (automatic method selection), `pdfo_uobyqa`, `pdfo_newuoa`,
`pdfo_bobyqa`, `pdfo_lincoa`, and `pdfo_cobyla`.

PDFO is a deployment capability, not a local-development requirement. The
wrappers live in the repo, but the API hides them unless
`OP_ENABLE_PDFO_COMPETITORS=1` is set and the Python sandbox image contains
`pdfo`. **PDFO is Python-only**: there is no MATLAB `pdfo` competitor, because
PDFO and PRIMA export the same MATLAB function names (`uobyqa`/`newuoa`/…) and
shadow each other on a single path (`pdfo:InvalidOutput`). PRIMA is the modern
reimplementation and supersedes PDFO, so on MATLAB we ship PRIMA only. This
matters because local Mac development currently does not have to run PDFO.

Independent PRIMA is exposed separately from SciPy. The Python catalogue has
`prima`, `prima_uobyqa`, `prima_newuoa`, `prima_bobyqa`, `prima_lincoa`, and
`prima_cobyla`; the MATLAB catalogue has `prima`. These entries are hidden by
default because they require different deployment prerequisites:

- Python PRIMA requires the sandbox image to be built with `WITH_PRIMA=1`,
  which installs the official `libprima/prima` Python source build pinned to
  commit `1d76fb88...`, plus `OP_ENABLE_PRIMA_COMPETITORS=1`. The `v0.7.2`
  tree predates the Python package entry point, so it must not be used for
  the Python image. The Python-facing API is SciPy-like
  (`prima.minimize(...)`), but it is the independent PRIMA package, not
  SciPy's `COBYLA` / `COBYQA` wrappers.
- MATLAB PRIMA requires a CLEAN `libprima/prima` checkout built once with
  `setup`, pointed at via `OP_MATLAB_PRIMA_ROOT` (the runner addpaths
  `<root>/matlab/interfaces`), a passing `op_prima` smoke test, plus
  `OP_ENABLE_MATLAB_PRIMA_COMPETITORS=1`. Do not reuse a dirty/dev tree — a
  partial build is what segfaulted MATLAB before.

The PyPI names `prima`/`libprima`/`pyprima` are currently unusable `0.0`
placeholder packages, so production deployments must use the pinned official
source build or a vetted internal wheel. The sandbox base image is pinned to
`python:3.11-slim-bookworm` because the official Python PRIMA extension
segfaults when built on the newer trixie / GCC 14 base behind the floating
`python:3.11-slim` tag. Do not clear PRIMA's `GNU_STACK` marker with
`patchelf --clear-execstack`: the tested source build imports after that edit
but `prima.minimize(...)` segfaults. This makes Python PRIMA a hosted-beta
capability only; keep it behind the opt-in flag and Docker hardening until a
non-executable-stack upstream build or vetted internal wheel exists.

Free-mode defaults are deliberately smaller than the leaderboard baseline
cohort so ordinary hosted submissions finish quickly: Python defaults to
`scipy_nelder_mead` for pure `u` and `scipy_cobyla` for constrained/mixed
ptypes; MATLAB defaults to `fminsearch` for pure `u` and `fmincon` for
constrained/mixed ptypes. Users can opt into more competitors on the submit
form.

### Ranking metric

Use an evaluation-count based scalar, not wall-clock time:

- Primary rank: area under the data profile over fixed tolerances, averaged
  across problems and seeds. Higher is better.
- Secondary display: performance-profile score at representative tolerances
  (`1e-2`, `1e-4`, `1e-6`) and solved-problem fraction.
- Tie-breakers: fewer evaluation failures, then better median score at `1e-6`,
  then earlier submission time.

Do not rank by runtime in v1. Runtime depends on language bridge overhead,
server load, and container scheduling; evaluation-count profiles are the fair
cross-language metric OptiProfiler already produces.

### Storage layout

```
backend/data/leaderboard/
└── <combo_id>/
    ├── combo.json            ← the spec + lib_versions
    ├── problems.json         ← resolved problem list (so a CUTEst
    │                           rebuild without changing version
    │                           is still detected)
    └── baselines/
        └── <solver_name>.json   ← profile_scores + scalar summary
```

Per-combo size: a few hundred KB. Total v1: < 10 MB. No new
storage tier needed.

### Submission flow (user solver vs leaderboard)

1. On `/submit`, expose a checkbox **"Submit to leaderboard"** (off
   by default). When on, the form's combo selector replaces the
   free-form ptype/dim/feature inputs.
2. Worker reads `combo.json`, runs **only** the user's solver against
   that combo's problems. No baselines re-run.
3. Aggregate into the same `profile_scores` shape.
4. To control variance for stochastic solvers, default to **3 seeds**
   per leaderboard submission. Display
   median ± [min, max].
5. Persist a `LeaderboardEntry` row:
   ```
   id, combo_id, solver_display_name,
   submitted_by_user_id (nullable for the seed entries),
   scores_median, scores_min, scores_max, n_seeds,
   curves_json (compressed),
   public bool, submitted_at, withdrawn_at nullable,
   solver_source_sha, solver_language,
   source_archive_path, runtime_versions_json
   ```
6. New leaderboard view: `GET /api/leaderboard/<combo_id>` returns
   `(rank, name, score, is_yours)` rows.

Leaderboard source code is public by default for ranked submissions. The normal
private "cloud run" path remains separate.

### Determinism

Three independent sources of variance, treated separately:

| Source                      | Mitigation                                            |
|-----------------------------|-------------------------------------------------------|
| Feature randomness          | Seed pinned in `feature_kwargs`; baked into combo.    |
| Problem-library churn       | `lib_versions` cache key; rebuild on bump.            |
| User-solver randomness      | Run K times, store median + range; show on chart.    |

If any of the first two changes, **all** leaderboard entries on that
combo are invalidated and the baselines are recomputed (likely via a
GitHub Action, not the live worker, so it doesn't compete with users).

### Anti-cheat posture

The most obvious attack is `if problem.name in KNOWN: return
optimum`. We treat this the same way Kaggle does:

- v1: every leaderboard submission's source code is **public**.
  Hard-coded answers are visible to reviewers and the community.
- v1: an off-the-record **hold-out problem set** is *not* shown on
  the public combo definition; we re-rank the top 5 against it
  monthly and demote suspicious entries.
- Out of scope for v1: automated overfitting detection, problem
  obfuscation, captcha-style proof-of-work.

## Why opt-in, not auto-leaderboard

Three reasons users will hate auto-publishing their solver:

1. **Privacy.** A research prototype is not ready for a public
   benchmark. Sharing must be explicit.
2. **IP.** Code uploaded for testing might be unpublished work.
3. **Failure noise.** A solver that crashes on three problems
   shouldn't appear on the board at all; opt-in lets us reject
   submissions that don't meet a minimum completeness bar.

## Consequences (if accepted)

- **Faster submissions.** Leaderboard runs skip the baseline phase
  → 2–5× faster wall-clock for the same chart.
- **Repeat traffic.** Climbing a leaderboard is a recurring
  motivation; one-shot submissions are not.
- **A real artefact to cite.** "OptiProfiler Leaderboard, combo
  `u-dim-1-50-noisy-2pct`" can be referenced from papers without
  asking us to maintain anything beyond the combo spec.
- **A new failure mode.** Cached baselines drift silently if we
  forget to bump `lib_versions` on a SciPy upgrade. The `lifespan`
  startup hook should warn loudly when the *running* SciPy version
  doesn't match any stored combo's `lib_versions`.
- **More LLM exposure.** Once leaderboards exist, expect users to
  ask Agent A "why did my solver lose to BOBYQA on combo X" —
  the AdvisorAgent's knowledge base needs a section on
  per-combo strengths of each baseline.

## Open questions (must resolve before the M1 build starts)

1. Combo list freeze for v1 — exactly which 4–6.
2. Baseline solver list freeze — what installs reliably in the
   sandbox image, and at which version pin.
3. Where do we run the baseline pre-computation — a GitHub Action
   on a beefy runner, or a one-off `make seed-leaderboard` on the
   prod host during a quiet hour?
4. Do anonymous (not signed-in) submissions appear on the public
   leaderboard? Probably yes, attributed as "Anonymous".
5. Do we let users **withdraw** a published entry? Probably yes,
   GDPR-friendly default.

## Out of scope (deferred to a future ADR)

- Cached-everything model (option 3 above).
- Cross-combo aggregate ranks ("best overall solver").
- Solver fingerprinting / duplicate-submission detection.
- Automated overfitting detection.
- Hardware-comparable scoring (right now we report `profile_scores`
  which is hardware-independent for derivative-free; once we let
  in derivative-based or wall-clock-sensitive solvers, the metric
  needs to change).

## Update — 2026-05-17 (multi-plib & combo refinements)

Two changes ride along with ADR 0010:

1. **Plib axis on every combo.** A combo is now keyed by
   `(ptype, dim range, feature, plib, max_eval_factor)`. The same
   benchmark run on S2MPJ vs. PyCUTEst is two combos, not one, because
   the underlying problem set is not identical. The v1 board ships with
   six combos (was five): the original five all on `s2mpj`, plus
   `u-cutest-plain` on `pycutest` (gated on Phase 3 install automation).
2. **Problem-parameter snapshot in the cache key.** S2MPJ's `argins`,
   CUTEst SIF parameters, and similar per-problem configuration knobs
   change a problem's effective dimension and constraint structure.
   `problems.json` already records `{name, dim, type, ...}`; we add a
   `problem_kwargs` field per entry so an upstream library bumping its
   default parameters does not silently invalidate the leaderboard.
   This is in the same cache-key tuple as `lib_versions`.

The "Submission flow" and "Storage layout" sections do not change beyond
adding `plib` to `combo_id` and `problem_kwargs` to `problems.json`.
Frozen baseline JSON files are partitioned per combo as before.
```

## Source: docs/adr/0009-matlab-solver-upload.md

```markdown
# 0009 — MATLAB solver uploads (Proposed, partly superseded)

Date: 2026-05-16
Status: **Proposed** — design first; no runtime support is built yet.
**The "Octave-first hosted preview" decision is superseded by ADR 0010
(2026-05-17): the lab has a real MATLAB licence, so v1 hosted execution
uses `matlab -batch` directly. The contract, security policy, and
implementation stages below remain valid; substitute "MATLAB CLI" for
"Octave" except where Octave is explicitly called out as a fallback.**

## Context

The platform currently accepts one uploaded Python file. That path is deeply
Python-specific:

- The API only accepts filenames ending in `.py`.
- The upload gate uses Python AST static analysis.
- The sandbox runner imports a top-level `solver` function from the uploaded
  file and passes that Python callable directly to `optiprofiler.benchmark()`.
- Auto-debug, reproduce ZIPs, and baseline bundling all assume Python source.

Supporting MATLAB/Octave `.m` files is valuable because many derivative-free
optimization solvers are still distributed as MATLAB code. But it is not a file
extension toggle: we need a runtime, a cross-language callable adapter, and a
separate security policy.

## Options considered

1. **MATLAB Engine for Python inside the sandbox.**
   - Best compatibility with real MATLAB solvers.
   - Requires a licensed MATLAB installation on the host/image, large image
     size, slower cold start, and license-server operations.
   - Hard to make available to arbitrary public submissions.

2. **GNU Octave CLI bridge.**
   - Open-source and practical for a public hosted preview.
   - Most MATLAB-style `.m` solvers that avoid proprietary toolboxes can run.
   - Compatibility is not perfect; we must label it clearly as "MATLAB/Octave
     compatible", not guaranteed MathWorks MATLAB.

3. **Local reproduce only for MATLAB.**
   - The hosted platform packages the user's `.m` file plus a generated Python
     or shell driver, but does not execute it.
   - Safe and simple, but less useful than cloud runs.

We choose **Option 2 for the first hosted implementation**, with Option 3 as a
fallback when a solver needs proprietary MATLAB. True MATLAB Engine support can
be added later for private deployments that have a license.

## MATLAB solver contract

Users upload a single `.m` file whose basename is the solver function name. For
clarity in docs and examples, recommend `solver.m`.

The function should return the best point found:

```matlab
function x = solver(fun, x0)
    x = x0;
end
```

Supported signatures mirror the Python UI:

```matlab
function x = solver(fun, x0)
function x = solver(fun, x0, xl, xu)
function x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)
function x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
```

The MATLAB/Octave bridge will pass arrays as column vectors/matrices. Nonlinear
constraint callbacks are more expensive because every callback crosses the
Python↔Octave process boundary.

## Runtime architecture

Do **not** ask `optiprofiler.benchmark()` to call MATLAB directly. Instead, the
Python sandbox runner creates a Python wrapper callable:

```python
def solver(fun, x0, *args):
    return matlab_bridge.run_solver(uploaded_m_file, fun, x0, *args)
```

The bridge should:

- Start Octave once per benchmark task, not once per objective call.
- Create a per-task temporary working directory.
- Serialize `x0`, bounds, matrices, and callback requests through a narrow
  protocol.
- Keep the network disabled and the workspace mounted exactly like Python
  submissions.
- Enforce the same task timeout; if Octave hangs, kill the whole process group.

The simplest first bridge is file-based IPC:

1. Python writes input arrays to `.mat` or JSON files.
2. Octave runs the uploaded function through a generated wrapper script.
3. Objective callbacks are handled by a local Python callback server over a Unix
   socket or by batched file polling.

For v1, prefer a simpler but slower design over a fragile fast design. If the
callback bridge becomes too complex, restrict hosted MATLAB v1 to solvers that
evaluate `fun` through a generated Octave-side proxy and benchmark only small
dimensions.

## Security policy

Python AST checks do not apply to MATLAB source. MATLAB/Octave support needs a
separate static gate and the same Docker hard boundary:

- Accept `.m` text only, size-limited like Python uploads.
- Reject obvious shell/file/network escape functions before execution, with a
  coarse user-facing message.
- Run only in Docker mode; no local-mode MATLAB execution for public previews.
- Keep `--network=none`, memory limit, CPU limit, read-only platform files, and
  per-task temp directories.
- Do not support multi-file ZIP MATLAB packages until we have a package-level
  static scan and dependency policy.

The static gate is a defense-in-depth filter, not the sandbox. The real security
boundary remains Docker.

## UI and API changes

- Add a language segmented control on Submit: `Python` / `MATLAB/Octave`.
- Accept `.py` for Python and `.m` for MATLAB/Octave.
- Store `solver_language` on `Task`, e.g. `python` or `matlab_octave`.
- Disable auto-debug for MATLAB v1; Agent B currently emits Python patches.
- Update examples and signature hints for MATLAB.
- Reproduce ZIPs should include:
  - `repro/solver.m`
  - `repro/run_repro_matlab.m` or `run_repro_octave.sh`
  - a README that says the hosted runtime is GNU Octave unless configured
    otherwise.

## Implementation stages

### M0 — Local reproduce package

Accept `.m` in a separate endpoint path only for ZIP generation. No cloud run.
This lets MATLAB users download a clean local package without adding hosted
runtime risk.

### M1 — Hosted Octave smoke

Build a separate sandbox image, e.g. `optiprofiler-sandbox-octave:latest`, with
GNU Octave installed. Run a tiny unconstrained `solver.m` on `ptype=u`,
`mindim=maxdim=1`, `feature=plain`, `max_eval_factor=5`.

### M2 — Public hosted beta

Expose MATLAB/Octave cloud runs for small `u` and `b` configurations only. Keep
constraint-heavy `l/n` disabled until callback overhead and failure modes are
understood.

### M3 — MATLAB Engine for private deployments

Add an optional `OP_MATLAB_RUNTIME=engine` mode for licensed environments. Do
not enable it on the public preview without license and operations review.

## Open questions

1. Which Octave version do we pin in the sandbox image? (Now moot for v1 —
   v1 uses MATLAB CLI per ADR 0010. Octave remains a fallback path for
   private deployments without a licence.)
2. Do we require the uploaded file to be named `solver.m`, or allow any basename
   and generate the call from that basename?
3. What is the minimum useful callback bridge for constrained problems?
4. Do we show MATLAB/Octave entries on the same leaderboard as Python solvers?
   Probably yes, if the benchmark metric is evaluation-based, but runtime
   failures and language labels should be visible.

## Update — 2026-05-17 (decision delta)

Implementation choice flipped from Octave to MATLAB CLI because the lab has a
licence and the public preview will run on lab hardware. Concrete deltas vs.
the original "Runtime architecture" section:

- Drop the file-based Python↔Octave callback bridge plan. Instead the
  Python-side runner generates a MATLAB wrapper script (`run_benchmark_wrapper.m`)
  that calls `optiprofiler.benchmark()` *inside MATLAB* with a function handle
  to the user's solver. No cross-process callback shuttling per objective call.
- New entry-point rule: single `.m` file → basename = function name; ZIP →
  top-level `solver.m` or `manifest.json -> entry`. The runner does
  `addpath('/workspace/solver_project')` so multi-file projects work.
- Sandbox image: `optiprofiler-sandbox-matlab` (separate from the Python
  image), running `matlab -batch` in `--network=none`. Network license check
  happens once when the worker boots, not per task.
- Auto-debug and interpret integration depend on changes inside
  `optiprofiler-agent`. Tracked in
  [`docs/agent-changes-required.md`](../agent-changes-required.md). Until
  those land, MATLAB tasks skip the auto-debug branch and Agent C
  interprets MATLAB results in a degraded mode (text summary only; PDF
  curve extraction may return empty until a MATLAB-specific PDF parser
  is added).
- M0 (local reproduce) and M3 (private MATLAB Engine) stay as written;
  M1/M2 are the same stages but with MATLAB CLI replacing Octave.
```

## Source: docs/adr/0010-multi-language-backend.md

```markdown
# 0010 — Multi-language solver backend, ZIP uploads, and problem-library selection

Date: 2026-05-17
Status: **Accepted / implemented through Phase 3**. Phase 1 (backend
abstraction, ZIP upload, plib selector), Phase 2 (MATLAB CLI runner; see ADR
0012), and Phase 3 (leaderboard + PyCUTEst install hook; see ADR 0011 and
`docs/leaderboard.md`) have landed. Phase 4 (`requirements.txt` ZIPs) remains
deferred. Supersedes part of ADR 0009 (the choice of Octave) and refines ADR
0008 (problem-library knob).

## Context

The platform was built single-language (Python) end to end. Three pressures from
the May 2026 product review push us beyond that:

1. **MATLAB users.** Many derivative-free optimisation solvers in active use
   are MATLAB code. The lab has a real MATLAB licence, so the previous
   "Octave-only public preview" plan in ADR 0009 is no longer the right
   default.
2. **Multi-file solvers.** Real research codebases come as folders with
   helper modules, not single `.py` files. Submit must accept a ZIP project,
   with a documented entry-point convention.
3. **Problem-library choice.** OptiProfiler can run benchmarks on **S2MPJ**
   (pure source, default), **PyCUTEst** (Fortran-backed Python bindings),
   or **MatCUTEst** (MATLAB CUTEst bindings). The three test sets overlap but
   are not identical. Users want to choose; today the platform hardcodes
   S2MPJ.

These are three problems but they all hit the same handful of files
(`backend/sandbox/run_task.py` — now `run_task_python.py` and a sibling
`run_task_matlab.{sh,m}` — `app/services/sandbox.py`, the `Task` model,
the submit form, the sandbox image). Solving them one at a time produces
three incompatible refactors. We solve them together so the next language
(Julia, R, etc.) is a derivative, not another rewrite.

> **Background — what S2MPJ / PyCUTEst / MatCUTEst really are.** All three
> are *implementations* of the CUTEst test problem set, not separate
> catalogues. S2MPJ translates SIF problems into pure Python / MATLAB /
> Julia code (no Fortran, easiest to deploy). PyCUTEst calls the original
> CUTEst Fortran library from Python. MatCUTEst is the MATLAB equivalent
> (`macup`/`secup`). Their problem sets overlap but each has unique
> instances and parameterisations, which is why letting users pick matters.

## Decision

We introduce a **language backend abstraction** in `backend/app/services/`.
Python and MATLAB now both implement it; full ZIP support with
`requirements.txt` still sits on this foundation as Phase 4.

### Phase 1 — shipped

#### 1. `language_backends/` abstraction layer

```
backend/app/services/language_backends/
├── __init__.py          # registry: get_backend(lang) -> LanguageBackend
├── base.py              # abstract class
├── python_backend.py    # Python validation/workspace backend
└── matlab_backend.py    # MATLAB validation/workspace backend (ADR 0012)
```

Abstract surface (`base.py`):

```python
class LanguageBackend(Protocol):
    name: str
    def accepted_extensions(self) -> list[str]: ...
    def validate_upload(self, path: Path) -> ValidationResult: ...
    def sandbox_image_name(self) -> str: ...
    def prepare_workspace(self, task: Task, work_dir: Path) -> None: ...
    def parse_result(self, output_dir: Path) -> BenchmarkResult: ...
```

`backend/sandbox/run_task_python.py` (Python) and `run_task_matlab.{sh,m}`
(MATLAB) implement a shared CLI (`--solver / --config / --output /
--language`). Both names follow the convention
`run_task_<lang>.<ext>` ↔ `Dockerfile.<lang>` ↔
`baseline_solvers_<lang>/` ↔ `<lang>_backend.py` ↔
`optiprofiler-sandbox-<lang>:latest`. (The Python runner used to be
called `run_task.py` while it was the only language; the rename came
when MATLAB landed so all five surfaces stay symmetric.) Adding a
third language is a derivative drop-in.

#### 2. `Task` model fields

In [`backend/app/models.py`](../../backend/app/models.py), `Task` gains:

- `solver_language: str` — `"python"` (default) | `"matlab"`.
- `upload_type: str` — `"single_file"` (default) | `"zip_project"`.
- `problem_libraries: str` — JSON-serialised list, e.g. `'["s2mpj"]'` or
  `'["s2mpj","pycutest"]'`. Default `'["s2mpj"]'`.

SQLite migration is an additive `ALTER TABLE` (all three are nullable with
defaults), so no Alembic infrastructure is needed yet.

#### 3. ZIP upload (same-language only)

The platform accepts ZIPs whose contents are **all one solver language**
(Python or MATLAB). No `requirements.txt`, no PyPI install. The contract:

- Max size: 10 MB.
- File-type allowlist: `.py`, `.m`, `.txt`, `.md`, `.json`, `.yaml`, `.yml`.
  Any binary, archive, or executable is rejected by extension and by a
  magic-number sniff on the first 512 bytes.
- Path traversal blocked: every entry's `Path(name).resolve()` must stay
  within the extraction directory.
- Entry-point rule:
  - Python: top-level `solver.py` exporting `solver(...)`.
  - MATLAB: top-level `solver.m` defining `function ... = solver(...)`.
  - Optional `manifest.json` at the ZIP root with `{"entry": "<filename>"}`
    overrides the default basename.
- AST gate is run against **every** `.py` file in the archive (not just the
  entry point) before sandbox execution.

Multi-file ZIP with `requirements.txt` is **deferred** — see "Out of scope".

#### 4. Problem-library selection

API surface:

- `POST /api/submit` accepts a new field `problem_libraries` (multi-select).
- Validation: at least one entry; hosted Python and MATLAB runs currently
  accept `{"s2mpj"}`. `pycutest` / `matcutest` remain visible as staged
  affordances until the corresponding host/runtime manifests are seeded.
- Worker writes the chosen list into `config.json` as `plibs`.
- `run_task_python.py` (and the MATLAB equivalent) read `plibs` and pass
  it to `optiprofiler.benchmark(..., plibs=plibs)`.

Frontend:

- Submit page gains a multi-select control (chips) under the ptype selector.
- `S2MPJ` is enabled for Python and MATLAB. `PyCUTEst` and `MatCUTEst`
  render disabled with tooltips until their runtime prerequisites are ready.

### Phase 2 — MATLAB CLI sandbox (shipped; see ADR 0012)

Decision change vs. ADR 0009: we use **real MATLAB CLI** (`matlab -batch`),
not Octave, because the lab has a licence. The sandbox image is a separate
image (`optiprofiler-sandbox-matlab`) so the Python image stays slim. In
practice, licence-bound lab hosts default to host-direct MATLAB execution via
`OP_MATLAB_RUNNER_MODE=host`; the Docker image path remains available for
deployments with a container-friendly licence flow.

Solver entry-point rules:

- Single `.m`: file basename = function name.
- ZIP: top-level `solver.m`, or `manifest.json -> entry`.
- The runner adds the user-file directory to `MATLABPATH` before
  `matlab -batch`, so helper functions in the same project can be called.

Auto-debug and interpret integration for MATLAB still depend on
`optiprofiler-agent` changes that are tracked separately (see
[agent-changes-required.md](../agent-changes-required.md)).

### Phase 3 — Leaderboard + PyCUTEst

`scripts/install_pycutest.sh` lands inside the Python sandbox image,
gated by a Phase 3 build flag. Once it's stable the submit page enables
the `PyCUTEst` chip. See ADR 0008 for the leaderboard design itself; the
only delta this ADR introduces is the `plib` axis on each combo.

### Phase 4 — ZIP with `requirements.txt` (out of scope for v1)

Two-stage container: networked install stage → offline run stage. Pulls
in PyPI threat-model questions we do not want to answer for the first
public preview. Tracked but not scheduled.

## Why one abstraction instead of three patches

Concrete cost estimate, ordered by effort if we treated each as a one-off:

- "Just add `.m` to the upload allow-list": touches sandbox runner,
  baseline registry, Task model, AST checker, frontend mime — every
  Python assumption gets a copy in MATLAB and they drift.
- "Just add ZIP upload": same files plus archive extraction logic, plus
  a per-language entry-point convention that needs to live somewhere.
- "Just add a plib selector": the worker-to-runner contract changes, and
  the runner is per-language, so it has to stay behind the backend contract
  instead of being patched separately per runner.

The abstraction is ~300 LOC of Python and replaces three independent diffs
with one centralised contract. The Python-only path through Phase 1 is a
move-not-rewrite of existing code; the risk is low.

## Consequences

- **Cleaner extension story.** Adding Julia is "write a `julia_backend.py`
  + a runner script + an image". No surgery in `tasks.py` / submit form
  beyond the language enum.
- **One more model field migration.** SQLite is forgiving but the change
  must land before any new column references go into routes.
- **Frontend disabled affordances.** The submit form shows options that are
  staged but not yet enabled in the hosted tier (PyCUTEst, MatCUTEst). They
  must be visibly disabled with a tooltip; otherwise users will report bugs
  that are actually deployment/runtime prerequisites.
- **Plib axis on the leaderboard.** ADR 0008's combo definition gains a
  `plib` field; an existing `u-small-plain` combo becomes
  `u-small-plain-s2mpj`, with `u-small-plain-pycutest` as a new combo
  once Phase 3 ships. This forces a manifest re-issue, which is OK —
  we have no public leaderboard yet.

## Out of scope

- `requirements.txt` resolution (Phase 4).
- Multi-language ZIP (e.g. mixed Python + MATLAB in one project).
- True MATLAB Engine for Python (deferred to private deployments;
  `matlab -batch` covers the public-preview need).
- Inter-language solver chaining (e.g. a Python wrapper calling a
  MATLAB primitive). Possible later but adds bridge cost we do not
  need to pay now.

## Open questions

1. **Where does the language switch live in the UI?** Top-of-form
   segmented control, or per-upload-card? Phase 1 lock: top-of-form.
2. **What's the exact set of "binary-like" magic numbers we sniff for
   in ZIPs?** ELF, PE, Mach-O, common archive signatures. The list
   lives in `language_backends/zip_validator.py`.
3. **Per-language AST/static check ordering.** Today AST gate runs in
   the request handler. The plan keeps it there for Python; MATLAB
   uses a keyword blacklist instead and shares the same handler hook.
   Verified in Phase 2 implementation.

## References

- [ADR 0008 — Public solver leaderboard](0008-leaderboard.md)
- [ADR 0009 — MATLAB solver uploads](0009-matlab-solver-upload.md) (decision
  on Octave vs. MATLAB superseded by this ADR)
- [`docs/agent-changes-required.md`](../agent-changes-required.md) — the
  optiprofiler-agent changes that unblock Phase 2 auto-debug & interpret
  for MATLAB
```

## Source: docs/adr/0011-leaderboard-pairwise-scoring.md

```markdown
# 0011 — Leaderboard scoring: pairwise data profile, no single-solver cache

Date: 2026-05-19.
Status: **Accepted** — supersedes ADR 0008 §"Ranking metric" and §"Submission flow".

See `docs/leaderboard.md` for the runbook.

## Why we are revisiting ADR 0008

ADR 0008 v1 chose option (2) "cached-baseline leaderboard": pre-compute every
known solver's `profile_scores` on each combo once, then on a user submission
run **only the user solver** against the combo's pinned problems and join
against the cached baseline scores in the API.

Implementing it surfaced a **fundamental error in the original reasoning**.
The Moré-Wild *convergence test* used by both performance and data profiles
is

```
phi(x) ≤ tau · phi(x_0)  +  (1 − tau) · phi_min
```

where `phi_min` is the **minimum merit observed across the set of solvers
in this benchmark run**. Concretely:

* In a 2-solver run `(A, B)`, `phi_min = min(best(A), best(B))` per
  `(problem, run)` cell.
* In a 3-solver run `(A, B, C)`, the same cell's `phi_min` can drop
  further if `C` finds a better minimum, **tightening the convergence
  threshold for A and B retroactively**.

Two consequences make ADR 0008's design unsound:

1. **The "join cached baseline scores" step is mathematically meaningless.**
   A cached baseline score that was computed in a 4-baseline experiment
   cannot be combined with a user score computed in a single-solver
   (or solver + padding) experiment, because the two experiments used
   different `phi_min` values and therefore evaluated against different
   convergence thresholds.

2. **The padding solver is a footgun.** To satisfy OptiProfiler's
   ≥2-solver API contract we added a no-op `__padding__` that returns
   `x_0`. Its merit history is `[phi(x_0)]`, so `phi_min` collapses to
   `phi(x_0)` and the threshold becomes
   `tau·phi(x_0) + (1−tau)·phi(x_0) = phi(x_0)` — i.e. the test
   degenerates into "any function decrease counts as convergence".
   The user's reported score is no longer comparable to baselines.

The platform shipped a non-trivial first cut on top of this wrong
foundation in Phase 3. ADR 0011 documents what we replace it with and
why a stored single-solver score cache cannot work.

## Decision

### Scoring: pairwise data-profile mean

For a combo with solver set `S = {s_1, …, s_K}` (baselines + ranked user
submissions), each solver's leaderboard score is the **mean of its
pairwise data-profile scores against every other solver in `S`**:

```
for each ordered pair (s_i, s_j) where i ≠ j:
    run benchmark([s_i, s_j], problems = combo.problems,
                  feature = combo.feature, …, n_runs = combo.n_runs)
    record pairwise_score[s_i][s_j] = data_profile_score_of_s_i

leaderboard_score(s_i) = mean over j ≠ i of pairwise_score[s_i][s_j]
```

The pairwise data profile is OptiProfiler's
`profile_scores[i, :, 0, 1].mean(axis=0)` for the 2-solver experiment
that pair belongs to — averaging the **history-based data profile**
across tolerances, picking the slice for solver `i`. We pass this via
`benchmark(score_fun=…)`.

Properties:

- The convergence threshold of every score is well-defined: it depends
  on exactly two solvers, both of which are present in the experiment.
- Two solvers can be compared without re-running every other solver:
  the score of `s_i` only needs the K−1 experiments where `s_i`
  participates, not the full `K(K−1)/2` matrix.
- The score is a **mean over [0, 1] values**, so it is itself in
  `[0, 1]` and ranks higher = better.
- It is *not* fully absolute: changing the solver set in a combo does
  shift everyone's score because each solver's mean is taken over the
  current opponent set. v1 mitigates this by capping the comparison
  set per combo (see "Solver-set cap" below).

### Why this is a "rough first cut" — and what comes next

A truly **absolute** score requires that the convergence threshold not
depend on the solver set at all. The cleanest construction is to record,
for each `(combo, problem, run, eval_count)` cell, the **best merit ever
observed by any solver that has been run on that combo** — call it
`phi_lower_bound(problem, run, k)`. The convergence threshold then
becomes

```
phi(x) ≤ tau · phi(x_0) + (1 − tau) · phi_lower_bound
```

independent of which solvers are in the current experiment. The lower
bound only ever decreases when a stronger solver is added, at which point
all entries are re-scored consistently against the new bound.

This is **out of scope for the platform** because it requires a change
inside OptiProfiler itself: the package's `compute_scores` reads
`merit_min` from the current run's `merit_histories`, not from an
external file. ADR 0011 therefore deliberately ships the pairwise mean
as v1, and tracks `phi_lower_bound` as a future OptiProfiler upstream
proposal in `docs/leaderboard.md` §"Future direction".

### Seeding (revised)

Seeding is **still** valuable, but its content changes:

| ADR 0008 (v0, wrong)                                | ADR 0011 (v1)                                        |
|------------------------------------------------------|------------------------------------------------------|
| Run each baseline once on the combo's problems       | Run each baseline pair `(B_i, B_j)` once on the combo|
| Store `<solver>.json` with single-solver score       | Store pairwise score matrix `baselines/pairwise.json`|
| Cannot be joined with user scores                    | Combines naturally with on-submit user pair runs     |

Cost: `K` baselines per combo → `K·(K−1)/2` 2-solver experiments. For
the v1 baseline set (`scipy_neldermead`, `scipy_lbfgsb`, `scipy_cobyla`,
`pdfo_newuoa`) that is 6 experiments per combo, ≈ 5–8 minutes per
combo at v1 dimensions.

### Submission flow (revised)

When a user opts into a combo:

1. The worker reads `combo.json` and the seeded `baselines/pairwise.json`.
2. The runner runs **K experiments**, one per baseline:
   `benchmark([user_solver, baseline_i], problems = combo.problems, …)`.
   Each call returns a 2-element score vector; we keep the user's slot.
3. The user's leaderboard score is the **mean** of those K pairwise
   scores. The K individual pairwise breakdowns are persisted on the
   `LeaderboardEntry` row (JSON column) so the UI can show
   "user vs `scipy_lbfgsb` = 0.71, vs `pdfo_newuoa` = 0.42, …".
4. No padding solver. No single-solver runs. Every benchmark in the
   leaderboard pipeline is a real 2-solver experiment.

This is **K times more expensive than ADR 0008's single-solver run**.
Concrete budget for `ubln-micro-plain` (30 problems, 4 baselines):

- ADR 0008 v0: 1 run × 30 problems × 200 max-eval-factor ≈ 60 s
- ADR 0011 v1: 4 runs × 30 problems × 200 max-eval-factor ≈ 4 min

### Solver-set cap

Because a user submission's score depends on the current opponent set,
that set must be stable across submissions. v1 caps each combo's
**comparison opponent set** at the **K seeded baselines** (4 today). User
submissions all rank against the same K baselines, never against each
other. This means:

- Adding a new user submission does **not** re-rank existing entries.
- The "leaderboard" is technically two superimposed rankings:
  baselines ranked among themselves (`K(K−1)/2` baseline pairs),
  user submissions ranked vs the baselines (one user × K baselines per
  submit). They share the same `[0, 1]` scale; the displayed rank
  merges both lists by score.
- A future v2 can introduce "user-vs-user" pairwise runs on demand
  ("compare my solver to user X's"), but they are **not** baked into
  the headline number.

We also cap **displayed user submissions per combo at the top 50** to
keep the public list scannable.

### Stochastic combos: `n_runs > 1`

For combos with stochastic features (`noisy`, `perturbed_x0`), each
2-solver pairwise experiment runs with the combo's `n_runs` (≥ 3 per
ADR 0008 §"Determinism"). OptiProfiler aggregates over runs internally
inside `compute_scores`, so the pairwise score we extract is already a
multi-seed median. We additionally persist `scores_min`/`scores_max`
across the K pairs (i.e. the variance comes from the opponent set, not
extra seeds).

### Sandbox-mode logging

The runner now emits a verbose pre-flight banner so users can verify
what their score actually compared against:

```
[leaderboard] combo: ubln-micro-plain
[leaderboard] problem set: 30 problems (s2mpj, ptype=ubln, dim 1..10)
[leaderboard] feature: plain (kwargs={})
[leaderboard] n_runs: 1, max_eval_factor: 200
[leaderboard] opponents (4): scipy_neldermead, scipy_lbfgsb, scipy_cobyla, pdfo_newuoa
[leaderboard] running pair 1/4: user_solver vs scipy_neldermead
…
[leaderboard] user score per opponent:
              scipy_neldermead = 0.612
              scipy_lbfgsb     = 0.481
              scipy_cobyla     = 0.703
              pdfo_newuoa      = 0.429
[leaderboard] mean = 0.556
```

## Migration

The Phase 3 v0 design left these now-unused artefacts:

* `backend/data/leaderboard/<combo>/baselines/<name>.json` — old single-solver
  score caches. Deleted on first ADR 0011 seed run; no migration logic
  needed because they were never displayed (the API only ever read the
  DB row, not the JSON).
* `backend/app/services/leaderboard_scoring.py:leaderboard_score_fun` —
  removed. The new runner uses
  `pairwise_data_profile_score_fun` defined in the same module.
* `_leaderboard_padding_solver` in `sandbox/run_task_python.py` — removed.
  Every pair has two real solvers.

The `LeaderboardEntry` table grows one new column,
`pairwise_scores_json TEXT`, holding `{"opponent_name": score, …}`.
SQLite migration is additive (`_migrate_sqlite_leaderboardentry_columns`).

## Open questions

1. **What if a baseline pair fails on some problems?** OptiProfiler
   already tolerates per-problem solver failures (`merit_history`
   marks them as NaN; `compute_scores` handles them). No change needed.

2. **What's the right K?** For v1: K = 4 baselines. We can grow to 6–8
   when an admin is willing to pay the seeding cost; the scoring math
   is identical.

3. **MATLAB combos (Phase 2).** Per ADR 0010 §"Language axis", a MATLAB
   combo carries Python baselines today (because OptiProfiler's MATLAB
   side does not yet ship the baseline registry). For ADR 0011, MATLAB
   combos run the same pairwise pipeline against MATLAB-resident
   baselines (`fminunc`, `fminsearch`, `fmincon`, `lsqnonlin`) selected
   by `language`. The pairwise math is unchanged; only the runner
   dispatch changes.

## Consequences (if accepted)

- The leaderboard is **K× slower per submission** than the broken v0
  design implied (factor of 4 today). Acceptable: still under 5 minutes
  for the smoke combos, well under the 30-minute sandbox cap.
- Tasks page can show real comparison data (per-opponent score table)
  instead of the bare scalar v0 returned. Closes a UX gap that was
  itself a symptom of the wrong design.
- The "lower-bound reference" upstream proposal (see "Why this is a
  rough first cut") is now a tracked Phase 4 item with a concrete
  scope: change `compute_scores` to read an external bound vector
  per (problem, run, eval_count).
```

## Source: docs/adr/0012-matlab-cli-sandbox.md

```markdown
# 0012 — MATLAB CLI sandbox

Date: 2026-05-19.
Status: **Accepted**, shipped in Phase 2 alongside the ADR 0011
leaderboard refactor. Promotes MATLAB from "preview" (ADR 0009 / 0010
§"Phase 2 placeholder") to a fully runnable language backend with the
same submit / sandbox / leaderboard shape as Python.

See `docs/operations.md` §"MATLAB sandbox image" for the deploy
runbook (license, build args, host-mount mode).

## Context

ADR 0009 + 0010 carved out the multi-language abstraction (a
`LanguageBackend` Protocol + a per-language sandbox image) but only
shipped Python execution. The MATLAB backend was a placeholder that
refused uploads at the submit endpoint with "coming in Phase 2".

Phase 2 finishes the feature. Three things are now real:

1. `MatlabBackend.runnable = True` — the platform accepts `.m` and ZIP
   uploads, runs the same source-text gate Python uses (different
   keyword list, same shape), and prepares the workspace exactly like
   the Python backend.
2. A real MATLAB sandbox runner — `backend/sandbox/run_task_matlab.sh`
   wraps `matlab -batch` against `run_task_matlab.m`, which:
   - parses `config.json` exactly like the Python runner;
   - branches on `leaderboard_mode` (free vs ADR-0011 pairwise);
   - uses OptiProfiler's MATLAB `benchmark()` API
     (`matlab/optiprofiler/src/benchmark.m`);
   - writes a `result.json` whose schema is byte-identical to the
     Python runner's, so the worker's `_promote_to_leaderboard_entry`
     reads either side without conditionals.
3. A `Dockerfile.matlab` that builds an image around either:
   - MathWorks's official `mathworks/matlab-deps:<release>` base + `mpm`
     to install MATLAB (Online License at runtime), or
   - a host-mounted MATLAB tree (`MATLAB_INSTALL_MODE=mounted`).

ADR 0011 already added a `language` field to every `ComboSpec`. v1
ships **only Python combos**; MATLAB combos with the parallel `m-*`
prefix arrive in a follow-up once we have at least two MATLAB-side
baselines we trust under the pairwise pipeline.

## Options considered

1. **Stay in preview** (status quo before this ADR). User can preview
   the form layout but `.m` uploads are refused. Rejected: the user
   explicitly asked for Phase 2 to be a finished product.

2. **Octave instead of MATLAB.** Free + already in apt repos, but its
   Optimization toolbox surface is a strict subset of MATLAB's, and
   OptiProfiler's MATLAB API uses MATLAB-specific class metadata for
   the problem objects. Rejected: silent semantic divergence is worse
   than a paid licence.

3. **Run MATLAB through a host-side process pool, no Docker.**
   Simpler than Docker for licensing (network license server seen
   directly), but loses the cgroup isolation we rely on in Python
   (`mem_limit`, `cpu_count`, `network_disabled=True`). Acceptable as
   a `local` fallback for dev (mirrors what Python's "local" sandbox
   mode already does), not the production path.

4. **MATLAB inside Docker, MathWorks `mpm` install during build**
   (this ADR). Image is heavy (~6 GB), but built once; subsequent
   submissions re-use it. License activation happens at container
   startup against the host's network license server or a bind-mounted
   `~/.matlab/MathWorks` directory.

We choose **(4)** for production deploys, with **(3)** as a local-mode
fallback. The runner script is the same in both modes — only the
`SandboxService.sandbox_mode` toggle differs.

## Decision

### File layout

```
backend/sandbox/
├── run_task_python.py                # Python runner (renamed from
│                                     # run_task.py to mirror MATLAB)
├── run_task_matlab.sh                # MATLAB CLI wrapper (new)
├── run_task_matlab.m                 # MATLAB-side dispatcher (new)
├── baseline_solvers_python/          # Python baselines (renamed from
│   │                                 # baseline_solvers/)
│   ├── scipy_cobyla.py …
└── baseline_solvers_matlab/          # MATLAB baselines (new)
    ├── op_fminsearch.m               # always-on baseline
    └── op_fmincon.m                  # bound/constrained baseline

backend/sandbox/Dockerfile.python      # Python sandbox (renamed from
                                       # Dockerfile so the two languages
                                       # are symmetric)
backend/sandbox/Dockerfile.matlab      # production MATLAB image (new)

backend/app/services/language_backends/matlab_backend.py
        — full implementation replacing the Phase-1 stub
```

### CLI contract (mirrors `run_task_python.py`)

```
run_task_matlab.sh \
    --solver  /workspace/solver_project/solver.m \
    --config  /workspace/config.json \
    --output  /workspace/output \
    --language matlab
```

The wrapper validates flags, ensures `matlab` is on `PATH`, and
shells into:

```
matlab -nosplash -nodesktop -nodisplay -batch \
   "addpath('/opt/optiprofiler-platform/sandbox'); \
    run_task_matlab('<solver>','<config>','<output>');"
```

`run_task_matlab.m` then:

1. Reads `<config>` via `jsondecode`.
2. `addpath(fileparts(solver_path))` so `function solver(...)` is
   visible by name (mirroring how `run_task_python.py` resolves the user's
   Python callable).
3. Adds `baseline_solvers_matlab/` to the path.
4. Dispatches to `run_free_mode` or `run_leaderboard_pairwise`.
5. Writes `<output>/result.json` with the **same key set** the Python
   runner emits, including `leaderboard_pairwise_scores`,
   `leaderboard_user_score`, `leaderboard_combo_id`, etc.

### Free vs leaderboard mode

* **Free mode.** A user-selected competitor list joins the user's solver in
  a single `benchmark()` call. When the user has not customised the list, the
  hosted default is intentionally one solver for speed: `fminsearch` for
  pure unconstrained (`ptype="u"`), and `fmincon` for bound / linear /
  nonlinear / mixed ptypes. Additional MATLAB baselines can be ticked by the
  user once they are added to `baseline_registry.py`.
  `benchmark()` call. The result panel on `/tasks` shows scalar
  per-solver scores using the existing `Cohort baseline N` label.
* **Leaderboard mode.** ADR 0011 pairwise — for each
  `opponent_solver_names` entry, run a 2-solver `benchmark()` and
  collect the user's mean-over-tau data-profile score. Aggregate is
  the mean. Score is then plumbed through the same
  `_promote_to_leaderboard_entry` worker code as Python tasks.

### Validation gate (text-based, not AST)

MATLAB has no AST surface as friendly as Python's. The v1 gate is a
**source-text adapter** over the shared hosted security policy in
`app/services/security_policy.py`, applied per `.m` file (single upload or
every `.m` inside a ZIP). It strips comments/strings first and blocks both
function-call syntax (`system(...)`) and MATLAB command syntax (`system id`)
for dangerous tokens. It blocks these capability classes:

* process / shell calls (`system`, `unix`, `dos`, `!cmd`, ...);
* filesystem reads/writes and path/environment mutation (`fopen`, `fileread`,
  `load`, `save`, `delete`, `addpath`, `cd`, `getenv`, ...);
* network APIs (`webread`, `tcpclient`, ...);
* Java/Python/native-library bridges (`java*`, `py.`, `loadlibrary`, `mex`,
  ...);
* dynamic dispatch/code construction (`eval`, `evalc`, `feval`, `str2func`,
  `builtin`, `run`, ...);
* background/parallel execution (`parpool`, `batch`, `parfeval`, ...).

Rationale per category:

* **OS / shell** (`system`, `unix`, `dos`, `!`): never legitimate for
  a solver inside a sealed sandbox.
* **Java** (`java*`, `loadlibrary`, `calllib`, `mex*`): allow the user
  to load a library we haven't audited.
* **Eval-style** (`eval`, `evalc`, `evalin`, `feval`, `str2func`, `builtin`,
  `assignin`): obfuscate banned calls trivially (`feval(['sys','tem'], …)`);
  easier to ban outright than to chase token reconstruction.
* **`addpath` / `rmpath` / `cd`**: the wrapper sets the path before
  calling user code; legitimate solvers never need to mutate it.
* **File/network/parallel APIs**: host-direct MATLAB is a public beta, so
  uploaded solvers receive all data through function arguments and cannot
  open arbitrary host files, contact remote services, or spawn background
  workers.

Refusals carry a user-safe one-liner ("`solver.m` uses `system`,
which is not allowed inside the sandbox.") plus an internal log line
the security log keeps for review.

### Sandbox image

`backend/sandbox/Dockerfile.matlab` ships two install modes:

* `MATLAB_INSTALL_MODE=mathworks` (default) — install MATLAB +
  Optimization Toolbox via `mpm` from MathWorks during build.
  ~6 GB image, ~30 min build first time; license activation happens
  at run time.
* `MATLAB_INSTALL_MODE=mounted` — skip the install step and rely on
  `-v /usr/local/MATLAB:/opt/matlab:ro` at run time. Useful when the
  host already has MATLAB installed for other workloads.

Either mode symlinks `/opt/matlab/bin/matlab` into `/usr/local/bin`
so the wrapper can call `matlab -batch` without further plumbing.

The image's `ENTRYPOINT` is the wrapper. `SandboxService._run_docker`
already forwards `--solver/--config/--output/--language` directly,
so dispatching by language is a one-line change in
`backend/app/services/sandbox.py`. The image tag itself is read from
the per-language config field `settings.sandbox_image_<lang>`
(`OP_SANDBOX_IMAGE_PYTHON` / `OP_SANDBOX_IMAGE_MATLAB`); the legacy
`OP_SANDBOX_IMAGE` env var is still honoured for the Python image
with a deprecation log so old `.env` files keep working.

### Local-mode fallback

`SandboxService._run_local` now picks the runner script by language
(Python → `run_task_python.py`, MATLAB → `run_task_matlab.sh`). On a host
without MATLAB the shell wrapper writes an `error.json` with a
specific message ("matlab binary is not on PATH inside the sandbox;
rebuild the matlab sandbox image …"). The platform happily accepts
the upload, queues the task, and surfaces the failure — local-dev
machines without MATLAB can still exercise the validation + workspace
prep paths even though the run itself fails.

### Submit-page integration

`MatlabBackend.runnable` is now `True`, so:

* `runnable_languages()` returns `["matlab", "python"]`.
* The submit form's MATLAB chip drops the "Preview · Phase 2" label.
* The cloud-run button is enabled for signed-in GitHub users in the hosted
  beta. Anonymous users can still generate local reproduce ZIPs, but cannot
  execute MATLAB on the server.
* The MATLAB plib chip group exposes `s2mpj` as enabled (default) and
  `matcutest` as disabled with a "Phase 2.1" tooltip.

### Leaderboard cross-language constraint

ADR 0011's combo language tag is now enforced at submit time:

* Python combos accept Python solvers only.
* MATLAB combos accept MATLAB solvers only.

Mixing the two would compare scores from different language bridges
(MATLAB's `benchmark()` evaluator vs Python's), which is itself a
confound — the language is part of the cache key for a reason.

### OptiProfiler MATLAB install — `setup` is the **single source of truth**

The OptiProfiler MATLAB tree ships a real installer at the repo
**root**: `setup.m`. It clones S2MPJ into
`matlab/optiprofiler/problem_libs/s2mpj/`, `addpath`s the two
directories `benchmark()` needs, optionally pulls MatCUTEst on
Linux, and tries to `savepath`. **The platform calls `setup`
unconditionally** in every code path that touches MATLAB; no
component duplicates what `setup` does internally.

Why this matters: the v0 Phase 2 design had Dockerfile.matlab
manually `addpath(genpath(.../src))` + the wrapper had a "fall back
to raw addpath if env var not set" branch. Both are subtle copies
of `setup`'s behaviour. The first time we shipped one of them, we
silently dropped S2MPJ — every v1 combo uses `plibs={'s2mpj'}`, so a
fresh machine ran with an empty problem set and nobody noticed
until it hit a real submission. Any future change upstream makes to
`setup.m` (new problem libraries, license validation, layout
shuffles) must propagate to **every** MATLAB entry point without
manual edits, or we're back in the same trap.

The contract is therefore:

* **Production Docker image (`Dockerfile.matlab`).** Carries only
  the things that have to live in the image (MATLAB binary itself
  in `mathworks` mode, system deps, the platform-side runner). The
  OptiProfiler tree itself is **host-mounted, not baked**: the
  image's `ENV OP_MATLAB_OPTIPROFILER_ROOT=/opt/optiprofiler`
  points at a bind-mount target, and `SandboxService._run_docker`
  injects `-v <host-clone>:/opt/optiprofiler:ro` automatically when
  `language=matlab`. This is the same pattern already used for
  MATLAB itself via `MATLAB_INSTALL_MODE=mounted`.

  Why not bake the clone into the image? Three reasons: (1)
  updating OptiProfiler then requires a full image rebuild instead
  of `git pull`; (2) image bloat (~hundreds of MB once we cache
  S2MPJ); (3) the in-image clone would lock to a specific ref while
  the host could already be on a newer one — drift risk. The
  symmetry with how we treat MATLAB itself makes the operator's
  story consistent: install once on the host, share across
  containers.

  **The image's `startup.m` intentionally does NOT addpath any
  OptiProfiler directory** — upstream `setup` at runtime owns that
  list. We can't run `setup` at image build time anyway because
  that needs a licensed MATLAB and licences are resolved per-deploy
  at runtime.

  Regression tests in `tests/test_phase2_acceptance.py` assert
  three things to keep this design from drifting:
  `test_dockerfile_matlab_shape` checks the Dockerfile contains
  neither `addpath('/opt/optiprofiler/matlab...` nor `git clone
  optiprofiler/optiprofiler` nor `s2mpj_matlab` (no in-image
  install).  `test_docker_bind_mounts_optiprofiler_root` checks
  `_run_docker` injects the bind-mount.
* **Wrapper (`run_task_matlab.sh`).** Always invokes
  `setup('install', struct('install_matcutest', false))` before
  dispatching to `run_task_matlab.m`. Reads
  `OP_MATLAB_OPTIPROFILER_ROOT` (production path baked in;
  developer's `.env` provides it locally). Refuses to start without
  this var, with a clear `error.json`. Legacy
  `OP_MATLAB_OPTIPROFILER_SRC` (pointing at `matlab/optiprofiler/src`)
  is still accepted via a three-level walk-up + deprecation
  warning, so existing dev machines keep working. The Phase 2 test
  `test_wrapper_calls_setup_unconditionally` asserts every CMD
  assignment in the wrapper includes a `setup(` call — there is no
  fall-back branch.
* **Local-reproduce ZIP (`run_repro.m`).** The README walks the
  user through `cd ~/OptiProfiler && matlab -batch "setup"` as the
  one-time install. If they prefer not to mutate the global
  `pathdef.m`, they set `OP_OPTIPROFILER=~/OptiProfiler` and
  `run_repro.m` calls `setup` for that session only. The driver
  explicitly refuses to fall back to `addpath(genpath(.../src))`.

Cost of "always call `setup`": ~50–100 ms per `matlab -batch`
session once S2MPJ is pre-cloned. Negligible compared to the
benchmark itself, and bought safety against the silent-drop class
of bugs that triggered this redesign.

### Local-reproduce ZIP for MATLAB

Symmetric with the Python `/api/repro-package` flow, MATLAB users can
download a ZIP that re-runs the same benchmark off-platform without
ever submitting to the cloud. Two new helpers in
`backend/app/routes/tasks.py`:

* `_build_matlab_repro_payload(task)` — assembles `repro/solver.m` (or
  `repro/solver_project/<...>` for ZIP uploads) + the matching baseline
  shims under `repro/baseline_solvers_matlab/` + `repro/config.json`
  (same schema as the Python repro) + `repro/run_repro.m`.
* `_make_matlab_run_repro_script(...)` — generates a `run_repro.m`
  that, if `benchmark()` isn't already on the path, runs OptiProfiler's
  upstream `setup` from the repo root the user passes via
  `OP_OPTIPROFILER` (or an inline edit). The driver then wires
  `solver` and baselines into a `solvers` cell and calls `benchmark()`
  with the exact `ptype/mindim/maxdim/feature/max_eval_factor` that
  ran on the cloud.

The `/api/repro-package` route accepts `solver_language=matlab` and
dispatches to these helpers. The post-run `/results/{id}/download-all`
endpoint also reads `task.solver_language` and ships the MATLAB repro
+ a MATLAB-specific top-level README. We deliberately do **not** ship
the OptiProfiler MATLAB tree itself inside the ZIP — it's tens of MB,
updates faster than this snapshot, and a single `git clone` + `setup`
once per user is the cleaner story.

End-to-end smoke test on a developer Mac (R2023b), with a fully
defaulted MATLAB path (`restoredefaultpath`) to prove `setup` is
the only path-resolution step that matters:

```
curl -F solver_file=@examples/toy_solver.m -F solver_language=matlab \
     ...other form fields... http://localhost:8000/api/repro-package > zip
unzip zip; cd repro
OP_OPTIPROFILER=~/OptiProfiler \
     matlab -batch "restoredefaultpath; rehash toolboxcache; run_repro"
# → out/.../summary_*.pdf, scores printed, exit 0
```

Acceptance test: `tests/test_phase2_acceptance.py::test_matlab_repro_payload`.

### Zombie-task reaper on worker startup

Lesson learnt during the Phase 2 → Phase 3 transition: when an
operator restarts the Celery worker mid-flight (e.g. `pkill -f
"celery.*workers.celery_app"` to pick up freshly-edited code), the
prefork child running the user's task receives `SIGTERM` and dies,
but the `Task` row stays at `RUNNING` — there's no `finally:` block
on a process that just got `kill -15`-ed. The frontend then polls a
`sandbox.log` no one is writing and shows the task as "running" for
hours.

`workers/celery_app.py` now wires up a `worker_ready` signal handler
that scans the `Task` table at startup, marks any `RUNNING` rows as
`FAILED` with a specific error message, and logs how many it reaped.
Live tasks are unaffected (a freshly-booted worker can't have
started anything yet by definition); the cost is one O(zombies)
table scan per worker boot. This makes "worker restart" a safe ops
operation — abandoned tasks are surfaced to the user immediately
with a Retry-able status, instead of haunting `/tasks` forever.

## Consequences

* **Every UI / API path now lights up for MATLAB.** No "coming soon"
  banner anywhere.
* **Production deploys must build the MATLAB image.** The image is
  *not* in the standard `docker compose up` flow because licensing is
  per-deploy and the image is too big to ship by default. The runbook
  in `docs/operations.md` walks an admin through it.
* **Failure mode when MATLAB is missing is loud.** A MATLAB submission
  on a host without the image fails with `error.json` carrying a
  specific human-readable message; the Celery worker writes that to
  `task.error_message` and `/tasks` renders it. There is no silent
  fall-through.
* **Phase 2.1 work is unblocked.** MatCUTEst install (mirroring
  `scripts/install_pycutest.sh`), additional MATLAB baselines
  (PDFO, BOBYQA wrappers), and MATLAB combos all sit on top of this
  ADR without re-touching the platform layer.
```

## Source: docs/adr/0013-dfo-ecosystem-module-registry.md

```markdown
# 0013 - DFO ecosystem module registry

Date: 2026-06-02.
Status: **Proposed**.

OptiProfiler Platform should grow from a single hosted submit form into a DFO
ecosystem surface. The platform already has a benchmarking engine, problem
libraries, feature perturbations, competitor solvers, reproduce bundles, and a
leaderboard. The next expansion should make those parts explicit modules with
metadata, validation, provenance, and smoke-test contracts instead of adding
one-off dropdown entries.

## Context

The current platform has several useful but separately managed registries:

- problem libraries (`s2mpj`, future CUTEst-family backends);
- feature names (`plain`, `noisy`, `perturbed_x0`, ...);
- solver competitors (SciPy, PDFO, PRIMA, MATLAB built-ins);
- benchmark runners (Python and MATLAB);
- report/scoring logic (performance profile, data profile, leaderboard score);
- leaderboard combos tying language, ptype, dimensions, features, and
  baselines together.

This works for the current hosted benchmark. It will not scale cleanly if the
site accepts community problem libraries, new feature generators, C/Fortran
wrapped real-world problems, or alternate benchmark reports. A real-world DFO
collection such as a SOLAR-style C-only library should not be bolted directly
onto the submit form; it needs licensing/provenance review, a deterministic
bridge, metadata extraction, and smoke tests before it is eligible for hosted
execution or leaderboard use.

## Decision

Introduce a platform-level module registry concept. The first implementation can
remain code-backed, but each module type should share the same lifecycle:

1. `submitted` - visible only to maintainers, not executed for public users.
2. `verified` - passes static checks, dependency checks, and smoke tests.
3. `hosted` - allowed in public hosted tasks under resource limits.
4. `leaderboard_eligible` - frozen enough to appear in ranked combos.

Every module record should carry:

- stable id, display label, version, maintainers, license, citation, upstream
  URL, and provenance notes;
- language/runtime bridge (`python`, `matlab`, `c_executable`, `mex`, etc.);
- resource envelope: expected time, memory, disk, network, and license needs;
- compatibility metadata: ptype, dimensions, constraints, stochasticity,
  smoothness/noise if known, and supported feature perturbations;
- validation hooks: static gate, smoke test, deterministic seed test, and
  optional benchmark micro-run;
- curation status and reviewer notes.

## Module Types

### Problem Libraries

Problem libraries expose problem metadata and a callable objective/constraint
adapter. Hosted problem libraries must provide deterministic smoke tests and a
manifest that lets the platform filter by dimension, ptype, bounds, linear
constraints, nonlinear constraints, and feature compatibility.

For C-only or executable-style collections, prefer a narrow bridge process with
fixed stdin/stdout or a compiled shared library wrapper. Do not let uploaded or
third-party problem code mutate host paths, open arbitrary files, or contact the
network.

### Feature Generators

Features are transformations around a base problem: noise, x0 perturbation,
permutation, truncation, NaN injection, quantization, and future transforms.
Each feature should declare whether it is deterministic, seedable, compatible
with constraints, and meaningful for leaderboard scoring.

### Solver Families

Competitor solvers should be grouped by family rather than scattered flat names:
SciPy, PDFO, PRIMA, MATLAB built-ins, and future packages. A family owns wrapper
files, upstream installation requirements, smoke tests, method-level metadata,
and capability flags. Cross-language names such as `pdfo` and `prima` must be
resolved by `(language, name)`, not by name alone.

### Benchmark Runners

Runners bind a language/runtime to OptiProfiler's benchmark API and produce the
same result schema. The registry should track whether a runner is local-only,
hosted, Docker-isolated, host-direct, or license-gated.

### Scoring And Reports

Performance/data profiles are the first reports, not the final product. Future
reports should be registered modules too: robustness summaries, failure
taxonomy, resource usage, per-problem traces, and alternate leaderboard scores.
Each score must declare what data it consumes and whether it is comparable
across languages and problem libraries.

### Leaderboard Combos

A leaderboard combo should reference versioned module ids: problem library,
problem subset, feature, language, ptype, dimension range, max-eval policy,
baseline set, and scoring module. Changing any dependency invalidates or
versions the combo cache.

## Public Contribution Flow

The public site should eventually offer a "Contribute" path for problem
libraries, solvers, features, and reports. Contributions should create review
records, not immediately execute arbitrary code in the hosted sandbox.

Minimum review path:

1. Upload metadata, source/reference URL, license, citation, and maintainer
   contact.
2. Run static validation and dependency detection.
3. Run isolated smoke tests on tiny deterministic examples.
4. Review provenance and resource envelope.
5. Promote to `verified`, then separately decide hosted and leaderboard status.

## Consequences

- The submit page can stay compact while the ecosystem grows; it reads grouped
  module metadata instead of hard-coded flat lists.
- New problem collections such as SOLAR-style real-world DFO problems enter
  through a maintained bridge and review process rather than direct execution.
- The leaderboard becomes easier to reason about because each ranked combo is a
  frozen composition of module versions.
- There is more upfront schema work, but it prevents the platform from turning
  into a pile of special cases as more DFO assets are added.
```

## Source: docs/problem-libraries-industrial-dfo.md

```markdown
# Industrial DFO Problem Library Survey

Date: 2026-06-07.

This is a first-pass map of public real-world, engineering, and
industry-flavoured black-box / derivative-free optimization problem
collections that could broaden OptiProfiler beyond the current CUTEst-family
libraries (`s2mpj`, `pycutest`, `matcutest`). It is intentionally practical:
each entry records the likely bridge we would need, not just the citation.

OptiProfiler already has the right architectural direction for this in
ADR 0013: new problem libraries should enter as reviewed modules with metadata,
resource envelopes, smoke tests, and versioned leaderboard eligibility.

## Current Baseline

The repository currently exposes only CUTEst-family libraries to users:

| Platform surface | Current hosted plibs | Notes |
| --- | --- | --- |
| Python backend | `s2mpj` | `pycutest` exists in leaderboard specs but is gated until the sandbox image is rebuilt with CUTEst support. |
| MATLAB backend | `s2mpj` | `matcutest` is staged but disabled in hosted uploads. |
| Local research checkout | `s2mpj_python`, `s2mpj_matlab`, `pycutest`, `matcutest` | Found under the sibling `problem_libs/` research directory, not as hosted platform modules. |

## Recommended Intake Tiers

| Tier | Candidate | Why it fits | Main bridge |
| --- | --- | --- | --- |
| 1 | SOLAR | Current, explicitly DFO/black-box, executable interface, strong GERAD/NOMAD provenance. | Build C++ executable in sandbox and wrap stdin/stdout as constrained problems. |
| 1 | STYRENE | Classic industrial chemical-process DFO benchmark; compact dimensions; true + surrogate variants. | Build C++ executables and expose truth/surrogate as separate fidelity modes. |
| 1 | EXPObench | Python package for expensive real-life black-box optimization; maintained and documented. | Python adapter per problem, with Docker-gated heavy cases. |
| 2 | CEC 2020 real-world constrained suite | Broad engineering constrained suite, many analytic engineering design cases. | Port/wrap MATLAB/C code; check license/provenance before hosted use. |
| 2 | Sahinidis SO/DFO real-world library | Explicitly simulation optimization + DFO with application problems. | Manual download/provenance review, then executable or Python wrappers per problem. |
| 2 | MECHBench | Very realistic structural mechanics / finite-element black-box cases. | Heavy Docker/OpenRadioss runtime; better as opt-in offline suite first. |
| 3 | Nevergrad / OptimSuite / Bayesmark / HPOBench style suites | Useful for ML, control, and black-box optimization, but not classical DFO profiles. | Treat as separate benchmark family, not default DFO leaderboard. |

## Candidate Details

### SOLAR

- Upstream: <https://github.com/bbopt/solar>
- Paper: "SOLAR: A solar thermal power plant simulator for blackbox
  optimization benchmarking", Optimization and Engineering / GERAD technical
  report, arXiv 2406.00140.
- Runtime: C++ project with a `make` build; produces a `solar` executable.
- License: LGPL-2.1.
- Status checked: repository updated in 2026; README reports `SOLAR v1.0.8`
  from September 2025.
- Scope: 11 solar-thermal plant instances, including single- and
  multi-objective cases.
- Dimensions/constraints: single-objective instances include 5 to 31 variables
  and 0 to 16 constraints; two listed instances are biobjective.
- Special features: stochastic instances, deterministic seeding,
  multi-fidelity static surrogates via `-fid`, replications via `-rep`.
- Interface: command-line call
  `solar pb_id x.txt -seed=S -fid=F -rep=R`.
- Fit for OptiProfiler: excellent. This should be the first executable-style
  non-CUTEst intake because it is modern, documented, and already designed as a
  black-box benchmarking problem.
- Caution: OptiProfiler's current `Problem` abstraction needs a clear policy
  for constraint outputs, multiobjective cases, stochastic histories, and
  fidelity. Start with deterministic single-objective full-fidelity instances.

Your memory that SOLAR was "only C" is directionally right historically, but
the current public repository is C++ and executable-oriented rather than a
Python/MATLAB-native library.

### STYRENE

- Upstream: <https://github.com/bbopt/styrene>
- Reference: Audet, Bechard, Le Digabel, "Nonsmooth optimization through Mesh
  Adaptive Direct Search and Variable Neighborhood Search", Journal of Global
  Optimization, 2008.
- Runtime: standard C++ with Makefiles.
- License: GPL-2.0.
- Scope: styrene production process simulation.
- Variables/constraints: 8 variables scaled to `[0, 100]`; 11 constraints.
- Outputs: 11 constraint values plus objective. Feasible means all constraints
  are nonpositive.
- Special features: separate truth and static surrogate executables.
- Fit for OptiProfiler: excellent small industrial benchmark. It is much more
  compact than SOLAR and therefore good for validating an executable bridge.
- Caution: GPL-2.0 license may affect whether we vendor code or require an
  external installation step. Prefer not vendoring until reviewed.

### EXPObench

- Upstream: <https://github.com/AlgTUDelft/ExpensiveOptimBenchmark>
- Docs: <https://algtudelft.github.io/ExpensiveOptimBenchmark/>
- Paper: "EXPObench: Benchmarking surrogate-based optimisation algorithms on
  expensive black-box functions", Applied Soft Computing, 2023.
- Runtime: Python package; some problems require Docker or Singularity.
- License: MIT.
- Scope: expensive real-life optimization problems for surrogate-based
  algorithms.
- Known heavy cases: README names ESP and Pitzdaily as Docker-dependent.
- Fit for OptiProfiler: strong candidate for a Python-native
  `expobench` plib, especially for a small curated subset that can run under
  hosted resource limits.
- Caution: EXPObench is built around surrogate-algorithm experiments, so we
  should map its evaluation-budget semantics carefully to OptiProfiler's
  performance/data profiles.

### CEC 2020 Real-World Constrained Optimization

- Upstream: <https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation>
- Scope: real-world single-objective constrained optimization benchmark used
  in CEC competitions.
- Runtime: repository exists publicly, with problem definitions in PDF form.
- License: not declared in GitHub metadata at the time of this survey.
- Fit for OptiProfiler: good source of engineering design problems once
  licensing and source provenance are settled. Many problems are analytic enough
  to be lightweight compared with SOLAR/MECHBench.
- Caution: competition suites are often distributed as reference code with
  unclear reuse terms. Treat as "wrap or reimplement after review", not a
  vendored dependency by default.

### CEC 2011 Real-World Numerical Optimization

- Source page: NTU/Suganthan CEC 2011 real-world optimization page, mirrored in
  several benchmark archives.
- Scope: older real-world numerical optimization suite used in evolutionary
  computation comparisons.
- Fit for OptiProfiler: useful historical engineering suite, but likely less
  clean than SOLAR/STYRENE for a first integration.
- Caution: source links and mirrors need manual verification. Do not use a
  random mirror as authoritative without checking license and checksums.

### Sahinidis SO/DFO Real-World Library

- Source page:
  <https://sahinidis.coe.gatech.edu/research/test-problems-so-and-dfo-algorithms>
- Page title: "A library of real world test problems for simulation
  optimization and derivative-free optimization algorithms".
- Stated scope: applications in science and engineering for testing simulation
  optimization and DFO algorithms.
- Confirmed example from the page: Cooling Crystallization Solvent Design.
- Fit for OptiProfiler: strategically important because it explicitly targets
  SO/DFO rather than general evolutionary benchmarking.
- Caution: the page is not a simple package repository. We need to download and
  inspect individual problem assets before estimating integration cost.

### MECHBench

- Upstream: <https://github.com/BayesOptApp/MECHBench>
- Paper: arXiv 2511.10821, "MECHBench: A Structural Mechanics Benchmark for
  Black-box Optimization".
- Runtime: Python-oriented benchmark around structural mechanics simulations;
  associated with OpenRadioss finite-element runs.
- Scope: realistic engineering design cases such as crashworthiness and
  structural mechanics.
- Fit for OptiProfiler: excellent "real engineering" story, but probably not
  suitable for default hosted public runs.
- Caution: expect heavy runtime, large dependencies, and careful sandboxing.
  Best first target is an offline/local benchmark pack or an unranked hosted
  demo with strict quotas.

### DIRECTGOLib

- Upstream: <https://github.com/blockchain-group/DIRECTGOLib>
- Scope: global optimization test library, including many standard analytic
  functions.
- Fit for OptiProfiler: useful if we want a broad global-optimization plib, but
  it is less "industrial DFO" than the candidates above.
- Caution: avoid duplicating problems already covered by S2MPJ/CUTEst unless we
  want a global-optimization-specific leaderboard.

### ML / AutoML Black-Box Suites

Candidate families:

- Bayesmark: ML hyperparameter optimization tasks.
- HPOBench / HPO-B: tabular and surrogate HPO benchmarks.
- Nevergrad benchmarks / OptimSuite-style tasks: broad black-box optimization
  tasks, including ML/control/noisy/discrete cases.

Fit for OptiProfiler: useful later, especially if the platform wants to compare
DFO-style solvers on ML tuning workloads. They should probably live under a
separate "black-box ML" category, not the default DFO problem-library dropdown,
because their budgets, noise models, categorical variables, and train/test
semantics differ from classical continuous DFO.

## Proposed First Implementation Path

1. Add an executable problem-library bridge type in the module-registry design:
   deterministic command, input-vector file, stdout parser, metadata manifest,
   and smoke test.
2. Integrate STYRENE first as the tiny executable pilot:
   8 variables, 11 constraints, truth only, one feasible starting point.
3. Integrate SOLAR second:
   start with deterministic, single-objective, full-fidelity instances; pin
   seed `0`; expose fidelity/replications only after base profiles work.
4. Add EXPObench as the first Python-native non-CUTEst library:
   choose a lightweight subset that does not require Docker-in-Docker.
5. Keep CEC/Sahinidis/MECHBench as reviewed backlog until license,
   dependencies, and resource envelopes are verified.

## Metadata We Should Require Before Hosting

Every non-CUTEst problem library should provide:

- stable `plib` id and upstream version or commit;
- license and citation;
- domain tags;
- problem ids, dimensions, bounds, constraints, starting points, known best
  values if available;
- deterministic smoke input and expected output tolerance;
- runtime envelope: expected time per evaluation, memory, disk, external
  binaries, container image needs;
- stochastic controls: seed, replications, noise/fidelity flags;
- OptiProfiler compatibility: single-objective vs multiobjective, constraint
  encoding, support for `plain`, `noisy`, and `perturbed_x0` features.

## Source Links

- SOLAR: <https://github.com/bbopt/solar>
- SOLAR paper page: <https://optimization-online.org/2024/06/solar-a-solar-thermal-power-plant-simulator-for-blackbox-optimization-benchmarking/>
- STYRENE: <https://github.com/bbopt/styrene>
- EXPObench: <https://github.com/AlgTUDelft/ExpensiveOptimBenchmark>
- EXPObench docs: <https://algtudelft.github.io/ExpensiveOptimBenchmark/>
- CEC 2020 real-world constrained optimization:
  <https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation>
- Sahinidis SO/DFO library:
  <https://sahinidis.coe.gatech.edu/research/test-problems-so-and-dfo-algorithms>
- MECHBench: <https://github.com/BayesOptApp/MECHBench>
- DIRECTGOLib: <https://github.com/blockchain-group/DIRECTGOLib>
```
