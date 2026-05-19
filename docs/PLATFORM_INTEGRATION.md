# Platform Integration — Language Parameters

This document describes how **optiprofiler-platform** should call
**optiprofiler-agent** for multi-language (Python / MATLAB) support.

## Agent B — Auto-Debug

### Python API

```python
from optiprofiler_agent.debugger.debugger import debug_script

result = debug_script(
    code=script_source,       # str: full script contents
    error=traceback_text,     # str: stderr / MATLAB command window output
    language="matlab",        # "python" (default) | "matlab"
)
# result.fixed_code          — corrected script, or None
# result.diagnostic_report   — Markdown diagnostic for the UI
# result.validation_passed   — bool
```

### Platform endpoint mapping

| Platform request field | Agent parameter |
|---|---|
| `script` / `code` | `code` |
| `error` / `traceback` | `error` |
| `language` (`python` \| `matlab`) | `language` |

When the sandbox runs a `.m` file, pass `language="matlab"`.

### Deferred: multi-file debug (B-10)

`auxiliary_files: dict[str, str]` is **not** implemented yet. Wait until
the platform ZIP-upload feature ships, then extend `debug_script` in
lockstep.

### Local MATLAB sandbox (for dev / L3 eval)

`run_and_debug(language="matlab", …)` will use `matlab -batch` directly
when a MATLAB binary is reachable. Resolution order:

1. Explicit `matlab_bin` keyword argument
2. `MATOP_MATLAB_BIN` environment variable (preferred for CI / contributors)
3. `matlab` on `PATH`

If none resolves, the runner returns a synthetic `RunResult` whose
`stderr` explains the gap, so the diagnose loop can still emit a static
report. The platform sandbox does **not** depend on this — it ships its
own MATLAB image. This is purely for local development and the
`tests/test_debugger_eval_matlab.py` Pass@1 gate.

```bash
# macOS dev box
export MATOP_MATLAB_BIN=/Applications/MATLAB_R2023b.app/bin/matlab
```

CI defaults to skipping the `requires_matlab`-marked tests. Set the
env var (or install MATLAB Runtime in the runner image) to opt in.

---

## Agent C — Interpret Report

### Python API

```python
from optiprofiler_agent.interpreter.interpreter import interpret, generate_report_object

report_md = interpret(
    results_dir="/path/to/experiment",
    language="English",       # report *output* language (English, 中文, …)
    read_profiles=True,       # attempt PDF curve extraction
    llm_enabled=True,
)

# Or typed object for native UI rendering:
report = generate_report_object(summary, language="English")
```

### Language fields (two different concepts)

| Field | Meaning | Set by |
|---|---|---|
| `BenchmarkSummary.language` | How the benchmark was **run** (`python` / `matlab`) | Auto-detected from `test_log/scratch.m` or `_scratch.py` |
| `interpret(..., language=...)` | **Report prose** language | Platform query param, e.g. `?language=中文` |

The platform should:

1. Pass the user's locale preference as `language` to `interpret()`.
2. **Not** confuse it with the code language used for debugging.

### MATLAB PDF degradation

When MATLAB-generated profile PDFs cannot be parsed, the summary sets
`profile_curves_available = false` and the rendered report includes a
one-line notice. Text-based analysis (scores, rankings, problem table)
still works.

---

## Unified agent tools (online chat)

| Tool | New / updated parameter |
|---|---|
| `validate_script` | `language: "python" \| "matlab"` |
| `debug_error` | `language: "python" \| "matlab"` |
| `interpret_results` | `report_language: str` (default `"English"`) |

---

## CLI equivalents (for manual testing)

```bash
# Debug MATLAB solver
opagent debug solver.m -l matlab -e "Undefined function or variable 'foo'."

# Validate MATLAB script
opagent check benchmark_run.m -l matlab

# Interpret results (report language)
opagent interpret ./out/experiment_xxx --language 中文
```

---

## Recommended platform rollout

1. **Phase 2a** — ship MATLAB sandbox + raw PDF/logs (no auto-debug).
2. **Phase 2b** — wire `POST /api/debug` with `language=matlab` once agent
   P0 tasks are deployed.
3. **Phase 2c** — wire `POST /api/interpret?language=...` for localized reports.
