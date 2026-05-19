# Agent System — Task Breakdown

> Corresponds to Roadmap Step 3

---

## Phase A-0: Knowledge Base & Agent A Core (Week 1)

- [x] Initialize repository: `pyproject.toml`, LICENSE, `.gitignore`, `.env.example`
- [x] Implement `config.py`: multi-provider LLM config (Kimi / MiniMax / OpenAI / DeepSeek / MiMo / Anthropic)
- [x] Implement `common/llm_client.py`: unified LLM call wrapper via LangChain
- [x] Implement `common/knowledge_base.py`: load JSON knowledge + markdown guides + query interface
- [x] Write `knowledge/api_params.json`: benchmark() parameters (feature/profile/problem options + core solver/fun spec)
- [x] Write `knowledge/enums.json`: FeatureName / ProfileOption / ProblemOption / FeatureOption enum values
- [x] Write `knowledge/solver_interface_spec.md`: solver function signature spec (4 problem types + DFO context)
- [x] Write `knowledge/examples.md`: 6 Python + 2 MATLAB runnable examples (DFO-only)
- [x] Write `knowledge/problem_libs_guide.md`: built-in and custom problem libraries
- [x] Write `knowledge/matlab_guide.md`: MATLAB API, solver signatures, differences from Python
- [x] Write `advisor/prompts/system_prompt.md`: role definition + DFO constraint + knowledge injection
- [x] Write `advisor/prompts/few_shots.md`: 7 Q&A pairs (factual, config, interface adaptation, DFO guidance)
- [x] Implement `advisor/advisor.py`: core Agent A class (prompt assembly + LLM call + think-tag stripping)
- [x] Write `scripts/chat.py`: interactive CLI for Agent A testing
- [x] Write `scripts/test_llm_connection.py`: LLM API connectivity test
- [x] Implement `common/interface_adapter.py`: solver signature analysis + wrapper generation
- [ ] Manual testing with 15+ Q&A scenarios, iterate prompts

**Deliverable**: `advisor.chat("...")` works in Python REPL with correct answers — **DONE (basic)**

---

## Phase A-0.5: Knowledge Automation & Quality Assurance (Week 1-2)

- [x] Write `scripts/extract_knowledge.py`: auto-extract from OptiProfiler source
  - Parse `benchmark()` docstring via `numpydoc` → `api_params.json`
  - Parse enum classes from `utils.py` → `enums.json`
  - Extract return value structure, raises, notes
  - Pin to a specific OptiProfiler version/commit for reproducibility
- [x] Add missing knowledge categories:
  - [x] Return values (`solver_scores`, `profile_scores`, `curves`) structure
  - [x] Error conditions (TypeError, ValueError triggers)
  - [x] Output file/directory structure description
  - [x] `Problem` / `Feature` / `FeaturedProblem` class reference
- [x] Write `scripts/run_eval.py`: automated Agent evaluation harness
  - [x] Keyword matching + code quality scoring
  - [x] Adversarial test cases (wrong premises, boundary conditions, DFO violations)
  - [x] LLM-as-Judge scoring (accuracy, completeness, helpfulness)
  - [x] Regression test mode: re-run after prompt/knowledge changes, compare scores
  - [x] Output: JSON report with per-question scores + overall accuracy
- [x] Write `tests/eval_cases/factual.json`: 12 factual test cases
- [x] Write `tests/eval_cases/adversarial.json`: 8 adversarial test cases

**Deliverable**: `python scripts/extract_knowledge.py` regenerates knowledge from source; `python scripts/run_eval.py` reports accuracy metrics

---

## Phase A-1: RAG + Validation + CLI (Week 2-3)

- [x] Implement `common/rag.py`: document chunking + embedding + ChromaDB vector store
- [x] Implement RAG retrieval + prompt injection logic (integrated into advisor.py)
- [x] Implement `formatters/input_parser.py`: intent classification (factual_query / interface_help / config_suggestion / script_gen / general)
- [x] Implement `validators/syntax_checker.py`: AST syntax validation for LLM-generated code
- [x] Implement `validators/api_checker.py`: benchmark() parameter validation (solver count, param names, enum values)
- [x] Implement `common/interface_adapter.py`: AST-based solver signature analysis + wrapper generation
- [x] Write CLI entry point: `opagent` / `optiprofiler-agent` — `chat` / `index` / `check` (click-based)
- [ ] Publish to PyPI: `pip install optiprofiler-agent`

**Deliverable**: local CLI is usable after `pip install`

---

## Phase A-2: Online Chat UI (Week 3-4)

- [ ] FastAPI backend: `POST /api/chat` (receive message + return reply)
- [ ] Chat UI widget (`web/chat-widget/`)
- [ ] Multi-turn conversation context management
- [ ] Integrate GitHub OAuth + rate limiting
- [ ] Token usage monitoring and budget alerts
- [ ] Deploy to `app.optprof.com/agent` or embed in website

**Deliverable**: online chat accessible from the website

---

## Phase B-0: Agent B Auto-Debug Core (Weeks 3-4, depends on sandbox platform)

- [x] Implement `debugger/error_classifier.py`: error classifier
  - Interface mismatch (signature analysis)
  - Runtime error (traceback pattern matching)
  - Missing dependency (ModuleNotFoundError)
  - Timeout
  - Numerical issue (NaN/Inf)
- [x] Implement `debugger/debugger.py`:
  - Interface mismatch → call `interface_adapter.py` to generate wrapper
  - Runtime error → LLM analysis (code + traceback → fix)
  - Retry orchestration (max 2-3 attempts)
  - Diagnostic report generation (structured Markdown)
- [x] Write `debugger/prompts/system_prompt.md`: Python debugging expert role
- [x] Write `debugger/prompts/fix_templates.md`: common fix patterns
- [ ] Write test cases (covering 5 error types)

**Deliverable**: given code + traceback, outputs a fix or diagnostic report — **DONE (core)**

---

## Phase B-1: Sandbox Platform Integration (Weeks 4-5)

- [ ] Connect to sandbox platform's task failure callback
- [ ] Implement in-sandbox retry (re-run in Docker after each fix)
- [ ] Frontend display of diagnostic reports (failed task detail page)

---

## Phase C-0: Agent C Data Analysis Engine (Weeks 3-4, independent)

- [x] Implement `interpreter/result_loader.py`:
  - Parse log.txt (experiment config, solver scores, per-run results)
  - Parse report.txt (problem table, convergence failures)
  - Auto-detect Python/MATLAB language
  - Discover PDF file paths (profiles, history plots)
- [x] Implement `interpreter/profile_reader.py`:
  - Extract step-function curves from performance/data profile PDFs (PyMuPDF)
  - Extract bar chart data from log-ratio profile PDFs
  - Support single-page and multi-page (summary) PDFs
- [x] Implement `interpreter/score_analyzer.py`:
  - Solver rankings from log.txt scores
  - Head-to-head comparison from profile curves
  - Precision cliff detection across tolerances
  - Convergence failure pattern analysis
  - Timing outlier detection
  - Curve crossover detection
- [x] Implement `interpreter/anomaly_detector.py`:
  - Extreme function values (solver failure)
  - Total evaluation failures
  - Universal convergence failure detection
  - Timing anomalies
  - Profile curve plateaus
  - Solver divergence at tight tolerances
- [x] Implement `interpreter/summary.py`: combine all analyzers into BenchmarkSummary JSON
- [ ] Write unit tests

**Deliverable**: given results_dir, outputs structured JSON summary — **DONE (core)**

---

## Phase C-1: Natural Language Report (Weeks 4-5)

- [x] Write `interpreter/prompts/system_prompt.md`: optimization benchmark expert (DFO + Dolan-Moré methodology)
- [x] Write `interpreter/prompts/report_template.md`: report template
- [x] Implement `interpreter/interpreter.py`: JSON summary → LLM polish → Markdown report (with no-LLM fallback)
- [x] CLI: `opagent interpret <results_dir>` (with --no-llm, --no-profiles, --latest, --output)
- [x] CLI: `opagent debug <script> --traceback <file>`
- [ ] Write test cases

**Deliverable**: CLI outputs a natural-language analysis report — **DONE (core)**

---

## Phase C-2: Online Integration (Weeks 5-6)

- [ ] Auto-trigger Agent C on sandbox task success
- [ ] Embed analysis report panel in frontend results page
- [ ] Support follow-up questions (e.g., "Why is solver A worse on high-dim problems?")

---

## MATLAB Support (Agent B + C) — platform-driven

> Requirement source: optiprofiler-platform multi-language sandbox spec.
> Integration contract: [`docs/PLATFORM_INTEGRATION.md`](PLATFORM_INTEGRATION.md).

### Phase 1 — Infrastructure

- [x] Pre-flight: fix `_handle_interface_mismatch` dataclass attribute access
- [x] B-6: `debugger/prompts/system_prompt_matlab.md`
- [x] B-7: `debugger/prompts/fix_templates_matlab.md`
- [x] B-8: `validators/matlab_checker.py` + `tests/test_matlab_checker.py`

### Phase 2 — Agent B pipeline (P0)

- [x] B-1: `debug_script` / `run_and_debug` accept `language` parameter
- [x] B-2: MATLAB regex patterns in `error_classifier.py`
- [x] B-3: LLM fix path loads MATLAB prompts
- [x] B-4: `_extract_code_from_reply` supports ` ```matlab` / ` ```m`
- [x] B-5: `_validate_code` dispatches to `matlab_checker`
- [x] Unified agent: `validate_script` + `debug_error` language routing
- [x] CLI: `opagent debug/check -l matlab` with auto-detect from `.m`
- [x] Tests: `test_debugger.py`, `test_error_classifier.py` MATLAB cases

### Phase 3 — Adapter + Interpreter (P1)

- [x] B-9: MATLAB branch in `interface_adapter.py`
- [x] C-1: Synthetic MATLAB fixture + `tests/test_result_loader_matlab.py`
- [ ] C-1 follow-up: replace synthetic fixture with real platform sandbox output
- [x] C-3: Language-aware paragraph in `interpreter/prompts/system_prompt.md`
- [x] C-3: `interpret_results(report_language=...)` in unified agent

### Phase 4 — PDF degradation + docs (P1/P2)

- [x] C-2 short-term: `profile_curves_available` flag + report template notice
- [x] C-4: `docs/PLATFORM_INTEGRATION.md`
- [x] ROADMAP N5 entry
- [ ] B-10: multi-file `auxiliary_files` debug (blocked on platform ZIP upload)
- [ ] Medium-term: MATLAB PDF vector-path parser for full curve extraction

### Phase 5 — Local MATLAB sandbox + L3 eval (post-roadmap)

- [x] `debugger/matlab_runner.py` — `matlab -batch` subprocess runner with timeout + process-tree kill + `getReport`-style traceback extraction.
- [x] `run_and_debug` dispatches by language via `_run_code_for_language`; `MatlabNotAvailable` degrades to a static diagnostic instead of crashing.
- [x] `pytest` marker `requires_matlab` + auto-skip when `MATOP_MATLAB_BIN` / `matlab` is missing (`tests/conftest.py`).
- [x] 15 broken `.m` fixtures with golden fix under `tests/fixtures/broken_matlab/` (interface reorder, undefined function/variable, index out of bounds, NaN/Inf/complex objective, syntax, timeout, shape and field errors).
- [x] 15 broken `.py` fixtures with golden fix under `tests/fixtures/broken_python/` (interface mismatch, missing dependency, runtime errors, numerical errors, timeout, pickle hazard).
- [x] `scripts/run_debugger_eval.py` — Pass@1 harness with `--strategy {golden,llm}` and `--language {matlab,python}`.
- [x] `tests/test_debugger_eval_matlab.py` — pytest gate at ≥70% Pass@1.
- [x] `tests/test_debugger_eval_python.py` — pytest gate at ≥70% Pass@1.
- [x] `tests/test_interpreter_eval_matlab.py` — fact-check on real MATLAB experiments.

---

## Long-term Iterations

- [ ] MATLAB script generation support (Advisor; distinct from B/C language plumbing above)
- [ ] Multi-turn conversations (modify based on previous run)
- [ ] Agent-generated scripts submitted directly to the cloud platform
- [ ] RAG enhancements (as documentation grows)
- [ ] Collect user feedback, expand test case library

---

## Reliability Rubric — "Completion Grades" for Agents A/B/C

Definition of done is layered so each Agent's reliability can be claimed
unambiguously. CI gates pass at L0; release gates publish at L1+L2; the
"Reliable" public claim requires L3 with a stated Pass@1 floor.

| Grade | Name | What we measure | Tooling | Pass bar |
|-------|------|-----------------|---------|----------|
| **L0** | Deterministic correctness | Parsers, classifiers, validators, wrappers — no LLM, no network | `pytest tests/` | **100%** green |
| **L1** | Structured I/O on real-shape data | `load_results` / `build_summary` on real & synthetic benchmark dirs (Python and MATLAB) | `pytest tests/test_result_loader*.py`, optional `MATOP_REAL_RESULTS_DIR=…` | language, scores, run_results, problems all non-empty; PDF curve fallback flag correctly set |
| **L2** | Tool routing & no-LLM eval | Unified ReAct routes to the right tool; deterministic eval cases under `tests/eval_cases/*` | `pytest tests/test_deterministic_agent_eval.py` + `scripts/run_eval.py --mode unified` | **routing ≥ 95%**, **deterministic eval ≥ 100%** of the dataset |
| **L3** | Task-level success with LLM | Agent B fixes a broken script end-to-end (Pass@1/Pass@3); Agent C produces a report whose key facts match the experiment | `scripts/run_debugger_eval.py`, `scripts/run_interpreter_eval.py`, broken-script fixtures, report fact-checks | **Pass@1 ≥ 70%**, Pass@3 ≥ 85% on ≥ 15 debugger cases per language; Interpreter report fact pass = 100% |
| **L4** | LLM-as-Judge quality | Multi-dimensional rubric over answers/reports (correctness / completeness / grounded / hallucination / instruction following) | `scripts/run_eval.py --judge`, `scripts/run_interpreter_eval.py --judge`, `scripts/run_eval_suite.py --judge` | mean ≥ 0.8, hallucination ≥ 0.9, judge coverage ≥ 0.8 on a release sample |

### Per-Agent current grade (Python vs MATLAB)

| Agent | Python | MATLAB |
|-------|--------|--------|
| A — Advisor | L0 ✅ · L2 ✅ (factual/adversarial/tool_routing eval) · L3/L4 not gated | L0 ✅ · L2 partial (`language: matlab` cases) · L3/L4 not gated |
| B — Debugger | L0 ✅ · L1 ✅ (mocked local_runner) · L2 ✅ (deterministic) · **L3 ✅ Pass@1 15/15 (golden strategy) — see `tests/test_debugger_eval_python.py`** | L0 ✅ · L1 ✅ (real `matlab -batch` sandbox via `MATOP_MATLAB_BIN`) · L2 ✅ (15 deterministic cases) · **L3 ✅ Pass@1 15/15 (golden strategy) — see `tests/test_debugger_eval_matlab.py`** |
| C — Interpreter | L0 ✅ · L1 ✅ (synthetic + thinking-model JSON path) · L2 ✅ · **L3 ✅ report fact-check runner (`scripts/run_interpreter_eval.py`)** · L4 wired, release sample pending | L0 ✅ · L1 ✅ (synthetic + **real** experiment via `MATOP_REAL_RESULTS_DIR`) · L2 ✅ · **L3 ✅ fact-check on real output — see `tests/test_interpreter_eval_matlab.py` and `scripts/run_interpreter_eval.py`** · L4 wired, release sample pending |

### How to run the L3 gates locally

```bash
# Agent B — broken-script Pass@1 (≈3 min, requires local MATLAB)
export MATOP_MATLAB_BIN=/Applications/MATLAB_R2023b.app/bin/matlab
python scripts/run_debugger_eval.py --language matlab --output /tmp/dbg.json
pytest tests/test_debugger_eval_matlab.py -v       # threshold-gated

# Agent C — fact-check against a real experiment
export MATOP_REAL_RESULTS_DIR=~/Desktop/tmp/matlab_op/out/fminsearch_fminunc_u_1_2_plain_…
pytest tests/test_interpreter_eval_matlab.py -v

# Release-grade deterministic suite with subprocess hard timeouts
python scripts/run_eval_suite.py --skip-advisor --output-dir docs/eval/latest_deterministic
```

### Open D1 work to reach the full L3+L4 bar

- [x] Expand the broken-script set to **≥ 15** per language (more interface_mismatch variants, more numerical cases, timeout/pickle cases).
- [x] Mirror the curated set on the **Python** side under `tests/fixtures/broken_python/` and wire the same Pass@1 gate.
- [x] Add multi-node deterministic eval runner for Debugger/Interpreter internal workflow nodes.
- [x] Add release-suite orchestrator with subprocess hard timeouts and aggregate reports.
- [x] Wire Agent C report fact-checking and report-specific LLM-as-Judge rubric.
- [ ] Run `scripts/run_debugger_eval.py --strategy llm` end-to-end on each provider; record Pass@1 per provider in `docs/eval/last_run.md`.
- [ ] Run provider/Judge release sample via `scripts/run_eval_suite.py --judge` and record the latest accepted artifact in `docs/eval/latest/summary.md`.
