# Agent System — Task Breakdown

> Corresponds to Roadmap Step 3

---

## Phase A-0: Knowledge Base & Agent A Core (Week 1)

- [x] Initialize repository: `pyproject.toml`, LICENSE, `.gitignore`, `.env.example`
- [x] Implement `config.py`: multi-provider LLM config (Kimi / MiniMax / OpenAI / DeepSeek / Anthropic)
- [x] Implement `common/llm_client.py`: unified LLM call wrapper via LangChain
- [x] Implement `common/knowledge_base.py`: load JSON knowledge + markdown guides + query interface
- [x] Write `knowledge/api_params.json`: benchmark() parameters (feature/profile/problem options + core solver/fun spec)
- [x] Write `knowledge/enums.json`: FeatureName / ProfileOption / ProblemOption / FeatureOption enum values
- [x] Write `knowledge/solver_interface_spec.md`: solver function signature spec (4 problem types + DFO context)
- [x] Write `knowledge/examples.md`: 6 Python + 2 MATLAB runnable examples (DFO-only)
- [x] Write `knowledge/problem_libs_guide.md`: built-in and custom problem libraries
- [x] Write `knowledge/matlab_guide.md`: MATLAB API, solver signatures, differences from Python
- [x] Write `agent_a/prompts/system_prompt.md`: role definition + DFO constraint + knowledge injection
- [x] Write `agent_a/prompts/few_shots.md`: 7 Q&A pairs (factual, config, interface adaptation, DFO guidance)
- [x] Implement `agent_a/advisor.py`: core Agent A class (prompt assembly + LLM call + think-tag stripping)
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

- [x] Implement `agent_b/error_classifier.py`: error classifier
  - Interface mismatch (signature analysis)
  - Runtime error (traceback pattern matching)
  - Missing dependency (ModuleNotFoundError)
  - Timeout
  - Numerical issue (NaN/Inf)
- [x] Implement `agent_b/debugger.py`:
  - Interface mismatch → call `interface_adapter.py` to generate wrapper
  - Runtime error → LLM analysis (code + traceback → fix)
  - Retry orchestration (max 2-3 attempts)
  - Diagnostic report generation (structured Markdown)
- [x] Write `agent_b/prompts/system_prompt.md`: Python debugging expert role
- [x] Write `agent_b/prompts/fix_templates.md`: common fix patterns
- [ ] Write test cases (covering 5 error types)

**Deliverable**: given code + traceback, outputs a fix or diagnostic report — **DONE (core)**

---

## Phase B-1: Sandbox Platform Integration (Weeks 4-5)

- [ ] Connect to sandbox platform's task failure callback
- [ ] Implement in-sandbox retry (re-run in Docker after each fix)
- [ ] Frontend display of diagnostic reports (failed task detail page)

---

## Phase C-0: Agent C Data Analysis Engine (Weeks 3-4, independent)

- [x] Implement `agent_c/result_loader.py`:
  - Parse log.txt (experiment config, solver scores, per-run results)
  - Parse report.txt (problem table, convergence failures)
  - Auto-detect Python/MATLAB language
  - Discover PDF file paths (profiles, history plots)
- [x] Implement `agent_c/profile_reader.py`:
  - Extract step-function curves from performance/data profile PDFs (PyMuPDF)
  - Extract bar chart data from log-ratio profile PDFs
  - Support single-page and multi-page (summary) PDFs
- [x] Implement `agent_c/score_analyzer.py`:
  - Solver rankings from log.txt scores
  - Head-to-head comparison from profile curves
  - Precision cliff detection across tolerances
  - Convergence failure pattern analysis
  - Timing outlier detection
  - Curve crossover detection
- [x] Implement `agent_c/anomaly_detector.py`:
  - Extreme function values (solver failure)
  - Total evaluation failures
  - Universal convergence failure detection
  - Timing anomalies
  - Profile curve plateaus
  - Solver divergence at tight tolerances
- [x] Implement `agent_c/summary.py`: combine all analyzers into BenchmarkSummary JSON
- [ ] Write unit tests

**Deliverable**: given results_dir, outputs structured JSON summary — **DONE (core)**

---

## Phase C-1: Natural Language Report (Weeks 4-5)

- [x] Write `agent_c/prompts/system_prompt.md`: optimization benchmark expert (DFO + Dolan-Moré methodology)
- [x] Write `agent_c/prompts/report_template.md`: report template
- [x] Implement `agent_c/interpreter.py`: JSON summary → LLM polish → Markdown report (with no-LLM fallback)
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

## Long-term Iterations

- [ ] MATLAB script generation support
- [ ] Multi-turn conversations (modify based on previous run)
- [ ] Agent-generated scripts submitted directly to the cloud platform
- [ ] RAG enhancements (as documentation grows)
- [ ] Collect user feedback, expand test case library
