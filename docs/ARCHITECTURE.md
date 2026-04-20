# OptiProfiler-Agent — Technical Architecture

> **Audience.** Maintainers, contributors, and engineers who want to fork
> or rebuild a similar **domain-specific AI agent** (RAG + tool-use +
> persistent memory + validation) on top of LangChain / LangGraph.
>
> **Scope.** Top-down architectural overview. For deep-dives into
> persistence (`~/.opagent/`, MEMORY.md, sessions.db, agent-writable
> wiki) see [`HERMES_INSPIRED.md`](HERMES_INSPIRED.md). For future work,
> see [`ROADMAP.md`](ROADMAP.md). For the user-facing CLI, see the
> top-level [`README.md`](../README.md).
>
> Code references throughout cite the file path so you can `rg` the
> corresponding symbol immediately.

---

## 1. Why this design?

OptiProfiler-Agent helps users go through the full **DFO benchmarking
loop** — *ask questions → write a script → debug a failing run →
interpret the resulting profile plots*. Three observations shaped the
architecture:

1. **Optimization library users hate hallucinations.** A wrong kwarg in
   a 30-minute benchmark is a ruined afternoon. We pay the engineering
   cost of an L0–L2 anti-hallucination stack (RAG + post-generation
   AST validation + lint-and-retry) up front, even though it adds
   complexity.
2. **The four phases each have a different "ideal" interface.**
   *Asking* wants a chatty advisor; *debugging* wants a deterministic
   classifier-then-LLM pipeline; *interpreting* wants strict structured
   output. Forcing one giant prompt to do all three loses on every axis.
   So we ship four agents (three specialists + one unified router) that
   share a common runtime.
3. **Pip-installable agents must Just Work on a fresh laptop.** No
   "clone this repo, edit this YAML, source these vars" rituals — that
   loses 80% of users. Hence `opagent init`, `~/.opagent/` bootstrap,
   and an auto-init hook on first invocation.

---

## 2. High-level layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI / Python API                           │
│  cli.py (click)                                __init__.py (LLMConfig)  │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Specialist agents   │  │   Unified ReAct      │  │   Onboarding         │
│                      │  │   agent              │  │                      │
│  advisor/   chat Q&A │  │                      │  │  onboarding.py       │
│  debugger/  fix code │  │  unified_agent.py    │  │  (`opagent init`)    │
│  interpreter/ report │  │  → 9 tools           │  │                      │
└──────┬──────┬────────┘  └─────────┬────────────┘  └─────────┬────────────┘
       │      │                     │                         │
       ▼      ▼                     ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Common services                              │
│  llm_client    rag    knowledge_base    text_clean    input_loop        │
│  interface_adapter    quiet_ml                                          │
└──────────┬─────────────────────────────────┬────────────────────────────┘
           │                                 │
           ▼                                 ▼
┌──────────────────────┐         ┌────────────────────────────────────────┐
│  Validators          │         │  Runtime (~/.opagent/)                 │
│                      │         │                                        │
│  syntax_checker      │         │  paths   bootstrap   memory            │
│  api_checker         │         │  session_log   wiki_local              │
│  lint_loop (L2)      │         │  trajectory   plugin                   │
└──────────────────────┘         └────────────────────────────────────────┘
```

Every arrow is **stateless from the caller's perspective**: agents read
from the runtime and validators, but never mutate them implicitly. The
only mutating side-effect on disk is `bootstrap.ensure()` (idempotent)
plus the explicit tools `remember`, `add_wiki_page`, `update_user_profile`.

---

## 3. Core architectural decisions

### 3.1 Function-calling architecture (the unified agent)

The unified agent uses **LangGraph's `create_react_agent`**, the standard
ReAct loop: the model emits a tool call, the runtime executes it, the
result is appended to the conversation, and the model decides what to
do next. We picked it for three reasons:

- **Provider-portable**: works identically against Kimi, MiniMax,
  DeepSeek, OpenAI, Anthropic — anything that exposes function-calling
  through `langchain-openai` or `langchain-anthropic`.
- **Stateless tools**: each tool is a pure-ish Python function that
  reads from disk / runtime and returns a string. No tool needs to know
  about turns, memory windows, or other tools.
- **Replaceable**: when LangGraph V1 ships `langchain.agents.create_agent`,
  we swap one import — there is no graph customisation to migrate.

The tool registry is built per-agent in
`unified_agent.py::_build_tools(config)`. `config` is captured in a
closure so each `@tool` becomes a zero-argument-from-the-LLM-perspective
function (the LLM only sees tool-relevant args, never `config`).

```
optiprofiler_agent/unified_agent.py:128  _build_tools(config) -> list[Tool]
optiprofiler_agent/unified_agent.py:301  create_unified_agent(config) -> compiled_graph
```

The system prompt is **composed at agent build time** (not per turn):
`_compose_system_prompt()` prepends `runtime.memory.frozen_snapshot()`
to the base prompt so the persistent memory is part of the static prompt
the ReAct loop sees on every iteration. Re-running `opagent agent`
rebuilds the agent and picks up newly-remembered facts — no per-turn
re-injection needed.

### 3.2 Tool catalog (the nine tools the LLM can call)

Inspired by Hermes Agent's small, opinionated tool set. Each tool has
**one** verb in its name and a docstring scoped to one use case so the
model rarely picks the wrong one. Source of truth is
`unified_agent.py::_build_tools`.

| # | Tool                | What it does                                                  | Reads from              | Writes to                       |
|---|---------------------|---------------------------------------------------------------|-------------------------|---------------------------------|
| 1 | `knowledge_search`  | RAG over the bundled OptiProfiler knowledge base              | `knowledge/wiki/`       | —                               |
| 2 | `validate_script`   | AST-level lint of a benchmark script (syntax + API + imports) | source code (in args)   | —                               |
| 3 | `debug_error`       | Diagnose a traceback, suggest a fix                           | source code + error     | —                               |
| 4 | `interpret_results` | Summarise a `out/<experiment>/` directory into Markdown       | results dir on disk     | —                               |
| 5 | `remember`          | Append a fact to long-term memory                             | —                       | `~/.opagent/MEMORY.md`          |
| 6 | `update_user_profile` | Set a whitelisted profile field                             | —                       | `~/.opagent/USER.md` (frontmatter) |
| 7 | `recall_past`       | Full-text search past chat turns (FTS5)                       | `~/.opagent/sessions.db`| —                               |
| 8 | `add_wiki_page`     | Author a new knowledge page                                   | —                       | `~/.opagent/wiki/auto/*.md`     |
| 9 | `web_search`        | External search (Tavily) — guard-railed scope                 | external API            | —                               |

**Routing discipline.** The system prompt contains a hard rule that the
model must call the tool **before** claiming it is unavailable, to
prevent a common failure mode where the model "imagines" that web
search is disabled and refuses pre-emptively. The tool itself returns a
`web_search disabled: ...` string when keys are missing, which the
model is then explicitly allowed to relay.

**Why these nine?** They map 1:1 to the user journey we ship for, and
to the L4 follow-up in the roadmap (constrained decoding can later
replace `validate_script`'s post-hoc check with a generation-time
guarantee). Adding a tenth tool means adding routing reasoning to the
prompt — a real cost — so the bar is high.

### 3.3 Specialist agents (when the unified loop is overkill)

The unified agent is convenient but pays one extra LLM round-trip per
turn (the routing decision). For workflows where the routing is fixed
in advance, we ship three specialists, each invokable directly from the
CLI:

| Agent          | Module                              | When the unified router would just route here |
|----------------|-------------------------------------|------------------------------------------------|
| `advisor`      | `optiprofiler_agent/advisor/`       | "How do I set `n_jobs`?"                       |
| `debugger`     | `optiprofiler_agent/debugger/`      | `--run` mode, automated CI                     |
| `interpreter`  | `optiprofiler_agent/interpreter/`   | Headless report generation                     |

All three share the same `AgentConfig` + `LLMConfig` and the same
`common/` services (RAG, llm_client, knowledge_base). The split is
purely for **prompt focus and CLI ergonomics**, not for capability.

#### 3.3.1 Advisor (`advisor/advisor.py`)

A single LLM call wrapped in a thin loop. The novel piece is
`_detect_language()`: it inspects the user message for Python or MATLAB
fingerprints (`fminunc`, `parfor`, `from prima import ...`) and only
injects the matching language's knowledge into the system prompt. This
keeps the prompt small (≈ 4–8 K tokens vs. the 20 K it would be if we
shipped both languages every turn) and stops cross-language confusion
("here's a Python answer for your MATLAB question").

#### 3.3.2 Debugger (`debugger/debugger.py`)

A **classify-then-route** pipeline rather than a single LLM call:

```
traceback ──► error_classifier (LLM, JSON-out)
                       │
   ┌───────────────────┼─────────────────────┐
   ▼                   ▼                     ▼
interface_adapter  pip-install advice   LLM diagnose+fix
(deterministic     (deterministic       (validators retry up to N)
 wrapper gen)       template)
```

`interface_adapter.generate_wrapper()` solves the most common DFO
debugging case (solver signature ≠ `(fun, x0, ...)`) **without** an LLM
call, by parsing the user's solver AST and emitting a known-good
adapter. We only fall back to LLM when classification is `runtime_error`
or `unknown`, and even then the suggested fix is re-validated by
`validate_benchmark_call` before being returned.

#### 3.3.3 Interpreter (`interpreter/interpreter.py`)

Strict structured-output pipeline:

```
out/<experiment>/  ──► summary.build_summary()  ──► BenchmarkSummary (verified facts, no LLM)
                                                         │
                                                         ▼
                                  llm.with_structured_output(BenchmarkReport)
                                                         │
                                          (parse fail? thinking model?)
                                                         │
                                                         ▼
                                  _extract_json_blob() + Pydantic.model_validate
                                                         │
                                                         ▼
                                    report_validator.validate_report()
                                                         │
                                            (errors? one retry with feedback)
                                                         │
                                                         ▼
                                  renderer.render_markdown / _html / _json
```

Two design notes worth copying for similar projects:

- **Rule-engine first, LLM second.** `summary.py` extracts solver
  scores, plot-curve ranks, and log anomalies from the experiment dir
  *before* any LLM call. The LLM sees a struct of *facts*; it cannot
  hallucinate that solver X won when solver Y did.
- **Thinking-model-aware JSON extraction.** Reasoning models (DeepSeek-R1,
  Kimi-thinking, MiniMax-M2) routinely wrap structured output in
  `<think>...</think>` blocks or markdown fences. Rather than fighting
  this with prompting, we pre-clean: `text_clean.strip_thinking()` then
  `_extract_json_blob()` (fenced first, then balanced-brace scan), then
  `Pydantic.model_validate`. This single fallback path turned a 40 %
  parse-failure rate on R1 into < 2 %.

### 3.4 RAG layer

`common/rag.py` — ChromaDB + `sentence-transformers`, optional
dependency under the `rag` extra. Two retrieval modes:

- `retrieve(query)`: full vector search across all wiki chunks.
- `retrieve_with_index(query)`: two-stage — first reads
  `wiki/index.md` to narrow the topic set, then runs vector search
  scoped to those wiki sections. Used by `knowledge_search` because it
  cuts noisy distractors on broad questions like *"What is ptype?"*

Chunking is deliberately coarse (`H2`-delimited, max 2 KB), one chunk
per benchmark parameter for the JSON sources, and frontmatter is
stripped. Index identity is keyed on `hash(content + chunker_version)`,
so any change to chunking forces a rebuild without manual flag-flipping.

The graceful-degradation path matters: when the `[rag]` extra is not
installed (`chromadb` missing), `KnowledgeRAG` returns an empty list
instead of raising — so a user who only does `pip install
optiprofiler-agent` still gets a working `chat` command, just without
retrieval.

### 3.5 Validation pipeline (the L0–L2 anti-hallucination stack)

Three layers, each pluggable.

```
L0 (prompts):    knowledge injection in system prompt (advisor)
                 verbatim "facts to never paraphrase" sections
L1 (retrieval):  RAG (knowledge_search tool) with wiki/index.md gating
L2 (validation): post-generation AST lint (api_checker)
                 lint-and-retry loop in chat (validators/lint_loop.py)
```

`validators/api_checker.py` does two things:

- `_BenchmarkCallVisitor`: validates the shape of `benchmark()` calls
  (≥ 2 solvers, kwarg names, enum values).
- `_ImportVisitor`: catches the most-common reference hallucination
  patterns in the wild — `optiprobe` typos and fake submodules
  (`optiprofiler.solvers` does not exist). The whitelist is **derived
  automatically** from `knowledge/_sources/python/*.json`, so the
  validator stays in sync with upstream API without a separate
  maintenance task.

`validators/lint_loop.py` is the L2 *lint-and-retry* path used by
`opagent chat --validate`: extract code blocks → run the constraint
backend → if errors, build a `ToolMessage` describing them and ask the
model once more for a corrected reply. The retry budget is **1**
(matches Cursor's reported ≈ 95 % repair rate) and only *errors* (not
warnings) trigger feedback — feeding warnings back tends to make the
model over-correct on legitimate kwargs.

The constraint backend is intentionally a `Protocol`
(`CodeConstraintBackend`) so a future L4 implementation (vLLM
grammar-constrained decoding) can be wired in by replacing
`ASTValidatorBackend` — no changes to the lint loop or to any agent.

### 3.6 LLM provider abstraction

`common/llm_client.py` is one file, < 60 lines, intentionally:

```python
def create_llm(cfg: LLMConfig) -> BaseChatModel:
    if cfg.provider == "anthropic":
        return ChatAnthropic(...)        # different SDK, separate path
    return ChatOpenAI(model=..., base_url=cfg.base_url, ...)
```

Every other provider — Kimi, MiniMax, DeepSeek, OpenAI, plus the
`custom` slot for any OpenAI-compatible endpoint we don't ship a preset
for — flows through `ChatOpenAI` with a custom `base_url`. Adding a new
preset is a one-row addition to `PROVIDER_REGISTRY` in `config.py`; no
agent code changes.

`config.py::LLMConfig.__post_init__` resolves defaults in a documented
precedence chain (kwarg → env → registry default) and supports two
escape hatches that matter in practice:

- `OPAGENT_DEFAULT_MODEL` — pin a specific model version
  (`kimi-k2-thinking` instead of the registry's `kimi-k2.5`) without
  switching to `provider=custom`.
- `OPAGENT_DEFAULT_BASE_URL` — route any provider through a corp
  proxy / Azure / internal gateway.

These mean you almost never need to reach for `provider=custom`, which
is reserved for genuinely unlisted endpoints.

### 3.7 Persistence layer (runtime)

The runtime sub-package is the closest thing to a "shared state" in the
system. Everything user-writable lives under `OPAGENT_HOME`
(default `~/.opagent/`). Full layout and rationale are in
[`HERMES_INSPIRED.md`](HERMES_INSPIRED.md); the headline points:

| Module                | Responsibility                                                            |
|-----------------------|---------------------------------------------------------------------------|
| `paths.py`            | Single source of truth for every writable path; reads `OPAGENT_HOME` lazy |
| `bootstrap.py`        | Idempotent first-run: `mkdir`, copy seeds, `chmod 0600` on `.env`         |
| `memory.py`           | `MEMORY.md` (facts) + `USER.md` (whitelisted profile)                     |
| `session_log.py`      | SQLite + FTS5 transcript store; LIKE fallback when FTS5 absent            |
| `wiki_local.py`       | Agent-authored pages with YAML frontmatter and slug-collision suffixing   |
| `trajectory.py`       | Optional JSONL dump of (sid, role, content) for SFT data prep             |
| `plugin.py`           | External skill / wiki dirs declared in `~/.opagent/config.yaml`           |

Three properties this layer guarantees:

1. **`pip install --upgrade` never overwrites user files.** The bundled
   seeds live under `runtime/_seed/` (read-only, shipped in the wheel)
   and `bootstrap.ensure()` only copies *missing* files into the home
   directory.
2. **Thinking-model contamination cannot reach storage.** Both
   `session_log.log_turn()` (FTS index) and `trajectory.append()` (SFT
   dump) call `text_clean.strip_thinking()` on assistant turns *before*
   writing. Without this, `recall_past` would re-feed the model its
   own private chain-of-thought on the next turn — a confabulation
   cascade we observed in early dogfooding.
3. **Secrets don't leak across users.** `bootstrap.ensure()`
   `chmod 0600`s `.env` after copying, so a misconfigured umask on a
   multi-tenant box doesn't expose API keys.

### 3.8 Onboarding & config plumbing

`onboarding.py::run_init()` is the wizard behind `opagent init`. It is
written without hardcoded prompt text per provider — the menu is
generated from `PROVIDER_REGISTRY` in `config.py` so adding a new
provider doesn't touch the wizard. The two helper functions
`detect_configured_providers()` and `active_default_provider()` are the
single source of truth for "what would `LLMConfig()` use right now?",
and both the wizard and `_default_provider()` call into them — they
cannot disagree.

`cli.py::_maybe_run_first_time_init()` is the auto-init hook on the
root command. It bails out cleanly when:

- `OPAGENT_NO_AUTO_INIT=1` (CI / Docker opt-out),
- the subcommand doesn't need an LLM (`init`, `home`, `wiki`, `memory`,
  `session`, `skills`, `index`, `check`),
- a key is already configured, **or**
- stdin is not a tty (degrades to a friendly warning, never blocks on
  `input()`).

### 3.9 Cross-cutting: `text_clean.strip_thinking()`

A small utility that pays back its weight every day. Reasoning models
emit `<think>...</think>` blocks that must be:

- **stripped** before showing to the user (renderer),
- **stripped** before persisting to `sessions.db` (so `recall_past`
  doesn't re-feed CoT),
- **stripped** before parsing structured output (interpreter JSON
  extraction),
- **stripped** before writing trajectories (SFT data leakage).

We deliberately centralised it in `common/text_clean.py` so all four
sites share the same regex; `interpreter.interpreter` re-exports it
under the legacy private name `_strip_thinking` to keep older imports
green.

---

## 4. Configuration & precedence

The configuration surface area is deliberately small. Three concepts:

| Concept             | Where it lives                                                  |
|---------------------|-----------------------------------------------------------------|
| **LLM config**      | `LLMConfig` dataclass (`config.py`)                             |
| **Agent config**    | `AgentConfig` dataclass (`config.py`) — knowledge dirs, RAG paths |
| **Runtime state**   | `~/.opagent/` (managed by `runtime/`)                           |

Env-variable precedence (documented top-to-bottom in `config.py`):

```
1. Explicit kwarg          LLMConfig(provider="kimi", api_key="...")
2. CLI flag                opagent agent --provider kimi
3. Real shell env          export KIMI_API_KEY=...
4. Project-local ./.env    (per-repo override; OK to commit if no secrets)
5. ~/.opagent/.env         (user-level, written by `opagent init`, mode 0600)
6. PROVIDER_REGISTRY       (built-in defaults)
```

The reason cwd `.env` is loaded **before** the user-level file in
`_load_env_files()` (despite "later wins" being the more common
convention) is that `python-dotenv` uses `override=False`: the *first*
loader to set a key wins. Loading cwd first means project-local
`.env` shadows `~/.opagent/.env` for keys it overrides, but real
`export FOO=...` (set before either loader runs) still beats both.

---

## 5. Public Python API

Most users only need three names; everything else is internal:

```python
from optiprofiler_agent import AgentConfig, LLMConfig
from optiprofiler_agent.unified_agent import create_unified_agent

agent = create_unified_agent(AgentConfig(
    llm=LLMConfig(provider="kimi", api_key="..."),
))
result = agent.invoke({"messages": [("user", "What is ptype?")]})
print(result["messages"][-1].content)
```

The three specialist agents are also public (re-exported through their
sub-package `__init__.py`):

```python
from optiprofiler_agent.advisor import AdvisorAgent
from optiprofiler_agent.debugger import debug_script, run_and_debug
from optiprofiler_agent.interpreter import interpret, interpret_from_summary
```

Re-exports are kept narrow on purpose: every name added here becomes a
backwards-compat commitment. Internal helpers (`_BenchmarkCallVisitor`,
`_extract_json_blob`, `_FakeStdin`) stay private and may be moved freely.

---

## 6. Testing strategy

Pytest is the single test driver. Categories:

| Suite                                | Coverage                                                   |
|--------------------------------------|-------------------------------------------------------------|
| `test_config*.py`                    | `LLMConfig` defaults, env precedence, multi-source loading |
| `test_onboarding.py`                 | `opagent init` flows, secret-file permissions              |
| `test_cli_first_run.py`              | Auto-init hook (CI, non-tty, key-already-set, etc.)        |
| `test_runtime.py`                    | `paths`, `bootstrap`, `memory`, `session_log`, `wiki_local`|
| `test_validators.py`, `test_lint_loop.py` | AST visitors, lint-and-retry                          |
| `test_interpreter.py` (+ schema/validator/renderer) | Structured-output + render path             |
| `test_unified_agent.py`              | Tool list shape, agent build sanity                        |
| `test_web_search.py`                 | Graceful degradation when Tavily key / package missing     |
| `test_rag.py`                        | RAG indexing + retrieval (gated behind `[rag]` extra)      |

The CI workflow (`.github/workflows/ci.yml`) runs the core suite on
**Ubuntu × macOS × Python 3.10/3.12/3.13**, and a separate job runs the
RAG suite once on Linux + Python 3.12 (HuggingFace cache makes it
expensive to fan out). A "Verify new test modules are collected" gate
guards against accidental rename / move of the freshly-added test
files, with a hard floor of 230 collected tests so a missing import
doesn't silently drop coverage.

For end-to-end behavioural smoke we use `scripts/run_eval.py` (factual
Q&A + tool-routing eval cases under `tests/eval_cases/`). It is not on
CI by default (it costs LLM tokens) but is the canonical "before
release" checklist.

---

## 7. How to extend

A few common change patterns and where they land:

| Goal                                 | Touchpoints                                                |
|--------------------------------------|-------------------------------------------------------------|
| Add a new LLM provider               | One row in `PROVIDER_REGISTRY` (`config.py`)                |
| Add a new tool to the unified agent  | New `@tool` in `unified_agent.py::_build_tools` + system-prompt entry |
| Tighten validation rules             | Extend `_BenchmarkCallVisitor` / `_ImportVisitor` (`validators/api_checker.py`) |
| Replace L2 with L4 constrained decode| New `CodeConstraintBackend` impl, plug into `lint_loop`     |
| New persistent data type             | New module under `runtime/` + path in `paths.py` + seed file in `_seed/` |
| New CLI subcommand                   | `@main.command()` in `cli.py`; if it doesn't need an LLM, add the name to `_NO_KEY_REQUIRED` |
| Bundle a new knowledge page          | Drop a Markdown file under `optiprofiler_agent/knowledge/wiki/`; rebuild index with `opagent index --force` |

The "minimum surface area" rule applies to all of these: adding to the
above is cheap; adding a new abstraction layer (a third agent kind, a
parallel runtime store, a second LLM client) is expensive and almost
always avoidable.

---

## 8. What lives elsewhere

To keep this document focused on the *system architecture*, three
neighbouring docs cover narrower angles:

- [`HERMES_INSPIRED.md`](HERMES_INSPIRED.md) — full design rationale
  for the persistence layer (`~/.opagent/`), Hermes-vs-ours
  comparisons, and `opagent init` plumbing details.
- [`ROADMAP.md`](ROADMAP.md) — near-term, mid-term, and long-term work
  including L4 constrained decoding, debugger web-search integration,
  and the (developer-only) trajectory → SFT pipeline.
- [`llm-wiki-design.md`](llm-wiki-design.md) — original design notes
  for the wiki layer that the unified agent's `add_wiki_page` tool
  writes into.

If you find yourself wanting to add a fourth, please ask first whether
it is really new content — or just a refactor of one of the above.
