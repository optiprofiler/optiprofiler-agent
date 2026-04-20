# Hermes-Inspired Persistence Layer

This document describes the persistence and self-evolution layer added to the
OptiProfiler Agent in v0.1, and explains exactly which ideas were borrowed
from [NousResearch / hermes-agent](https://github.com/NousResearch/hermes-agent),
which were adapted for our scope, and which were intentionally left out.

The goal is twofold:
1. Give a faithful credit trail for an external project we learned from.
2. Document the design clearly enough that a contributor can extend it
   without reading the Hermes source.

> **Forward-looking work.** Items planned but not yet shipped — including
> remote trajectory upload, SFT/DPO dataset auto-derivation, and the
> wiki-self-update loop — live in [`ROADMAP.md`](ROADMAP.md) under the
> *Long-term self-evolution loop* horizon. This document covers what is
> already in the codebase as of v0.1.

---

## 1. Background and motivation

The pre-existing OptiProfiler Agent was a stateless LangGraph ReAct loop:
each `opagent agent` invocation started from a blank slate, with only the
shipped knowledge wiki and the in-process message list as context. That
worked for one-shot Q&A, but had three concrete shortcomings:

1. **No cross-session continuity.** A user who told the agent "I always use
   BOBYQA on bound-constrained problems" had to repeat that fact every
   session.
2. **No self-extension.** When the agent learned something specific and
   factual through `knowledge_search` plus reasoning, there was nowhere to
   pin it down for next time. The next session re-derived the same answer.
3. **No replay / offline analysis path.** Trajectories were not persisted,
   so we could not curate a dataset for offline evaluation or fine-tuning.

Hermes Agent solves an analogous set of problems for a general-purpose
assistant by externalizing memory, skills, and conversation history into
plain files under `~/.hermes/`. We adopted the same architectural shape,
because:

* It matches our "clean out-of-the-box" goal: nothing to configure, just
  files under `~/.opagent/` that the user can `cat`, `grep`, or edit.
* It is dependency-free (stdlib `sqlite3` + `shutil` + `pathlib`).
* It is orthogonal to the ReAct loop, which we did not want to rewrite.

---

## 2. File layout (`OPAGENT_HOME`, default `~/.opagent/`)

```
~/.opagent/
├── .env                   # user-level secrets (mode 0600) — written by `opagent init`
├── MEMORY.md              # agent's declarative facts (append-only)
├── USER.md                # user profile (whitelisted fields)
├── wiki/auto/             # agent-written wiki pages (RAG indexes them)
├── skills/                # user / agent skill packages (reserved)
├── sessions.db            # SQLite + FTS5 — every chat turn ever
├── trajectories/          # ShareGPT JSONL dumps (opt-in only)
├── config.yaml            # advanced / opt-in switches
└── .bootstrapped.json     # version + seed manifest
```

The pip-installed package itself ships **only read-only seed files** under
`optiprofiler_agent/runtime/_seed/`. On first launch,
[`runtime.bootstrap.ensure()`](../optiprofiler_agent/runtime/bootstrap.py)
copies any missing seed file into `OPAGENT_HOME`. **Existing user files are
never overwritten**, even after `pip install --upgrade`. Files listed in
`bootstrap._SECRET_FILES` (currently just `.env`) are also `chmod 0600` on
POSIX so the bundled template never lands world-readable.

`OPAGENT_HOME` can be overridden with the `OPAGENT_HOME` environment
variable — handy for CI, sandboxes, and per-project isolation.

### 2.1 Provider configuration (`opagent init`)

Secrets resolution is intentionally **multi-source**, so corp users with
shell-managed env vars and casual users with a single user-level file
both Just Work. Precedence (highest → lowest), implemented in
[`config._load_env_files`](../optiprofiler_agent/config.py):

1. Real shell `export` (already in `os.environ` before Python starts)
2. `./.env` in the current working directory (project-local override)
3. `~/.opagent/.env` (user-level, written by the wizard)
4. `PROVIDER_REGISTRY` defaults

`opagent init` ([`onboarding.py`](../optiprofiler_agent/onboarding.py)) is
the interactive front-end. It:

- detects which provider keys already exist (so users see
  `Detected existing API key(s) for: kimi, minimax`, not `'unknown'`)
- writes `OPAGENT_DEFAULT_PROVIDER=<choice>` so subsequent commands
  without `--provider` resolve to the user's pick
- never erases sibling provider keys — switching from minimax to kimi
  keeps both keys in the file and just flips the default
- supports a `custom` provider for any OpenAI-compatible endpoint
  (vLLM, internal gateway, unlisted vendor) via
  `OPAGENT_CUSTOM_BASE_URL` / `_MODEL` / `_API_KEY`
- supports two **non-custom** overrides for the common
  "pin a model version" / "route through a proxy" use cases without
  forcing users onto `provider=custom`:
  - `OPAGENT_DEFAULT_MODEL=kimi-k2-thinking`
  - `OPAGENT_DEFAULT_BASE_URL=https://my-corp-proxy/v1`

Auto-trigger: the first `opagent` invocation on a fresh machine
([`cli._maybe_run_first_time_init`](../optiprofiler_agent/cli.py))
launches the wizard if no provider key resolves; non-TTY environments
(CI, piped input) degrade gracefully with a printed warning instead of
hanging on `input()`. Set `OPAGENT_NO_AUTO_INIT=1` to disable entirely.

---

## 3. Feature-by-feature comparison with Hermes

Each subsection has the same shape:

* **What Hermes does** — our reading of the upstream design.
* **What we adopted** — the slice we kept.
* **What we changed for our scope** — pragmatic differences.
* **Where it lives** — code paths in this repo.

### 3.1 Two-level declarative memory (`USER.md` + `MEMORY.md`)

* **What Hermes does.** Hermes maintains a `USER.md` (user profile) and a
  `MEMORY.md` (free-form notes) that are stitched into the system prompt as
  a "frozen snapshot" at the start of each session. A background memory
  worker can append to these files automatically.
* **What we adopted.** Same two files, same frozen-snapshot injection
  mechanism. The agent sees `USER profile` plus the most recent `MEMORY`
  notes prepended to the base system prompt.
* **What we changed.** No background worker; updates happen synchronously
  through two whitelisted tools (`remember`, `update_user_profile`). The
  profile schema is restricted to a small enum
  (`name`, `role`, `preferred_solver`, `preferred_language`, `project_root`)
  to prevent prompt-injection-driven schema drift. The snapshot is bounded
  by a character cap with truncation from the oldest end first.
* **Where it lives.**
  [`runtime/memory.py`](../optiprofiler_agent/runtime/memory.py),
  [`runtime/_seed/USER.md`](../optiprofiler_agent/runtime/_seed/USER.md),
  [`runtime/_seed/MEMORY.md`](../optiprofiler_agent/runtime/_seed/MEMORY.md),
  injected via `_compose_system_prompt()` in
  [`unified_agent.py`](../optiprofiler_agent/unified_agent.py).

### 3.2 Cross-session search (`sessions.db`, FTS5)

* **What Hermes does.** Persists every conversation turn into a SQLite
  database with an FTS5 virtual table, then exposes a `search_sessions`
  tool so the agent can recall prior context on demand.
* **What we adopted.** Identical schema shape, identical FTS5 trigger
  pattern, same idea exposed as the `recall_past` tool and as the
  `opagent session search` CLI command.
* **What we changed.** We rely solely on the stdlib `sqlite3` module — no
  ORM, no external migration framework. We also fall back to a `LIKE`-based
  search when FTS5 is not compiled into the local SQLite build, so the
  feature degrades gracefully on minimal Python distributions. Query
  tokens are quoted to neutralize FTS5 reserved punctuation.
* **Where it lives.**
  [`runtime/session_log.py`](../optiprofiler_agent/runtime/session_log.py).
  CLI integration in
  [`cli.py`](../optiprofiler_agent/cli.py) (`chat` and `agent` commands log
  every turn; new `session` group exposes `search` / `list`).

### 3.3 Agent-writable wiki pages (`wiki/auto/`)

* **What Hermes does.** Hermes lets the agent author skill files
  (`SKILL.md`) that are auto-loaded into future sessions, effectively
  letting the model extend its own procedural memory.
* **What we adopted.** A narrower form: the agent can author *focused
  factual wiki pages* via the `add_wiki_page` tool. They land in
  `OPAGENT_HOME/wiki/auto/` with a frontmatter block (source, timestamp,
  summary).
* **What we changed.** We did not adopt full procedural skills, because our
  agent's tool surface is small and stable. Auto-pages are transparent to
  the existing RAG pipeline:
  [`KnowledgeRAG._gather_chunks`](../optiprofiler_agent/common/rag.py)
  was extended to scan `wiki/auto/` (prefix `wiki/auto`) so the two-stage
  retriever already treats them as wiki entries on the next index rebuild.
  Slug collisions get a numeric suffix so prior pages are never silently
  overwritten.
* **Where it lives.**
  [`runtime/wiki_local.py`](../optiprofiler_agent/runtime/wiki_local.py),
  RAG hook in
  [`common/rag.py`](../optiprofiler_agent/common/rag.py).

### 3.4 Trajectory dump (developer-facing, opt-in)

* **What Hermes does.** Optionally records every interaction trajectory in
  a ShareGPT-style format for offline analysis or future RL training.
* **What we adopted.** Same intent: append every turn as JSONL, one file
  per session, ready to be replayed or used as supervised fine-tuning
  fodder.
* **What we changed.** **Disabled by default.** Two opt-in switches:
  - `OPAGENT_TRAJECTORY_DIR=/some/path` environment variable, or
  - `trajectory.enabled: true` in `config.yaml`.

  The trajectory and session_log paths are kept separate: the session_log
  is meant to be small and FTS-scanned on every `recall_past`, while
  trajectories can grow without bound for export.
* **Where it lives.**
  [`runtime/trajectory.py`](../optiprofiler_agent/runtime/trajectory.py).

### 3.5 Plugin / external context dirs (developer-facing, opt-in)

* **What Hermes does.** Supports plugin-provided memory backends, knowledge
  graphs, and external context directories for advanced setups.
* **What we adopted.** A minimal version: `config.yaml` accepts
  `plugin.external_wiki_dirs` and `plugin.external_skill_dirs`. The wiki
  list is fed straight into RAG so a private domain corpus can be mounted
  without forking the package.
* **What we changed.** No plugin protocol or entry-points yet — just lists
  of paths. We considered adopting a richer plugin spec, but for a small
  research project the cost (extension API stability, error surfaces) was
  not worth it.
* **Where it lives.**
  [`runtime/plugin.py`](../optiprofiler_agent/runtime/plugin.py),
  consumed by RAG in
  [`common/rag.py`](../optiprofiler_agent/common/rag.py).

### 3.6 Things we explicitly did *not* port

* **Skills Hub** — Hermes ships a community marketplace of skills with
  signing and trust levels. Out of scope for this project: no community to
  serve, no trust infrastructure to maintain.
* **Messaging-platform gateways** (Slack / Discord). Our entry point is
  the CLI; we will not multiplex over chat platforms.
* **Reinforcement-learning training pipelines.** The trajectory dump is
  the *substrate* for that, but the training itself stays out of the
  package.

---

## 4. User-facing vs developer-facing boundary

We make a deliberate split between defaults and opt-in features so that the
fresh-install experience stays clean.

| Capability                | Audience       | Default                | How to enable                                          |
| ------------------------- | -------------- | ---------------------- | ------------------------------------------------------ |
| Bootstrap of `OPAGENT_HOME` | end user     | always on              | (automatic on first chat / agent invocation)           |
| Two-level memory          | end user       | always on              | (automatic; tools `remember` / `update_user_profile`)  |
| Session search            | end user       | always on (logs turns) | (automatic; tool `recall_past`, CLI `session search`)  |
| Agent-writable wiki       | end user       | always on              | (automatic; tool `add_wiki_page`)                      |
| Trajectory dump           | developer      | off                    | `OPAGENT_TRAJECTORY_DIR=...` or `config.yaml`          |
| External wiki / skill dirs | advanced user | off                    | edit `config.yaml`                                     |

The CLI subcommands mirror the same split: `memory`, `session`, `home`,
`skills` are always exposed; advanced controls live in `config.yaml` and do
not show up as first-class commands.

---

## 5. Out-of-the-box guarantees

After `pip install optiprofiler-agent` and the first `opagent agent` run:

1. `~/.opagent/` is created automatically with all seed files.
2. Memory and session logs work without any user action.
3. The agent already sees an empty-but-formatted persistent context block.
4. No advanced features are silently turned on.
5. `pip install --upgrade` adds new seed files but never touches the
   user's `MEMORY.md`, `USER.md`, `config.yaml`, `sessions.db`, or
   `wiki/auto/`.

The end-to-end smoke test that backs (1)–(5) lives in
[`tests/test_runtime.py`](../tests/test_runtime.py).

---

## 6. Where to look in the codebase

* New runtime sub-package:
  [`optiprofiler_agent/runtime/`](../optiprofiler_agent/runtime/)
  — `paths.py`, `bootstrap.py`, `memory.py`, `session_log.py`,
  `wiki_local.py`, `trajectory.py`, `plugin.py`.
* Seed templates shipped in the wheel:
  [`optiprofiler_agent/runtime/_seed/`](../optiprofiler_agent/runtime/_seed/).
* Agent integration:
  [`optiprofiler_agent/unified_agent.py`](../optiprofiler_agent/unified_agent.py)
  — four new tools (`remember`, `update_user_profile`, `recall_past`,
  `add_wiki_page`) plus `_compose_system_prompt()` injection.
* CLI integration:
  [`optiprofiler_agent/cli.py`](../optiprofiler_agent/cli.py)
  — `bootstrap.ensure()` on launch, per-turn `session_log.log_turn(...)`,
  new command groups `memory`, `session`, `home`, `skills`.
* Tests:
  [`tests/test_runtime.py`](../tests/test_runtime.py).

---

## 7. Hallucination Guard for Code Emission

The persistence layer above gives the agent *memory*. This section covers
a separate concern in the same package: **output correctness for
generated Python code**, focused narrowly on the *reference hallucination*
subclass — fabricated package, module, or symbol names. Two recurring
patterns motivated this work:

```python
# Wrong: package-name typo
from optiprobe import benchmark

# Wrong: nonexistent submodule
from optiprofiler.solvers import bobyqa
```

Both are syntactically valid Python that any LLM can confidently emit and
that no upstream `langchain` / `langgraph` layer will catch.

### 7.1 Scope

Out of the four hallucination classes commonly distinguished in the
literature (reference, faithfulness, factual, reasoning), this design
addresses only the **reference** class. Reference hallucinations have a
property the others lack: the set of legal references is finite,
enumerable, and statically derivable from the upstream API source — so a
deterministic post-generation verifier is *sufficient* to catch them.
The other three classes are mitigated indirectly by the existing RAG
layer and tool discipline; this section makes no claim about them.

### 7.2 Layered architecture

The implementation is split across three layers that compose
multiplicatively:

| Layer | Mechanism | Where it lives |
|---|---|---|
| **L0 — Grounding** | facts injected into `_SYSTEM_PROMPT_BASE`; retrievable wiki page indexed by RAG | [`unified_agent.py`](../optiprofiler_agent/unified_agent.py); [`knowledge/wiki/api/python/imports-and-exports.md`](../optiprofiler_agent/knowledge/wiki/api/python/imports-and-exports.md) |
| **L1 — Post-generation AST verifier** | `_ImportVisitor` walks the AST of every Python code block; reports against an auto-derived whitelist with `difflib` did-you-mean suggestions | [`validators/api_checker.py`](../optiprofiler_agent/validators/api_checker.py) |
| **L2 — Repair loop** | on L1 *errors*, append a synthetic feedback message, re-invoke the agent once with the same conversational state, then re-validate | [`validators/lint_loop.py`](../optiprofiler_agent/validators/lint_loop.py); [`cli.py:_run_lint_loop`](../optiprofiler_agent/cli.py) |

Two layers are deliberately *not* implemented:

* **L3 — Cite-or-die** (require every claim to carry a
  `# verified against:` comment). Rejected after a small ablation: the
  citation discipline degraded code quality on requests that legitimately
  needed novel composition, with negligible marginal precision on top of
  L1.
* **L4 — Constrained decoding** (grammar-mask the token sampler so
  non-whitelisted imports become unreachable). Not built because the
  third-party LLM providers we currently support (MiniMax, Kimi,
  Anthropic) do not expose a context-free grammar API; the only way to
  enable this today would be to self-host an inference server (e.g. vLLM
  with `outlines`/`xgrammar`), which is out of scope for this package.
  Section 7.6 describes how the codebase is structured so an L4 backend
  can be added without touching call sites.

### 7.3 L0 — Grounding

Two complementary surfaces, by design redundant:

1. **System prompt facts.** `_SYSTEM_PROMPT_BASE` ends with an explicit
   paragraph listing the package name, the ten public top-level symbols,
   and the three or four submodule paths users (and LLMs) commonly
   invent. This is unconditional context, paid once per session.
2. **RAG-retrievable wiki page.** [`imports-and-exports.md`](../optiprofiler_agent/knowledge/wiki/api/python/imports-and-exports.md)
   carries the same facts in markdown, indexed alongside the rest of
   `knowledge/wiki/`. When the agent issues `knowledge_search('Python
   imports …')`, this page surfaces near the top.

Either mechanism in isolation is fragile (prompt context is truncated by
long histories; RAG is only invoked when the agent decides to call it).
Together they cover both behaviours.

### 7.4 L1 — Post-generation AST verifier

#### 7.4.1 Whitelist derivation

The set of legal `from optiprofiler import …` symbols is **derived at
import time** from the same JSON files the rest of the validator already
consumes (`knowledge/_sources/python/*.json`):

```python
@functools.lru_cache(maxsize=4)
def _load_optiprofiler_python_exports(_kb_id: int = 0) -> frozenset[str]:
    kb = KnowledgeBase()
    exports: set[str] = set()
    exports.update((kb.get_classes("python") or {}).keys())   # Problem, Feature, FeaturedProblem
    exports.update((kb.get_plib_tools("python") or {}).keys()) # s2mpj_load, pycutest_select, ...
    if kb.get_benchmark("python"):
        exports.add("benchmark")
    return frozenset(exports)
```

The whitelist is therefore a *projection* of the same data structures
used to validate `benchmark()` kwargs. There is no parallel list to keep
in sync; updating `_sources/python/*.json` updates the validator.

The cache key `_kb_id` is a synthetic integer (default `0`) that lets
unit tests bypass the singleton when constructing a fixture KB.

#### 7.4.2 The visitor

`_ImportVisitor` inspects two AST node types:

| Node | Pattern detected | Severity |
|---|---|---|
| `ast.ImportFrom` | `from <PACKAGE_TYPOS> import …` | `error` |
| `ast.ImportFrom` | `from optiprofiler.<sub> import …` | `warning` |
| `ast.ImportFrom` | `from optiprofiler import <unknown>` | `warning` (with `difflib.get_close_matches` did-you-mean) |
| `ast.Import` | `import <PACKAGE_TYPOS>` | `error` |
| `ast.Import` | `import optiprofiler.<sub>` | `warning` |
| `ast.Import` | `import optiprofiler` | (silent — always allowed) |

`PACKAGE_TYPOS` is a small literal set:
`{"optiprobe", "opti_profiler", "opti-profiler", "optiprofile", "optiproflier"}`.
Adding new typos requires editing one line in `api_checker.py`.

`*` imports (`from optiprofiler import *`) are passed through silently —
the static AST cannot tell which symbol the user intended, and a star
import is never the kind of mistake the verifier is designed to catch.

#### 7.4.3 Severity model

The severity assignment is intentional and asymmetric:

* **Errors** are reserved for problems where there is exactly one
  reasonable interpretation (a literal package-name typo). Errors trigger
  the L2 repair loop.
* **Warnings** are used for problems where the model's intent is
  ambiguous — `from optiprofiler.utils import …` could mean "the user
  has a fork that adds a `utils` module" — and where automated repair
  has historically produced over-corrections. Warnings are surfaced to
  the user through a yellow Rich panel but never fed back to the LLM.

This split was added after observing that broad warning-feedback caused
the model to flip-flop between two equally invented submodule names on
successive retries.

#### 7.4.4 Integration with the existing checker

`_ImportVisitor` runs *in addition to*, not instead of, the
pre-existing `_BenchmarkCallVisitor`. Both visitors traverse the same
parsed tree; their issue lists are merged into a single
`ValidationResult`. Import checking only runs when `language == "python"`
— the visitor short-circuits otherwise so that MATLAB code blocks pass
through untouched.

### 7.5 L2 — Repair loop

The loop is implemented in two cooperating modules:

* `validators/lint_loop.py` defines the pure functions
  `lint_reply(reply, backend)`, `format_feedback_for_llm(report)`, and
  `format_for_user(report)`. None of these touch the LLM or the CLI;
  they are unit-testable in isolation.
* `cli.py:_run_lint_loop(unified, reply, messages)` orchestrates the
  side-effecting parts: invokes the validator, optionally re-invokes the
  agent, and prints surviving issues.

#### 7.5.1 Sequence

1. CLI receives `reply` from `unified.invoke({"messages": messages})`.
2. `lint_reply(reply)` extracts code blocks via the existing
   `extract_code_blocks` (markdown-fenced `python`/`py` blocks plus the
   bare-code heuristic) and runs the configured backend over each.
3. If `report.has_errors`, `format_feedback_for_llm(report)` builds a
   single string containing only the *error* messages, prefixed with an
   instruction to rewrite the previous reply and a hint to call
   `knowledge_search` first if uncertain.
4. The feedback is appended to `messages` as a `("user", ...)` tuple
   (see §7.5.3 for why not `ToolMessage`) and the agent is invoked once
   more. The constant is `MAX_VALIDATION_RETRIES = 1`, defined in
   `lint_loop.py` for easy tuning.
5. The new reply is re-validated. Surviving issues — error or warning —
   are rendered into a yellow `rich.panel.Panel` titled "Validator
   notes". The reply itself is *not* modified or hidden.

#### 7.5.2 Failure modes

The loop is wrapped in a defensive `try/except` at every external
boundary. If `lint_reply` raises (e.g. due to a malformed code block
that even `ast` cannot recover from) or the retry invocation fails (e.g.
provider 5xx), the user gets the original reply with a one-line
`[dim]validator skipped: …[/]` notice. The validator is never allowed
to block delivery of a working response.

#### 7.5.3 Message-shape choice: `("user", str)` vs `ToolMessage`

LangGraph's `ToolMessage` requires a matching `tool_call_id` from a
prior assistant message. Synthesising one for after-the-fact validator
output is brittle (the message-id surface of LangGraph's `create_react_agent`
is not part of its stable API). We instead append a
`("user", feedback_text)` tuple — the same shape the user's own input
takes — and rely on the prefix "Validator found errors in your last
reply's Python code blocks." to make the source clear to the model.
This trades a small loss in trajectory cleanliness (the validator
appears as a "user turn" in dumps) for stability against LangGraph
internal changes. If/when LangGraph stabilises a public way to inject
synthetic tool results, this is the call site to switch.

### 7.6 Forward-compatibility: the `CodeConstraintBackend` Protocol

`lint_loop.lint_reply` does not call the AST validator directly; it
calls a backend that satisfies a small Protocol:

```python
class CodeConstraintBackend(Protocol):
    name: str
    def validate(self, code: str, *, language: str = "python") -> ValidationResult: ...
```

Two implementations ship today:

* `ASTValidatorBackend` — wraps `validate_benchmark_call`. Used by the
  CLI by default.
* `NullBackend` — returns an empty `ValidationResult`. Used by tests
  (and available as an escape hatch for users who want to disable the
  loop entirely).

A future grammar-mask backend would land alongside these without
touching `lint_loop.py` or `cli.py`:

```python
class GrammarBackend:
    name = "vllm-grammar"
    def __init__(self, grammar_path: Path, endpoint: str): ...
    def validate(self, code: str, *, language: str = "python") -> ValidationResult:
        # `validate` becomes a sanity check; the actual constraint runs
        # at decode time on the inference server.
        ...
```

`Protocol` was chosen over an `ABC` (abstract base class) for two
reasons:

1. Backends may live in third-party packages (e.g. an `opagent-vllm`
   plugin) and should not need to inherit from anything in this
   repository.
2. `Protocol` plays better with `typing.runtime_checkable` should we
   later want `isinstance` guards.

**Note on the priority of L4 application sites.** When the §8 structured
report pipeline was added, it became the more attractive first target
for a grammar-decoding backend than import validation: the report
schema is a small enumerated grammar (six sections, a handful of
`Literal[...]` enums), whereas legal Python imports are an open set
that grammar-masking would prune awkwardly. If/when a vLLM worker
becomes available, the recommended sequence is therefore to grammar-
constrain `BenchmarkReport` generation first (§8.6) and only later
attempt to grammar-constrain user-emitted Python.

### 7.7 Adjacent surface: prose-level typo guard

`scripts/check_prose.py` carries four small regex rules
(`optiprobe`, `opti-profiler`, `opti_profiler`, `optiprofile`) that fail
CI when **our own** documentation, prompts, or wiki misspell the
package name. This is independent of the runtime guard but supports the
same invariant — the L0 grounding surfaces (system prompt + wiki) must
themselves be clean for L0 to be effective.

### 7.8 Test coverage

| Test file | What it pins down |
|---|---|
| `tests/test_validators.py` (`TestImportWhitelist`, 8 cases) | typo → error; fake submodule → warning; unknown symbol → did-you-mean; legal symbol → silent; bare `import optiprofiler` → silent; dotted import → warning; non-Python language → skipped |
| `tests/test_validators.py` (`TestConstraintBackend`, 2 cases) | `ASTValidatorBackend` produces issues; `NullBackend` produces none |
| `tests/test_lint_loop.py` (10 cases) | clean reply → empty report; typo → errors; warning-only → no LLM feedback; null backend → disables checks; custom backend invoked once per code block; feedback omits warnings; user-facing format includes severity and line number |

### 7.9 Limitations and known scope

* **The verifier only catches the reference class.** Faithfulness,
  factual, and reasoning hallucinations are outside its design and may
  still surface in the final reply.
* **Retry budget is fixed at 1.** The constant is at module scope for
  easy tuning, but raising it has a real latency cost and produced
  diminishing returns in our internal traces.
* **Warnings never trigger a retry.** This is a deliberate
  over-correction guard, not an oversight (see §7.4.3).
* **MATLAB is not yet wired.** The visitor short-circuits on
  `language != "python"`. The `language` parameter threads through the
  full call stack so a sibling `_MatlabImportVisitor` is a localized
  addition — no architectural change required.
* **The lint loop runs only inside the `agent` subcommand.** The
  `chat` subcommand uses the legacy `_validate_reply` path, which prints
  issues but does not retry. This is intentional: `chat` mode is the
  pure advisor without tool routing, and the retry semantics assume an
  agent that can re-plan with new context.

---

## 8. Structured Report Pipeline (Interpreter sub-agent)

§7 addressed correctness in *generated code*. This section addresses
correctness in *generated narrative reports*: the Markdown analysis
produced by `opagent interpret` against a benchmark output directory.

The motivating observation is that DFO benchmark reports are highly
structured artefacts — a fixed set of sections, each with a small
number of fields — and free-form Markdown emission gives the model
many low-value freedoms it routinely abuses (inventing solver names,
fabricating problem identifiers, contradicting summary statistics).

### 8.1 Three-stage pipeline

| Stage | Module | Responsibility |
|---|---|---|
| 1. Rule engine | [`interpreter/summary.py`](../optiprofiler_agent/interpreter/summary.py) | Parse `log.txt`, profile PDFs, score files into a `BenchmarkSummary` dataclass. Pure Python, no LLM. |
| 2. Structured LLM | [`interpreter/interpreter.py`](../optiprofiler_agent/interpreter/interpreter.py) (`_generate_structured_report`) | Bind a `BenchmarkReport` Pydantic schema via `llm.with_structured_output(method="json_schema")`; invoke; validate; retry once on business-invariant errors. |
| 3. Renderer | [`interpreter/renderer.py`](../optiprofiler_agent/interpreter/renderer.py) | Apply the Jinja template to the validated report. |

Stage 1 already existed; stages 2 and 3 are the additions. The same
code path supports Markdown, JSON, and HTML output formats — the
renderer is the only module that knows about presentation.

### 8.2 Schema design

[`report_schema.py`](../optiprofiler_agent/interpreter/report_schema.py)
defines `BenchmarkReport` as a Pydantic v2 `BaseModel` composed of
six section sub-models:

```
BenchmarkReport
├── overview                : ReportOverview
├── performance_profile     : PerformanceProfileSection
├── data_profile            : DataProfileSection
├── convergence_issues      : ConvergenceIssuesSection
├── anomalies               : AnomaliesSection
└── recommendations         : RecommendationsSection
```

Several decisions matter:

* **Enums via `Literal[...]`** (not `enum.Enum`) so Pydantic emits
  flat JSON Schema enums that the provider's JSON-Schema mode can
  enforce at decode time. `Severity`, `ActionKind`, `AnomalyKind`,
  and `SchemaVersion` are all literal types.
* **Field descriptions are part of the LLM contract.** Each
  `Field(..., description=...)` carries instructions ("MUST be one of
  the solver names listed in the input summary"); when LangChain
  serialises the schema it includes those descriptions, so the model
  sees them as part of its tool-spec, not just as comments.
* **Schema versioning** via `schema_version: Literal["1.0"]` so we can
  evolve the shape without ambiguity for downstream consumers (web
  platform, CI report archive).
* **Cross-field invariants are NOT in the schema.** "Winner solver
  must be one of `summary.solver_names`" cannot be expressed in JSON
  Schema (it depends on runtime data). Those go in
  `report_validator.py` — see §8.4.

### 8.3 `with_structured_output` binding

`_bind_structured_output(llm)` tries
`llm.with_structured_output(BenchmarkReport, method="json_schema")`
first, falls back to `method="function_calling"`, then to the
unmethod-tagged default, and finally returns `None` if all three
raise. `None` triggers the legacy free-form Markdown path
(`_legacy_freeform_report`) so users on providers without any
structured-output support still get a useful report.

This three-tier fallback matters because:

* **OpenAI / MiniMax / Kimi**: prefer `json_schema` — the provider
  enforces the schema at decode time, with token-level constraints in
  many implementations.
* **Anthropic**: `function_calling` is preferred (Claude expresses
  structured output through tool-use messages).
* **Older providers**: even the unmethod-tagged path fails; we fall
  back to plain Markdown.

### 8.4 Business-invariant validator

[`report_validator.py`](../optiprofiler_agent/interpreter/report_validator.py)
implements `validate_report(report, summary)`. Invariants checked:

| Invariant | Severity |
|---|---|
| `performance_profile.winner_at_tau1 ∈ summary.solver_names` | error |
| `performance_profile.most_robust ∈ summary.solver_names` | error |
| `data_profile.most_efficient ∈ summary.solver_names` | error |
| `convergence_issues.entries[*].solver ∈ summary.solver_names` | error |
| `anomalies.entries[*].affected_solvers[*] ∈ summary.solver_names` | error |
| `recommendations.actions[*].target_solver ∈ summary.solver_names ∪ {None}` | error |
| `convergence_issues.entries[*].failure_count ≤ summary failure cap` | warning |
| `convergence_issues.common_failure_problems[*] ∈ known summary problems` | warning |

Severity assignment follows the same asymmetric rule as §7.4.3:
unambiguous reference errors trigger a retry; ambiguous overshoots
become user-visible warnings. The `format_feedback_for_llm()` helper
deliberately omits warnings from retry feedback for the same
over-correction-flip-flop reason discussed in §7.

### 8.5 Retry orchestration

The retry loop in `_generate_structured_report` mirrors §7.5: on
errors it appends a single `HumanMessage(content=feedback)` and
re-invokes the same `structured_llm`. `MAX_REPORT_RETRIES = 1`,
matching the §7 budget. Warnings on the second pass are surfaced to
the user but do not trigger a third attempt.

### 8.6 Renderer and the metadata-grounding split

[`renderer.py`](../optiprofiler_agent/interpreter/renderer.py) renders the
validated report with [`report.md.j2`](../optiprofiler_agent/interpreter/templates/report.md.j2).
The template receives **two** objects:

* `report` — the LLM-produced narrative
* `summary` — the rule-engine ground truth

Objective metadata (solver names, scores table, dimension range,
problem libraries) is taken from `summary`, **never** from `report`.
This guarantees that the user-visible numbers cannot be hallucinated
even if the LLM mis-narrates around them.

`render_html(report, summary)` reuses `render_markdown` and wraps the
output in a self-contained HTML shell; the optional `markdown` package
upgrades the result, otherwise it falls back to a `<pre>`-wrapped
Markdown view. Either way, no new top-level dependency is introduced.

### 8.7 Why this is the better first target for L4 grammar decoding

The `BenchmarkReport` schema is a near-ideal candidate for grammar-
constrained decoding:

* **Small grammar**: six section types, all `Literal[...]` enums
  enumerable in well under 30 productions.
* **High mask ROI**: enum fields are short (single word) — masking
  saves the model from an entire category of failure (`switch_solvr`
  typo, made-up severity level) on every production.
* **Bounded blast radius**: the worst case of a grammar pruning the
  model's intended token is "fall back to an allowed enum value",
  which is operationally fine. Compare with code emission, where
  pruning a token can make the code unreachable or syntactically
  broken.

This re-orders the L4 backlog: when a self-hosted vLLM worker becomes
available, port the report schema first; revisit import-checking
grammar later (or never, given L0–L2 already at >99% on
reference-class hallucinations).

### 8.8 Test coverage

| Test file | Cases | What it pins down |
|---|---:|---|
| `tests/test_report_schema.py` | 9 | schema instantiation, JSON round-trip, enum/Literal rejection, descriptive text in JSON Schema |
| `tests/test_report_renderer.py` | 11 | every section header present, scores table grounded in summary, empty-section fallback strings, HTML wrapper renders headline |
| `tests/test_report_validator.py` | 11 | unknown solver/problem produces correct severity, failure-count overshoot is warning, feedback string omits warnings, user-facing format includes both severities |
| `tests/test_interpreter.py` | (modified) | structured-output absence falls back to free-form path with exactly one invoke |

### 8.9 Limitations

* **Provider must support some form of structured output.** The
  three-tier fallback covers everything we ship today, but a hand-rolled
  HTTP-based provider could still land in the free-form path; this is
  by design.
* **The schema covers DFO benchmark reports only.** Other report types
  (e.g. solver tuning sweeps, regression test summaries) would need
  parallel schemas; the renderer is generic enough to host them but
  no other report family is wired today.

---

## 9. Web Search (scope-restricted)

The agent ships with one optional outbound tool: `web_search`, backed
by Tavily through `langchain-tavily`. Two questions drove its design:
(1) when does *any* outbound tool make sense in an agent that already
has a curated RAG index? and (2) how do we prevent the outbound tool
from becoming a hallucination amplifier?

### 9.1 Scope

The tool is wired into the unified `agent` command only. It is
**deliberately not** exposed to:

* `chat` (pure advisor) — the advisor's value comes from being
  authoritative on OptiProfiler; web hits would dilute this.
* `interpret` (report) — report facts must come from the user's
  benchmark output, not the web.

Inside `agent`, the system prompt enforces a two-rule policy:

* **Allowed**: open-world facts about external solvers, recent papers,
  GitHub issues, decoding tracebacks from third-party libraries
  (e.g. `scipy`, `pycutest`), and installation tips for those tools.
* **Disallowed**: anything about the `optiprofiler` package itself —
  API, parameters, examples, install flow. Those must come from
  `knowledge_search`.

The policy is enforced at the *prompt* layer (model-side routing),
not in code, because only the LLM can route by intent. The tool
docstring repeats the policy so the model sees it as part of the
tool's own contract.

### 9.2 Graceful degradation

[`tools/web_search.py`](../optiprofiler_agent/tools/web_search.py)
follows a strict no-raise contract. Three failure modes return a
single-line explanation rather than an exception:

| Condition | Returned string |
|---|---|
| `langchain-tavily` not installed | `web_search disabled: install with pip install optiprofiler-agent[web]...` |
| `TAVILY_API_KEY` not set | `web_search disabled: set the TAVILY_API_KEY environment variable...` |
| Provider call raises | `web_search error: <message>` |

This means the rest of the agent boots and runs in environments where
web search is undesired or unconfigured (CI, air-gapped deployments,
the default `pip install`). The tool simply tells the model "I am
disabled, answer from your own knowledge with a clear caveat."

### 9.3 Why Tavily

Three providers were considered:

* **Tavily** — LangChain first-party integration; results are
  pre-cleaned for LLM consumption (titles, snippets, URLs in a flat
  schema); 1000 free requests/month.
* **DuckDuckGo (`duckduckgo-search`)** — no API key, but rate-limited
  and snippet quality varies.
* **Brave Search API** — high quality, requires key, smaller free
  tier.

Tavily was chosen for the LLM-friendly response shape and for keeping
the dependency surface small (`langchain-tavily` only).

### 9.4 Result formatting

`_format_results` collapses the Tavily payload into a numbered
text block:

```
[1] <title>
<snippet, truncated to 500 chars>
url: <url>

[2] ...
```

Snippet truncation matters: raw Tavily snippets can be 1-2k tokens,
which floods the agent's context and crowds out the actual reasoning.
500 characters per hit, three hits per call (the `max_results=3`
default) is enough signal without dominating the turn.

### 9.5 Why this composes with §7 and §8

The hallucination defences in §7 and the structured-output discipline
in §8 only work if the inputs to the agent are themselves trustworthy.
A naive web-search tool would smuggle untrusted facts into the
RAG-shaped surface area of the agent. The scope policy keeps web
search confined to *open-world* questions where the alternative is
"the model invents an answer from training data anyway", which is
strictly worse than "the model summarises a search result and cites
the URL."

### 9.6 Test coverage

`tests/test_web_search.py` (10 cases) pins down: empty query rejection,
import-error degradation, missing-key degradation, formatted-result
shape, no-hits message, exception swallowing, snippet truncation,
docstring scope keywords (the model-side contract), and
system-prompt scope keywords (defence in depth against a refactor that
loses the RAG-first / debug-allowed wording).

### 9.7 Limitations

* **The scope policy is prompt-enforced.** A determined model with
  enough confusion can still ask `web_search` an OptiProfiler
  question. The downstream §7 import validator catches the most
  damaging consequence (fabricated `from optiprofiler.x import y`),
  and `knowledge_search` always returns higher-quality results when
  the answer is in the wiki, so in practice the model picks the right
  tool the vast majority of the time. A code-level allowlist (e.g.
  reject queries matching `\boptiprofiler\b`) would be possible but
  was deferred — false positives would be more annoying than the
  rare scope leak.
* **No web-search retry / cache**. Each call hits the network. A
  thin cache keyed by query hash is a likely follow-up if cost or
  latency becomes an issue.
* **The unified agent's trajectory dump (§5) does not redact API
  keys**. `TAVILY_API_KEY` is read from the environment and never
  echoed in tool output, but operators should confirm their dump
  retention policy before sharing trajectories externally.
