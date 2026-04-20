# OptiProfiler Agent

AI Agent system for [OptiProfiler](https://www.optprof.com), covering the full user journey of derivative-free optimization benchmarking: **ask questions → write scripts → debug errors → interpret results**.

## Architecture

| Sub-agent | Role | When to Use |
|-----------|------|-------------|
| **Advisor** ([`advisor/`](optiprofiler_agent/advisor/)) | Answer usage questions, adapt solver interfaces, generate benchmark scripts | Before testing |
| **Debugger** ([`debugger/`](optiprofiler_agent/debugger/)) | Analyze test failures, auto-fix code, retry execution | During testing (on failure) |
| **Interpreter** ([`interpreter/`](optiprofiler_agent/interpreter/)) | Analyze profile scores and curves, generate structured reports | After testing (on success) |
| **Unified Agent** ([`unified_agent.py`](optiprofiler_agent/unified_agent.py)) | Single conversational interface that dynamically selects tools from all three sub-agents | Anytime |

> Roadmap of upcoming work — including L4 constrained decoding, web search in the debugger path, and the long-term self-evolution loop — lives in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Installation

```bash
pip install optiprofiler-agent

# With RAG support (recommended)
pip install 'optiprofiler-agent[rag]'

# With PDF profile reading
pip install 'optiprofiler-agent[interpret]'

# Everything
pip install 'optiprofiler-agent[all]'
```

### CLI command name (`opagent` vs `optiprofiler-agent`)

After `pip install`, two **equivalent** executables are on your `PATH`:

| Command | When to use |
|---------|-------------|
| **`opagent`** | **Recommended** — short and easy to type. |
| `optiprofiler-agent` | Same program; name matches the [PyPI](https://pypi.org/project/optiprofiler-agent/) package. |

Running **`opagent` with no subcommand** starts the **unified agent** (ReAct tool-use loop). That is the same as `opagent agent` / `optiprofiler-agent agent`.

Examples below use **`opagent`**. Substitute `optiprofiler-agent` anywhere if you prefer the long name.

### Configure the LLM (required)

**Without a valid API key, any command that calls an LLM will fail.** You only need **one** provider configured.

#### Quickest path — `opagent init` (recommended)

```bash
opagent init
```

Interactively pick a provider (or `custom` for any OpenAI-compatible endpoint we don't ship presets for), paste your key, and the wizard writes `~/.opagent/.env` with mode `0600`. The next time you run `opagent`, your provider is the default; no flags needed.

If you skip this and just run `opagent` on a fresh machine, the wizard fires automatically the first time. To suppress this in CI / Docker, set `OPAGENT_NO_AUTO_INIT=1`.

#### Where keys are read from (precedence, highest first)

The agent loads secrets from multiple places — same convention as `gh`, `codex`, `aider`:

1. **`--provider` / `--model`** flags on the command line
2. **Real shell environment** (`export FOO=...`)
3. **`./.env`** in the current working directory (project-local override)
4. **`~/.opagent/.env`** (user-level, written by `opagent init`)
5. Built-in defaults from `PROVIDER_REGISTRY`

A real `export` always beats every dotenv file; a project-local `.env` overrides the user-level one only for that project. Useful when you want different keys per repo (e.g. client A vs client B billing) without touching the global default — copy [`.env.example`](.env.example) to `.env` in your project root and fill in only the values you need to override.

#### Provider table

| Provider (`--provider`) | Environment variable | Notes |
|-------------------------|------------------------|--------|
| `minimax` | `MINIMAX_API_KEY` | Historical fallback when nothing else is configured. |
| `kimi` | `KIMI_API_KEY` | |
| `openai` | `OPENAI_API_KEY` | |
| `deepseek` | `DEEPSEEK_API_KEY` | |
| `anthropic` | `ANTHROPIC_API_KEY` | Install extras: `pip install 'optiprofiler-agent[anthropic]'` |
| `custom` | `OPAGENT_CUSTOM_BASE_URL` + `OPAGENT_CUSTOM_MODEL` + `OPAGENT_CUSTOM_API_KEY` | Any OpenAI-compatible endpoint (self-hosted vLLM, internal gateway, unlisted vendor) — no code changes required |

When `--provider` is omitted, the default is resolved as: `OPAGENT_DEFAULT_PROVIDER` → first provider whose API key is set → `minimax`. The `init` wizard writes `OPAGENT_DEFAULT_PROVIDER` for you.

#### Pinning model / routing through a proxy (without `provider=custom`)

Built-in providers ship with reasonable model defaults (see the [Default models](#default-models-and-endpoints) table), but model names change ("kimi-k2" → "kimi-k2-thinking") and corp networks often need a proxy. Two optional env vars override **without** forcing you onto `provider=custom`:

```bash
# Pin a specific model version for the active provider:
OPAGENT_DEFAULT_MODEL=kimi-k2-thinking

# Route any provider's calls through your proxy / Azure / internal gateway:
OPAGENT_DEFAULT_BASE_URL=https://my-corp-proxy.example.com/v1
```

Precedence: `--model` flag > `OPAGENT_DEFAULT_MODEL` > registry default. Same shape for `--base-url` / `OPAGENT_DEFAULT_BASE_URL`.

#### Python API — explicit config (no `.env` required)

If you embed this library in another app, pass the key in code:

```python
import os
from optiprofiler_agent.config import AgentConfig, LLMConfig

config = AgentConfig(
    llm=LLMConfig(
        provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        # model="gpt-4o",  # optional override
    ),
)
```

If `api_key` is omitted, `LLMConfig` falls back through the same precedence chain.

---

## Quick Start

### CLI (Primary Interface)

```bash
# One-time setup (auto-fires on first run; rerun any time to switch providers)
opagent init

# Unified agent (default if you run plain `opagent`; combines Advisor / Debug / Interpret via tools)
opagent

# Same as above, explicit subcommand
opagent agent

# Interactive advisor chat (Advisor sub-agent only)
opagent chat

# Validate a benchmark script
opagent check my_script.py

# Interpret benchmark results
opagent interpret path/to/out --latest

# Debug a failing script (auto-run + fix)
opagent debug my_script.py --run

# Wiki knowledge base management
opagent wiki stats
```

Inside the unified agent, slash shortcuts include `/chat`, `/agent`, `/debug <file.py>`, `/interpret <dir> [--latest]`, `/help`, `/quit`. Run `opagent --help` for all subcommands.

### Python Library

```python
from optiprofiler_agent.config import AgentConfig, LLMConfig

config = AgentConfig(
    llm=LLMConfig(provider="minimax"),  # or "kimi", "openai", "deepseek", "anthropic"
    rag_enabled=True,
)

# Agent A: Product Advisor
from optiprofiler_agent.advisor.advisor import AdvisorAgent
advisor = AdvisorAgent(config)
reply = advisor.chat("How do I benchmark COBYLA vs Nelder-Mead on unconstrained problems?")

# Agent C: Results Interpreter
from optiprofiler_agent.interpreter.interpreter import interpret
report = interpret("path/to/results_dir", config=config, language="English")

# Unified Agent (LangGraph ReAct)
from optiprofiler_agent.unified_agent import create_unified_agent
agent = create_unified_agent(config)
result = agent.invoke({"messages": [("user", "What is ptype?")]})
```

---

## CLI Command Reference

The following use **`opagent`**; **`optiprofiler-agent ...`** is identical.

### `opagent chat`

Interactive conversation with Agent A (Product Advisor).

```bash
opagent chat [OPTIONS]

Options:
  --provider [kimi|minimax|openai|deepseek|anthropic|custom]
                        LLM provider (default: $OPAGENT_DEFAULT_PROVIDER → first
                        configured key → minimax)
  --model TEXT          Model name (overrides $OPAGENT_DEFAULT_MODEL + registry default)
  --rag                 Enable RAG retrieval for more detailed answers
  --rag-top-k INT      Number of RAG chunks to retrieve (default: 3)
  --validate            Validate code blocks in responses
  --verbose             Show system prompt size
```

In-chat commands: `/reset` (clear history), `/prompt` (show system prompt), `/quit`.

### `opagent` / `opagent agent`

Unified tool-use agent with ReAct reasoning. Automatically selects among knowledge search, script validation, error debugging, and result interpretation. **Plain `opagent` (no subcommand) runs this mode.**

> **Note:** The unified agent uses the `knowledge_search` tool internally, which requires `[rag]` extras (`pip install 'optiprofiler-agent[rag]'`). Without it, knowledge search calls will silently return no results.

```bash
opagent [OPTIONS]
opagent agent [OPTIONS]

Options:
  --provider [kimi|minimax|openai|deepseek|anthropic|custom]
  --model TEXT
```

### `opagent check`

Validate a Python benchmark script for syntax errors and API usage issues.

```bash
opagent check FILEPATH [OPTIONS]

Options:
  --language [python|matlab]  (default: python)
```

### `opagent interpret`

Analyze benchmark results and generate a natural-language report.

> **Note:** To read PDF profile curves (the default), install `[interpret]` extras: `pip install 'optiprofiler-agent[interpret]'`. Use `--no-profiles` to skip PDF reading without the extra.

```bash
opagent interpret RESULTS_DIR [OPTIONS]

Options:
  --latest              Auto-detect the latest experiment in the directory
  --provider TEXT       LLM provider for report generation
  --model TEXT          Model name
  --language TEXT       Report language (default: English)
  --no-llm             Output raw JSON summary instead of LLM-generated report
  --no-profiles        Skip PDF profile reading (faster, less detailed)
  -o, --output PATH    Write report to file instead of stdout
```

### `opagent debug`

Diagnose and fix benchmark script errors. Supports two modes:

**Manual mode** — provide code + error traceback:

```bash
opagent debug script.py --traceback error.log
opagent debug script.py --error "ValueError: at least 2 solvers"
```

**Auto mode** — run the script, catch errors, fix, and re-run:

```bash
opagent debug script.py --run [OPTIONS]

Options:
  --run                 Run the script first, then auto-debug if it fails
  --timeout INT         Timeout per run in seconds (default: 120)
  --max-retries INT     Maximum fix attempts (default: 3)
  --save-fixed PATH     Save the fixed code to a file
  --code-limit INT      Max code chars sent to LLM (0 = no limit)
  --provider TEXT       LLM provider
  --model TEXT          Model name
```

### `opagent index`

Build or rebuild the RAG vector index.

```bash
opagent index [OPTIONS]

Options:
  --force       Force rebuild even if index is up-to-date
  --no-persist  Use in-memory index only (do not save to disk)
```

### `opagent wiki`

Wiki knowledge base management commands.

```bash
opagent wiki stats           # Page count, size, category breakdown
opagent wiki lint            # Check for broken links, missing index entries
opagent wiki rebuild-index   # Rebuild RAG index from wiki content
opagent wiki rebuild-index --force  # Force full rebuild
```

---

## Default models and endpoints

API keys are documented above. Each row below is the **fallback** — both columns can be overridden globally via `OPAGENT_DEFAULT_MODEL` / `OPAGENT_DEFAULT_BASE_URL`, or per-call via `--model` (no `--base-url` flag yet; use the env var).

| Provider | Default model | API base |
|----------|---------------|----------|
| minimax  | MiniMax-M2.7  | api.minimaxi.com |
| kimi     | kimi-k2.5     | api.moonshot.cn |
| openai   | gpt-4o        | OpenAI official |
| deepseek | deepseek-chat | api.deepseek.com |
| anthropic | claude-sonnet-4-20250514 | Anthropic official |
| custom   | `$OPAGENT_CUSTOM_MODEL` | `$OPAGENT_CUSTOM_BASE_URL` |

---

## Knowledge Architecture (LLM Wiki)

The knowledge base follows the [LLM Wiki pattern](docs/llm-wiki-design.md) — a three-layer structure:

```
knowledge/
├── SCHEMA.md              # Wiki conventions and maintenance rules
├── enums.json             # Enum constants (utility file)
├── _sources/              # Raw, immutable source extractions
│   ├── python/*.json      # Extracted from Python docstrings
│   ├── matlab/*.json      # Extracted from MATLAB help comments
│   └── refs/              # Reference metadata
└── wiki/                  # Compiled, interlinked markdown pages
    ├── index.md           # Master page catalog (used for two-stage RAG)
    ├── log.md             # Chronological change log
    ├── concepts/          # Core domain concepts (DFO, benchmark, solver interface)
    ├── api/               # API reference (python/, matlab/)
    ├── guides/            # Quickstart guides, custom solver howto
    ├── profiles/          # Profile methodology and interpretation
    ├── solvers/           # Per-solver entity pages (NEWUOA, COBYLA, etc.)
    └── troubleshooting/   # Error patterns and fixes
```

RAG retrieval uses **two-stage search**: index scan → targeted vector search.

---

## Project Structure

```
optiprofiler_agent/
├── config.py            # LLMConfig + multi-source dotenv loader (cwd → ~/.opagent → defaults)
├── onboarding.py        # `opagent init` wizard (interactive provider setup)
├── cli.py               # CLI entry point (all commands incl. auto first-run init)
├── unified_agent.py     # Unified ReAct agent (LangGraph)
├── common/              # Shared modules
│   ├── llm_client.py    # Unified LLM wrapper (langchain)
│   ├── knowledge_base.py # Structured knowledge loader
│   ├── rag.py           # RAG retrieval (ChromaDB + two-stage search)
│   ├── text_clean.py    # Strips <think>...</think> before persistence
│   ├── input_loop.py    # prompt_toolkit chat input
│   └── interface_adapter.py
├── advisor/             # Advisor sub-agent (product Q&A, script generation)
│   ├── __init__.py      # re-exports AdvisorAgent
│   ├── advisor.py
│   └── prompts/
├── debugger/            # Debugger sub-agent (failure → fix loop)
│   ├── __init__.py      # re-exports debug_script, run_and_debug
│   ├── debugger.py
│   ├── local_runner.py  # Subprocess execution with process tree cleanup
│   ├── error_classifier.py
│   └── prompts/
├── interpreter/         # Interpreter sub-agent (results → BenchmarkReport)
│   ├── __init__.py      # re-exports interpret, generate_report_object
│   ├── interpreter.py
│   ├── result_loader.py # Parse log.txt, report.txt
│   ├── profile_reader.py # Extract curves from PDF profiles
│   ├── score_analyzer.py
│   ├── anomaly_detector.py
│   ├── summary.py
│   └── prompts/
├── runtime/             # ~/.opagent persistence (memory, sessions, trajectories)
│   ├── paths.py         # Canonical layout incl. env_path()
│   ├── bootstrap.py     # First-run seed copy + chmod 0600 on .env
│   └── _seed/           # Bundled defaults (.env.template, MEMORY.md, USER.md, config.yaml)
├── validators/          # Code validation (AST syntax + API parameter checks)
├── formatters/          # Input/output normalization
└── knowledge/           # LLM Wiki knowledge base (see above)
```

For what's coming next, see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Development

```bash
git clone https://github.com/optiprofiler/optiprofiler-agent.git
cd optiprofiler-agent
pip install -e '.[dev,all]'

# Unit + integration tests (250+ cases, no LLM calls; safe for CI)
pytest tests/ -v

# Optional: end-to-end eval against a real LLM (costs tokens)
python scripts/run_eval.py --mode advisor    # Agent A
python scripts/run_eval.py --mode unified    # Full ReAct loop
```

Contributor protocol (when to run what, PR checklist, release gating) lives in [`docs/ROADMAP.md`](docs/ROADMAP.md) under "Developer-facing follow-up".

## License

BSD-3-Clause
