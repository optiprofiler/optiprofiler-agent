# OptiProfiler Agent

AI Agent system for [OptiProfiler](https://www.optprof.com), covering the full user journey of derivative-free optimization benchmarking: **ask questions → write scripts → debug errors → interpret results**.

## Architecture

| Agent | Role | When to Use |
|-------|------|-------------|
| **Agent A** — Advisor | Answer usage questions, adapt solver interfaces, generate benchmark scripts | Before testing |
| **Agent B** — Debugger | Analyze test failures, auto-fix code, retry execution | During testing (on failure) |
| **Agent C** — Interpreter | Analyze profile scores and curves, generate natural-language reports | After testing (on success) |
| **Unified Agent** | Single conversational interface that dynamically selects tools from all three agents | Anytime |

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

**Without a valid API key, any command that calls an LLM will fail.** You must configure credentials for the provider you use (you only need **one** provider).

#### 1. Environment variables (recommended for CLI and servers)

Each provider reads its key from a **fixed environment variable** (set in the shell, in your hosting dashboard, or in a `.env` file — see below):

| Provider (`--provider`) | Environment variable | Notes |
|-------------------------|------------------------|--------|
| `minimax` (CLI default) | `MINIMAX_API_KEY` | |
| `kimi` | `KIMI_API_KEY` | |
| `openai` | `OPENAI_API_KEY` | |
| `deepseek` | `DEEPSEEK_API_KEY` | |
| `anthropic` | `ANTHROPIC_API_KEY` | Install extras: `pip install 'optiprofiler-agent[anthropic]'` or `[all]` |

Example (current shell session only):

```bash
export MINIMAX_API_KEY="your-key-here"
opagent chat --provider minimax
```

#### 2. `.env` file (local development)

The package uses `python-dotenv` and loads a file named **`.env`** from the **current working directory** when you run a command (the folder you `cd` into), not from inside the installed package.

1. Create a file named `.env` in your project directory (same place you run `opagent`).
2. Put **only the key(s) you need**, for example:

```bash
# .env — use the variable that matches your chosen --provider
MINIMAX_API_KEY=your-key-here
```

3. Run the CLI from that directory, or ensure the process cwd is that directory.

For a template, copy from the repository’s [`.env.example`](https://github.com/optiprofiler/optiprofiler-agent/blob/main/.env.example) if you clone the repo; when using **only** `pip install`, create `.env` manually with the variable names from the table above.

#### 3. Python API — explicit config (no `.env` required)

If you embed this library in another app, pass the key in code or inject it from your own settings:

```python
import os
from optiprofiler_agent.config import AgentConfig, LLMConfig

config = AgentConfig(
    llm=LLMConfig(
        provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],  # or a literal string for testing only
        # model="gpt-4o",  # optional override
    ),
)
```

If `api_key` is omitted, `LLMConfig` falls back to the same environment variable as in the table.

---

## Quick Start

### CLI (Primary Interface)

```bash
# Unified agent (default if you run plain `opagent`; combines Advisor / Debug / Interpret via tools)
opagent

# Same as above, explicit subcommand
opagent agent

# Interactive advisor chat (Agent A only)
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
from optiprofiler_agent.agent_a.advisor import AdvisorAgent
advisor = AdvisorAgent(config)
reply = advisor.chat("How do I benchmark COBYLA vs Nelder-Mead on unconstrained problems?")

# Agent C: Results Interpreter
from optiprofiler_agent.agent_c.interpreter import interpret
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
  --provider [kimi|minimax|openai|deepseek|anthropic]  LLM provider (default: minimax)
  --model TEXT          Model name (overrides provider default)
  --rag                 Enable RAG retrieval for more detailed answers
  --rag-top-k INT      Number of RAG chunks to retrieve (default: 5)
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
  --provider [kimi|minimax|openai|deepseek|anthropic]
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

API keys are documented above. Defaults when you do not pass `--model`:

| Provider | Default model | API base (handled for you) |
|----------|---------------|----------------------------|
| minimax  | MiniMax-M2.7  | api.minimaxi.com |
| kimi     | kimi-k2.5     | api.moonshot.cn |
| openai   | gpt-4o        | OpenAI official |
| deepseek | deepseek-chat | api.deepseek.com |
| anthropic | claude-sonnet-4-20250514 | Anthropic official |

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
├── config.py            # Global config (LLM providers, RAG settings)
├── cli.py               # CLI entry point (all commands)
├── unified_agent.py     # Unified ReAct agent (LangGraph)
├── common/              # Shared modules
│   ├── llm_client.py    # Unified LLM wrapper (langchain)
│   ├── knowledge_base.py # Structured knowledge loader
│   ├── rag.py           # RAG retrieval (ChromaDB + two-stage search)
│   ├── quiet_ml.py      # Suppress HF/transformers noise during RAG model load
│   └── interface_adapter.py
├── agent_a/             # Product Advisor
│   ├── advisor.py
│   └── prompts/
├── agent_b/             # Auto-Debugger
│   ├── debugger.py
│   ├── local_runner.py  # Subprocess execution with process tree cleanup
│   ├── error_classifier.py
│   └── prompts/
├── agent_c/             # Results Interpreter
│   ├── interpreter.py
│   ├── result_loader.py # Parse log.txt, report.txt
│   ├── profile_reader.py # Extract curves from PDF profiles
│   ├── score_analyzer.py
│   ├── anomaly_detector.py
│   ├── summary.py
│   └── prompts/
├── validators/          # Code validation (AST syntax + API parameter checks)
├── formatters/          # Input/output normalization
└── knowledge/           # LLM Wiki knowledge base (see above)
```

## Development

```bash
git clone https://github.com/optiprofiler/optiprofiler-agent.git
cd optiprofiler-agent
pip install -e '.[dev,all]'
pytest tests/ -v
```

## License

BSD-3-Clause
