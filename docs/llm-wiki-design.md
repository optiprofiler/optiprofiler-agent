# LLM Wiki Knowledge Architecture for OptiProfiler Agent

## Background: The LLM Wiki Pattern

Traditional RAG (Retrieval-Augmented Generation) treats documents as static
artifacts — chunked, embedded, and retrieved at query time. The LLM
re-discovers knowledge from scratch on every question. There is no
accumulation; nothing is built up between queries.

The **LLM Wiki** pattern (inspired by Andrej Karpathy's proposal) takes a
fundamentally different approach. Instead of retrieving from raw documents,
the LLM **incrementally builds and maintains a persistent wiki** — a
structured, interlinked collection of markdown files that sits between the
user and the raw sources. When new source material arrives, the LLM reads
it, extracts key information, and **integrates** it into the existing wiki:
updating entity pages, revising topic summaries, flagging contradictions,
and strengthening the evolving synthesis.

The critical insight: **the wiki is a persistent, compounding artifact.**
Cross-references are pre-built. Contradictions are already flagged.
Synthesis already reflects everything ingested. The wiki grows richer with
every source added and every question asked.

## Three-Layer Architecture

```
┌───────────────────────────────────────┐
│  Layer 3: Schema (SCHEMA.md)          │  ← Conventions & rules
├───────────────────────────────────────┤
│  Layer 2: Wiki (wiki/)                │  ← Compiled, interlinked pages
│    index.md  log.md                   │
│    concepts/  api/  guides/           │
│    profiles/  solvers/  troubleshoot/ │
├───────────────────────────────────────┤
│  Layer 1: Raw Sources (_sources/)     │  ← Immutable extractions
│    python/*.json  matlab/*.json       │
│    refs/bibliography.md               │
└───────────────────────────────────────┘
```

### Layer 1 — Raw Sources (`_sources/`)

Immutable source material. The LLM reads from these but never modifies
them. Contains JSON files extracted from OptiProfiler source code
(docstrings, class definitions, API signatures) and reference metadata.

### Layer 2 — Wiki (`wiki/`)

LLM-generated and LLM-maintained markdown pages. Each page focuses on one
concept, entity, or topic. Pages are interlinked via relative markdown
links. The LLM owns this layer entirely — it creates pages, updates them
when new sources arrive, and maintains cross-references.

### Layer 3 — Schema (`SCHEMA.md`)

The governance document that tells the LLM (or human maintainer) how the
wiki is structured: naming conventions, page templates, frontmatter format,
linking rules, and maintenance workflows.

## How We Adapted It for OptiProfiler Agent

### Page Organization

Wiki pages are organized into six categories:

| Category         | Purpose                                      | Example Pages                           |
|------------------|----------------------------------------------|-----------------------------------------|
| `concepts/`      | Core domain concepts                         | DFO, benchmark function, problem types  |
| `api/`           | API reference (per-language)                 | benchmark parameters, Problem class     |
| `guides/`        | Step-by-step instructions                    | Python quickstart, custom solver guide  |
| `profiles/`      | Profile methodology and interpretation       | Performance profiles, data profiles     |
| `solvers/`       | Per-solver entity pages                      | NEWUOA, COBYLA, Nelder-Mead             |
| `troubleshooting/` | Error diagnosis and fixes                  | Common errors, solver compatibility     |

### Page Format

Every wiki page follows this template:

```markdown
---
tags: [concept, dfo]
sources: [_sources/python/benchmark.json]
related: [concepts/benchmark-function.md, concepts/problem-types.md]
last_updated: 2025-04-13
---

# Page Title

Content organized by H2 sections...

## See Also

- [Related Page](relative-link.md)
```

### Two-Stage RAG Retrieval

Instead of flat vector search over all chunks, we use a two-stage approach:

1. **Stage 1 — Index Scan**: Read `wiki/index.md` (a lightweight catalog of
   all pages with one-line summaries). Identify 2-3 relevant wiki sections
   or pages based on the query.

2. **Stage 2 — Vector RAG**: Perform vector search only within the targeted
   pages, returning top-k chunks. Since wiki pages are already compiled,
   high-quality text, retrieval precision is higher.

This reduces noise, saves tokens, and improves answer quality.

## Wiki Maintenance Workflows

### Ingest — Adding New Knowledge

1. Place raw source material in `_sources/` (or update existing files)
2. LLM reads the raw source and extracts key information
3. Create or update wiki pages with cross-references
4. Update `wiki/index.md` with new/modified entries
5. Append an entry to `wiki/log.md`
6. Rebuild the RAG vector index (`build_index(force=True)`)

### Lint — Health Check

Periodically verify wiki integrity:

- Orphan pages with no inbound links
- Stale claims superseded by newer sources
- Missing pages referenced by wiki links
- `index.md` consistency with actual files
- Gaps that could be filled with new sources

### Query — Answering Questions

1. Read `index.md` to locate relevant pages
2. Retrieve fine-grained chunks via vector search
3. Synthesize answer with citations to wiki sources
4. Optionally file valuable answers back as new wiki pages

## Why This Works for OptiProfiler

OptiProfiler's knowledge domain is **compact but deep**: a single
`benchmark()` function with 50+ parameters, 4 problem types, 10+ features,
multiple profile methodologies, and language-specific nuances (Python vs
MATLAB). This makes it an ideal candidate for the wiki pattern:

- **Parameter pages** compile type, default, description, related parameters,
  and common pitfalls into one place — no need to re-derive from raw JSON.
- **Solver entity pages** accumulate knowledge about each solver's behavior
  across different profile types and features.
- **Cross-references** between concepts, API, and troubleshooting pages let
  the LLM follow association trails rather than relying on embedding
  similarity alone.
- **The index** serves as a table of contents that fits in a single prompt,
  enabling the LLM to navigate efficiently without vector search.

## Updating the Wiki Going Forward

The wiki should be updated whenever:

1. **OptiProfiler releases a new version** — re-run `extract_knowledge.py`
   to refresh `_sources/`, then update affected wiki pages.
2. **New solver is added to benchmarks** — create a new solver entity page
   in `wiki/solvers/`.
3. **User discovers a common error pattern** — add to
   `wiki/troubleshooting/` and update index.
4. **New research paper is published** — add reference to
   `_sources/refs/bibliography.md`, update methodology pages.
5. **Agent answers a novel question** — if the answer required synthesizing
   multiple sources, consider filing it as a new wiki page.
6. **CLI checks** — from the repo root, use `opagent wiki stats`, `opagent wiki lint`,
   and `opagent wiki rebuild-index` (same as the `optiprofiler-agent` command name).

The key principle: **the wiki is never "done"** — it compounds with every
interaction and every new source.
