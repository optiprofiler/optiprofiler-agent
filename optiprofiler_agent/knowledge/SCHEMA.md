# OptiProfiler Wiki Schema

This document defines the conventions for maintaining the OptiProfiler
knowledge wiki. Any LLM or human editor must follow these rules when
creating or updating wiki pages.

## Directory Structure

```
knowledge/
├── SCHEMA.md              # This file
├── enums.json             # Enum constants (utility, not part of wiki)
├── _sources/              # Raw, immutable source extractions
│   ├── python/            # JSON extracted from Python docstrings
│   ├── matlab/            # JSON extracted from MATLAB help comments
│   ├── platform/          # Snapshot from optiprofiler-platform docs
│   └── refs/              # Reference metadata (papers, URLs)
└── wiki/                  # Compiled wiki pages (LLM-maintained)
    ├── index.md           # Master page catalog
    ├── log.md             # Chronological change log
    ├── concepts/          # Core domain concepts
    ├── api/               # API reference (python/, matlab/ subdirs)
    ├── reference/         # Generated lossless mirrors of bundled sources
    ├── guides/            # How-to guides and quickstarts
    ├── platform/          # Hosted platform workflows and agent role
    ├── profiles/          # Profile methodology and interpretation
    ├── solvers/           # Solver entity pages (one per solver)
    └── troubleshooting/   # Error patterns and fixes
```

## Page Naming

- Use lowercase kebab-case: `solver-interface.md`, `performance-profile.md`
- Name should be descriptive and unique within its category
- Solver pages use the solver's common name: `newuoa.md`, `cobyla.md`

## Page Template

Every wiki page must include YAML frontmatter and follow this structure:

```markdown
---
tags: [category-tag, topic-tag]
sources: [_sources/python/benchmark.json, _sources/refs/bibliography.md]
related: [concepts/dfo.md, api/python/benchmark.md]
last_updated: YYYY-MM-DD
---

# Page Title

Brief introduction (1-2 sentences).

## Section Heading

Content...

## See Also

- [Related Page Title](../relative-path.md) — one-line description
```

## Frontmatter Fields

| Field          | Required | Description                                    |
|----------------|----------|------------------------------------------------|
| `tags`         | Yes      | List of topic tags for categorization           |
| `sources`      | Yes      | Which `_sources/` files this page derives from  |
| `related`      | Yes      | Links to related wiki pages (relative paths)    |
| `last_updated` | Yes      | Date of last modification (YYYY-MM-DD)          |

## Linking Rules

- Use relative markdown links: `[text](../category/page.md)`
- All links must resolve to existing files
- Every page should have at least one inbound link from another page
- `index.md` links to every page in the wiki

## index.md Format

The index groups pages by category with a one-line summary:

```markdown
## Concepts
- [Page Title](concepts/page.md) — One-line summary

## API Reference
### Python
- [Page Title](api/python/page.md) — One-line summary
```

## log.md Format

Append-only. Each entry starts with a consistent heading:

```markdown
## [YYYY-MM-DD] action | Page Title
Brief description of what changed.
```

Actions: `ingest`, `update`, `create`, `lint`, `migrate`.

## Lossless Coverage Contract

The wiki is structured knowledge, not a lossy summary. Human-authored
narrative pages may reorganize or explain source material, but exact API
facts must remain available inside `wiki/`.

1. `knowledge/_sources/**/*.json`, `knowledge/_sources/refs/*.md`,
   `knowledge/_sources/platform/*.md`, and bundled legacy Markdown
   docs/examples must have generated mirrors under `wiki/reference/`.
2. Generated reference pages are source-backed and marked with
   `generated: true`; do not hand-edit them.
3. If any source changes, run:

   ```bash
   python scripts/sync_knowledge.py
   ```

4. `scripts/audit_wiki_coverage.py` is the guardrail for hidden
   compression loss. It fails when a reference page is missing or stale.
5. Narrative pages should link to the relevant reference page whenever a
   topic includes feature options, defaults, choices, callable contracts,
   examples, or language-specific API differences.

## Content Rules

1. **One concept per page** — do not combine unrelated topics
2. **Cite sources** — reference `_sources/` files or external URLs
3. **Cross-reference liberally** — link to related concepts, API pages
4. **Keep pages focused** — aim for 200-800 words per page
5. **Use H2 for sections** — H1 is reserved for the page title
6. **Code examples** should be minimal and runnable
7. **Never modify `_sources/`** — those are immutable raw data

## Maintenance Workflows

### Adding a New Source

1. Place file in `_sources/` appropriate subdirectory
2. Read and extract key information
3. Create or update wiki pages
4. Add cross-references to/from related pages
5. Update `index.md`
6. Regenerate source-backed reference pages with `python scripts/sync_wiki_reference.py`
7. Audit exact source coverage with `python scripts/audit_wiki_coverage.py`
8. Append entry to `log.md`

### Syncing Package And Platform Sources

Use the combined command when the sibling OptiProfiler package or platform
repository changes:

```bash
python scripts/sync_knowledge.py
```

This command refreshes package API JSON, platform documentation snapshots,
generated reference pages, the lossless coverage audit, and wiki lint.

### Lint Check

Verify:
- All pages have valid frontmatter
- All internal links resolve
- No orphan pages (unreferenced by any other page)
- `index.md` lists every wiki page
- No contradictions between pages
- `wiki/reference/` is synchronized with all bundled source facts
