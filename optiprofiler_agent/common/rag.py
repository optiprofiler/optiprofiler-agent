"""RAG retrieval layer — chunk, embed, and retrieve knowledge on demand.

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Falls back gracefully when optional dependencies are not installed.

The RAG layer operates on the **wiki/** compiled knowledge pages (not raw
``_sources/``). It supports two retrieval modes:

1. **Full vector search** (``retrieve``): searches all wiki chunks.
2. **Two-stage retrieval** (``retrieve_with_index``): first reads
   ``wiki/index.md`` to narrow scope, then does targeted vector search.

Usage::

    from optiprofiler_agent.common.rag import KnowledgeRAG
    rag = KnowledgeRAG(knowledge_dir)
    rag.build_index()                       # one-time indexing
    chunks = rag.retrieve("What is ptype?") # query
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from optiprofiler_agent.common.quiet_ml import suppress_hf_transformers_noise

logger = logging.getLogger(__name__)

_CHUNK_SEPARATOR = re.compile(r"\n#{1,2}\s")

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from markdown text."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def _chunk_wiki_page(text: str, source: str, max_chars: int = 2000) -> list[dict]:
    """Split a wiki markdown page into H2-delimited chunks.

    Wiki pages are shorter and more focused than the old flat files, so we
    use H2 boundaries (coarser grain) and a larger max_chars budget.
    """
    text = _strip_frontmatter(text)
    parts = _CHUNK_SEPARATOR.split(text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 20:
            continue
        if len(part) > max_chars:
            part = part[:max_chars]
        chunks.append({"text": part, "source": source})
    return chunks


def _chunk_json_params(data: dict, source: str) -> list[dict]:
    """Flatten a benchmark JSON into one chunk per parameter."""
    chunks = []
    cat_keys = ("parameters", "feature_options", "profile_options", "problem_options")
    for cat in cat_keys:
        params = data.get(cat, {})
        if not isinstance(params, dict):
            continue
        for name, info in params.items():
            desc = info.get("description", "")
            ptype = info.get("type", "")
            default = info.get("default", "")
            text = f"Parameter `{name}` ({ptype})"
            if default:
                text += f", default={default}"
            text += f": {desc}"
            chunks.append({"text": text, "source": f"{source}#{name}"})

    for key in ("solver_signatures", "solver_notes", "calling_convention",
                "returns", "raises", "notes"):
        val = data.get(key)
        if not val:
            continue
        if isinstance(val, dict):
            text = json.dumps(val, indent=2, ensure_ascii=False)
        elif isinstance(val, list):
            text = "\n".join(str(v) for v in val)
        else:
            text = str(val)
        chunks.append({"text": f"[{key}] {text}", "source": f"{source}#{key}"})

    return chunks


def _chunk_json_classes(data: dict, source: str) -> list[dict]:
    """One chunk per class with its properties and methods."""
    chunks = []
    for cls_name, cls_data in data.items():
        lines = [f"Class {cls_name}: {cls_data.get('description', '')}"]
        for prop, pinfo in cls_data.get("properties", {}).items():
            lines.append(f"  property {prop}: {pinfo.get('description', '')[:200]}")
        for meth, minfo in cls_data.get("methods", {}).items():
            lines.append(f"  method {meth}: {minfo.get('description', '')[:200]}")
        chunks.append({"text": "\n".join(lines), "source": f"{source}#{cls_name}"})
    return chunks


def _content_hash(chunks: list[dict]) -> str:
    h = hashlib.md5()
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()


def _walk_wiki_dir(wiki_dir: Path, base_prefix: str = "wiki") -> list[dict]:
    """Recursively walk wiki/ and chunk all .md files."""
    chunks: list[dict] = []
    if not wiki_dir.exists():
        return chunks

    for f in sorted(wiki_dir.rglob("*.md")):
        rel = f"{base_prefix}/{f.relative_to(wiki_dir)}"
        text = f.read_text(encoding="utf-8")
        chunks.extend(_chunk_wiki_page(text, rel))

    return chunks


def _walk_sources_dir(sources_dir: Path) -> list[dict]:
    """Walk _sources/ and chunk JSON files (for backward compatibility)."""
    chunks: list[dict] = []
    if not sources_dir.exists():
        return chunks

    for lang in ("python", "matlab"):
        lang_dir = sources_dir / lang
        if not lang_dir.exists():
            continue
        for f in sorted(lang_dir.iterdir()):
            if f.is_dir() or f.suffix != ".json":
                continue
            rel = f"_sources/{lang}/{f.name}"
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if f.stem == "benchmark":
                chunks.extend(_chunk_json_params(data, rel))
            elif f.stem == "classes":
                chunks.extend(_chunk_json_classes(data, rel))
            else:
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if len(text) > 2000:
                    text = text[:2000]
                chunks.append({"text": text, "source": rel})

    return chunks


class KnowledgeRAG:
    """Retrieval-Augmented Generation layer over the knowledge wiki."""

    def __init__(
        self,
        knowledge_dir: Path | str,
        collection_name: str = "optiprofiler_wiki",
        persist_dir: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._dir = Path(knowledge_dir)
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._embedding_model_name = embedding_model
        self._collection = None
        self._client = None
        self._index_text: str | None = None

    def _ensure_deps(self):
        try:
            import chromadb  # noqa: F401
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            return chromadb, SentenceTransformerEmbeddingFunction
        except ImportError as exc:
            raise ImportError(
                "RAG requires optional dependencies. Install with:\n"
                "  pip install 'optiprofiler-agent[rag]'"
            ) from exc

    def _load_index(self) -> str:
        """Load wiki/index.md content for Stage 1 retrieval."""
        if self._index_text is not None:
            return self._index_text

        index_path = self._dir / "wiki" / "index.md"
        if index_path.exists():
            self._index_text = _strip_frontmatter(
                index_path.read_text(encoding="utf-8")
            )
        else:
            self._index_text = ""
        return self._index_text

    def _gather_chunks(self) -> list[dict]:
        """Walk wiki/ pages and _sources/ JSON to produce chunks.

        Primary source: wiki/ compiled markdown pages.
        Secondary source: _sources/ raw JSON (for parameter-level detail).
        Old flat directories (common/, python/, matlab/, etc.) are no longer
        scanned — the wiki replaces them.
        """
        chunks: list[dict] = []

        wiki_dir = self._dir / "wiki"
        chunks.extend(_walk_wiki_dir(wiki_dir))

        sources_dir = self._dir / "_sources"
        chunks.extend(_walk_sources_dir(sources_dir))

        enums_path = self._dir / "enums.json"
        if enums_path.exists():
            with open(enums_path, encoding="utf-8") as fh:
                data = json.load(fh)
            for cls_name, members in data.items():
                vals = ", ".join(members.values())
                chunks.append({
                    "text": f"Enum {cls_name}: {vals}",
                    "source": f"enums.json#{cls_name}",
                })

        return chunks

    def build_index(self, force: bool = False) -> int:
        """Build (or rebuild) the vector index from knowledge files.

        Returns the number of chunks indexed.
        """
        suppress_hf_transformers_noise()
        chromadb, SentenceTransformerEF = self._ensure_deps()

        chunks = self._gather_chunks()
        if not chunks:
            logger.warning("No knowledge chunks found in %s", self._dir)
            return 0

        new_hash = _content_hash(chunks)

        ef = SentenceTransformerEF(model_name=self._embedding_model_name)

        if self._persist_dir:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            self._client = chromadb.Client()

        existing = {c.name for c in self._client.list_collections()}
        if self._collection_name in existing and not force:
            col = self._client.get_collection(
                self._collection_name, embedding_function=ef)
            meta = col.metadata or {}
            if meta.get("content_hash") == new_hash:
                self._collection = col
                logger.info("Index up-to-date (%d chunks), skipping rebuild.", col.count())
                return col.count()
            self._client.delete_collection(self._collection_name)

        col = self._client.create_collection(
            name=self._collection_name,
            embedding_function=ef,
            metadata={"content_hash": new_hash},
        )

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"]} for c in chunks]

        batch_size = 256
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            col.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        self._collection = col
        logger.info("Built index with %d chunks.", len(chunks))
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        language: str | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant knowledge chunks for *query*.

        Returns a list of dicts with keys ``text``, ``source``, ``distance``.
        """
        if self._collection is None:
            self.build_index()
        if self._collection is None:
            return []

        fetch_k = top_k * 3 if language else top_k
        results = self._collection.query(
            query_texts=[query],
            n_results=fetch_k,
        )

        items = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            exclude_prefix = None
            if language and language in ("python", "matlab"):
                other = "matlab" if language == "python" else "python"
                exclude_prefix = f"wiki/api/{other}/"

            for doc, meta, dist in zip(docs, metas, dists):
                source = meta.get("source", "")
                if exclude_prefix and source.startswith(exclude_prefix):
                    continue
                items.append({
                    "text": doc,
                    "source": source,
                    "distance": dist,
                })
                if len(items) >= top_k:
                    break
        return items

    def retrieve_with_index(
        self,
        query: str,
        top_k: int = 5,
        language: str | None = None,
    ) -> list[dict]:
        """Two-stage retrieval: index scan + targeted vector search.

        Stage 1: Use wiki/index.md to identify relevant wiki sections
        via keyword matching against the query.
        Stage 2: Filter vector search to chunks from matched sections.
        """
        if self._collection is None:
            self.build_index()
        if self._collection is None:
            return []

        index_text = self._load_index()

        relevant_prefixes: list[str] = []
        if index_text:
            query_lower = query.lower()
            for line in index_text.splitlines():
                line_stripped = line.strip()
                if not line_stripped.startswith("- ["):
                    continue
                paren_start = line_stripped.find("](")
                paren_end = line_stripped.find(")", paren_start + 2)
                if paren_start == -1 or paren_end == -1:
                    continue
                link = line_stripped[paren_start + 2: paren_end]
                label_end = line_stripped.find("]")
                label = line_stripped[3:label_end].lower() if label_end > 3 else ""
                summary_start = line_stripped.find("—")
                summary = line_stripped[summary_start + 1:].strip().lower() if summary_start != -1 else ""

                combined = f"{label} {summary} {link}"
                words = re.findall(r"[a-z_]+", query_lower)
                if any(w in combined for w in words if len(w) > 2):
                    relevant_prefixes.append(f"wiki/{link}")

        if not relevant_prefixes:
            return self.retrieve(query, top_k=top_k, language=language)

        fetch_k = top_k * 4
        results = self._collection.query(
            query_texts=[query],
            n_results=fetch_k,
        )

        items = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            exclude_prefix = None
            if language and language in ("python", "matlab"):
                other = "matlab" if language == "python" else "python"
                exclude_prefix = f"wiki/api/{other}/"

            for doc, meta, dist in zip(docs, metas, dists):
                source = meta.get("source", "")
                if exclude_prefix and source.startswith(exclude_prefix):
                    continue
                matched = any(source.startswith(p.rsplit(".", 1)[0])
                              for p in relevant_prefixes)
                if not matched:
                    continue
                items.append({
                    "text": doc,
                    "source": source,
                    "distance": dist,
                })
                if len(items) >= top_k:
                    break

        if len(items) < top_k:
            fallback = self.retrieve(query, top_k=top_k - len(items), language=language)
            seen = {i["source"] for i in items}
            for fb in fallback:
                if fb["source"] not in seen:
                    items.append(fb)
                    if len(items) >= top_k:
                        break

        return items

    def get_index_text(self) -> str:
        """Return the wiki index content for direct prompt injection."""
        return self._load_index()

    def retrieve_as_text(
        self,
        query: str,
        top_k: int = 5,
        language: str | None = None,
        max_chars: int = 3000,
        use_index: bool = True,
    ) -> str:
        """Retrieve chunks and format them as a text block for prompt injection."""
        if use_index:
            items = self.retrieve_with_index(query, top_k=top_k, language=language)
        else:
            items = self.retrieve(query, top_k=top_k, language=language)

        if not items:
            return ""

        lines = ["### Retrieved Knowledge\n"]
        total = 0
        for item in items:
            entry = f"[{item['source']}] {item['text']}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            lines.append("")
            total += len(entry)

        return "\n".join(lines)

    @property
    def is_ready(self) -> bool:
        return self._collection is not None and self._collection.count() > 0
