"""Tests for the RAG retrieval layer."""

import pytest
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "optiprofiler_agent" / "knowledge"


def _has_chromadb():
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_chromadb(), reason="chromadb not installed")
class TestKnowledgeRAG:

    def test_build_index_returns_nonzero(self):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        rag = KnowledgeRAG(KNOWLEDGE_DIR)
        count = rag.build_index()
        assert count > 0

    def test_profiles_and_debugging_indexed(self):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        rag = KnowledgeRAG(KNOWLEDGE_DIR)
        rag.build_index()

        results = rag.retrieve("performance profile Dolan More", top_k=3)
        assert len(results) > 0
        sources = [r["source"] for r in results]
        assert any("profiles/" in s for s in sources)

    def test_debugging_knowledge_indexed(self):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        rag = KnowledgeRAG(KNOWLEDGE_DIR)
        rag.build_index()

        results = rag.retrieve("TypeError solver wrong number arguments PicklingError lambda", top_k=5)
        assert len(results) > 0
        sources = [r["source"] for r in results]
        assert any("troubleshooting/" in s or "debugging/" in s for s in sources)

    def test_persistence_avoids_rebuild(self, tmp_path):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        persist_dir = str(tmp_path / "chroma_db")

        rag1 = KnowledgeRAG(KNOWLEDGE_DIR, persist_dir=persist_dir)
        count1 = rag1.build_index()
        assert count1 > 0

        rag2 = KnowledgeRAG(KNOWLEDGE_DIR, persist_dir=persist_dir)
        count2 = rag2.build_index()
        assert count2 == count1

    def test_language_filtering(self):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        rag = KnowledgeRAG(KNOWLEDGE_DIR)
        rag.build_index()

        results = rag.retrieve("solver interface", top_k=5, language="python")
        for r in results:
            assert not r["source"].startswith("matlab/")
            assert not r["source"].startswith("wiki/api/matlab/")

    def test_wiki_pages_indexed(self):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        rag = KnowledgeRAG(KNOWLEDGE_DIR)
        rag.build_index()

        results = rag.retrieve("derivative free optimization", top_k=3)
        assert len(results) > 0
        sources = [r["source"] for r in results]
        assert any("wiki/" in s for s in sources)

    def test_retrieve_with_index(self):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        rag = KnowledgeRAG(KNOWLEDGE_DIR)
        rag.build_index()

        results = rag.retrieve_with_index("performance profile", top_k=3)
        assert len(results) > 0

    def test_get_index_text(self):
        from optiprofiler_agent.common.rag import KnowledgeRAG
        rag = KnowledgeRAG(KNOWLEDGE_DIR)

        index_text = rag.get_index_text()
        assert len(index_text) > 0
        assert "Concepts" in index_text
        assert "Troubleshooting" in index_text


class TestWikiStructure:

    def test_wiki_dir_exists(self):
        assert (KNOWLEDGE_DIR / "wiki").is_dir()

    def test_wiki_index_exists(self):
        assert (KNOWLEDGE_DIR / "wiki" / "index.md").exists()

    def test_wiki_log_exists(self):
        assert (KNOWLEDGE_DIR / "wiki" / "log.md").exists()

    def test_wiki_categories_exist(self):
        for cat in ("concepts", "api", "guides", "profiles", "solvers", "troubleshooting"):
            assert (KNOWLEDGE_DIR / "wiki" / cat).is_dir(), f"Missing wiki/{cat}/"

    def test_sources_dir_exists(self):
        assert (KNOWLEDGE_DIR / "_sources").is_dir()

    def test_sources_python_json(self):
        sources_py = KNOWLEDGE_DIR / "_sources" / "python"
        assert sources_py.is_dir()
        assert (sources_py / "benchmark.json").exists()

    def test_schema_exists(self):
        assert (KNOWLEDGE_DIR / "SCHEMA.md").exists()

    def test_wiki_pages_have_frontmatter(self):
        import re
        fm_re = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
        wiki_dir = KNOWLEDGE_DIR / "wiki"
        for f in wiki_dir.rglob("*.md"):
            if f.name in ("index.md", "log.md"):
                continue
            content = f.read_text(encoding="utf-8")
            assert fm_re.match(content), f"{f.name} missing YAML frontmatter"
