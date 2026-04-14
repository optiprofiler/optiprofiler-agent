"""Tests for the knowledge base structure and loading."""

from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "optiprofiler_agent" / "knowledge"


class TestKnowledgeStructure:

    def test_common_dir_exists(self):
        assert (KNOWLEDGE_DIR / "common").is_dir()

    def test_python_dir_exists(self):
        assert (KNOWLEDGE_DIR / "python").is_dir()

    def test_matlab_dir_exists(self):
        assert (KNOWLEDGE_DIR / "matlab").is_dir()

    def test_profiles_dir_exists(self):
        assert (KNOWLEDGE_DIR / "profiles").is_dir()

    def test_debugging_dir_exists(self):
        assert (KNOWLEDGE_DIR / "debugging").is_dir()

    def test_profiles_methodology(self):
        path = KNOWLEDGE_DIR / "profiles" / "methodology.md"
        assert path.exists()
        content = path.read_text()
        assert "performance profile" in content.lower()
        assert "data profile" in content.lower()
        assert "Dolan" in content

    def test_profiles_feature_effects(self):
        path = KNOWLEDGE_DIR / "profiles" / "feature_effects.md"
        assert path.exists()
        content = path.read_text()
        assert "noisy" in content
        assert "plain" in content

    def test_debugging_common_errors(self):
        path = KNOWLEDGE_DIR / "debugging" / "common_errors.md"
        assert path.exists()
        content = path.read_text()
        assert "two solvers" in content.lower()
        assert "TypeError" in content

    def test_debugging_solver_compat(self):
        path = KNOWLEDGE_DIR / "debugging" / "solver_compat.md"
        assert path.exists()
        content = path.read_text()
        assert "fun" in content
        assert "x0" in content


class TestKnowledgeBase:

    def test_load_knowledge_base(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(KNOWLEDGE_DIR)
        text = kb.to_prompt_text()
        assert len(text) > 100

    def test_language_filtering(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(KNOWLEDGE_DIR)
        python_text = kb.to_prompt_text(language="python")
        matlab_text = kb.to_prompt_text(language="matlab")
        assert len(python_text) > 0
        assert len(matlab_text) > 0

    def test_loads_from_sources(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(KNOWLEDGE_DIR)
        bm = kb.get_benchmark("python")
        assert "solver_signatures" in bm
        assert "parameters" in bm or "feature_options" in bm

    def test_enums_loaded(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(KNOWLEDGE_DIR)
        enum = kb.get_enum("FeatureName")
        assert enum is not None
        assert "PLAIN" in enum
