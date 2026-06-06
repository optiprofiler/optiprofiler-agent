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

    def test_python_public_exports_match_current_package_root(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(KNOWLEDGE_DIR)
        notes = kb.get_api_notes("python")
        exports = set(notes["public_exports"])

        assert {
            "benchmark",
            "Problem",
            "Feature",
            "FeaturedProblem",
            "show_versions",
            "get_plib_config",
            "set_plib_config",
        } <= exports
        assert "s2mpj_load" not in exports
        assert "pycutest_select" not in exports

    def test_worker_defaults_synced_from_source(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(KNOWLEDGE_DIR)
        for lang in ("python", "matlab"):
            n_jobs = kb.get_param(lang, "n_jobs")
            assert n_jobs is not None
            text = f"{n_jobs.get('default', '')} {n_jobs.get('description', '')}".lower()
            assert "conservative" in text
            assert "half" in text
            assert "available workers" in text

    def test_matlab_draw_hist_plots_default_is_source_synced(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(KNOWLEDGE_DIR)
        opt = kb.get_param("matlab", "draw_hist_plots")
        assert opt is not None
        text = f"{opt.get('default', '')} {opt.get('description', '')}".lower()
        assert "parallel" in text
        assert "load" in text
        assert "sequential" in text

    def test_output_report_diagnostics_are_documented(self):
        from optiprofiler_agent.common.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(KNOWLEDGE_DIR)
        for lang in ("python", "matlab"):
            artifacts = kb.get_benchmark(lang).get("output_artifacts", {})
            report = artifacts.get("test_log_report", "").lower()
            assert "selected problem" in report
            assert "abnormal solver" in report
            assert "output fallback" in report
            assert "solver scores" in report

    def test_matlab_custom_problem_library_uses_folder_convention(self):
        path = KNOWLEDGE_DIR / "wiki" / "guides" / "custom-problem-library-matlab.md"
        content = path.read_text()
        assert "options.custom_problem_libs_path" not in content
        assert "optiprofiler/problem_libs" in content
        assert "options.plibs" in content
        assert "MATLAB has no `custom_problem_libs_path` option" in content

    def test_solver_compat_scipy_nonlinear_constraints_are_vector_safe(self):
        path = KNOWLEDGE_DIR / "wiki" / "troubleshooting" / "solver-compat.md"
        content = path.read_text()
        assert "np.atleast_1d(cub(x0))" in content
        assert "np.zeros_like(c_ub_x0)" in content
        assert "np.atleast_1d(ceq(x0))" in content
        assert "np.zeros_like(c_eq_x0)" in content
        assert "NonlinearConstraint(cub, -np.inf, 0)" not in content
        assert "NonlinearConstraint(ceq, 0, 0)" not in content

    def test_distribution_mapping_is_rag_visible(self):
        api = (KNOWLEDGE_DIR / "wiki" / "api" / "python" / "benchmark.md").read_text()
        features = (KNOWLEDGE_DIR / "wiki" / "concepts" / "features.md").read_text()
        profiles = (KNOWLEDGE_DIR / "wiki" / "profiles" / "feature-effects.md").read_text()
        combined = "\n".join([api, features, profiles])

        assert "`feature_name='noisy'`" in api
        assert "'gaussian'" in combined
        assert "'uniform'" in combined
        assert "'spherical'" in combined
        assert "distribution(random_stream) -> scalar" in combined
        assert "distribution(random_stream, dimension) -> random vector" in combined
        assert "Do not use `distribution='normal'`" in features
        assert "max(1, abs(f)) * noise_level * noise" in combined
