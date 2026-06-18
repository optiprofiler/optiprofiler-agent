"""Tests for source-backed wiki coverage."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_wiki_coverage import audit_wiki_coverage  # noqa: E402
from sync_wiki_reference import generate_reference_pages  # noqa: E402

KNOWLEDGE_DIR = REPO_ROOT / "optiprofiler_agent" / "knowledge"


def test_source_backed_reference_pages_are_in_sync():
    assert audit_wiki_coverage() == []


def test_reference_pages_cover_all_bundled_sources():
    pages = generate_reference_pages()
    rel_targets = {path.relative_to(KNOWLEDGE_DIR / "wiki").as_posix() for path in pages}

    assert {
        "reference/python-benchmark.md",
        "reference/python-classes.md",
        "reference/python-plib_tools.md",
        "reference/matlab-benchmark.md",
        "reference/matlab-classes.md",
        "reference/enums.md",
        "reference/legacy-docs.md",
        "reference/bibliography.md",
        "reference/platform-manifest.md",
        "reference/platform-docs.md",
    } <= rel_targets


def test_python_benchmark_reference_preserves_all_option_names_and_choices():
    source = json.loads((KNOWLEDGE_DIR / "_sources" / "python" / "benchmark.json").read_text(encoding="utf-8"))
    reference = (KNOWLEDGE_DIR / "wiki" / "reference" / "python-benchmark.md").read_text(encoding="utf-8")

    for category in ("parameters", "feature_options", "profile_options", "problem_options"):
        for name, info in source.get(category, {}).items():
            assert f"`{category}.{name}`" in reference
            assert f"## {category}.{name}" in reference
            for choice in info.get("choices", []) or []:
                assert str(choice) in reference


def test_lossless_reference_is_indexed_for_two_stage_rag():
    index = (KNOWLEDGE_DIR / "wiki" / "index.md").read_text(encoding="utf-8")

    for page in (
        "reference/python-benchmark.md",
        "reference/matlab-benchmark.md",
        "reference/legacy-docs.md",
        "reference/platform-docs.md",
    ):
        assert page in index

    assert "Source-Backed Reference" in index


def test_platform_reference_preserves_snapshot_metadata():
    manifest = json.loads((KNOWLEDGE_DIR / "_sources" / "platform" / "manifest.json").read_text(encoding="utf-8"))
    reference = (KNOWLEDGE_DIR / "wiki" / "reference" / "platform-manifest.md").read_text(encoding="utf-8")
    docs_reference = (KNOWLEDGE_DIR / "wiki" / "reference" / "platform-docs.md").read_text(encoding="utf-8")

    assert manifest["git_commit"] in reference
    assert "docs/api.md" in reference
    assert "docs/adr/0013-dfo-ecosystem-module-registry.md" in reference
    assert "OptiProfiler Platform Source Snapshot" in docs_reference
    assert "DFO ecosystem module registry" in docs_reference
