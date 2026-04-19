"""Tests for the ``optiprofiler_agent.runtime`` sub-package.

Covers:
* ``paths`` honors the ``OPAGENT_HOME`` env var
* ``bootstrap.ensure`` is idempotent and copies seed files
* ``memory`` round-trip: append → read → frozen_snapshot
* ``memory.update_user_profile`` rejects unknown fields
* ``session_log`` log + search round-trip (FTS5 path or LIKE fallback)
* ``wiki_local.add_page`` writes a frontmatter-tagged file
* ``trajectory`` is silent unless explicitly enabled
* ``plugin.external_*_dirs`` returns [] without yaml / config
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Force every runtime module to use a fresh ``OPAGENT_HOME``."""
    monkeypatch.setenv("OPAGENT_HOME", str(tmp_path / "opagent_home"))
    monkeypatch.delenv("OPAGENT_TRAJECTORY_DIR", raising=False)

    # Reload cached config in plugin.py so the fresh path is picked up.
    from optiprofiler_agent.runtime import plugin

    plugin.reload()
    yield


def test_paths_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPAGENT_HOME", str(tmp_path / "custom"))
    from optiprofiler_agent.runtime import paths

    assert paths.home() == tmp_path / "custom"
    assert paths.memory_path().parent == tmp_path / "custom"


def test_bootstrap_idempotent_and_copies_seeds():
    from optiprofiler_agent.runtime import bootstrap, paths

    m1 = bootstrap.ensure()
    assert paths.home().exists()
    assert paths.memory_path().exists()
    assert paths.user_path().exists()
    assert paths.config_path().exists()
    assert paths.auto_wiki_dir().exists()
    assert paths.skills_dir().exists()
    assert paths.manifest_path().exists()

    user_text = paths.memory_path().read_text(encoding="utf-8")
    paths.memory_path().write_text(user_text + "\n- user-edit-marker\n", encoding="utf-8")

    m2 = bootstrap.ensure()
    assert m1["seeded"] == m2["seeded"]
    assert "user-edit-marker" in paths.memory_path().read_text(encoding="utf-8")


def test_memory_append_and_snapshot():
    from optiprofiler_agent.runtime import bootstrap, memory

    bootstrap.ensure()
    memory.append_fact("BOBYQA is preferred for bound-constrained DFO", tags=["solver"])
    memory.append_fact("User runs OptiProfiler on macOS")

    facts = memory.read_facts()
    assert any("BOBYQA" in f for f in facts)

    snap = memory.frozen_snapshot()
    assert "BOBYQA" in snap
    assert "Persistent Context" in snap


def test_user_profile_whitelist_rejects_unknown():
    from optiprofiler_agent.runtime import bootstrap, memory

    bootstrap.ensure()
    memory.update_user_profile("preferred_solver", "BOBYQA")

    with pytest.raises(ValueError):
        memory.update_user_profile("ssh_key", "rm -rf /")

    profile = memory.read_user_profile()
    assert profile["preferred_solver"] == "BOBYQA"
    assert "ssh_key" not in profile


def test_session_log_roundtrip():
    from optiprofiler_agent.runtime import bootstrap, session_log

    bootstrap.ensure()
    sid = session_log.new_session(label="test")
    session_log.log_turn(sid, "user", "How do I run BOBYQA in OptiProfiler?")
    session_log.log_turn(sid, "assistant", "Use benchmark([BOBYQA, NEWUOA]).")

    hits = session_log.search("BOBYQA", limit=10)
    assert any("BOBYQA" in h.content for h in hits)

    sessions = session_log.list_sessions()
    assert any(s["session_id"] == sid for s in sessions)


def test_session_search_handles_punctuation():
    """FTS5 reserved characters in the query must not raise."""
    from optiprofiler_agent.runtime import bootstrap, session_log

    bootstrap.ensure()
    sid = session_log.new_session()
    session_log.log_turn(sid, "user", "What is ptype='u'?")
    hits = session_log.search("ptype='u'")
    assert isinstance(hits, list)


def test_wiki_local_add_page_writes_frontmatter():
    from optiprofiler_agent.runtime import bootstrap, paths, wiki_local

    bootstrap.ensure()
    p = wiki_local.add_page(
        slug="My Note!",
        content="Some body text.",
        summary="hello",
    )
    assert p.exists()
    assert p.parent == paths.auto_wiki_dir()
    text = p.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "source: agent" in text
    assert "summary: hello" in text


def test_wiki_local_collision_suffix():
    from optiprofiler_agent.runtime import bootstrap, wiki_local

    bootstrap.ensure()
    p1 = wiki_local.add_page("dup", "first")
    p2 = wiki_local.add_page("dup", "second")
    assert p1 != p2
    assert "dup-2" in p2.name


def test_trajectory_disabled_by_default(tmp_path):
    from optiprofiler_agent.runtime import bootstrap, trajectory

    bootstrap.ensure()
    assert trajectory.enabled() is False
    trajectory.append("sid", "user", "hello")
    assert not any(tmp_path.rglob("sid.jsonl"))


def test_trajectory_enabled_via_env(tmp_path, monkeypatch):
    target = tmp_path / "trajdump"
    monkeypatch.setenv("OPAGENT_TRAJECTORY_DIR", str(target))
    from optiprofiler_agent.runtime import bootstrap, trajectory

    bootstrap.ensure()
    assert trajectory.enabled() is True
    trajectory.append("sid", "user", "hello")
    files = list(target.glob("*.jsonl"))
    assert files
    assert "hello" in files[0].read_text(encoding="utf-8")


def test_plugin_no_yaml_returns_empty():
    from optiprofiler_agent.runtime import bootstrap, plugin

    bootstrap.ensure()
    plugin.reload()
    assert plugin.external_wiki_dirs() == []
    assert plugin.external_skill_dirs() == []
