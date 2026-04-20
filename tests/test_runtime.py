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
    # `.env` is the renamed-on-copy seed (`.env.template` -> `.env`). It MUST
    # land here or the onboarding wizard has no commented hints to merge into,
    # and a brand-new install would also start with an empty secrets file.
    # This guards against the .gitignore `.env.*` pattern silently dropping
    # the seed from the source tree on a fresh checkout.
    env_file = paths.env_path()
    assert env_file.exists(), (
        f"{env_file} missing — `.env.template` did not get copied. "
        "Most likely the seed was excluded from git by an over-broad "
        ".gitignore rule. See .gitignore exception for "
        "optiprofiler_agent/runtime/_seed/.env.template."
    )
    env_text = env_file.read_text(encoding="utf-8")
    # Spot-check that template hints made it through — the wizard's merge
    # step relies on these comment lines being preserved.
    assert "# KIMI_API_KEY=" in env_text or "# MINIMAX_API_KEY=" in env_text, (
        "Seed `.env.template` was copied but is empty / lacks commented "
        "provider hints. Did someone replace it with an empty file?"
    )

    user_text = paths.memory_path().read_text(encoding="utf-8")
    paths.memory_path().write_text(user_text + "\n- user-edit-marker\n", encoding="utf-8")

    m2 = bootstrap.ensure()
    assert m1["seeded"] == m2["seeded"]
    assert "user-edit-marker" in paths.memory_path().read_text(encoding="utf-8")


def test_seed_template_present_in_source_tree():
    """Source-tree guard: catch the .gitignore-eats-seed-file bug at import time.

    The wheel ships ``runtime/_seed/.env.template`` via package-data, but the
    source tree must also contain it — every test that exercises the
    onboarding flow assumes ``bootstrap.ensure()`` can find a non-empty
    template to copy. Catches the failure mode where a `.env.*` ignore
    pattern silently strips the file from a fresh ``git clone``.
    """
    from optiprofiler_agent.runtime import paths

    seed = paths.bundled_seed_dir() / ".env.template"
    assert seed.is_file(), (
        f"missing {seed}. If you just adjusted .gitignore, make sure it "
        "still allows this path (see the explicit `!` exception)."
    )
    body = seed.read_text(encoding="utf-8")
    assert "MINIMAX_API_KEY" in body and "OPAGENT_DEFAULT_PROVIDER" in body, (
        "seed template is present but does not look like the expected "
        "skeleton — refusing to ship a degenerate template."
    )


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


def test_session_log_strips_thinking_from_assistant():
    """Thinking-model reasoning blocks must never reach ``sessions.db``.

    If they did, ``recall_past`` would re-feed the model its own private
    chain-of-thought on the next turn, polluting both retrieval and
    downstream responses.
    """
    from optiprofiler_agent.runtime import bootstrap, session_log

    bootstrap.ensure()
    sid = session_log.new_session()
    session_log.log_turn(
        sid,
        "assistant",
        "<think>scheming about secrets</think>Final answer: use BOBYQA.",
    )
    # User turns must NOT be stripped — they're verbatim user input.
    session_log.log_turn(sid, "user", "<think>not a tag I emitted</think>keep me")

    # Search by a token that's *only* present inside the stripped block;
    # if stripping worked, the assistant turn should not match.
    leaked = session_log.search("scheming")
    assert all(h.role != "assistant" for h in leaked), (
        "assistant <think> content leaked into FTS index"
    )

    # The visible portion of the assistant turn must still be searchable.
    visible = session_log.search("BOBYQA")
    assert any("Final answer" in h.content for h in visible)
    assert all("<think>" not in h.content for h in visible if h.role == "assistant")

    # User turn was preserved verbatim.
    user_hits = session_log.search("keep")
    assert any(h.role == "user" and "<think>" in h.content for h in user_hits)


def test_trajectory_strips_thinking_from_assistant(tmp_path, monkeypatch):
    target = tmp_path / "trajdump"
    monkeypatch.setenv("OPAGENT_TRAJECTORY_DIR", str(target))
    from optiprofiler_agent.runtime import bootstrap, trajectory

    bootstrap.ensure()
    trajectory.append(
        "sid",
        "assistant",
        "<think>private plan</think>Hello, user.",
    )
    files = list(target.glob("*.jsonl"))
    assert files
    body = files[0].read_text(encoding="utf-8")
    assert "Hello, user." in body
    assert "<think>" not in body
    assert "private plan" not in body


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
