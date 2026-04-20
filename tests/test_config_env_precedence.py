"""Verify the documented precedence of secrets sources.

Documented order (highest first):
    1. Real shell env (``os.environ`` already populated)
    2. ``./.env`` in the current working directory
    3. ``~/.opagent/.env`` (user-level)
    4. ``PROVIDER_REGISTRY`` defaults

Each test isolates a single boundary so failures pinpoint exactly which
layer regressed.
"""

from __future__ import annotations


import pytest


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Three independent locations for the three .env layers under test."""
    home = tmp_path / "opagent_home"
    home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()

    monkeypatch.setenv("OPAGENT_HOME", str(home))
    monkeypatch.chdir(cwd)

    # Strip any ambient provider env so the test starts from a clean slate.
    for k in (
        "MINIMAX_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
        "OPAGENT_CUSTOM_BASE_URL", "OPAGENT_CUSTOM_MODEL",
        "OPAGENT_CUSTOM_API_KEY", "OPAGENT_DEFAULT_PROVIDER",
        "OPAGENT_DEFAULT_MODEL", "OPAGENT_DEFAULT_BASE_URL",
    ):
        monkeypatch.delenv(k, raising=False)

    return {"home": home, "cwd": cwd}


def _reload_config():
    """Reload optiprofiler_agent.config so its module-level _load_env_files
    fires against the freshly-arranged filesystem."""
    from importlib import reload
    import optiprofiler_agent.runtime.paths as paths
    import optiprofiler_agent.config as config
    reload(paths)
    reload(config)
    return config


def test_user_level_env_only(isolated_paths):
    """When only ~/.opagent/.env defines a key, LLMConfig must see it."""
    (isolated_paths["home"] / ".env").write_text(
        "MINIMAX_API_KEY=user-level-value\n", encoding="utf-8"
    )
    config = _reload_config()
    cfg = config.LLMConfig(provider="minimax")
    assert cfg.api_key == "user-level-value"


def test_cwd_env_overrides_user_env(isolated_paths):
    """A project-local .env wins over the user-level file."""
    (isolated_paths["home"] / ".env").write_text(
        "MINIMAX_API_KEY=user-level\n", encoding="utf-8"
    )
    (isolated_paths["cwd"] / ".env").write_text(
        "MINIMAX_API_KEY=project-local\n", encoding="utf-8"
    )
    config = _reload_config()
    cfg = config.LLMConfig(provider="minimax")
    assert cfg.api_key == "project-local"


def test_real_env_wins_over_both_files(isolated_paths, monkeypatch):
    """A real os.environ value beats every dotenv file."""
    (isolated_paths["home"] / ".env").write_text(
        "MINIMAX_API_KEY=user-level\n", encoding="utf-8"
    )
    (isolated_paths["cwd"] / ".env").write_text(
        "MINIMAX_API_KEY=project-local\n", encoding="utf-8"
    )
    monkeypatch.setenv("MINIMAX_API_KEY", "shell-export")
    config = _reload_config()
    cfg = config.LLMConfig(provider="minimax")
    assert cfg.api_key == "shell-export"


def test_default_provider_honors_env(isolated_paths, monkeypatch):
    """OPAGENT_DEFAULT_PROVIDER picks the default provider for new LLMConfigs."""
    monkeypatch.setenv("OPAGENT_DEFAULT_PROVIDER", "kimi")
    config = _reload_config()
    cfg = config.LLMConfig()
    assert cfg.provider == "kimi"


def test_default_provider_falls_back_when_invalid(isolated_paths, monkeypatch):
    """A typo'd OPAGENT_DEFAULT_PROVIDER must not break startup."""
    monkeypatch.setenv("OPAGENT_DEFAULT_PROVIDER", "nonexistent-vendor")
    config = _reload_config()
    cfg = config.LLMConfig()
    assert cfg.provider == "minimax"


def test_opagent_default_model_overrides_registry(isolated_paths, monkeypatch):
    """Built-in provider should accept OPAGENT_DEFAULT_MODEL override.

    Lets users pin "kimi-k2-thinking" without switching to provider=custom
    or hard-coding ``--model`` on every command.
    """
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("OPAGENT_DEFAULT_MODEL", "kimi-k2-thinking")
    config = _reload_config()
    cfg = config.LLMConfig(provider="kimi")
    assert cfg.model == "kimi-k2-thinking"
    # base_url should still come from the registry — we only overrode model
    assert cfg.base_url == "https://api.moonshot.cn/v1"


def test_opagent_default_base_url_overrides_registry(isolated_paths, monkeypatch):
    """OPAGENT_DEFAULT_BASE_URL routes a built-in provider through a proxy."""
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("OPAGENT_DEFAULT_BASE_URL", "https://proxy.example.com/v1")
    config = _reload_config()
    cfg = config.LLMConfig(provider="kimi")
    assert cfg.base_url == "https://proxy.example.com/v1"


def test_explicit_kwarg_beats_opagent_default_model(isolated_paths, monkeypatch):
    """``--model`` (explicit kwarg) must outrank OPAGENT_DEFAULT_MODEL."""
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("OPAGENT_DEFAULT_MODEL", "from-env")
    config = _reload_config()
    cfg = config.LLMConfig(provider="kimi", model="from-flag")
    assert cfg.model == "from-flag"


def test_default_provider_falls_back_to_first_configured_key(isolated_paths, monkeypatch):
    """No OPAGENT_DEFAULT_PROVIDER set, but KIMI key present → default = kimi.

    Regression: before the fix, _default_provider() always returned
    'minimax' even when the only configured key was for a different
    provider, so 'opagent init' and 'opagent agent' disagreed about
    which provider was active.
    """
    monkeypatch.setenv("KIMI_API_KEY", "k")
    config = _reload_config()
    assert config._default_provider() == "kimi"


def test_default_provider_and_init_ux_agree(isolated_paths, monkeypatch):
    """The 'default' shown by `opagent init` must match what LLMConfig() uses."""
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    config = _reload_config()
    from optiprofiler_agent import onboarding
    assert config._default_provider() == onboarding.active_default_provider()


def test_custom_provider_reads_three_env_vars(isolated_paths, monkeypatch):
    """provider=custom maps to OPAGENT_CUSTOM_{BASE_URL,MODEL,API_KEY}."""
    monkeypatch.setenv("OPAGENT_CUSTOM_BASE_URL", "https://api.foo.com/v1")
    monkeypatch.setenv("OPAGENT_CUSTOM_MODEL", "foo-1")
    monkeypatch.setenv("OPAGENT_CUSTOM_API_KEY", "sk-foo")
    config = _reload_config()

    cfg = config.LLMConfig(provider="custom")
    assert cfg.base_url == "https://api.foo.com/v1"
    assert cfg.model == "foo-1"
    assert cfg.api_key == "sk-foo"
