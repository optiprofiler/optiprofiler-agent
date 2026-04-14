"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from optiprofiler_agent.config import (
    PROVIDER_REGISTRY,
    AgentConfig,
    LLMConfig,
)


class TestLLMConfig:

    def test_default_provider_is_minimax(self):
        cfg = LLMConfig()
        assert cfg.provider == "minimax"
        assert cfg.model == "MiniMax-M2.7"
        assert cfg.base_url == "https://api.minimaxi.com/v1"

    @pytest.mark.parametrize("provider", list(PROVIDER_REGISTRY.keys()))
    def test_all_providers_have_defaults(self, provider):
        if provider == "anthropic":
            cfg = LLMConfig(provider=provider, api_key="test-key")
        else:
            cfg = LLMConfig(provider=provider, api_key="test-key")
        info = PROVIDER_REGISTRY[provider]
        assert cfg.model == info["default_model"]

    def test_kimi_provider_defaults(self):
        cfg = LLMConfig(provider="kimi", api_key="test")
        assert cfg.base_url == "https://api.moonshot.cn/v1"
        assert cfg.model == "kimi-k2.5"
        assert cfg.temperature == 1.0, "Kimi reasoning model enforces temperature=1.0"

    def test_openai_provider_no_base_url(self):
        cfg = LLMConfig(provider="openai", api_key="test")
        assert cfg.base_url is None
        assert cfg.model == "gpt-4o"

    def test_deepseek_provider_defaults(self):
        cfg = LLMConfig(provider="deepseek", api_key="test")
        assert cfg.base_url == "https://api.deepseek.com/v1"
        assert cfg.model == "deepseek-chat"

    def test_anthropic_provider_defaults(self):
        cfg = LLMConfig(provider="anthropic", api_key="test")
        assert cfg.base_url is None
        assert cfg.model == "claude-sonnet-4-20250514"

    def test_custom_model_overrides_default(self):
        cfg = LLMConfig(provider="openai", model="gpt-3.5-turbo", api_key="test")
        assert cfg.model == "gpt-3.5-turbo"

    def test_custom_base_url_overrides_default(self):
        cfg = LLMConfig(provider="deepseek", base_url="http://localhost:8080", api_key="test")
        assert cfg.base_url == "http://localhost:8080"

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-123"}):
            cfg = LLMConfig(provider="openai")
            assert cfg.api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            cfg = LLMConfig(provider="openai", api_key="explicit-key")
            assert cfg.api_key == "explicit-key"

    def test_unknown_provider_uses_fallback(self):
        cfg = LLMConfig(provider="unknown_provider", model="my-model", api_key="k")
        assert cfg.model == "my-model"
        assert cfg.api_key == "k"

    def test_unknown_provider_env_fallback(self):
        with patch.dict(os.environ, {"UNKNOWN_PROVIDER_API_KEY": "fallback-key"}):
            cfg = LLMConfig(provider="unknown_provider")
            assert cfg.api_key == "fallback-key"

    def test_kimi_fixed_temperature_overrides_user(self):
        cfg = LLMConfig(provider="kimi", temperature=0.5, api_key="k")
        assert cfg.temperature == 1.0

    def test_non_kimi_temperature_respected(self):
        cfg = LLMConfig(provider="openai", temperature=0.7, api_key="k")
        assert cfg.temperature == 0.7


class TestAgentConfig:

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.rag_enabled is False
        assert cfg.rag_top_k == 3
        assert cfg.max_debug_retries == 3
        assert cfg.code_char_limit == 0
        assert cfg.verbose is False

    def test_knowledge_dir_exists(self):
        cfg = AgentConfig()
        assert cfg.knowledge_dir.exists()

    def test_wiki_dir_property(self):
        cfg = AgentConfig()
        assert cfg.wiki_dir == cfg.knowledge_dir / "wiki"

    def test_sources_dir_property(self):
        cfg = AgentConfig()
        assert cfg.sources_dir == cfg.knowledge_dir / "_sources"

    def test_rag_persist_dir_auto_set(self):
        cfg = AgentConfig()
        assert cfg.rag_persist_dir is not None
        assert ".chroma_db" in cfg.rag_persist_dir

    def test_rag_persist_dir_explicit(self):
        cfg = AgentConfig(rag_persist_dir="/tmp/my_chroma")
        assert cfg.rag_persist_dir == "/tmp/my_chroma"

    def test_custom_llm_config(self):
        llm = LLMConfig(provider="openai", api_key="test")
        cfg = AgentConfig(llm=llm)
        assert cfg.llm.provider == "openai"
