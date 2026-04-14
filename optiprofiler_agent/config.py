"""Configuration for the OptiProfiler Agent system.

Supports multiple LLM providers via a unified configuration.
All OpenAI-compatible providers (Kimi, MiniMax, DeepSeek, etc.)
use the same ``langchain-openai`` backend — only base_url differs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Pre-configured provider registry
# Users only need to set provider name + API key; base_url and default model
# are filled in automatically from this table.
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, dict] = {
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2.5",
        "env_key": "KIMI_API_KEY",
        "fixed_temperature": 1.0,  # reasoning model, only temperature=1 allowed
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "default_model": "MiniMax-M2.7",
        "env_key": "MINIMAX_API_KEY",
    },
    "openai": {
        "base_url": None,  # use official default
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "anthropic": {
        "base_url": None,
        "default_model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
    },
}


@dataclass
class LLMConfig:
    """Configuration for a single LLM connection."""

    provider: str = "minimax"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096

    def __post_init__(self):
        info = PROVIDER_REGISTRY.get(self.provider, {})
        if self.model is None:
            self.model = info.get("default_model", self.provider)
        if self.base_url is None and self.provider != "openai":
            self.base_url = info.get("base_url")
        if self.api_key is None:
            env_key = info.get("env_key", f"{self.provider.upper()}_API_KEY")
            self.api_key = os.getenv(env_key)
        if "fixed_temperature" in info:
            self.temperature = info["fixed_temperature"]


@dataclass
class AgentConfig:
    """Global configuration shared across all agents."""

    llm: LLMConfig = field(default_factory=LLMConfig)

    knowledge_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "knowledge"
    )

    rag_enabled: bool = False
    rag_top_k: int = 3
    rag_persist_dir: Optional[str] = field(default=None)
    rag_use_index: bool = True

    max_debug_retries: int = 3
    code_char_limit: int = 0

    verbose: bool = False

    @property
    def wiki_dir(self) -> Path:
        return self.knowledge_dir / "wiki"

    @property
    def sources_dir(self) -> Path:
        return self.knowledge_dir / "_sources"

    def __post_init__(self):
        if self.rag_persist_dir is None:
            self.rag_persist_dir = str(
                Path(__file__).parent / ".chroma_db"
            )
