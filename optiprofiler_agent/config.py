"""Configuration for the OptiProfiler Agent system.

Supports multiple LLM providers via a unified configuration.
All OpenAI-compatible providers (Kimi, MiniMax, DeepSeek, etc.)
use the same ``langchain-openai`` backend — only base_url differs.

Environment / secrets resolution order, matching the conventions used by
``gh``, ``codex``, ``claude``, and ``aider``:

    1. Explicit kwarg passed to ``LLMConfig`` (highest)
    2. CLI flag (``--provider``, ``--model``)
    3. Real shell environment variables (``export FOO=...``)
    4. Project-local ``./.env`` in the current working directory
    5. User-level ``~/.opagent/.env`` (created by ``opagent init``)
    6. Built-in defaults from ``PROVIDER_REGISTRY`` (lowest)

``python-dotenv``'s default ``load_dotenv()`` *never* overrides keys that
already live in ``os.environ`` (i.e. shell ``export`` always wins). We
load both files with ``override=False`` and load cwd ``.env`` last so it
beats the user-level file but still loses to a real ``export`` — exactly
the precedence above.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Multi-source dotenv loading
# ---------------------------------------------------------------------------

def _user_env_path() -> Path:
    """Path to the user-level secrets file (``~/.opagent/.env`` by default).

    Reads ``OPAGENT_HOME`` lazily so tests / CI can isolate via ``monkeypatch``.
    Imported lazily to avoid a circular dependency on ``runtime.paths`` (which
    itself imports nothing from this module, but keeping it lazy keeps the
    config module self-contained for users who only want the dataclasses).
    """
    raw = os.environ.get("OPAGENT_HOME")
    base = Path(raw).expanduser() if raw else Path.home() / ".opagent"
    return base / ".env"


def _load_env_files() -> None:
    """Load ``./.env`` then ``~/.opagent/.env``; never override real env.

    Order matters: the *first* call to ``load_dotenv`` for a given key
    wins (because ``override=False``). We therefore load the project-local
    file *first* so it shadows the user-level file when both define the
    same variable, then fall back to the user-level file for keys the
    project didn't override. Real ``os.environ`` values were set before
    this function ran, so ``override=False`` keeps them on top of both.

    We pass an explicit cwd path rather than letting ``load_dotenv()``
    auto-discover, because the auto-discovery (``find_dotenv()``) walks
    upward from *this module's source directory*, not from the user's
    working directory — which would silently pick up a stale ``.env``
    living next to the installed package. Explicit beats implicit here.
    """
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        load_dotenv(cwd_env, override=False)
    user_env = _user_env_path()
    if user_env.is_file():
        load_dotenv(user_env, override=False)


_load_env_files()


# ---------------------------------------------------------------------------
# Pre-configured provider registry
# Users only need to set provider name + API key; base_url and default model
# are filled in automatically from this table.
#
# To support providers we don't ship presets for, use the ``custom`` entry
# and set the matching ``OPAGENT_CUSTOM_*`` env vars (see ``opagent init``
# for the interactive flow). This keeps the registry small while letting
# any OpenAI-compatible endpoint plug in without code changes.
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
    "custom": {
        # All three are read from env at __post_init__ time; see below.
        "base_url": None,
        "default_model": None,
        "env_key": "OPAGENT_CUSTOM_API_KEY",
    },
}


def _default_provider() -> str:
    """Pick the default provider used when ``--provider`` is not given.

    Resolution order (the same one shown to the user by
    ``opagent init`` so they don't see two different "default"s):

    1. ``OPAGENT_DEFAULT_PROVIDER`` if set and recognised
    2. The first provider whose ``env_key`` resolves to a value
       (lets ``MINIMAX_API_KEY=... opagent`` Just Work without also
       requiring ``OPAGENT_DEFAULT_PROVIDER=minimax``)
    3. ``minimax`` as the historical fallback when nothing is configured
    """
    explicit = os.environ.get("OPAGENT_DEFAULT_PROVIDER", "").strip().lower()
    if explicit and explicit in PROVIDER_REGISTRY:
        return explicit

    for name, info in PROVIDER_REGISTRY.items():
        env_key = info.get("env_key")
        if env_key and os.environ.get(env_key):
            return name

    return "minimax"


@dataclass
class LLMConfig:
    """Configuration for a single LLM connection."""

    provider: str = field(default_factory=_default_provider)
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096

    def __post_init__(self):
        # ``provider=None`` is the CLI's way of saying "no --provider flag
        # was passed, please honor OPAGENT_DEFAULT_PROVIDER / first
        # configured key". We resolve here rather than at click-decorator
        # time so a freshly-written ``~/.opagent/.env`` is picked up
        # without restarting the process.
        if self.provider is None:
            self.provider = _default_provider()

        info = PROVIDER_REGISTRY.get(self.provider, {})

        # ``custom`` delegates everything to OPAGENT_CUSTOM_* so users can
        # plug any OpenAI-compatible endpoint in without touching the
        # registry. ``opagent init`` writes these for them.
        if self.provider == "custom":
            if self.base_url is None:
                self.base_url = os.getenv("OPAGENT_CUSTOM_BASE_URL")
            if self.model is None:
                self.model = os.getenv("OPAGENT_CUSTOM_MODEL")
            if self.api_key is None:
                self.api_key = os.getenv("OPAGENT_CUSTOM_API_KEY")
            return

        # Order for model / base_url:
        #   1. explicit kwarg (already set)
        #   2. OPAGENT_DEFAULT_MODEL / OPAGENT_DEFAULT_BASE_URL — lets users
        #      pin a specific model version (e.g. ``kimi-k2-thinking``) or
        #      route an internal proxy without editing code or switching to
        #      ``provider=custom`` and re-typing every field.
        #   3. PROVIDER_REGISTRY default
        if self.model is None:
            self.model = os.getenv("OPAGENT_DEFAULT_MODEL") or info.get(
                "default_model", self.provider
            )
        if self.base_url is None and self.provider != "openai":
            self.base_url = os.getenv("OPAGENT_DEFAULT_BASE_URL") or info.get(
                "base_url"
            )
        elif self.base_url is None and self.provider == "openai":
            # OpenAI's SDK accepts None as "use api.openai.com", but a user
            # with ``OPAGENT_DEFAULT_BASE_URL`` set almost certainly means
            # "route my OpenAI calls through this proxy too" — honor it.
            self.base_url = os.getenv("OPAGENT_DEFAULT_BASE_URL")
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
