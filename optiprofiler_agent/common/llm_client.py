"""Unified LLM client — a factory that returns a LangChain ``BaseChatModel``.

Every provider supported in :data:`config.PROVIDER_REGISTRY` is
OpenAI-compatible **except** Anthropic, which needs its own SDK.
The factory transparently handles this so callers always get the
same ``BaseChatModel`` interface.

Usage::

    from optiprofiler_agent.config import LLMConfig
    from optiprofiler_agent.common.llm_client import create_llm

    llm = create_llm(LLMConfig(provider="kimi"))
    response = llm.invoke("Hello!")
    print(response.content)
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from optiprofiler_agent.config import LLMConfig


def create_llm(cfg: LLMConfig) -> BaseChatModel:
    """Create a LangChain chat model from *cfg*.

    For OpenAI-compatible providers (Kimi, MiniMax, DeepSeek, OpenAI itself)
    we use ``ChatOpenAI`` with a custom ``base_url``.

    For Anthropic we use ``ChatAnthropic`` (requires ``pip install
    langchain-anthropic``).
    """
    if cfg.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "pip install langchain-anthropic  to use Anthropic models"
            ) from exc
        return ChatAnthropic(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": cfg.model,
        "api_key": cfg.api_key,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url

    return ChatOpenAI(**kwargs)
