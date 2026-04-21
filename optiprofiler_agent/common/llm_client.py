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

from typing import Any

from langchain_core.language_models import BaseChatModel

from optiprofiler_agent.config import LLMConfig


# MiniMax / Moonshot Kimi (and some proxied stacks) enable *thinking* alongside
# ``tool_calls``. Their HTTP API rejects replayed assistant turns that contain
# ``tool_calls`` but omit the sibling ``reasoning_content`` field.
#
# LangChain's OpenAI-compat path does not round-trip that field into the next
# request's JSON, so multi-turn ReAct loops hit HTTP 400.  We inject a minimal
# non-empty placeholder on the wire-format ``messages`` list (after LangChain's
# own conversion), matching a widely used community workaround.
_REASONING_PLACEHOLDER = "."

# Lazily defined — ``langchain_openai`` is only imported when an OpenAI-compat
# model is actually constructed (keeps ``import optiprofiler_agent`` lighter).
_thinking_tool_replay_compat_cls: type | None = None


def _needs_thinking_tool_replay_patch(cfg: LLMConfig) -> bool:
    """True when the upstream stack is known to 400 without ``reasoning_content``."""
    if cfg.provider in ("minimax", "kimi"):
        return True
    if cfg.provider != "custom" or not cfg.base_url:
        return False
    u = cfg.base_url.lower()
    # Custom proxies / regional mirrors for the same vendors.
    return any(
        n in u
        for n in (
            "minimax",
            "minimaxi",
            "moonshot",
            "kimi",
        )
    )


def inject_reasoning_content_placeholders(messages: list[dict[str, Any]] | None) -> None:
    """Mutate *messages* in-place for thinking+tool-call provider compatibility.

    For each assistant dict that has ``tool_calls`` but no non-empty
    ``reasoning_content``, set ``reasoning_content`` to a one-character
    placeholder.  Safe to call repeatedly (idempotent once applied).
    """
    if not messages:
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        rc = msg.get("reasoning_content")
        if rc is not None and str(rc).strip():
            continue
        msg["reasoning_content"] = _REASONING_PLACEHOLDER


def _get_thinking_tool_replay_compat_cls() -> type:
    global _thinking_tool_replay_compat_cls
    if _thinking_tool_replay_compat_cls is not None:
        return _thinking_tool_replay_compat_cls
    from langchain_openai import ChatOpenAI

    class _ThinkingToolReplayCompatChatOpenAI(ChatOpenAI):
        def _get_request_payload(  # type: ignore[override]
            self,
            input_,
            *,
            stop=None,
            **kw: Any,
        ) -> dict[str, Any]:
            payload = super()._get_request_payload(input_, stop=stop, **kw)
            inject_reasoning_content_placeholders(payload.get("messages"))
            return payload

    _thinking_tool_replay_compat_cls = _ThinkingToolReplayCompatChatOpenAI
    return _thinking_tool_replay_compat_cls


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

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "api_key": cfg.api_key,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url

    if _needs_thinking_tool_replay_patch(cfg):
        return _get_thinking_tool_replay_compat_cls()(**kwargs)

    return ChatOpenAI(**kwargs)
