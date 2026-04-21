"""Unit tests for OpenAI-compat LLM client quirks (thinking + tool_calls)."""

from optiprofiler_agent.common.llm_client import create_llm, inject_reasoning_content_placeholders
from optiprofiler_agent.config import LLMConfig


def test_inject_reasoning_content_adds_placeholder_for_tool_calls():
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function"}]},
    ]
    inject_reasoning_content_placeholders(msgs)
    assert msgs[0]["reasoning_content"] == "."


def test_inject_skips_when_reasoning_content_nonempty():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1"}],
            "reasoning_content": "real chain-of-thought",
        },
    ]
    inject_reasoning_content_placeholders(msgs)
    assert msgs[0]["reasoning_content"] == "real chain-of-thought"


def test_inject_skips_user_and_assistant_without_tools():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": []},
    ]
    inject_reasoning_content_placeholders(msgs)
    assert "reasoning_content" not in msgs[0]
    assert "reasoning_content" not in msgs[1]


def test_create_llm_minimax_wraps_compat_subclass():
    llm = create_llm(LLMConfig(provider="minimax", api_key="sk-test"))
    assert type(llm).__name__ == "_ThinkingToolReplayCompatChatOpenAI"


def test_create_llm_openai_is_plain_chat_openai():
    from langchain_openai import ChatOpenAI

    llm = create_llm(LLMConfig(provider="openai", api_key="sk-test"))
    assert type(llm) is ChatOpenAI
