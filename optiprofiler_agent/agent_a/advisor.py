"""Agent A — Product Advisor: the user's first point of contact.

This module implements the core conversation loop for Agent A.
It loads structured knowledge, assembles prompts, and manages
multi-turn dialogue via LangChain message history.

The knowledge injection is language-aware: if the user asks about
Python or MATLAB, only the relevant language knowledge is included.

Usage::

    from optiprofiler_agent.config import AgentConfig
    from optiprofiler_agent.agent_a.advisor import AdvisorAgent

    agent = AdvisorAgent(AgentConfig())
    reply = agent.chat("What is the default n_jobs?")
    print(reply)
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from optiprofiler_agent.config import AgentConfig
from optiprofiler_agent.common.knowledge_base import KnowledgeBase
from optiprofiler_agent.common.llm_client import create_llm
from optiprofiler_agent.common.rag import KnowledgeRAG

_MATLAB_PATTERNS = re.compile(
    r"\bmatlab\b|\.m\b|function\s+\w+\s*=|@\w+|@\(|fminsearch|fminunc|patternsearch"
    r"|optimoptions|optimset|cell\s+array|struct\b|parfor\b|matcutest",
    re.IGNORECASE,
)

_PYTHON_PATTERNS = re.compile(
    r"\bpython\b|\.py\b|import\s|from\s+\w+\s+import|def\s+\w+|scipy"
    r"|numpy|minimize\(|pip\s+install|pycutest",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str | None:
    """Detect whether the user is asking about Python or MATLAB.

    Returns "python", "matlab", or None (ambiguous/general).
    """
    mat = bool(_MATLAB_PATTERNS.search(text))
    py = bool(_PYTHON_PATTERNS.search(text))
    if mat and not py:
        return "matlab"
    if py and not mat:
        return "python"
    return None


class AdvisorAgent:
    """OptiProfiler Product Advisor — answers questions about the platform."""

    def __init__(self, config: AgentConfig | None = None):
        self._config = config or AgentConfig()
        self._llm = create_llm(self._config.llm)
        self._kb = KnowledgeBase(self._config.knowledge_dir)
        self._rag: KnowledgeRAG | None = None
        self._history: list = []
        self._current_language: str | None = None

        if self._config.rag_enabled:
            try:
                self._rag = KnowledgeRAG(
                    self._config.knowledge_dir,
                    persist_dir=self._config.rag_persist_dir,
                )
                self._rag.build_index()
            except ImportError:
                self._rag = None

        self._prompt_template = self._load_prompt_template()
        self._few_shots = self._load_few_shots()
        self._system_prompt = self._build_system_prompt(language=None)

    def _load_prompt_template(self) -> str:
        path = Path(__file__).parent / "prompts" / "system_prompt.md"
        return path.read_text(encoding="utf-8")

    def _load_few_shots(self) -> str:
        path = Path(__file__).parent / "prompts" / "few_shots.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _build_system_prompt(
        self, language: str | None = None, rag_context: str = "",
    ) -> str:
        knowledge_text = self._kb.to_prompt_text(language=language)
        if rag_context:
            knowledge_text += "\n\n" + rag_context
        system_text = self._prompt_template.replace("{knowledge_text}", knowledge_text)

        if self._few_shots:
            system_text += "\n\n## Example Conversations\n\n"
            system_text += self._few_shots

        return system_text

    def chat(self, user_message: str) -> str:
        """Send a message and get a reply. Maintains conversation history."""
        detected = _detect_language(user_message)
        if detected:
            self._current_language = detected

        rag_context = ""
        if self._rag and self._rag.is_ready:
            rag_context = self._rag.retrieve_as_text(
                user_message,
                top_k=self._config.rag_top_k,
                language=self._current_language,
            )

        prompt = self._build_system_prompt(
            language=self._current_language, rag_context=rag_context)
        self._system_prompt = prompt

        messages = [SystemMessage(content=prompt)]
        messages.extend(self._history)
        messages.append(HumanMessage(content=user_message))

        response: AIMessage = self._llm.invoke(messages)
        reply = response.content

        if "<think>" in reply and "</think>" in reply:
            end = reply.index("</think>") + len("</think>")
            reply = reply[end:].strip()

        self._history.append(HumanMessage(content=user_message))
        self._history.append(AIMessage(content=reply))

        return reply

    def reset(self):
        """Clear conversation history and language detection."""
        self._history.clear()
        self._current_language = None

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def detected_language(self) -> str | None:
        return self._current_language
