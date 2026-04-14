"""Intent classification for user queries.

Classifies each user message into one of several intents so the agent
can route it to the appropriate handler:

- ``factual_query``: parameter defaults, enum values, API facts
  -> answer from knowledge base directly
- ``interface_help``: solver signature adaptation, wrapper generation
  -> invoke interface_adapter
- ``config_suggestion``: benchmark configuration advice
  -> RAG + LLM generation
- ``script_gen``: generate a complete benchmarking script
  -> template filling + LLM
- ``general``: greetings, off-topic, clarification
  -> direct LLM response

The classifier uses a lightweight keyword + regex approach first.
If ambiguous, it falls back to LLM-based classification.

Usage::

    from optiprofiler_agent.formatters.input_parser import classify_intent
    intent = classify_intent("What is the default value of n_jobs?")
    # -> Intent(category="factual_query", confidence=0.9, ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class IntentCategory(str, Enum):
    FACTUAL_QUERY = "factual_query"
    INTERFACE_HELP = "interface_help"
    CONFIG_SUGGESTION = "config_suggestion"
    SCRIPT_GEN = "script_gen"
    GENERAL = "general"


@dataclass
class Intent:
    category: IntentCategory
    confidence: float
    detected_params: list[str] = field(default_factory=list)
    detected_language: str | None = None


_FACTUAL_PATTERNS = [
    re.compile(r"(?:what|what's)\s+(?:is|are)\s+(?:the\s+)?(?:default|value|type|meaning)", re.I),
    re.compile(r"(?:default|type|value)\s+(?:of|for)\s+[`'\"]?\w+", re.I),
    re.compile(r"(?:how\s+many|which)\s+(?:enum|option|value|type)", re.I),
    re.compile(r"(?:list|show|tell)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:enum|option|parameter|value)", re.I),
    re.compile(r"what\s+does\s+[`'\"]?\w+[`'\"]?\s+(?:do|mean|control)", re.I),
    re.compile(r"(?:explain|describe)\s+(?:the\s+)?[`'\"]?\w+[`'\"]?\s+(?:parameter|option|argument)", re.I),
]

_INTERFACE_PATTERNS = [
    re.compile(r"(?:wrapper|interface|signature|adapt|convert)\b", re.I),
    re.compile(r"(?:how\s+(?:to|do\s+I)\s+)?(?:wrap|adapt|connect|plug\s*in)\s+(?:my|a|the)?\s*solver", re.I),
    re.compile(r"fun\s*\(\s*x\s*\)", re.I),
    re.compile(r"(?:function\s+handle|@\w+)\s*.*(?:benchmark|solver)", re.I),
]

_CONFIG_PATTERNS = [
    re.compile(r"(?:how\s+(?:to|should\s+I|do\s+I))\s+(?:configure|set\s*up|use|run|call)", re.I),
    re.compile(r"(?:recommend|suggest|best|optimal)\s+(?:setting|config|option|parameter)", re.I),
    re.compile(r"(?:which|what)\s+(?:ptype|feature|profile|problem)\s+(?:should|to)", re.I),
    re.compile(r"(?:compare|benchmark)\s+(?:my|these|two|multiple)\s+(?:solver|algorithm|method)", re.I),
]

_SCRIPT_PATTERNS = [
    re.compile(r"(?:generate|create|write|give\s+me|show\s+me)\s+(?:a\s+)?(?:complete\s+)?(?:script|code|example|snippet)", re.I),
    re.compile(r"(?:full|complete|runnable|working)\s+(?:script|code|example|program)", re.I),
    re.compile(r"(?:benchmark|profile|run)\s+.*(?:code|script|example)", re.I),
    re.compile(r"(?:can\s+you|please)\s+(?:write|generate|create)\s+(?:a\s+)?(?:python|matlab)?\s*(?:script|code)", re.I),
]

_PARAM_NAMES = re.compile(
    r"\b(n_jobs|n_runs|ptype|plibs|mindim|maxdim|max_eval_factor|"
    r"max_tol_order|seed|savepath|normalized_scores|score_only|silent|"
    r"load|custom_problem_libs_path|noise_level|noise_type|"
    r"perturbation_level|draw_hist_plots|feature_name)\b",
    re.I,
)

_PYTHON_KW = re.compile(
    r"\bpython\b|\.py\b|import\s|from\s+\w+\s+import|def\s+\w+|scipy|numpy|pip\s+install|pycutest",
    re.I,
)
_MATLAB_KW = re.compile(
    r"\bmatlab\b|\.m\b|function\s+\w+\s*=|@\w+|@\(|fminsearch|optimoptions|struct\b|matcutest",
    re.I,
)


def _detect_language(text: str) -> str | None:
    mat = bool(_MATLAB_KW.search(text))
    py = bool(_PYTHON_KW.search(text))
    if mat and not py:
        return "matlab"
    if py and not mat:
        return "python"
    return None


def _score_patterns(text: str, patterns: list[re.Pattern]) -> float:
    hits = sum(1 for p in patterns if p.search(text))
    return min(hits / max(len(patterns) * 0.3, 1), 1.0)


def classify_intent(text: str) -> Intent:
    """Classify user query into an intent category.

    Uses keyword/regex matching. Returns the best-matching intent
    with a confidence score.
    """
    detected_params = _PARAM_NAMES.findall(text)
    language = _detect_language(text)

    scores = {
        IntentCategory.FACTUAL_QUERY: _score_patterns(text, _FACTUAL_PATTERNS),
        IntentCategory.INTERFACE_HELP: _score_patterns(text, _INTERFACE_PATTERNS),
        IntentCategory.CONFIG_SUGGESTION: _score_patterns(text, _CONFIG_PATTERNS),
        IntentCategory.SCRIPT_GEN: _score_patterns(text, _SCRIPT_PATTERNS),
    }

    if detected_params:
        scores[IntentCategory.FACTUAL_QUERY] = max(
            scores[IntentCategory.FACTUAL_QUERY], 0.6)

    if re.search(r"(?:generate|create|write|give)\b.*(?:script|code|example|snippet)", text, re.I):
        scores[IntentCategory.SCRIPT_GEN] = max(scores[IntentCategory.SCRIPT_GEN], 0.85)

    best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_cat]

    if best_score < 0.2:
        best_cat = IntentCategory.GENERAL
        best_score = 0.5

    return Intent(
        category=best_cat,
        confidence=best_score,
        detected_params=[p.lower() for p in detected_params],
        detected_language=language,
    )


def classify_intent_with_llm(text: str, llm) -> Intent:
    """Use an LLM to classify ambiguous queries.

    Falls back to keyword classification if the LLM response is
    unparseable. Requires a LangChain BaseChatModel instance.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    keyword_result = classify_intent(text)
    if keyword_result.confidence >= 0.6:
        return keyword_result

    system = (
        "Classify the user's question about OptiProfiler into exactly one category:\n"
        "- factual_query: asking about parameter defaults, enum values, API facts\n"
        "- interface_help: asking how to adapt/wrap a solver function\n"
        "- config_suggestion: asking for benchmark configuration advice\n"
        "- script_gen: asking to generate a complete script or code\n"
        "- general: greetings, off-topic, or unclear\n\n"
        "Reply with ONLY the category name, nothing else."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=text),
        ])
        cat_str = response.content.strip().lower()
        try:
            category = IntentCategory(cat_str)
        except ValueError:
            return keyword_result

        return Intent(
            category=category,
            confidence=0.8,
            detected_params=keyword_result.detected_params,
            detected_language=keyword_result.detected_language,
        )
    except Exception:
        return keyword_result
