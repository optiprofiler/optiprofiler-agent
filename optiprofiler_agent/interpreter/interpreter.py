"""Agent C — Results Interpreter: turns benchmark outputs into reports.

Pipeline:

1. **Rule engine** (``summary.py``): scores + curves + logs ->
   ``BenchmarkSummary`` (verified facts, no LLM).
2. **Structured LLM** with thinking-model-aware JSON extraction:
   summary -> typed ``BenchmarkReport`` Pydantic object. Tries
   ``llm.with_structured_output`` first; on parse failure (common with
   ``<think>...</think>`` reasoning models such as MiniMax-M2,
   DeepSeek-R1, Kimi-thinking) falls back to a manual extract-then-parse
   path that strips reasoning tags and unwraps fenced JSON blocks.
3. **Validator** (``report_validator.validate_report``) checks
   cross-field invariants; errors trigger one retry with feedback.
4. **Renderer** (``renderer.render_markdown``): typed report ->
   Markdown / JSON / HTML.

When *both* the structured path and the manual JSON path fail, we fall
back to legacy free-form Markdown (with thinking tags stripped) so
``opagent interpret`` always returns *something* useful.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from optiprofiler_agent.interpreter.renderer import render_html, render_markdown
from optiprofiler_agent.interpreter.report_schema import BenchmarkReport
from optiprofiler_agent.interpreter.report_validator import (
    format_feedback_for_llm,
    validate_report,
)
from optiprofiler_agent.interpreter.summary import BenchmarkSummary, build_summary
from optiprofiler_agent.config import AgentConfig


_LOG = logging.getLogger(__name__)
_PROMPTS_DIR = Path(__file__).parent / "prompts"

MAX_REPORT_RETRIES = 1


# ---------------------------------------------------------------------------
# Thinking-model output sanitisation
# ---------------------------------------------------------------------------

# Canonical implementation lives in ``common.text_clean`` so the same
# regex is shared by session_log / trajectory dump / report renderer.
# Re-exported under the legacy underscore name to keep existing imports
# (and the public surface used by tests) working without churn.
from optiprofiler_agent.common.text_clean import strip_thinking as _strip_thinking  # noqa: E402


_FENCED_JSON_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def _extract_json_blob(text: str) -> str:
    """Extract the first JSON object from a model reply.

    Tolerates: leading <think>...</think> blocks, ```json ... ``` fences,
    leading/trailing prose. Returns the largest balanced object found via
    a brace-counting scan when no fence matches.
    """
    text = _strip_thinking(text or "").strip()
    if not text:
        return ""

    # 1. Prefer a fenced ```json``` block — most reliable signal.
    match = _FENCED_JSON_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    # 2. If the text already starts with '{', return as-is.
    if text.startswith("{"):
        return text

    # 3. Brace-count scan from the first '{' to its matching '}'.
    start = text.find("{")
    if start < 0:
        return text  # nothing JSON-shaped; let the parser raise
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return text[start:]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _build_user_message(summary: BenchmarkSummary, language: str = "English") -> str:
    """Build the user message containing the JSON data for the LLM."""
    json_data = summary.to_json(indent=2)

    # Truncate if extremely large (> 30k chars) to fit context window
    if len(json_data) > 30000:
        d = summary.to_dict()
        for key in ["head_to_head", "curve_crossovers", "per_tolerance_scores"]:
            items = d.get(key)
            if isinstance(items, list) and len(items) > 10:
                d[key] = items[:10]
            elif isinstance(items, dict) and len(items) > 10:
                d[key] = dict(list(items.items())[:10])
        json_data = json.dumps(d, indent=2, ensure_ascii=False)

    return (
        f"Please analyse the following benchmark results and produce a "
        f"BenchmarkReport in {language}. Solver names referenced in your "
        f"output MUST exactly match the names in this JSON. Do NOT invent "
        f"problem names; only cite ones that appear in the input.\n\n"
        f"```json\n{json_data}\n```"
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def interpret(
    results_dir: str | Path,
    config: AgentConfig | None = None,
    language: str = "English",
    read_profiles: bool = True,
    llm_enabled: bool = True,
    output_format: str = "markdown",
) -> str:
    """Generate a report from benchmark results."""
    summary = build_summary(results_dir, read_profiles=read_profiles)

    if not llm_enabled:
        return summary.to_json()

    config = config or AgentConfig()
    report = _generate_structured_report(summary, config, language)
    if report is None:
        return _legacy_freeform_report(summary, config, language)

    return _render(report, summary, output_format)


def interpret_from_summary(
    summary: BenchmarkSummary,
    config: AgentConfig | None = None,
    language: str = "English",
    output_format: str = "markdown",
) -> str:
    """Generate a report from a pre-built summary (skips data loading)."""
    config = config or AgentConfig()
    report = _generate_structured_report(summary, config, language)
    if report is None:
        return _legacy_freeform_report(summary, config, language)
    return _render(report, summary, output_format)


def generate_report_object(
    summary: BenchmarkSummary,
    config: AgentConfig | None = None,
    language: str = "English",
) -> BenchmarkReport | None:
    """Public hook for callers that want the typed report directly
    (e.g. a future web platform that renders the schema natively).

    Returns ``None`` when no path produces a valid report.
    """
    config = config or AgentConfig()
    return _generate_structured_report(summary, config, language)


def _render(report: BenchmarkReport, summary: BenchmarkSummary, fmt: str) -> str:
    if fmt == "json":
        return report.model_dump_json(indent=2)
    if fmt == "html":
        return render_html(report, summary)
    return render_markdown(report, summary)


# ---------------------------------------------------------------------------
# Structured-report generation
# ---------------------------------------------------------------------------


def _build_messages(summary: BenchmarkSummary, language: str) -> tuple[list, str]:
    """Build the system + user messages and return them along with the
    full system prompt string (needed by the manual JSON path so it can
    re-augment with explicit schema instructions)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = _load_prompt("system_prompt.md")
    full_system = (
        f"{system_prompt}\n\n"
        "## Output Contract\n\n"
        "You MUST return a BenchmarkReport object that satisfies the "
        "provided JSON Schema. Solver names you reference MUST be drawn "
        "verbatim from the input JSON's `solver_names` field. Do NOT "
        "invent problem names; only cite ones present in the input."
    )

    user_message = _build_user_message(summary, language=language)
    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_message),
    ]
    return messages, full_system


def _generate_structured_report(
    summary: BenchmarkSummary,
    config: AgentConfig,
    language: str,
) -> BenchmarkReport | None:
    """Run the structured LLM stage with one validator-driven retry.

    Strategy:

    1. Try ``llm.with_structured_output(...)`` (provider-side schema).
    2. On parse error (common with ``<think>...</think>`` thinking
       models), fall back to a manual extract-and-parse path.
    3. Validate; on business-invariant errors, retry once with
       feedback.

    Returns ``None`` only when *both* the structured and manual paths
    fail entirely; the caller then falls back to free-form Markdown.
    """
    from optiprofiler_agent.common.llm_client import create_llm

    llm = create_llm(config.llm)
    messages, full_system = _build_messages(summary, language)

    report = _try_structured_output(llm, messages)
    if report is None:
        report = _try_manual_json(llm, messages, full_system)
    if report is None:
        return None

    validation = validate_report(report, summary)
    if not validation.has_errors:
        return report

    feedback = format_feedback_for_llm(validation)
    for _ in range(MAX_REPORT_RETRIES):
        retry = _retry_with_feedback(llm, messages, feedback, full_system)
        if retry is None:
            return report  # keep the best-so-far rather than nothing
        report = retry
        validation = validate_report(report, summary)
        if not validation.has_errors:
            return report

    return report


def _try_structured_output(llm, messages) -> BenchmarkReport | None:
    """First-attempt path: provider-side ``with_structured_output``.

    Returns ``None`` when the binding is unavailable or the call
    fails (e.g. JSON parser chokes on a leading ``<think>`` block).
    """
    structured_llm = _bind_structured_output(llm)
    if structured_llm is None:
        return None
    try:
        return structured_llm.invoke(messages)
    except Exception as exc:
        _LOG.info(
            "with_structured_output failed (%s); switching to manual JSON path",
            exc,
        )
        return None


def _try_manual_json(llm, messages, full_system: str) -> BenchmarkReport | None:
    """Fallback: invoke the model raw, strip ``<think>``, extract JSON.

    Designed for thinking-model outputs that ``with_structured_output``
    cannot parse.
    """
    from langchain_core.messages import SystemMessage

    schema_text = json.dumps(BenchmarkReport.model_json_schema())
    augmented_system = SystemMessage(
        content=(
            f"{full_system}\n\n"
            "## JSON Output Format\n\n"
            "Reply with EXACTLY one JSON object that matches the schema "
            "below. Wrap it in a ```json ... ``` fenced block. Do NOT "
            "include any prose outside the fenced block.\n\n"
            f"```json\n{schema_text}\n```"
        )
    )
    try:
        response = llm.invoke([augmented_system, *messages[1:]])
    except Exception as exc:
        _LOG.warning("Manual JSON path: LLM call failed: %s", exc)
        return None

    raw = _response_text(response)
    return _parse_report_json(raw)


def _retry_with_feedback(
    llm,
    messages,
    feedback: str,
    full_system: str,
) -> BenchmarkReport | None:
    """Single retry that mirrors the two-path strategy of the initial
    attempt: prefer with_structured_output, fall back to manual JSON."""
    from langchain_core.messages import HumanMessage

    structured_llm = _bind_structured_output(llm)
    retry_messages = list(messages) + [HumanMessage(content=feedback)]

    if structured_llm is not None:
        try:
            return structured_llm.invoke(retry_messages)
        except Exception as exc:
            _LOG.info(
                "Structured retry failed (%s); switching to manual JSON",
                exc,
            )
    return _try_manual_json(llm, retry_messages, full_system)


def _response_text(response: Any) -> str:
    """Extract a string from a LangChain message-like response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some providers return a list of content blocks.
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _parse_report_json(raw: str) -> BenchmarkReport | None:
    """Strip thinking tags, extract JSON, parse to ``BenchmarkReport``."""
    blob = _extract_json_blob(raw)
    if not blob:
        _LOG.warning("Manual JSON path: no JSON object found in response")
        return None
    try:
        return BenchmarkReport.model_validate_json(blob)
    except ValidationError as exc:
        _LOG.warning("Manual JSON path: schema validation failed: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 — defensive: never raise to caller
        _LOG.warning("Manual JSON path: unexpected parse failure: %s", exc)
        return None


def _bind_structured_output(llm):
    """Best-effort ``with_structured_output`` binding.

    Tries ``method='json_schema'`` first (preferred for OpenAI-compatible
    providers including Kimi / MiniMax), falls back to function-calling,
    and returns ``None`` if neither is available.
    """
    for method in ("json_schema", "function_calling"):
        try:
            return llm.with_structured_output(BenchmarkReport, method=method)
        except (TypeError, ValueError, NotImplementedError):
            continue
        except Exception as exc:
            _LOG.debug(
                "with_structured_output(method=%s) raised %s",
                method,
                exc,
            )
            continue
    try:
        return llm.with_structured_output(BenchmarkReport)
    except Exception as exc:
        _LOG.warning("with_structured_output unavailable: %s", exc)
        return None


def _legacy_freeform_report(
    summary: BenchmarkSummary,
    config: AgentConfig,
    language: str,
) -> str:
    """Last-resort fallback: free-form Markdown from the LLM.

    Used only when *both* the structured path and the manual JSON path
    fail. Output is run through ``_strip_thinking`` so chain-of-thought
    reasoning never leaks into the user's report file.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from optiprofiler_agent.common.llm_client import create_llm

    system_prompt = _load_prompt("system_prompt.md")
    report_template = _load_prompt("report_template.md")
    full_system = (
        f"{system_prompt}\n\n"
        f"## Report Template\n\n{report_template}"
    )
    user_message = _build_user_message(summary, language=language)

    llm = create_llm(config.llm)
    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return _strip_thinking(_response_text(response))


__all__ = [
    "MAX_REPORT_RETRIES",
    "generate_report_object",
    "interpret",
    "interpret_from_summary",
]
