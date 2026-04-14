"""Agent C — Results Interpreter: turns benchmark outputs into natural-language reports.

Two-stage pipeline:
1. Rule engine (summary.py): scores + curves + logs → structured JSON summary
2. LLM polish (this module): JSON summary → natural-language Markdown report

Supports a no-LLM mode that outputs the raw JSON summary for programmatic use.
"""

from __future__ import annotations

from pathlib import Path

from optiprofiler_agent.agent_c.summary import BenchmarkSummary, build_summary
from optiprofiler_agent.config import AgentConfig


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def _build_user_message(summary: BenchmarkSummary, language: str = "English") -> str:
    """Build the user message containing the JSON data for the LLM."""
    json_data = summary.to_json(indent=2)

    # Truncate if extremely large (> 30k chars) to fit context window
    if len(json_data) > 30000:
        # Keep essential fields, trim verbose ones
        d = summary.to_dict()
        for key in ["head_to_head", "curve_crossovers", "per_tolerance_scores"]:
            items = d.get(key)
            if isinstance(items, list) and len(items) > 10:
                d[key] = items[:10]
            elif isinstance(items, dict) and len(items) > 10:
                d[key] = dict(list(items.items())[:10])
        import json
        json_data = json.dumps(d, indent=2, ensure_ascii=False)

    return (
        f"Please analyze the following benchmark results and generate a report "
        f"in {language}. Follow the report template structure.\n\n"
        f"```json\n{json_data}\n```"
    )


def interpret(
    results_dir: str | Path,
    config: AgentConfig | None = None,
    language: str = "English",
    read_profiles: bool = True,
    llm_enabled: bool = True,
) -> str:
    """Generate a Markdown analysis report from benchmark results.

    Parameters
    ----------
    results_dir : str or Path
        Path to the experiment output directory.
    config : AgentConfig, optional
        Agent configuration (for LLM settings). Uses defaults if not given.
    language : str
        Language for the report output.
    read_profiles : bool
        Whether to read profile PDFs (slower but more detailed).
    llm_enabled : bool
        If False, returns the raw JSON summary instead of an LLM-generated report.

    Returns
    -------
    str
        Markdown report (if LLM enabled) or JSON summary string.
    """
    summary = build_summary(results_dir, read_profiles=read_profiles)

    if not llm_enabled:
        return summary.to_json()

    config = config or AgentConfig()

    system_prompt = _load_prompt("system_prompt.md")
    report_template = _load_prompt("report_template.md")

    full_system = (
        f"{system_prompt}\n\n"
        f"## Report Template\n\n{report_template}"
    )

    user_message = _build_user_message(summary, language=language)

    from optiprofiler_agent.common.llm_client import create_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = create_llm(config.llm)
    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    return response.content


def interpret_from_summary(
    summary: BenchmarkSummary,
    config: AgentConfig | None = None,
    language: str = "English",
) -> str:
    """Generate a report from a pre-built summary (skips data loading).

    Useful when the summary has already been computed and you want to
    regenerate the report with different LLM settings or language.
    """
    config = config or AgentConfig()

    system_prompt = _load_prompt("system_prompt.md")
    report_template = _load_prompt("report_template.md")

    full_system = f"{system_prompt}\n\n## Report Template\n\n{report_template}"
    user_message = _build_user_message(summary, language=language)

    from optiprofiler_agent.common.llm_client import create_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = create_llm(config.llm)
    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    return response.content
