"""Error classifier — categorize traceback into actionable error types.

Uses regex pattern matching first (fast, deterministic), then falls back
to LLM classification for ambiguous cases.

Error types:
- ``interface_mismatch``: solver signature doesn't match OptiProfiler's expected API
- ``dependency_missing``: required Python package not installed
- ``timeout``: benchmark exceeded wall-clock time limit
- ``numerical``: NaN/Inf/overflow in solver output
- ``runtime_error``: general Python exception during execution
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorClassification:
    """Result of error classification."""

    error_type: str
    confidence: float   # 0.0 to 1.0
    details: str
    module_name: Optional[str] = None  # for dependency_missing
    expected_signature: Optional[str] = None  # for interface_mismatch


# ---------------------------------------------------------------------------
# Regex-based classification rules
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern, str, float]] = [
    # Interface mismatch
    (
        "interface_mismatch",
        re.compile(
            r"TypeError:.*(?:takes|got|missing|unexpected|positional|keyword).*argument",
            re.IGNORECASE,
        ),
        "Function signature mismatch with OptiProfiler's expected solver interface.",
        0.95,
    ),
    (
        "interface_mismatch",
        re.compile(
            r"TypeError:.*(?:fun|x0|problem)\b.*(?:not callable|is not|has no)",
            re.IGNORECASE,
        ),
        "Solver does not accept the expected problem interface arguments.",
        0.90,
    ),
    # Dependency missing
    (
        "dependency_missing",
        re.compile(r"ModuleNotFoundError: No module named ['\"](\S+)['\"]"),
        "Required Python module is not installed.",
        0.99,
    ),
    (
        "dependency_missing",
        re.compile(r"ImportError: cannot import name ['\"](\S+)['\"]"),
        "Cannot import a specific name from a module.",
        0.85,
    ),
    # Timeout
    (
        "timeout",
        re.compile(r"(?:TimeoutError|timed?\s*out|wall.?clock.*exceed|time.*limit)", re.IGNORECASE),
        "Execution exceeded the time limit.",
        0.90,
    ),
    # Numerical issues
    (
        "numerical",
        re.compile(r"(?:nan|inf|overflow|underflow|divide by zero|invalid value)", re.IGNORECASE),
        "Numerical issue detected in solver output or computation.",
        0.85,
    ),
    (
        "numerical",
        re.compile(r"RuntimeWarning:.*(?:overflow|invalid|divide)", re.IGNORECASE),
        "Runtime numerical warning.",
        0.80,
    ),
    # Common runtime errors
    (
        "runtime_error",
        re.compile(r"NameError: name '(\S+)' is not defined"),
        "NameError — variable or function name not defined (likely a typo).",
        0.90,
    ),
    (
        "runtime_error",
        re.compile(r"ValueError:"),
        "ValueError in solver execution.",
        0.70,
    ),
    (
        "runtime_error",
        re.compile(r"IndexError:"),
        "IndexError — likely array dimension mismatch.",
        0.70,
    ),
    (
        "runtime_error",
        re.compile(r"AttributeError:"),
        "AttributeError — incorrect object usage.",
        0.70,
    ),
    (
        "runtime_error",
        re.compile(r"KeyError:"),
        "KeyError — missing dictionary key.",
        0.70,
    ),
    (
        "runtime_error",
        re.compile(r"(?:SyntaxError|IndentationError):"),
        "Syntax error in the script.",
        0.90,
    ),
]


def classify_error(traceback_text: str) -> ErrorClassification:
    """Classify an error from its traceback text using regex rules.

    Parameters
    ----------
    traceback_text : str
        The full traceback or error message.

    Returns
    -------
    ErrorClassification
        The classified error type with confidence and details.
    """
    best_match: ErrorClassification | None = None
    best_confidence = 0.0

    for error_type, pattern, details, confidence in _PATTERNS:
        m = pattern.search(traceback_text)
        if m and confidence > best_confidence:
            module_name = None
            if error_type == "dependency_missing" and m.lastindex:
                module_name = m.group(1)

            best_match = ErrorClassification(
                error_type=error_type,
                confidence=confidence,
                details=details,
                module_name=module_name,
            )
            best_confidence = confidence

    if best_match:
        return best_match

    return ErrorClassification(
        error_type="runtime_error",
        confidence=0.5,
        details="Unrecognized error — requires LLM analysis.",
    )


def classify_error_with_llm(
    traceback_text: str,
    code: str = "",
    config=None,
) -> ErrorClassification:
    """Classify an error using LLM when regex rules are insufficient.

    Falls back to regex classification if LLM is unavailable.
    """
    # Try regex first
    regex_result = classify_error(traceback_text)
    if regex_result.confidence >= 0.85:
        return regex_result

    # LLM fallback
    if config is None:
        return regex_result

    try:
        from optiprofiler_agent.common.llm_client import create_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = create_llm(config.llm)

        system = (
            "You are an error classifier for OptiProfiler benchmark scripts. "
            "Classify the error into exactly one of these types:\n"
            "- interface_mismatch: solver function signature doesn't match expected API\n"
            "- dependency_missing: a Python module is not installed\n"
            "- timeout: execution exceeded time limit\n"
            "- numerical: NaN/Inf/overflow in computation\n"
            "- runtime_error: other Python exception\n\n"
            "Respond with ONLY the error type name, nothing else."
        )

        user_msg = f"Traceback:\n```\n{traceback_text[-2000:]}\n```"
        if code:
            user_msg += f"\n\nCode:\n```python\n{code[-1000:]}\n```"

        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_msg),
        ])

        classified_type = response.content.strip().lower().replace(" ", "_")
        valid_types = {"interface_mismatch", "dependency_missing", "timeout", "numerical", "runtime_error"}
        if classified_type in valid_types:
            return ErrorClassification(
                error_type=classified_type,
                confidence=0.75,
                details=f"LLM-classified as {classified_type}.",
            )
    except Exception:
        pass

    return regex_result
