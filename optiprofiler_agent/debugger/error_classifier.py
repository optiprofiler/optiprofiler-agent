"""Error classifier — categorize traceback into actionable error types.

Uses regex pattern matching first (fast, deterministic), then falls back
to LLM classification for ambiguous cases.

Error types:
- ``interface_mismatch``: solver signature doesn't match OptiProfiler's expected API
- ``dependency_missing``: required package / function not available
- ``timeout``: benchmark exceeded wall-clock time limit
- ``numerical``: NaN/Inf/overflow in solver output
- ``runtime_error``: general exception during execution
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
    # Interface mismatch (Python)
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
    # Interface mismatch (MATLAB)
    (
        "interface_mismatch",
        re.compile(
            r"Error using\s+\S+\s*\n.*Too many input arguments",
            re.MULTILINE,
        ),
        "MATLAB function received too many input arguments.",
        0.95,
    ),
    (
        "interface_mismatch",
        re.compile(
            r"Error using\s+\S+\s*\n.*Not enough input arguments",
            re.MULTILINE,
        ),
        "MATLAB function needs more input arguments.",
        0.95,
    ),
    (
        "interface_mismatch",
        re.compile(r"^Not enough input arguments\.", re.MULTILINE),
        "MATLAB function needs more input arguments.",
        0.95,
    ),
    # Dependency missing (Python)
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
    # Dependency missing (MATLAB)
    (
        "dependency_missing",
        re.compile(r"Undefined function or variable '(\S+)'"),
        "MATLAB function or variable is not defined (may need addpath or missing toolbox).",
        0.90,
    ),
    (
        "dependency_missing",
        re.compile(r"Unrecognized function or variable '(\S+)'"),
        "MATLAB unrecognized function (check spelling or required toolbox).",
        0.90,
    ),
    # Newer MATLAB (R2018a+) — "Undefined function 'X' for input arguments of type 'Y'".
    (
        "dependency_missing",
        re.compile(r"Undefined function '(\S+?)' for input arguments"),
        "MATLAB function not defined for the given argument types (may need a toolbox or correct call).",
        0.90,
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
        re.compile(
            r"(?:\bnan\b|\binf\b|overflow|underflow|divide by zero|invalid value)",
            re.IGNORECASE,
        ),
        "Numerical issue detected in solver output or computation.",
        0.85,
    ),
    (
        "numerical",
        re.compile(r"RuntimeWarning:.*(?:overflow|invalid|divide)", re.IGNORECASE),
        "Runtime numerical warning.",
        0.80,
    ),
    # Common runtime errors (Python)
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
    # Runtime errors (MATLAB)
    (
        "runtime_error",
        re.compile(r"Error using\s+(\S+)"),
        "MATLAB runtime error in function call.",
        0.70,
    ),
    (
        "runtime_error",
        re.compile(r"Index exceeds the number of array elements"),
        "MATLAB index out of bounds.",
        0.80,
    ),
    (
        "runtime_error",
        re.compile(r"Dimensions of arrays being concatenated are not consistent"),
        "MATLAB dimension mismatch in array operation.",
        0.80,
    ),
]


def classify_error(traceback_text: str, language: str = "python") -> ErrorClassification:
    """Classify an error from its traceback text using regex rules."""
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
    language: str = "python",
) -> ErrorClassification:
    """Classify an error using LLM when regex rules are insufficient."""
    regex_result = classify_error(traceback_text, language=language)
    if regex_result.confidence >= 0.85:
        return regex_result

    if config is None:
        return regex_result

    lang = (language or "python").strip().lower()
    if lang in ("matlab", "m"):
        dep_desc = "a MATLAB function or toolbox is not available"
        runtime_desc = "other MATLAB error"
        code_tag = "matlab"
    else:
        dep_desc = "a Python module is not installed"
        runtime_desc = "other Python exception"
        code_tag = "python"

    try:
        from optiprofiler_agent.common.llm_client import create_llm
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = create_llm(config.llm)

        system = (
            "You are an error classifier for OptiProfiler benchmark scripts. "
            f"The code is written in {'MATLAB' if code_tag == 'matlab' else 'Python'}. "
            "Classify the error into exactly one of these types:\n"
            "- interface_mismatch: solver function signature doesn't match expected API\n"
            f"- dependency_missing: {dep_desc}\n"
            "- timeout: execution exceeded time limit\n"
            "- numerical: NaN/Inf/overflow in computation\n"
            f"- runtime_error: {runtime_desc}\n\n"
            "Respond with ONLY the error type name, nothing else."
        )

        user_msg = f"Traceback:\n```\n{traceback_text[-2000:]}\n```"
        if code:
            user_msg += f"\n\nCode:\n```{code_tag}\n{code[-1000:]}\n```"

        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_msg),
        ])

        classified_type = response.content.strip().lower().replace(" ", "_")
        valid_types = {
            "interface_mismatch", "dependency_missing", "timeout",
            "numerical", "runtime_error",
        }
        if classified_type in valid_types:
            return ErrorClassification(
                error_type=classified_type,
                confidence=0.75,
                details=f"LLM-classified as {classified_type}.",
            )
    except Exception:
        pass

    return regex_result
