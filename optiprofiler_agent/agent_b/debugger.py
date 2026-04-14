"""Agent B — Automatic Debugger: diagnoses and suggests fixes for benchmark errors.

Flow:
1. Receive: solver code + traceback
2. Classify error type (via error_classifier)
3. Route to specialized handler:
   - interface_mismatch → interface_adapter.generate_wrapper()
   - dependency_missing → pip install suggestion
   - runtime_error → LLM analysis + code fix
   - timeout / numerical → diagnostic advice (no auto-fix)
4. Validate fix with syntax_checker + api_checker
5. Retry up to N times if validation fails
6. Output: DebugResult with fixed code and diagnostic report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from optiprofiler_agent.agent_b.error_classifier import (
    ErrorClassification,
    classify_error,
    classify_error_with_llm,
)
from optiprofiler_agent.config import AgentConfig


_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class DebugResult:
    """Result of the debug process."""

    classification: ErrorClassification
    fixed_code: Optional[str] = None
    diagnostic_report: str = ""
    attempts: int = 0
    validation_passed: bool = False


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _handle_interface_mismatch(code: str, error: str) -> tuple[str | None, str]:
    """Handle interface mismatch by generating a wrapper."""
    try:
        from optiprofiler_agent.common.interface_adapter import analyze_solver, generate_wrapper

        analysis = analyze_solver(code)
        if analysis and analysis.get("needs_wrapper"):
            wrapper = generate_wrapper(analysis)
            report = (
                "## Interface Mismatch Detected\n\n"
                f"Your solver's signature doesn't match OptiProfiler's expected interface.\n\n"
                f"**Missing parameters:** {', '.join(analysis.get('missing', []))}\n"
                f"**Extra parameters:** {', '.join(analysis.get('extra', []))}\n\n"
                "A wrapper function has been generated to adapt your solver."
            )
            return wrapper, report
    except Exception:
        pass

    report = (
        "## Interface Mismatch Detected\n\n"
        "Your solver function signature doesn't match what OptiProfiler expects.\n\n"
        "OptiProfiler calls solvers with `solver(fun, x0)` for unconstrained problems.\n"
        "Make sure your solver accepts at least `fun` (callable) and `x0` (initial point).\n\n"
        f"**Error:** {error[:500]}"
    )
    return None, report


def _handle_dependency_missing(classification: ErrorClassification) -> tuple[str | None, str]:
    """Handle missing dependency with install instructions."""
    module = classification.module_name or "unknown"
    report = (
        "## Missing Dependency\n\n"
        f"The module `{module}` is not installed.\n\n"
        f"**Fix:** Run the following command:\n\n"
        f"```bash\npip install {module}\n```\n\n"
        "Then re-run your benchmark script."
    )
    return None, report


def _handle_timeout(error: str) -> tuple[str | None, str]:
    """Handle timeout errors with diagnostic advice."""
    report = (
        "## Timeout Detected\n\n"
        "Your benchmark exceeded the time limit. Possible causes:\n\n"
        "1. **Too many problems:** Reduce the dimension range or problem count.\n"
        "2. **Slow solver:** Your solver may be too slow for the problem set.\n"
        "3. **Infinite loop:** Check if your solver has proper termination conditions.\n\n"
        "**Suggestions:**\n"
        "- Set `maxfev` (max function evaluations) in your solver options.\n"
        "- Reduce `n_runs` to test fewer random starts.\n"
        "- Use `n_jobs` for parallel execution.\n\n"
        f"**Error excerpt:** {error[:300]}"
    )
    return None, report


def _handle_numerical(error: str) -> tuple[str | None, str]:
    """Handle numerical issues with diagnostic advice."""
    report = (
        "## Numerical Issue Detected\n\n"
        "Your solver produced NaN, Inf, or overflow values. Possible causes:\n\n"
        "1. **Unbounded objective:** The problem's objective function may be unbounded.\n"
        "2. **Poor initial point:** The starting point may be in a numerically unstable region.\n"
        "3. **Missing bounds handling:** Your solver may not handle bound constraints properly.\n\n"
        "**Suggestions:**\n"
        "- Add `try/except` around your solver to catch numerical errors.\n"
        "- Use `numpy.clip` to bound intermediate values.\n"
        "- Check if your solver handles the case where `fun(x)` returns very large values.\n\n"
        f"**Error excerpt:** {error[:300]}"
    )
    return None, report


def _handle_runtime_with_llm(
    code: str,
    error: str,
    config: AgentConfig,
    max_retries: int = 3,
    code_char_limit: int = 0,
) -> tuple[str | None, str, int]:
    """Use LLM to analyze and fix runtime errors.

    Parameters
    ----------
    code_char_limit : int
        Max characters of code sent to LLM. 0 means no limit.

    Returns (fixed_code, report, attempts).
    """
    from optiprofiler_agent.common.llm_client import create_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    system_prompt = _load_prompt("system_prompt.md")
    fix_templates = _load_prompt("fix_templates.md")

    if not system_prompt:
        system_prompt = (
            "You are a Python debugging expert specializing in OptiProfiler benchmark scripts. "
            "Analyze the error and provide a corrected version of the code. "
            "Return ONLY the corrected Python code in a code block."
        )

    full_system = system_prompt
    if fix_templates:
        full_system += f"\n\n## Common Fix Patterns\n\n{fix_templates}"

    llm = create_llm(config.llm)

    attempts = 0
    last_error = error
    current_code = code

    def _truncate_code(src: str, limit: int) -> str:
        """Truncate code smartly: keep both head (imports/defs) and tail (main logic)."""
        if limit <= 0 or len(src) <= limit:
            return src
        head_size = limit // 2
        tail_size = limit - head_size - 50
        return (
            src[:head_size]
            + "\n\n# ... (middle section omitted) ...\n\n"
            + src[-tail_size:]
        )

    for attempt in range(max_retries):
        attempts += 1

        code_for_llm = _truncate_code(current_code, code_char_limit)
        user_msg = (
            f"## Code\n\n```python\n{code_for_llm}\n```\n\n"
            f"## Error\n\n```\n{last_error[-2000:]}\n```\n\n"
            "Please fix the code. Return the COMPLETE corrected code in a Python code block. "
            "Include ALL imports and function definitions, not just the changed part."
        )

        try:
            response = llm.invoke([
                SystemMessage(content=full_system),
                HumanMessage(content=user_msg),
            ])

            reply = response.content
            fixed = _extract_code_from_reply(reply)

            if not fixed:
                continue

            validation_errors = _validate_code(fixed)
            if not validation_errors:
                report = (
                    f"## Runtime Error Fixed (attempt {attempts})\n\n"
                    f"The LLM identified and fixed the issue.\n\n"
                    f"**Original error:** {error[:200]}\n"
                )
                return fixed, report, attempts

            last_error = f"Validation failed: {'; '.join(validation_errors)}"
            current_code = fixed

        except Exception as e:
            last_error = str(e)

    report = (
        f"## Runtime Error — Fix Attempted ({attempts} attempts)\n\n"
        f"The automatic fix did not pass validation after {attempts} attempts.\n\n"
        f"**Original error:** {error[:300]}\n\n"
        "**Suggestion:** Review the error manually and check:\n"
        "1. Your solver function signature matches `solver(fun, x0)`.\n"
        "2. All required imports are present.\n"
        "3. The `benchmark()` call has at least 2 solvers.\n"
    )
    return None, report, attempts


def _extract_code_from_reply(reply: str) -> str | None:
    """Extract Python code from an LLM reply."""
    import re

    # Look for ```python ... ``` blocks
    pattern = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(reply)
    if matches:
        return matches[0].strip()

    # Look for ``` ... ``` blocks
    pattern = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(reply)
    if matches:
        return matches[0].strip()

    return None


def _validate_code(code: str) -> list[str]:
    """Validate code using syntax_checker and api_checker.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    try:
        from optiprofiler_agent.validators.syntax_checker import check_code_string
        result = check_code_string(code)
        if result.has_errors:
            for err in result.errors:
                errors.append(f"Syntax error at line {err.line}: {err.message}")
    except Exception:
        pass

    try:
        from optiprofiler_agent.validators.api_checker import validate_benchmark_call
        result = validate_benchmark_call(code)
        if result.has_errors:
            for issue in result.issues:
                if issue.severity == "error":
                    errors.append(f"API error: {issue.message}")
    except Exception:
        pass

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def debug_script(
    code: str,
    error: str,
    config: AgentConfig | None = None,
) -> DebugResult:
    """Diagnose and attempt to fix a benchmark script error.

    Parameters
    ----------
    code : str
        The Python source code that produced the error.
    error : str
        The full traceback or error message.
    config : AgentConfig, optional
        Agent configuration (for LLM settings and max retries).

    Returns
    -------
    DebugResult
        Classification, optional fixed code, and diagnostic report.
    """
    config = config or AgentConfig()

    # Step 1: Classify the error
    classification = classify_error_with_llm(error, code, config)

    # Step 2: Route to handler
    fixed_code: str | None = None
    report: str = ""
    attempts: int = 0

    if classification.error_type == "interface_mismatch":
        fixed_code, report = _handle_interface_mismatch(code, error)
        attempts = 1

    elif classification.error_type == "dependency_missing":
        fixed_code, report = _handle_dependency_missing(classification)
        attempts = 1

    elif classification.error_type == "timeout":
        fixed_code, report = _handle_timeout(error)
        attempts = 1

    elif classification.error_type == "numerical":
        fixed_code, report = _handle_numerical(error)
        attempts = 1

    else:
        # runtime_error or unknown — try LLM fix
        fixed_code, report, attempts = _handle_runtime_with_llm(
            code, error, config,
            max_retries=config.max_debug_retries,
            code_char_limit=config.code_char_limit,
        )

    # Step 3: Validate the fix if we have one
    validation_passed = False
    if fixed_code:
        errors = _validate_code(fixed_code)
        validation_passed = len(errors) == 0
        if errors:
            report += (
                "\n\n**Validation warnings on suggested fix:**\n"
                + "\n".join(f"- {e}" for e in errors)
            )

    return DebugResult(
        classification=classification,
        fixed_code=fixed_code,
        diagnostic_report=report,
        attempts=attempts,
        validation_passed=validation_passed,
    )


def run_and_debug(
    code: str,
    config: AgentConfig | None = None,
    timeout: int = 120,
    cwd: str | None = None,
    save_fixed: str | None = None,
    progress_callback: callable | None = None,
) -> DebugResult:
    """Run a script, and if it fails, automatically diagnose and fix.

    This implements the full run → diagnose → fix → re-run loop.

    Parameters
    ----------
    code : str
        Python source code to run.
    config : AgentConfig, optional
        Agent configuration.
    timeout : int
        Wall-clock timeout per run (seconds).
    cwd : str, optional
        Working directory for script execution.
    save_fixed : str, optional
        If set, write the final fixed code to this path.
    progress_callback : callable, optional
        Called with a status message string at each step for live output.

    Returns
    -------
    DebugResult
        Final result (may include the successfully-fixed code).
    """
    _progress_callback = progress_callback
    from optiprofiler_agent.agent_b.local_runner import run_script

    config = config or AgentConfig()
    max_rounds = config.max_debug_retries
    current_code = code
    all_reports: list[str] = []

    def _log(msg: str):
        if _progress_callback:
            _progress_callback(msg)

    for round_num in range(1, max_rounds + 1):
        _log(f"[Round {round_num}/{max_rounds}] Running script (timeout={timeout}s)...")
        run_result = run_script(current_code, timeout=timeout, cwd=cwd)

        if run_result.success:
            _log(f"[Round {round_num}] Script ran successfully!")
            summary = f"## Script ran successfully (round {round_num})\n\n"
            if round_num > 1:
                summary += "The fix was applied and the script now runs without errors.\n"
            else:
                summary += "No errors detected.\n"
            if run_result.stdout.strip():
                summary += f"\n**Output (last 500 chars):**\n```\n{run_result.stdout[-500:]}\n```\n"
            all_reports.append(summary)
            if save_fixed and round_num > 1:
                Path(save_fixed).write_text(current_code, encoding="utf-8")
                _log(f"Fixed code saved to {save_fixed}")
            return DebugResult(
                classification=ErrorClassification(
                    error_type="none", confidence=1.0,
                    details="Script ran successfully.",
                ),
                fixed_code=current_code if round_num > 1 else None,
                diagnostic_report="\n\n---\n\n".join(all_reports),
                attempts=round_num,
                validation_passed=True,
            )

        error_text = run_result.traceback or run_result.stderr
        error_preview = error_text.splitlines()[-1][:120] if error_text.strip() else "unknown"
        if run_result.timed_out:
            _log(f"[Round {round_num}] Timed out after {timeout}s.")
        else:
            _log(f"[Round {round_num}] Error: {error_preview}")

        all_reports.append(
            f"## Round {round_num}: Error detected\n\n"
            f"```\n{error_text[:1000]}\n```\n"
        )

        if run_result.timed_out and round_num > 1:
            all_reports.append(
                "## Timeout on re-run — likely not a code bug\n\n"
                "The previous code fix was applied, but the script timed out "
                "during execution. This typically means the benchmark takes "
                "longer than the configured timeout, not that the code is wrong.\n\n"
                "**Suggestion:** Re-run with a longer timeout:\n"
                "```bash\noptiprofiler-agent debug script.py --run --timeout 600\n```\n"
            )
            if save_fixed:
                Path(save_fixed).write_text(current_code, encoding="utf-8")
                _log(f"Fixed code saved to {save_fixed}")
                all_reports.append(f"\nFixed code saved to `{save_fixed}`.\n")
            return DebugResult(
                classification=ErrorClassification(
                    error_type="timeout", confidence=0.9,
                    details="Code was fixed but benchmark needs more time to run.",
                ),
                fixed_code=current_code,
                diagnostic_report="\n\n---\n\n".join(all_reports),
                attempts=round_num,
                validation_passed=True,
            )

        _log(f"[Round {round_num}] Diagnosing error...")
        result = debug_script(current_code, error_text, config)
        all_reports.append(result.diagnostic_report)

        if result.fixed_code and result.validation_passed:
            current_code = result.fixed_code
            _log(f"[Round {round_num}] Fix generated and validated. Retrying...")
        else:
            if run_result.timed_out:
                _log(f"[Round {round_num}] Timeout — script may need more time.")
                all_reports.append(
                    "\n**Timeout detected.** The script may need more time. "
                    "Try `--timeout 600` or higher.\n"
                )
            else:
                _log(f"[Round {round_num}] Could not produce a valid fix. Stopping.")
                all_reports.append(
                    f"\n**Could not produce a valid fix in round {round_num}. Stopping.**\n"
                )
            result.diagnostic_report = "\n\n---\n\n".join(all_reports)
            if save_fixed and current_code != code:
                Path(save_fixed).write_text(current_code, encoding="utf-8")
                _log(f"Partially fixed code saved to {save_fixed}")
            return result

    _log(f"[Final] Running verification...")
    final_run = run_script(current_code, timeout=timeout, cwd=cwd)
    if final_run.success:
        _log("[Final] Verification passed!")
        all_reports.append(
            "## Final verification: Success\n\n"
            "The fixed script ran without errors.\n"
        )
        if save_fixed:
            Path(save_fixed).write_text(current_code, encoding="utf-8")
            _log(f"Fixed code saved to {save_fixed}")
            all_reports.append(f"\nFixed code saved to `{save_fixed}`.\n")

        return DebugResult(
            classification=ErrorClassification(
                error_type="fixed", confidence=1.0,
                details="Script fixed and verified.",
            ),
            fixed_code=current_code,
            diagnostic_report="\n\n---\n\n".join(all_reports),
            attempts=max_rounds,
            validation_passed=True,
        )
    else:
        _log("[Final] Still failing after all attempts.")
        all_reports.append(
            "## Final verification: Still failing\n\n"
            f"```\n{(final_run.traceback or final_run.stderr)[:500]}\n```\n"
        )
        return DebugResult(
            classification=ErrorClassification(
                error_type="runtime_error", confidence=0.8,
                details="Could not fully fix the script.",
            ),
            fixed_code=current_code,
            diagnostic_report="\n\n---\n\n".join(all_reports),
            attempts=max_rounds,
            validation_passed=False,
        )
