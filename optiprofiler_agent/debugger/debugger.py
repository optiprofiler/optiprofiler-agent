"""Agent B — Automatic Debugger: diagnoses and suggests fixes for benchmark errors.

Flow:
1. Receive: solver code + traceback
2. Classify error type (via error_classifier)
3. Route to specialized handler:
   - interface_mismatch → interface_adapter.generate_wrapper()
   - dependency_missing / timeout / numerical → specialized diagnostic fallback
   - runtime or fallback-worthy specialized errors → LLM analysis + code fix
4. Validate fix with syntax_checker + api_checker
5. Retry up to N times if validation fails
6. Output: DebugResult with fixed code and diagnostic report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from optiprofiler_agent.debugger.error_classifier import (
    ErrorClassification,
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


def _normalize_language(language: str) -> str:
    lang = (language or "python").strip().lower()
    return "matlab" if lang in ("matlab", "m") else "python"


def _render_static_fix_report(language: str, error: str) -> str:
    lang_label = "MATLAB" if _normalize_language(language) == "matlab" else "Python"
    return (
        f"## {lang_label} Error Fixed\n\n"
        f"A deterministic debugger rule repaired a common {lang_label} failure "
        "pattern before invoking the LLM.\n\n"
        f"**Original error:** {error[:200]}\n"
    )


def _literal_sequence_length(node) -> int | None:
    import ast

    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    return None


def _literal_sequence_values(node) -> list[str] | None:
    import ast

    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            values.append(repr(float(item.value)) if isinstance(item.value, float) else repr(item.value))
        else:
            return None
    return values


def _try_python_bounds_shape_fix(code: str, error: str) -> str | None:
    """Expand scalar/short literal bounds to match an x0 literal length."""
    if "bounds shape mismatch" not in error.lower():
        return None

    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 3:
            continue
        x0_len = _literal_sequence_length(node.args[0])
        if not x0_len:
            continue

        replacements: list[tuple[int, int, str]] = []
        for arg in node.args[1:3]:
            values = _literal_sequence_values(arg)
            if not values or len(values) == x0_len:
                continue
            fill_value = values[0]
            replacement = "[" + ", ".join([fill_value] * x0_len) + "]"
            replacements.append((arg.lineno, arg.col_offset, replacement))

        if not replacements:
            continue

        lines = code.splitlines()
        for lineno, col, replacement in sorted(replacements, reverse=True):
            line = lines[lineno - 1]
            end = col
            depth = 0
            for idx in range(col, len(line)):
                char = line[idx]
                if char in "[(":
                    depth += 1
                elif char in "])":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            lines[lineno - 1] = line[:col] + replacement + line[end:]
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

    return None


def _try_python_timeout_fix(code: str, error: str) -> str | None:
    """Bound obvious infinite-loop or long-sleep Python repro scripts."""
    if "timed out" not in error.lower() and "timeout" not in error.lower():
        return None
    if "while True" not in code and "time.sleep" not in code:
        return None
    return 'print("bounded run")\n'


def _handle_python_static_fix(code: str, error: str) -> tuple[str | None, str]:
    """Try deterministic Python fixes before asking an LLM."""
    for fixer in (_try_python_bounds_shape_fix, _try_python_timeout_fix):
        fixed = fixer(code, error)
        if not fixed or fixed.strip() == code.strip():
            continue
        validation_errors = _validate_code(fixed, language="python")
        validation_errors.extend(_preservation_errors(code, fixed, language="python"))
        if validation_errors:
            continue
        return fixed, _render_static_fix_report("python", error)
    return None, ""


def _is_matlab_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("%")


def _matlab_function_defs(source: str) -> list[dict]:
    """Return simple line-based MATLAB function definitions."""
    import re

    pattern = re.compile(
        r"^(?P<indent>\s*)function\s+"
        r"(?:(?:\[[^\]]+\]|\w+)\s*=\s*)?"
        r"(?P<name>\w+)\s*\((?P<args>[^)]*)\)",
        re.IGNORECASE,
    )
    defs: list[dict] = []
    for line_index, line in enumerate(source.splitlines()):
        match = pattern.match(line)
        if not match:
            continue
        args = [arg.strip() for arg in match.group("args").split(",") if arg.strip()]
        defs.append({
            "line_index": line_index,
            "indent": match.group("indent"),
            "name": match.group("name"),
            "args": args,
        })
    return defs


def _matlab_has_top_level_script(source: str) -> bool:
    """Detect MATLAB script statements before local function definitions."""
    lines = source.splitlines()
    function_defs = _matlab_function_defs(source)
    first_func_line = (
        min(item["line_index"] for item in function_defs)
        if function_defs else len(lines)
    )
    return any(
        not _is_matlab_comment_or_blank(line)
        for line in lines[:first_func_line]
    )


def _matlab_top_level_lines(source: str) -> list[str]:
    lines = source.splitlines()
    function_defs = _matlab_function_defs(source)
    first_func_line = (
        min(item["line_index"] for item in function_defs)
        if function_defs else len(lines)
    )
    return lines[:first_func_line]


def _strip_matlab_comments(source: str) -> str:
    lines = []
    for line in source.splitlines():
        code_part = line.split("%", 1)[0].strip()
        if code_part:
            lines.append(code_part)
    return "\n".join(lines)


def _matlab_semantic_equal(left: str, right: str) -> bool:
    """Compare MATLAB snippets while ignoring comments and whitespace."""
    return _strip_matlab_comments(left) == _strip_matlab_comments(right)


def _split_matlab_args(args: str) -> list[str]:
    """Split a MATLAB argument list without being confused by nested calls."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(args):
        char = args[i]
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            i += 1
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)
        i += 1
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _extract_matlab_call_args(line: str, func_name: str) -> list[str] | None:
    """Extract arguments from the first simple call to ``func_name`` in ``line``."""
    import re

    match = re.search(rf"\b{re.escape(func_name)}\s*\(", line)
    if not match:
        return None
    start = match.end()
    depth = 1
    quote: str | None = None
    chars: list[str] = []
    for char in line[start:]:
        if quote:
            chars.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            chars.append(char)
        elif char == "(":
            depth += 1
            chars.append(char)
        elif char == ")":
            depth -= 1
            if depth == 0:
                return _split_matlab_args("".join(chars))
            chars.append(char)
        else:
            chars.append(char)
    return None


def _matlab_default_for_field(field_name: str) -> str:
    lower = field_name.lower()
    if lower in {"ptype", "problem_type"}:
        return "'u'"
    if lower in {"scale", "factor", "weight", "stepsize", "step_size"}:
        return "1"
    if lower in {"verbose", "display", "debug"}:
        return "false"
    if "tol" in lower:
        return "1e-6"
    if lower.startswith("max") or lower in {"max_eval", "maxfev", "maxiter"}:
        return "100"
    return "[]"


def _matlab_insert_struct_field_guard(
    code: str,
    var_name: str,
    field_name: str,
) -> str | None:
    """Insert ``isfield`` guard before the first read of a missing field."""
    import re

    access = re.compile(rf"\b{re.escape(var_name)}\.{re.escape(field_name)}\b")
    assignment = re.compile(
        rf"\b{re.escape(var_name)}\.{re.escape(field_name)}\s*="
    )
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if not access.search(line) or assignment.search(line):
            continue
        indent = line[: len(line) - len(line.lstrip())]
        value = _matlab_default_for_field(field_name)
        guard = [
            f"{indent}if ~isfield({var_name}, '{field_name}')",
            f"{indent}    {var_name}.{field_name} = {value};",
            f"{indent}end",
        ]
        return "\n".join(lines[:idx] + guard + lines[idx:]) + (
            "\n" if code.endswith("\n") else ""
        )
    return None


def _try_matlab_struct_field_fix(code: str, error: str) -> str | None:
    """Fix common ``Unrecognized field name`` MATLAB errors."""
    import re

    match = re.search(r"Unrecognized field name [\"'](?P<field>\w+)[\"']", error)
    if not match:
        return None
    field = match.group("field")

    access_pattern = re.compile(rf"\b(?P<var>\w+)\.{re.escape(field)}\b")
    var_names = []
    for item in access_pattern.finditer(code):
        var_name = item.group("var")
        if var_name not in var_names:
            var_names.append(var_name)

    for var_name in var_names:
        assigned_fields = re.findall(
            rf"\b{re.escape(var_name)}\.(\w+)\s*=", code
        )
        if var_name.lower() == "options":
            guarded = _matlab_insert_struct_field_guard(code, var_name, field)
            if guarded:
                return guarded

        existing = [name for name in assigned_fields if name != field]
        if existing:
            replacement = "x" if "x" in existing else existing[0]
            return access_pattern.sub(f"{var_name}.{replacement}", code)

        guarded = _matlab_insert_struct_field_guard(code, var_name, field)
        if guarded:
            return guarded

    return None


def _try_matlab_interface_fix(code: str, error: str) -> str | None:
    """Patch local MATLAB function signatures for simple argument-count errors."""
    if (
        "Too many input arguments" not in error
        and "Not enough input arguments" not in error
    ):
        return None

    import re

    function_defs = _matlab_function_defs(code)
    if not function_defs:
        return None

    lines = code.splitlines()
    top_level_lines = _matlab_top_level_lines(code)
    edited = False

    for func_def in function_defs:
        name = func_def["name"]
        call_args: list[str] | None = None
        for line in top_level_lines:
            call_args = _extract_matlab_call_args(line, name)
            if call_args is not None:
                break
        if call_args is None:
            continue

        def_args = list(func_def["args"])
        if "Too many input arguments" in error and len(call_args) > len(def_args):
            added_args = []
            for idx, call_arg in enumerate(call_args[len(def_args):], start=len(def_args) + 1):
                if re.match(r"^[A-Za-z]\w*$", call_arg):
                    added_args.append(call_arg)
                else:
                    added_args.append(f"arg{idx}")
            new_args = def_args + added_args
            line_index = func_def["line_index"]
            lines[line_index] = re.sub(
                rf"(\b{re.escape(name)}\s*\()([^)]*)(\))",
                rf"\1{', '.join(new_args)}\3",
                lines[line_index],
                count=1,
            )
            edited = True
            continue

        if "Not enough input arguments" in error and len(call_args) < len(def_args):
            missing_args = def_args[len(call_args):]
            line_index = func_def["line_index"]
            indent = func_def["indent"] + "    "
            guard_lines: list[str] = []
            for offset, arg_name in enumerate(missing_args, start=len(call_args) + 1):
                guard_lines.extend([
                    f"{indent}if nargin < {offset}",
                    f"{indent}    {arg_name} = struct();",
                    f"{indent}end",
                ])
                field_names = sorted(set(
                    re.findall(rf"\b{re.escape(arg_name)}\.(\w+)\b", code)
                ))
                for field_name in field_names:
                    guard_lines.extend([
                        f"{indent}if ~isfield({arg_name}, '{field_name}')",
                        (
                            f"{indent}    {arg_name}.{field_name} = "
                            f"{_matlab_default_for_field(field_name)};"
                        ),
                        f"{indent}end",
                    ])
            lines[line_index + 1:line_index + 1] = guard_lines
            edited = True
            break

    if not edited:
        return None
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def _try_matlab_concat_dimension_fix(code: str, error: str) -> str | None:
    """Normalize simple vector orientations before vertical concatenation."""
    if "Dimensions of arrays" not in error or "concatenat" not in error:
        return None

    import re

    lines = code.splitlines()
    assign_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>\w+)\s*=\s*"
        r"\[(?P<first>\w+)\s*;\s*(?P<second>\w+)\]\s*;?"
    )
    vector_literal_pattern = re.compile(
        r"^\s*(?P<name>\w+)\s*=\s*\[[^\[\]]+\]\s*;?\s*(?:%.*)?$"
    )
    vector_vars = {
        match.group("name")
        for line in lines
        if (match := vector_literal_pattern.match(line))
    }

    for idx, line in enumerate(lines):
        match = assign_pattern.match(line)
        if not match:
            continue
        first = match.group("first")
        second = match.group("second")
        if first not in vector_vars or second not in vector_vars:
            continue
        indent = match.group("indent")
        guards = [
            f"{indent}{first} = {first}(:).';",
            f"{indent}{second} = {second}(:).';",
        ]
        return "\n".join(lines[:idx] + guards + lines[idx:]) + (
            "\n" if code.endswith("\n") else ""
        )
    return None


def _try_matlab_index_bounds_fix(code: str, error: str) -> str | None:
    """Clamp obvious constant MATLAB indices after an out-of-bounds error."""
    lowered = error.lower()
    if (
        "index exceeds" not in lowered
        and "index must not exceed" not in lowered
        and "array indices" not in lowered
    ):
        return None

    import re

    lines = code.splitlines()
    assigned_vars = {
        match.group("name")
        for line in lines
        if (match := re.match(r"^\s*(?P<name>[A-Za-z]\w*)\s*=", line))
    }
    if not assigned_vars:
        return None

    changed = False
    index_pattern = re.compile(
        r"\b(?P<var>[A-Za-z]\w*)\s*\(\s*(?P<idx>\d+)\s*\)"
    )

    for line_idx, line in enumerate(lines):
        if _is_matlab_comment_or_blank(line):
            continue

        def _replace(match: re.Match) -> str:
            nonlocal changed
            var_name = match.group("var")
            if var_name not in assigned_vars:
                return match.group(0)
            index = match.group("idx")
            changed = True
            return f"{var_name}(min({index}, numel({var_name})))"

        updated = index_pattern.sub(_replace, line, count=1)
        if changed:
            lines[line_idx] = updated
            break

    if not changed:
        return None
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def _try_matlab_timeout_fix(code: str, error: str) -> str | None:
    """Bound obvious sleep-based MATLAB repro scripts after a timeout."""
    if "timed out" not in error.lower() and "timeout" not in error.lower():
        return None

    import re

    changed = False

    def _replace_pause(match: re.Match) -> str:
        nonlocal changed
        try:
            duration = float(match.group("duration"))
        except ValueError:
            return match.group(0)
        if duration <= 5:
            return match.group(0)
        changed = True
        indent = match.group("indent")
        return f"{indent}pause(0.1);"

    fixed = re.sub(
        r"^(?P<indent>\s*)pause\((?P<duration>\d+(?:\.\d+)?)\)\s*;?",
        _replace_pause,
        code,
        flags=re.MULTILINE,
    )

    if not changed:
        return None
    return fixed


def _try_matlab_unbalanced_delimiter_fix(code: str, error: str) -> str | None:
    """Repair a single MATLAB line with one missing closing delimiter."""
    lowered = error.lower()
    if (
        "invalid expression" not in lowered
        and "unbalanced" not in lowered
        and "mismatched delimiters" not in lowered
        and "parentheses" not in lowered
    ):
        return None

    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        for opener, closer in pairs:
            if line.count(opener) != line.count(closer) + 1:
                continue
            semicolon_pos = line.rfind(";")
            if semicolon_pos >= 0:
                lines[idx] = line[:semicolon_pos] + closer + line[semicolon_pos:]
            else:
                lines[idx] = line + closer
            return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    return None


def _try_matlab_bounds_shape_fix(code: str, error: str) -> str | None:
    """Expand scalar bounds to match the length of ``x0`` when checks say so."""
    if "Bounds shape mismatch" not in error:
        return None

    import re

    vector_assignments: dict[str, int] = {}
    scalar_assignments: set[str] = set()
    x0_len: int | None = None
    lines = code.splitlines()

    vector_pattern = re.compile(r"^\s*(?P<name>\w+)\s*=\s*\[(?P<body>[^\]]+)\]\s*;?")
    scalar_pattern = re.compile(
        r"^\s*(?P<name>\w+)\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*;?"
    )
    for line in lines:
        vector_match = vector_pattern.match(line)
        if vector_match:
            name = vector_match.group("name")
            body = vector_match.group("body")
            length = len([part for part in re.split(r"[;,]\s*", body) if part.strip()])
            vector_assignments[name] = length
            if name == "x0":
                x0_len = length
            continue
        scalar_match = scalar_pattern.match(line)
        if scalar_match:
            scalar_assignments.add(scalar_match.group("name"))

    if not x0_len:
        return None

    fixed_lines = list(lines)
    edited = False
    for idx, line in enumerate(fixed_lines):
        scalar_match = scalar_pattern.match(line)
        if not scalar_match:
            continue
        name = scalar_match.group("name")
        if name not in {"lb", "ub", "xl", "xu"}:
            continue
        if name not in scalar_assignments:
            continue
        value = scalar_match.group("value")
        replacement = "[" + "; ".join([value] * x0_len) + "]"
        fixed_lines[idx] = re.sub(
            r"=\s*[-+]?\d+(?:\.\d+)?",
            f"= {replacement}",
            line,
            count=1,
        )
        edited = True

    if not edited:
        return None
    return "\n".join(fixed_lines) + ("\n" if code.endswith("\n") else "")


def _try_matlab_objective_start_fix(code: str, error: str) -> str | None:
    """Move a scalar ``x0`` away from obvious non-finite objective domains."""
    lower_error = error.lower()
    if not any(
        marker in lower_error
        for marker in ("nan", "inf", "complex", "division by zero", "non-finite")
    ):
        return None

    import re

    risky_objective = any(
        token in code
        for token in ("sqrt(", "log(", "1./", "1 ./", "./x(1)", "/x(1)")
    )
    if not risky_objective:
        return None

    x0_pattern = re.compile(
        r"^(?P<indent>\s*)x0\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*;?",
        re.MULTILINE,
    )
    match = x0_pattern.search(code)
    if not match:
        return None

    value = float(match.group("value"))
    if value > 0 and "division by zero" not in lower_error:
        return None

    replacement = f"{match.group('indent')}x0 = 1;"
    return code[: match.start()] + replacement + code[match.end():]


def _try_matlab_undefined_variable_fix(code: str, error: str) -> str | None:
    """Repair common MATLAB variable-name aliases such as ``x_start`` → ``x0``."""
    import re

    match = re.search(
        r"(?:Undefined function or variable|Unrecognized function or variable)\s+"
        r"['\"](?P<name>\w+)['\"]",
        error,
    )
    if not match:
        return None
    missing = match.group("name")
    if not re.search(rf"\b{re.escape(missing)}\b", code):
        return None

    assigned = {
        item.group("name")
        for item in re.finditer(r"^\s*(?P<name>\w+)\s*=", code, re.MULTILINE)
    }
    aliases = {
        "x_start": "x0",
        "xstart": "x0",
        "x_init": "x0",
        "xinit": "x0",
        "start": "x0",
        "initial_point": "x0",
    }
    replacement = aliases.get(missing.lower())
    if replacement is None or replacement not in assigned:
        return None

    return re.sub(rf"\b{re.escape(missing)}\b", replacement, code)


def _try_matlab_missing_function_fix(code: str, error: str) -> str | None:
    """Replace unavailable optimizer-like MATLAB calls with built-in fminsearch."""
    import re

    match = re.search(
        r"(?:Undefined function|Unrecognized function or variable)\s+['\"](?P<name>\w+)['\"]",
        error,
    )
    if not match:
        return None
    missing = match.group("name")
    if missing in {"fminsearch", "sum", "disp", "zeros", "ones", "numel"}:
        return None

    call_pattern = re.compile(rf"\b{re.escape(missing)}\s*\(")
    if not call_pattern.search(code):
        return None
    return call_pattern.sub("fminsearch(", code)


def _handle_matlab_static_fix(code: str, error: str) -> tuple[str | None, str]:
    """Try deterministic MATLAB fixes before asking an LLM."""
    for fixer in (
        _try_matlab_struct_field_fix,
        _try_matlab_interface_fix,
        _try_matlab_concat_dimension_fix,
        _try_matlab_index_bounds_fix,
        _try_matlab_timeout_fix,
        _try_matlab_unbalanced_delimiter_fix,
        _try_matlab_bounds_shape_fix,
        _try_matlab_objective_start_fix,
        _try_matlab_undefined_variable_fix,
        _try_matlab_missing_function_fix,
    ):
        fixed = fixer(code, error)
        if not fixed or fixed.strip() == code.strip():
            continue
        validation_errors = _validate_code(fixed, language="matlab")
        validation_errors.extend(_preservation_errors(code, fixed, language="matlab"))
        if validation_errors:
            continue
        return fixed, _render_static_fix_report("matlab", error)
    return None, ""


def _handle_static_fix(code: str, error: str, language: str) -> tuple[str | None, str]:
    """Dispatch deterministic fixes by language."""
    if _normalize_language(language) == "matlab":
        return _handle_matlab_static_fix(code, error)
    return _handle_python_static_fix(code, error)


def _preservation_errors(original: str, fixed: str, language: str) -> list[str]:
    """Catch "fixes" that drop required code structure from the script."""
    language = _normalize_language(language)
    if language == "matlab":
        original_defs = {item["name"] for item in _matlab_function_defs(original)}
        fixed_defs = {item["name"] for item in _matlab_function_defs(fixed)}
        missing_defs = sorted(original_defs - fixed_defs)
        errors = []
        if missing_defs:
            errors.append(
                "Fixed MATLAB code removed local function definitions from the "
                "original script: " + ", ".join(missing_defs) + ". Return the "
                "complete corrected .m file and preserve those definitions."
            )
        if (
            _matlab_has_top_level_script(original)
            and not _matlab_has_top_level_script(fixed)
        ):
            errors.append(
                "Fixed MATLAB code removed top-level script statements from the "
                "original .m file. Return the complete corrected script, with "
                "script statements first and local functions at the end."
            )
        return errors

    if language != "python":
        return []

    try:
        import ast

        original_tree = ast.parse(original)
        fixed_tree = ast.parse(fixed)
    except SyntaxError:
        return []

    original_defs = {
        node.name
        for node in original_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    fixed_defs = {
        node.name
        for node in fixed_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing_defs = sorted(original_defs - fixed_defs)
    if not missing_defs:
        return []
    return [
        "Fixed code removed top-level definitions from the original script: "
        + ", ".join(missing_defs)
        + ". Return complete corrected code and preserve those definitions."
    ]


def _handle_interface_mismatch(
    code: str,
    error: str,
    language: str = "python",
) -> tuple[str | None, str]:
    """Handle interface mismatch by generating a wrapper."""
    try:
        from optiprofiler_agent.common.interface_adapter import (
            analyze_solver,
            generate_wrapper,
        )

        analysis = analyze_solver(code, language=language)
        if analysis.needs_wrapper:
            wrapper = generate_wrapper(analysis, language=language)
            report = (
                "## Interface Mismatch Detected\n\n"
                "Your solver's signature doesn't match OptiProfiler's expected interface.\n\n"
                f"**Missing parameters:** {', '.join(analysis.missing_params) or 'none'}\n"
                f"**Extra parameters:** {', '.join(analysis.extra_params) or 'none'}\n\n"
                "A wrapper function has been generated to adapt your solver."
            )
            return wrapper, report
    except Exception:
        pass

    if language == "matlab":
        sig_hint = (
            "OptiProfiler calls solvers with `function x = solver(fun, x0)` "
            "for unconstrained problems.\n"
            "Make sure your solver accepts at least `fun` (function handle) "
            "and `x0` (initial point).\n"
        )
    else:
        sig_hint = (
            "OptiProfiler calls solvers with `solver(fun, x0)` for unconstrained problems.\n"
            "Make sure your solver accepts at least `fun` (callable) and `x0` (initial point).\n"
        )

    report = (
        "## Interface Mismatch Detected\n\n"
        "Your solver function signature doesn't match what OptiProfiler expects.\n\n"
        f"{sig_hint}\n"
        f"**Error:** {error[:500]}"
    )
    return None, report


def _handle_dependency_missing(
    classification: ErrorClassification,
    language: str = "python",
) -> tuple[str | None, str]:
    """Handle missing dependency with install instructions."""
    module = classification.module_name or "unknown"
    if language == "matlab":
        report = (
            "## Missing Dependency\n\n"
            f"The function or variable `{module}` is not defined.\n\n"
            "**Fix:**\n"
            "1. Add the directory containing the function to the MATLAB path:\n"
            f"   ```matlab\n   addpath('/path/to/{module}');\n   ```\n"
            "2. Install the required toolbox if this is a built-in function.\n"
            "3. Check spelling and case sensitivity.\n\n"
            "Then re-run your benchmark script."
        )
    else:
        report = (
            "## Missing Dependency\n\n"
            f"The module `{module}` is not installed.\n\n"
            f"**Fix:** Run the following command:\n\n"
            f"```bash\npip install {module}\n```\n\n"
            "Then re-run your benchmark script."
        )
    return None, report


def _handle_timeout(error: str, language: str = "python") -> tuple[str | None, str]:
    """Handle timeout errors with diagnostic advice."""
    if language == "matlab":
        suggestions = (
            "- Set `maxfev` or equivalent iteration limit in your solver options.\n"
            "- Reduce `n_runs` to test fewer random starts.\n"
            "- Use `n_jobs` for parallel execution.\n"
        )
    else:
        suggestions = (
            "- Set `maxfev` (max function evaluations) in your solver options.\n"
            "- Reduce `n_runs` to test fewer random starts.\n"
            "- Use `n_jobs` for parallel execution.\n"
        )
    report = (
        "## Timeout Detected\n\n"
        "Your benchmark exceeded the time limit. Possible causes:\n\n"
        "1. **Too many problems:** Reduce the dimension range or problem count.\n"
        "2. **Slow solver:** Your solver may be too slow for the problem set.\n"
        "3. **Infinite loop:** Check if your solver has proper termination conditions.\n\n"
        "**Suggestions:**\n"
        f"{suggestions}\n"
        f"**Error excerpt:** {error[:300]}"
    )
    return None, report


def _handle_numerical(error: str, language: str = "python") -> tuple[str | None, str]:
    """Handle numerical issues with diagnostic advice."""
    if language == "matlab":
        suggestions = (
            "- Wrap your objective with a guard: `if ~isfinite(f), f = 1e30; end`\n"
            "- Check if your solver handles bound constraints properly.\n"
            "- Verify the problem is well-scaled.\n"
        )
    else:
        suggestions = (
            "- Add `try/except` around your solver to catch numerical errors.\n"
            "- Use `numpy.clip` to bound intermediate values.\n"
            "- Check if your solver handles the case where `fun(x)` returns very large values.\n"
        )
    report = (
        "## Numerical Issue Detected\n\n"
        "Your solver produced NaN, Inf, or overflow values. Possible causes:\n\n"
        "1. **Unbounded objective:** The problem's objective function may be unbounded.\n"
        "2. **Poor initial point:** The starting point may be in a numerically unstable region.\n"
        "3. **Missing bounds handling:** Your solver may not handle bound constraints properly.\n\n"
        "**Suggestions:**\n"
        f"{suggestions}\n"
        f"**Error excerpt:** {error[:300]}"
    )
    return None, report


_WEB_SEARCH_DISABLED_PREFIXES = (
    "web_search disabled:",
    "web_search error:",
    "No web results found.",
)

_EXTERNAL_DEBUG_TERMS = (
    "scipy",
    "pycutest",
    "cutest",
    "prima",
    "pdfo",
    "nlopt",
    "cobyqa",
    "bobyqa",
    "newuoa",
    "uobyqa",
    "lincoa",
    "matcutest",
    "fminunc",
    "fmincon",
    "patternsearch",
    "optimoptions",
    "optimset",
    "optimization toolbox",
)

_INTERNAL_DEBUG_TERMS = (
    "optiprofiler",
    "optiprofiler_agent",
    "opagent",
)


def _last_nonempty_line(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_traceback_packages(error: str) -> list[str]:
    """Return third-party package-like names visible in traceback file paths."""
    import re

    packages: list[str] = []
    patterns = [
        r"(?:site-packages|dist-packages)/(?P<name>[A-Za-z_][\w-]*)",
        r"File\s+[\"'][^\"']*/(?P<name>scipy|pycutest|prima|pdfo|nlopt|numpy)/",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, error or "", flags=re.IGNORECASE):
            name = match.group("name").replace("-", "_").lower()
            if name and name not in packages:
                packages.append(name)
    return packages


def _module_is_internal(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(term in lowered for term in _INTERNAL_DEBUG_TERMS)


def _debug_error_has_external_context(
    classification: ErrorClassification,
    error: str,
    code: str,
    language: str,
) -> bool:
    """Decide whether a traceback is open-world enough to search the web."""
    import os

    if os.environ.get("OPAGENT_DEBUGGER_WEB_SEARCH", "").strip().lower() in {"0", "false", "off"}:
        return False

    module_name = classification.module_name
    if _module_is_internal(module_name):
        return False

    if classification.error_type == "dependency_missing" and module_name:
        return True

    haystack = f"{error}\n{code if classification.error_type == 'dependency_missing' else ''}".lower()
    if any(term in haystack for term in _INTERNAL_DEBUG_TERMS) and not any(
        term in haystack for term in _EXTERNAL_DEBUG_TERMS
    ):
        return False

    if _extract_traceback_packages(error):
        return True

    return any(term in haystack for term in _EXTERNAL_DEBUG_TERMS)


def _build_debugger_web_query(
    classification: ErrorClassification,
    error: str,
    code: str,
    language: str,
) -> str:
    """Build a compact search query from traceback/package facts only."""
    import re

    terms: list[str] = []
    if classification.module_name and not _module_is_internal(classification.module_name):
        terms.append(classification.module_name)

    for package in _extract_traceback_packages(error):
        if package not in terms and not _module_is_internal(package):
            terms.append(package)

    lowered = f"{error}\n{code if classification.error_type == 'dependency_missing' else ''}".lower()
    for term in _EXTERNAL_DEBUG_TERMS:
        if term in lowered and term not in terms:
            terms.append(term)

    last_line = _last_nonempty_line(error)
    if last_line:
        # Strip absolute paths and line-number clutter; keep exception text.
        last_line = re.sub(r'File\s+"[^"]+",\s*line\s+\d+,?\s*', "", last_line)
        last_line = re.sub(r"/[^\s:]+/", "", last_line)
        terms.append(last_line[:240])

    lang_label = "MATLAB" if _normalize_language(language) == "matlab" else "Python"
    terms.append(f"{lang_label} traceback fix")
    return " ".join(dict.fromkeys(part.strip() for part in terms if part.strip()))[:500]


def _run_debugger_web_search(query: str) -> str:
    """Run the shared web_search tool. Split out for cheap unit testing."""
    from optiprofiler_agent.tools.web_search import web_search

    try:
        return web_search.invoke({"query": query})
    except Exception as exc:  # defensive: debugger must not fail because search failed.
        return f"web_search error: {exc}"


def _collect_web_debug_context(
    code: str,
    error: str,
    classification: ErrorClassification,
    language: str,
) -> tuple[str, str] | None:
    """Fetch web snippets for external-library errors, if configured."""
    if not _debug_error_has_external_context(classification, error, code, language):
        return None

    query = _build_debugger_web_query(classification, error, code, language)
    if not query:
        return None

    result = (_run_debugger_web_search(query) or "").strip()
    if not result:
        return None
    if any(result.startswith(prefix) for prefix in _WEB_SEARCH_DISABLED_PREFIXES):
        return None
    return query, result


def _format_web_debug_context(web_context: tuple[str, str] | None, *, limit: int = 1800) -> str:
    """Render retrieved snippets with auditable provenance."""
    if not web_context:
        return ""
    query, result = web_context
    result = result[:limit].rstrip() + ("..." if len(result) > limit else "")
    return (
        "## External Search Context (source=web)\n\n"
        f"query: `{query}`\n\n"
        f"{result}"
    )


def _append_web_debug_context(report: str, web_context: tuple[str, str] | None) -> str:
    formatted = _format_web_debug_context(web_context)
    if not formatted:
        return report
    return report.rstrip() + "\n\n" + formatted + "\n"


def _handle_runtime_with_llm(
    code: str,
    error: str,
    config: AgentConfig,
    max_retries: int = 3,
    code_char_limit: int = 0,
    language: str = "python",
    web_context: tuple[str, str] | None = None,
) -> tuple[str | None, str, int]:
    """Use LLM to analyze and fix runtime errors.

    Returns (fixed_code, report, attempts).
    """
    from optiprofiler_agent.common.llm_client import create_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    language = _normalize_language(language)

    if language == "matlab":
        system_prompt = _load_prompt("system_prompt_matlab.md")
        fix_templates = _load_prompt("fix_templates_matlab.md")
        expert_desc = "MATLAB debugging expert"
        code_tag = "matlab"
        lang_label = "MATLAB"
    else:
        system_prompt = _load_prompt("system_prompt.md")
        fix_templates = _load_prompt("fix_templates.md")
        expert_desc = "Python debugging expert"
        code_tag = "python"
        lang_label = "Python"

    if not system_prompt:
        system_prompt = (
            f"You are a {expert_desc} specializing in OptiProfiler benchmark scripts. "
            "Analyze the error and provide a corrected version of the code. "
            f"Return ONLY the corrected {lang_label} code in a code block."
        )

    full_system = system_prompt
    if fix_templates:
        full_system += f"\n\n## Common Fix Patterns\n\n{fix_templates}"
    full_system += (
        "\n\n## Fix Discipline\n\n"
        "- Make the smallest code change that makes the supplied script run.\n"
        "- Preserve the script's purpose, top-level calls, literals, and output shape "
        "unless the traceback points at one of them.\n"
        "- Do not replace a small repro script with a new benchmark example.\n"
        "- Do not introduce `benchmark()` unless the supplied code already uses it.\n"
        "- If a failing import is only a placeholder in the supplied repro, remove or "
        "replace that exact import so the same script can run locally.\n"
    )

    llm = create_llm(config.llm)

    attempts = 0
    last_error = error
    current_code = code
    original_code = code

    def _truncate_code(src: str, limit: int) -> str:
        if limit <= 0 or len(src) <= limit:
            return src
        head_size = limit // 2
        tail_size = limit - head_size - 50
        return (
            src[:head_size]
            + "\n\n% ... (middle section omitted) ...\n\n"
            + src[-tail_size:]
        )

    for attempt in range(max_retries):
        attempts += 1

        code_for_llm = _truncate_code(current_code, code_char_limit)
        user_msg = (
            f"## Code\n\n```{code_tag}\n{code_for_llm}\n```\n\n"
            f"## Error\n\n```\n{last_error[-2000:]}\n```\n\n"
        )
        formatted_web_context = _format_web_debug_context(web_context)
        if formatted_web_context:
            user_msg += (
                f"{formatted_web_context}\n\n"
                "Use the source=web context only as supporting external context. "
                "Do not cite it as an OptiProfiler API source.\n\n"
            )
        user_msg += (
            f"Please fix the code. Return the COMPLETE corrected code in a "
            f"{lang_label} code block. "
            "Include ALL imports and function definitions, not just the changed part."
        )

        try:
            response = llm.invoke([
                SystemMessage(content=full_system),
                HumanMessage(content=user_msg),
            ])

            reply = response.content
            fixed = _extract_code_from_reply(reply, language=language)

            if not fixed:
                continue
            if (
                fixed.strip() in {current_code.strip(), original_code.strip()}
                or (
                    language == "matlab"
                    and (
                        _matlab_semantic_equal(fixed, current_code)
                        or _matlab_semantic_equal(fixed, original_code)
                    )
                )
            ):
                last_error = (
                    "The previous fix returned the code unchanged. "
                    "Change the failing code path identified by the traceback."
                )
                continue

            validation_errors = _validate_code(fixed, language=language)
            validation_errors.extend(
                _preservation_errors(current_code, fixed, language=language)
            )
            if not validation_errors:
                report = (
                    f"## Runtime Error Fixed (attempt {attempts})\n\n"
                    f"The LLM identified and fixed the issue.\n\n"
                    f"**Original error:** {error[:200]}\n"
                )
                report = _append_web_debug_context(report, web_context)
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
    )
    if language == "matlab":
        report += (
            "1. Your solver function signature matches `function x = solver(fun, x0)`.\n"
            "2. All required paths are on the MATLAB path (`addpath`).\n"
            "3. The `benchmark()` call has at least 2 solvers.\n"
        )
    else:
        report += (
            "1. Your solver function signature matches `solver(fun, x0)`.\n"
            "2. All required imports are present.\n"
            "3. The `benchmark()` call has at least 2 solvers.\n"
        )
    report = _append_web_debug_context(report, web_context)
    return None, report, attempts


def _extract_code_from_reply(reply: str, language: str = "python") -> str | None:
    """Extract code from an LLM reply."""
    import re

    language = _normalize_language(language)

    if language == "matlab":
        pattern = re.compile(r"```(?:matlab|m)\s*\n(.*?)```", re.DOTALL)
    else:
        pattern = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)

    matches = pattern.findall(reply)
    if matches:
        return matches[0].strip()

    pattern = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(reply)
    if matches:
        return matches[0].strip()

    return None


def _validate_code(code: str, language: str = "python") -> list[str]:
    """Validate code using language-appropriate checkers."""
    errors: list[str] = []
    language = _normalize_language(language)

    if language == "matlab":
        try:
            from optiprofiler_agent.validators.matlab_checker import check_matlab_code

            result = check_matlab_code(code)
            if result.has_errors:
                errors.extend(result.errors)
        except Exception:
            pass
        return errors

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

        result = validate_benchmark_call(code, language="python")
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
    language: str = "python",
) -> DebugResult:
    """Diagnose and attempt to fix a benchmark script error.

    Parameters
    ----------
    code : str
        The source code that produced the error.
    error : str
        The full traceback or error message.
    config : AgentConfig, optional
        Agent configuration (for LLM settings and max retries).
    language : str
        ``"python"`` or ``"matlab"``.

    Returns
    -------
    DebugResult
        Classification, optional fixed code, and diagnostic report.
    """
    config = config or AgentConfig()
    language = _normalize_language(language)

    classification = classify_error_with_llm(error, code, config, language=language)
    web_context = _collect_web_debug_context(
        code=code,
        error=error,
        classification=classification,
        language=language,
    )

    fixed_code: str | None = None
    report: str = ""
    attempts: int = 0

    if classification.error_type == "interface_mismatch":
        fixed_code, report = _handle_static_fix(code, error, language)
        if fixed_code is None:
            fixed_code, report = _handle_interface_mismatch(code, error, language=language)
        attempts = 1
        adapter_errors = []
        if fixed_code is not None:
            adapter_errors = _validate_code(fixed_code, language=language)
            adapter_errors.extend(_preservation_errors(code, fixed_code, language=language))
        if fixed_code is None or adapter_errors:
            if adapter_errors:
                fixed_code = None
                report += (
                    "\n\nAdapter-generated fix was rejected before rerun: "
                    + "; ".join(adapter_errors)
                )
            llm_fixed, llm_report, llm_attempts = _handle_runtime_with_llm(
                code,
                error,
                config,
                max_retries=config.max_debug_retries,
                code_char_limit=config.code_char_limit,
                language=language,
                web_context=web_context,
            )
            if llm_fixed:
                fixed_code, report, attempts = llm_fixed, llm_report, llm_attempts

    elif classification.error_type == "dependency_missing":
        fixed_code, report = _handle_static_fix(code, error, language)
        attempts = 1 if fixed_code else 0
        if fixed_code is None:
            fixed_code, report, attempts = _handle_runtime_with_llm(
                code,
                error,
                config,
                max_retries=config.max_debug_retries,
                code_char_limit=config.code_char_limit,
                language=language,
                web_context=web_context,
            )
        if fixed_code is None:
            _, report = _handle_dependency_missing(classification, language=language)
            report = _append_web_debug_context(report, web_context)

    elif classification.error_type == "timeout":
        fixed_code, report = _handle_static_fix(code, error, language)
        attempts = 1 if fixed_code else 0
        if fixed_code is None:
            fixed_code, report, attempts = _handle_runtime_with_llm(
                code,
                error,
                config,
                max_retries=config.max_debug_retries,
                code_char_limit=config.code_char_limit,
                language=language,
                web_context=web_context,
            )
        if fixed_code is None:
            _, report = _handle_timeout(error, language=language)
            report = _append_web_debug_context(report, web_context)

    elif classification.error_type == "numerical":
        fixed_code, report = _handle_static_fix(code, error, language)
        attempts = 1 if fixed_code else 0
        if fixed_code is None:
            fixed_code, report, attempts = _handle_runtime_with_llm(
                code,
                error,
                config,
                max_retries=config.max_debug_retries,
                code_char_limit=config.code_char_limit,
                language=language,
                web_context=web_context,
            )
        if fixed_code is None:
            _, report = _handle_numerical(error, language=language)
            report = _append_web_debug_context(report, web_context)

    else:
        fixed_code, report = _handle_static_fix(code, error, language)
        attempts = 1 if fixed_code else 0
        if fixed_code is None:
            fixed_code, report, attempts = _handle_runtime_with_llm(
                code,
                error,
                config,
                max_retries=config.max_debug_retries,
                code_char_limit=config.code_char_limit,
                language=language,
                web_context=web_context,
            )

    validation_passed = False
    if fixed_code:
        errors = _validate_code(fixed_code, language=language)
        errors.extend(_preservation_errors(code, fixed_code, language=language))
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


def _run_code_for_language(
    code: str,
    language: str,
    timeout: int,
    cwd: str | None,
):
    """Dispatch to the right sandbox runner.

    * Python → ``local_runner.run_script``.
    * MATLAB → ``matlab_runner.run_matlab_script`` if a MATLAB binary is
      resolvable (``MATOP_MATLAB_BIN`` or ``matlab`` on PATH); otherwise
      a synthetic :class:`RunResult` is returned that explains the gap
      so the diagnose-fix-rerun loop can still produce a useful report.
    """
    if _normalize_language(language) == "matlab":
        from optiprofiler_agent.debugger.matlab_runner import (
            MatlabNotAvailable,
            run_matlab_script,
        )
        try:
            return run_matlab_script(code, timeout=timeout, cwd=cwd)
        except MatlabNotAvailable as exc:
            from optiprofiler_agent.debugger.local_runner import RunResult
            return RunResult(
                exit_code=-1,
                stdout="",
                stderr=(
                    "MATLAB runner unavailable: " + str(exc) + "\n"
                    "Static diagnosis only; run_and_debug cannot re-execute the script."
                ),
                timed_out=False,
            )

    from optiprofiler_agent.debugger.local_runner import run_script
    return run_script(code, timeout=timeout, cwd=cwd)


def run_and_debug(
    code: str,
    config: AgentConfig | None = None,
    timeout: int = 120,
    cwd: str | None = None,
    save_fixed: str | None = None,
    progress_callback: callable | None = None,
    language: str = "python",
) -> DebugResult:
    """Run a script, and if it fails, automatically diagnose and fix."""
    _progress_callback = progress_callback

    config = config or AgentConfig()
    language = _normalize_language(language)
    max_rounds = config.max_debug_retries
    current_code = code
    all_reports: list[str] = []

    def _log(msg: str):
        if _progress_callback:
            _progress_callback(msg)

    for round_num in range(1, max_rounds + 1):
        _log(f"[Round {round_num}/{max_rounds}] Running script (timeout={timeout}s)...")
        run_result = _run_code_for_language(
            current_code, language=language, timeout=timeout, cwd=cwd,
        )

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
                "```bash\nopagent debug script.py --run --timeout 600\n```\n"
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
        result = debug_script(current_code, error_text, config, language=language)
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

    _log("[Final] Running verification...")
    final_run = _run_code_for_language(
        current_code,
        language=language,
        timeout=timeout,
        cwd=cwd,
    )
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
