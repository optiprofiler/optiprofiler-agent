#!/usr/bin/env python
"""Prose & terminology linter for OptiProfiler documentation.

Checks Markdown files in the knowledge base, prompts, and docs for:
1. Domain-specific terminology consistency (e.g. "bound-constrained" not "bound constrained")
2. Common typos via codespell (if installed)
3. Style issues (inconsistent capitalization, missing hyphens, etc.)

Usage::

    python scripts/check_prose.py                     # check all .md files
    python scripts/check_prose.py --fix               # auto-fix what we can
    python scripts/check_prose.py path/to/file.md     # check specific file
    python scripts/check_prose.py --codespell         # also run codespell
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCAN_DIRS = [
    REPO_ROOT / "optiprofiler_agent" / "knowledge",
    REPO_ROOT / "optiprofiler_agent" / "agent_a" / "prompts",
    REPO_ROOT / "optiprofiler_agent" / "agent_b" / "prompts",
    REPO_ROOT / "optiprofiler_agent" / "agent_c" / "prompts",
    REPO_ROOT / "docs",
]

# ──────────────────────────────────────────────────────────────
# Terminology rules: (bad_pattern, replacement, explanation)
# Patterns are case-insensitive regexes. Use word boundaries.
# ──────────────────────────────────────────────────────────────

TERM_RULES: list[tuple[str, str, str]] = [
    # Hyphenation for compound adjectives modifying a noun
    (r"\bbound constrained\b", "bound-constrained",
     "Use hyphen: 'bound-constrained' when used as adjective"),
    (r"\bderivative free\b", "derivative-free",
     "Use hyphen: 'derivative-free'"),
    (r"\bname value pair", "name-value pair",
     "Use hyphen: 'name-value pair'"),

    # Adjective form for problem types
    (r"\blinear constraint(?:ed)?\s+problem", "linearly constrained problem",
     "Use adverb form: 'linearly constrained problem'"),
    (r"\bnonlinear constraint(?:ed)?\s+problem", "nonlinearly constrained problem",
     "Use adverb form: 'nonlinearly constrained problem'"),
    (r"\bbound constraint(?:ed)?\s+problem", "bound-constrained problem",
     "Use: 'bound-constrained problem'"),
    (r"\bunconstraint(?:ed)?\s+problem", "unconstrained problem",
     "Use: 'unconstrained problem'"),

    # OptiProfiler-specific
    (r"\bOpti\s+Profiler\b", "OptiProfiler",
     "No space: 'OptiProfiler'"),
    (r"(?<![-_/`])(?<!\w)\boptiprofiler\b(?![-_./`\]])", "OptiProfiler",
     "Capitalize: 'OptiProfiler' (except in code/paths)"),

    # Package-name typos that LLMs / docs occasionally produce. The package
    # is one word, lowercase, no hyphen, no underscore.
    (r"\boptiprobe\b", "optiprofiler",
     "Package typo: the package is 'optiprofiler', not 'optiprobe'"),
    (r"\bopti-profiler\b", "optiprofiler",
     "Package typo: 'opti-profiler' is not a valid Python identifier; use 'optiprofiler'"),
    (r"\bopti_profiler\b", "optiprofiler",
     "Package typo: 'opti_profiler' is not the package name; use 'optiprofiler'"),
    (r"\boptiprofile\b(?!r)", "optiprofiler",
     "Package typo: missing trailing 'r' in 'optiprofiler'"),

    # Common optimization terminology
    (r"\bobjective function value\b", None,
     None),  # correct — skip
    (r"\bfunction handle\b", None,
     None),  # correct — skip
]

# Filter out None rules (placeholders for "correct" patterns)
TERM_RULES = [(p, r, e) for p, r, e in TERM_RULES if r is not None]

# Lines matching these patterns are code blocks — skip them
_CODE_FENCE_RE = re.compile(r"^\s*```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


@dataclass
class Issue:
    file: str
    line: int
    col: int
    message: str
    replacement: str | None = None
    rule: str = "terminology"


@dataclass
class LintResult:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


def _strip_inline_code(text: str) -> str:
    """Replace inline code spans with spaces to avoid false positives."""
    return _INLINE_CODE_RE.sub(lambda m: " " * len(m.group()), text)


def check_file(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return issues

    lines = text.split("\n")
    in_code_block = False
    rel = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)

    for lineno, raw_line in enumerate(lines, start=1):
        if _CODE_FENCE_RE.match(raw_line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        check_line = _strip_inline_code(raw_line)

        for pattern, replacement, explanation in TERM_RULES:
            for match in re.finditer(pattern, check_line, re.IGNORECASE):
                matched_text = match.group()
                if matched_text.lower() == replacement.lower():
                    continue
                issues.append(Issue(
                    file=rel,
                    line=lineno,
                    col=match.start() + 1,
                    message=f"'{matched_text}' -> '{replacement}': {explanation}",
                    replacement=replacement,
                ))

    return issues


def check_all(paths: list[Path] | None = None) -> LintResult:
    if paths:
        md_files = [p for p in paths if p.suffix == ".md"]
    else:
        md_files = []
        for d in SCAN_DIRS:
            if d.exists():
                md_files.extend(sorted(d.rglob("*.md")))

    result = LintResult()
    for f in md_files:
        result.issues.extend(check_file(f))

    return result


def apply_fixes(result: LintResult) -> int:
    """Auto-fix issues in-place. Returns count of fixed issues."""
    from collections import defaultdict
    by_file: dict[str, list[Issue]] = defaultdict(list)
    for issue in result.issues:
        if issue.replacement:
            by_file[issue.file].append(issue)

    fixed = 0
    for rel_path, issues in by_file.items():
        path = REPO_ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        for issue in sorted(issues, key=lambda i: (-i.line, -i.col)):
            lines = text.split("\n")
            line_idx = issue.line - 1
            if line_idx < len(lines):
                old_line = lines[line_idx]
                new_line = re.sub(
                    re.escape(text.split("\n")[line_idx][issue.col - 1:issue.col - 1 + len(issue.replacement or "")]),
                    issue.replacement or "",
                    old_line,
                    count=1,
                    flags=re.IGNORECASE,
                )
                if new_line != old_line:
                    lines[line_idx] = new_line
                    text = "\n".join(lines)
                    fixed += 1
        path.write_text(text, encoding="utf-8")

    return fixed


def run_codespell(paths: list[Path] | None = None) -> int:
    """Run codespell and return exit code."""
    cmd = ["codespell", "--skip", "*.json,*.pyc,__pycache__,.venv,node_modules"]
    if paths:
        cmd.extend(str(p) for p in paths)
    else:
        for d in SCAN_DIRS:
            if d.exists():
                cmd.append(str(d))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout)
        if result.returncode != 0 and result.stderr.strip():
            print(result.stderr, file=sys.stderr)
        return result.returncode
    except FileNotFoundError:
        print("codespell not installed — run: pip install codespell", file=sys.stderr)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Prose & terminology linter for OptiProfiler docs")
    parser.add_argument("files", nargs="*", help="Specific .md files to check (default: all)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix terminology issues")
    parser.add_argument("--codespell", action="store_true", help="Also run codespell for typos")
    parser.add_argument("--ci", action="store_true", help="CI mode: non-zero exit on any issue")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files else None

    print("=== OptiProfiler Prose Lint ===\n")

    result = check_all(paths)

    if result.issues:
        for issue in result.issues:
            print(f"  {issue.file}:{issue.line}:{issue.col}: {issue.message}")
        print(f"\n  Found {len(result.issues)} terminology issue(s)")

        if args.fix:
            fixed = apply_fixes(result)
            print(f"  Auto-fixed {fixed} issue(s)")
    else:
        print("  No terminology issues found.")

    exit_code = 0
    if args.codespell:
        print("\n=== codespell ===\n")
        cs_code = run_codespell(paths)
        if cs_code != 0:
            exit_code = 1

    if args.ci and not result.ok:
        exit_code = 1

    if exit_code == 0 and result.ok:
        print("\n  All checks passed.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
