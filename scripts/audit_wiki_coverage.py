#!/usr/bin/env python
"""Audit lossless coverage of bundled knowledge sources in the wiki."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.sync_wiki_reference import REPO_ROOT, generate_reference_pages, sync_reference_pages
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sync_wiki_reference import REPO_ROOT, generate_reference_pages, sync_reference_pages


@dataclass(frozen=True)
class CoverageIssue:
    path: Path
    message: str


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def audit_wiki_coverage() -> list[CoverageIssue]:
    """Return coverage issues for generated source-backed wiki pages."""
    issues: list[CoverageIssue] = []

    for path, expected in generate_reference_pages().items():
        if not path.exists():
            issues.append(CoverageIssue(path, "missing generated reference page"))
            continue
        current = path.read_text(encoding="utf-8")
        if current != expected:
            issues.append(CoverageIssue(path, "generated reference page is out of sync with source"))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit source-vs-wiki knowledge coverage")
    parser.add_argument("--fix", action="store_true", help="Regenerate reference pages before auditing")
    args = parser.parse_args()

    if args.fix:
        sync_reference_pages(check=False)

    issues = audit_wiki_coverage()
    if issues:
        print("Wiki source coverage audit failed:")
        for issue in issues:
            print(f"  {_rel(issue.path)}: {issue.message}")
        print("\nRun: python scripts/sync_wiki_reference.py")
        return 1

    print("Wiki source coverage audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
