#!/usr/bin/env python
"""Generate source-backed wiki reference pages.

The narrative wiki pages are written for humans and RAG synthesis. These
reference pages are different: they are deterministic mirrors of the bundled
knowledge sources, so API facts, examples, and legacy docs have a lossless
representation inside ``knowledge/wiki``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = REPO_ROOT / "optiprofiler_agent" / "knowledge"
WIKI_DIR = KNOWLEDGE_DIR / "wiki"
REFERENCE_DIR = WIKI_DIR / "reference"
REFERENCE_LAST_UPDATED = "2026-06-07"

JSON_SOURCE_GLOBS = (
    "_sources/python/*.json",
    "_sources/matlab/*.json",
)
LEGACY_MD_DIRS = ("common", "python", "matlab", "profiles", "debugging")


def _rel(path: Path, base: Path = KNOWLEDGE_DIR) -> str:
    return path.relative_to(base).as_posix()


def _fence(text: str, language: str) -> tuple[str, str]:
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}", ticks


def _code_block(text: str, language: str = "text") -> str:
    opener, closer = _fence(text, language)
    return f"{opener}\n{text}\n{closer}"


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _frontmatter(tags: list[str], sources: list[str], related: list[str] | None = None) -> str:
    lines = ["---"]
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f"sources: [{', '.join(sources)}]")
    lines.append(f"related: [{', '.join(related or [])}]")
    lines.append(f"last_updated: {REFERENCE_LAST_UPDATED}")
    lines.append("generated: true")
    lines.append("---")
    return "\n".join(lines)


def _json_sections(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return RAG-sized sections for a JSON source."""
    sections: list[tuple[str, Any]] = []
    option_categories = {
        "parameters",
        "feature_options",
        "profile_options",
        "problem_options",
        "returns",
        "solver_signatures",
    }

    for key, value in data.items():
        if isinstance(value, dict) and key in option_categories:
            sections.append((key, value))
            for subkey, subvalue in value.items():
                sections.append((f"{key}.{subkey}", subvalue))
        elif isinstance(value, dict) and value and all(isinstance(item, dict) for item in value.values()):
            sections.append((key, value))
            for subkey, subvalue in value.items():
                sections.append((f"{key}.{subkey}", subvalue))
        else:
            sections.append((key, value))

    return sections


def _render_json_reference_page(source_path: Path) -> str:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    rel_source = _rel(source_path)
    lang = source_path.parent.name
    stem = source_path.stem
    canonical = _canonical_json(data)
    digest = _hash_text(canonical)
    sections = _json_sections(data)

    related = []
    if stem == "benchmark":
        related.append(f"../api/{lang}/benchmark.md")
    elif stem == "classes" and lang == "python":
        related.append("../api/python/problem-class.md")
    elif stem == "plib_tools" and lang == "python":
        related.append("../api/python/plib-tools.md")

    lines = [
        _frontmatter(
            tags=["reference", "source-backed", lang, stem.replace("_", "-")],
            sources=[rel_source],
            related=related,
        ),
        "",
        f"# Source Reference: {lang.title()} {source_path.name}",
        "",
        f"This page is auto-generated from `{rel_source}`. It is the lossless wiki mirror for this source.",
        "Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing the source.",
        "",
        "## Source Metadata",
        "",
        f"- Source path: `{rel_source}`",
        f"- Canonical SHA256: `{digest}`",
        f"- Top-level keys: {', '.join(f'`{key}`' for key in data)}",
        "",
        "## Path Index",
        "",
        "| Path | Kind |",
        "|---|---|",
    ]

    for path, value in sections:
        kind = type(value).__name__
        if isinstance(value, dict):
            kind = f"dict[{len(value)}]"
        elif isinstance(value, list):
            kind = f"list[{len(value)}]"
        lines.append(f"| `{path}` | {kind} |")

    for path, value in sections:
        lines.extend(["", f"## {path}", ""])
        if isinstance(value, (dict, list)):
            lines.append(_code_block(_canonical_json(value), "json"))
        else:
            lines.append(_code_block(str(value), "text"))

    lines.extend(["", "## Canonical JSON Mirror", "", _code_block(canonical, "json"), ""])
    return "\n".join(lines)


def _render_enums_reference_page() -> str:
    source_paths = [KNOWLEDGE_DIR / "enums.json", KNOWLEDGE_DIR / "common" / "enums.json"]
    existing = [path for path in source_paths if path.exists()]
    sources = [_rel(path) for path in existing]
    lines = [
        _frontmatter(tags=["reference", "source-backed", "enums"], sources=sources),
        "",
        "# Source Reference: Enums",
        "",
        "This page mirrors bundled enum JSON files exactly.",
    ]

    for path in existing:
        rel_source = _rel(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        canonical = _canonical_json(data)
        lines.extend(
            [
                "",
                f"## {rel_source}",
                "",
                f"- Canonical SHA256: `{_hash_text(canonical)}`",
                "",
                _code_block(canonical, "json"),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _render_markdown_reference_page(title: str, source_paths: list[Path], tags: list[str]) -> str:
    sources = [_rel(path) for path in source_paths]
    lines = [
        _frontmatter(tags=tags, sources=sources),
        "",
        f"# Source Reference: {title}",
        "",
        "This page mirrors bundled Markdown knowledge sources exactly.",
        "Do not hand-edit it; run `python scripts/sync_wiki_reference.py` after changing a source.",
    ]

    for path in source_paths:
        rel_source = _rel(path)
        text = path.read_text(encoding="utf-8")
        lines.extend(
            [
                "",
                f"## {rel_source}",
                "",
                f"- Source SHA256: `{_hash_text(text)}`",
                "",
                _code_block(text, "markdown"),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _legacy_markdown_sources() -> list[Path]:
    paths: list[Path] = []
    for dirname in LEGACY_MD_DIRS:
        source_dir = KNOWLEDGE_DIR / dirname
        if source_dir.exists():
            paths.extend(sorted(source_dir.glob("*.md")))
    return paths


def _bibliography_sources() -> list[Path]:
    refs_dir = KNOWLEDGE_DIR / "_sources" / "refs"
    if not refs_dir.exists():
        return []
    return sorted(refs_dir.glob("*.md"))


def generate_reference_pages() -> dict[Path, str]:
    pages: dict[Path, str] = {}

    for pattern in JSON_SOURCE_GLOBS:
        for source_path in sorted(KNOWLEDGE_DIR.glob(pattern)):
            lang = source_path.parent.name
            target = REFERENCE_DIR / f"{lang}-{source_path.stem}.md"
            pages[target] = _render_json_reference_page(source_path)

    pages[REFERENCE_DIR / "enums.md"] = _render_enums_reference_page()

    legacy_sources = _legacy_markdown_sources()
    if legacy_sources:
        pages[REFERENCE_DIR / "legacy-docs.md"] = _render_markdown_reference_page(
            "Legacy Docs And Examples",
            legacy_sources,
            tags=["reference", "source-backed", "legacy-docs", "examples"],
        )

    bibliography = _bibliography_sources()
    if bibliography:
        pages[REFERENCE_DIR / "bibliography.md"] = _render_markdown_reference_page(
            "Bibliography",
            bibliography,
            tags=["reference", "source-backed", "bibliography"],
        )

    return pages


def sync_reference_pages(check: bool = False) -> list[Path]:
    changed: list[Path] = []
    pages = generate_reference_pages()

    if not check:
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    for path, expected in pages.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == expected:
            continue
        changed.append(path)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate source-backed wiki reference pages")
    parser.add_argument("--check", action="store_true", help="Only report pages that would change")
    args = parser.parse_args()

    changed = sync_reference_pages(check=args.check)
    if changed:
        action = "Out of sync" if args.check else "Updated"
        for path in changed:
            print(f"{action}: {_rel(path, REPO_ROOT)}")
        return 1 if args.check else 0

    print("Source-backed wiki reference pages are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
