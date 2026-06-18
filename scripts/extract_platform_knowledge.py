#!/usr/bin/env python
"""Extract OptiProfiler Platform docs into the agent knowledge sources.

The platform repository is maintained separately from ``optiprofiler-agent``.
This script snapshots selected platform docs into ``knowledge/_sources`` so
Agent A/B/C can answer platform-specific questions from the same LLM wiki as
package API facts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLATFORM_ROOT = REPO_ROOT.parent / "optiprofiler-platform"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "optiprofiler_agent" / "knowledge" / "_sources" / "platform"

DOC_PATHS = (
    "README.md",
    "docs/api.md",
    "docs/architecture.md",
    "docs/leaderboard.md",
    "docs/adr/0005-agent-c-integration.md",
    "docs/adr/0006-chat-widget.md",
    "docs/adr/0007-auto-debug.md",
    "docs/adr/0008-leaderboard.md",
    "docs/adr/0009-matlab-solver-upload.md",
    "docs/adr/0010-multi-language-backend.md",
    "docs/adr/0011-leaderboard-pairwise-scoring.md",
    "docs/adr/0012-matlab-cli-sandbox.md",
    "docs/adr/0013-dfo-ecosystem-module-registry.md",
    "docs/problem-libraries-industrial-dfo.md",
)


@dataclass(frozen=True)
class PlatformSource:
    path: str
    exists: bool
    size_bytes: int = 0


def _git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _source_label(root: Path) -> str:
    return root.name


def _build_markdown(root: Path, sources: list[PlatformSource], commit: str, dirty: str) -> str:
    lines = [
        "# OptiProfiler Platform Source Snapshot",
        "",
        "This file is generated from the local `optiprofiler-platform` repository.",
        "Do not hand-edit it; run `python scripts/extract_platform_knowledge.py`.",
        "",
        "## Snapshot Metadata",
        "",
        f"- Source repository: `{_source_label(root)}`",
        f"- Git commit: `{commit}`",
        f"- Worktree status: `{dirty}`",
        "",
        "## Included Sources",
        "",
        "| Path | Status | Bytes |",
        "|---|---:|---:|",
    ]

    for source in sources:
        status = "included" if source.exists else "missing"
        lines.append(f"| `{source.path}` | {status} | {source.size_bytes} |")

    for source in sources:
        if not source.exists:
            continue
        path = root / source.path
        text = path.read_text(encoding="utf-8")
        lines.extend(
            [
                "",
                f"## Source: {source.path}",
                "",
                "```markdown",
                text.rstrip(),
                "```",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def extract_platform_knowledge(platform_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Write platform source snapshot files and return their paths."""
    platform_root = platform_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not platform_root.exists():
        raise FileNotFoundError(f"Platform root does not exist: {platform_root}")

    sources: list[PlatformSource] = []
    for rel_path in DOC_PATHS:
        path = platform_root / rel_path
        sources.append(
            PlatformSource(
                path=rel_path,
                exists=path.exists(),
                size_bytes=path.stat().st_size if path.exists() else 0,
            )
        )

    commit = _git_output(platform_root, "rev-parse", "--short", "HEAD")
    status = _git_output(platform_root, "status", "--short")
    dirty = "clean" if status == "unknown" or not status else status.replace("\n", "; ")

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_path = output_dir / "platform-docs.md"
    manifest_path = output_dir / "manifest.json"

    docs_path.write_text(_build_markdown(platform_root, sources, commit, dirty), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "source_repository": _source_label(platform_root),
                "git_commit": commit,
                "worktree_status": dirty,
                "sources": [asdict(source) for source in sources],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return docs_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract platform docs into knowledge/_sources/platform")
    parser.add_argument("--platform-root", type=Path, default=DEFAULT_PLATFORM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    docs_path, manifest_path = extract_platform_knowledge(args.platform_root, args.output_dir)
    print(f"Wrote {docs_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
