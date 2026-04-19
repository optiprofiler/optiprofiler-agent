"""Agent-writable wiki pages under ``OPAGENT_HOME/wiki/auto/``.

Hermes Agent allows the model to extend its own knowledge by writing skill
files. We adopt a narrower form: the agent can append focused, factual wiki
pages that the existing RAG pipeline picks up automatically on the next
indexing pass.

Each page gets a frontmatter block with the source ("agent" vs "user"), a
short summary, and the UTC timestamp. This keeps the pages first-class wiki
entries — ``opagent wiki lint`` treats them like any other page.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from optiprofiler_agent.runtime import paths

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    """Sanitize a free-form title into a safe filename slug."""
    value = (value or "").strip().lower()
    value = _SLUG_RE.sub("-", value).strip("-")
    return value or "untitled"


def add_page(
    slug: str,
    content: str,
    summary: str = "",
    source: str = "agent",
) -> Path:
    """Write one markdown page under ``OPAGENT_HOME/wiki/auto/`` and return it.

    If a file with the same slug already exists, a numeric suffix is appended
    so the agent's notes never silently overwrite an earlier version.
    """
    safe = slugify(slug)
    target_dir = paths.auto_wiki_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / f"{safe}.md"
    n = 2
    while target.exists():
        target = target_dir / f"{safe}-{n}.md"
        n += 1

    frontmatter = (
        "---\n"
        f"title: {slug}\n"
        f"source: {source}\n"
        f"created: {_now_iso()}\n"
    )
    if summary:
        frontmatter += f"summary: {summary}\n"
    frontmatter += "---\n\n"

    body = content if content.endswith("\n") else content + "\n"
    target.write_text(frontmatter + body, encoding="utf-8")
    return target


def list_pages() -> list[Path]:
    base = paths.auto_wiki_dir()
    if not base.exists():
        return []
    return sorted(base.rglob("*.md"))
