"""Two-level declarative memory: ``USER.md`` (profile) + ``MEMORY.md`` (notes).

Borrowed from Hermes Agent's ``USER.md`` / ``MEMORY.md`` pattern, but pared
down for our scope:

* No background "memory worker": writes happen synchronously through tools.
* No nested namespaces: just two flat files the user can grep / edit by hand.
* Profile updates are restricted to a small whitelist of fields to prevent
  accidental schema drift through prompt injection.

The ``frozen_snapshot`` function returns a small text block that the agent's
system prompt prepends, giving the LLM persistent context across sessions
without bloating every turn.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from optiprofiler_agent.runtime import paths

ALLOWED_PROFILE_FIELDS: tuple[str, ...] = (
    "name",
    "role",
    "preferred_solver",
    "preferred_language",
    "project_root",
)

_COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*$", re.DOTALL | re.MULTILINE)
_PROFILE_LINE_RE = re.compile(r"^\s*-\s+\*\*(?P<field>[a-z_]+)\*\*\s*:\s*(?P<value>.+)$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


# ---------------------------------------------------------------------------
# MEMORY.md — append-only declarative facts
# ---------------------------------------------------------------------------


def append_fact(fact: str, tags: list[str] | None = None) -> str:
    """Append one declarative fact to ``MEMORY.md``. Returns the line written."""
    fact = fact.strip()
    if not fact:
        return ""
    tags = tags or []
    tag_str = ", ".join(t.strip() for t in tags if t.strip())
    line = f"- [{_now_iso()}]"
    if tag_str:
        line += f" [{tag_str}]"
    line += f" {fact}"

    mp = paths.memory_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    if not mp.exists():
        mp.write_text("# Agent Memory\n\n", encoding="utf-8")
    with mp.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


def read_facts() -> list[str]:
    """Return all non-comment, non-blank lines from ``MEMORY.md``."""
    text = _strip_comments(_read_text(paths.memory_path()))
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def clear_facts() -> None:
    paths.memory_path().write_text("# Agent Memory\n\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# USER.md — structured profile
# ---------------------------------------------------------------------------


def read_user_profile() -> dict[str, str]:
    """Parse ``USER.md`` into a dict. Unknown fields are dropped."""
    text = _strip_comments(_read_text(paths.user_path()))
    out: dict[str, str] = {}
    for ln in text.splitlines():
        m = _PROFILE_LINE_RE.match(ln)
        if not m:
            continue
        field = m.group("field")
        value = m.group("value").strip()
        if field in ALLOWED_PROFILE_FIELDS:
            out[field] = value
    return out


def update_user_profile(field: str, value: str) -> str:
    """Set one whitelisted profile field. Returns the new line.

    Raises ``ValueError`` if the field is not in ``ALLOWED_PROFILE_FIELDS``.
    """
    if field not in ALLOWED_PROFILE_FIELDS:
        raise ValueError(
            f"Field '{field}' is not allowed. Choose one of: "
            f"{', '.join(ALLOWED_PROFILE_FIELDS)}"
        )
    value = value.strip()
    profile = read_user_profile()
    profile[field] = value

    up = paths.user_path()
    up.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# User Profile", ""]
    for f in ALLOWED_PROFILE_FIELDS:
        if f in profile:
            lines.append(f"- **{f}**: {profile[f]}")
    lines.append("")
    up.write_text("\n".join(lines), encoding="utf-8")
    return f"- **{field}**: {value}"


# ---------------------------------------------------------------------------
# Frozen snapshot — system-prompt injection
# ---------------------------------------------------------------------------


def frozen_snapshot(max_chars: int = 1500) -> str:
    """Compose USER profile + recent MEMORY notes into a prompt-ready block.

    Returns the empty string when both files are empty / unset, so the caller
    can ``if snap: prompt += snap`` without padding the system message.
    """
    profile = read_user_profile()
    facts = read_facts()
    if not profile and not facts:
        return ""

    parts: list[str] = ["### Persistent Context (frozen snapshot)"]
    parts.append(
        "These are facts the agent has previously stored. Treat them as "
        "background context; if the user's latest message contradicts them, "
        "prefer the latest message."
    )

    if profile:
        parts.append("\n**User profile**")
        for f in ALLOWED_PROFILE_FIELDS:
            if f in profile:
                parts.append(f"- {f}: {profile[f]}")

    if facts:
        parts.append("\n**Notes**")
        # Most recent at the bottom is more useful for LLMs (recency primacy)
        # but we hard-cap from the top so the *latest* are always preserved.
        keep: list[str] = []
        running = sum(len(p) for p in parts)
        for line in reversed(facts):
            if running + len(line) + 1 > max_chars:
                keep.insert(0, "- [...older notes truncated...]")
                break
            keep.insert(0, line)
            running += len(line) + 1
        parts.extend(keep)

    return "\n".join(parts) + "\n"
