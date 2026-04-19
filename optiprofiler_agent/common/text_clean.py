"""Shared text-sanitisation helpers used across the agent.

Centralising ``strip_thinking`` here lets us apply the same scrubbing rule
to every persistence boundary (session log, trajectory dump, structured
report, free-form report) without duplicating the regex or risking
drift.

Why this matters: reasoning models such as MiniMax-M2, DeepSeek-R1, and
Kimi-thinking emit chain-of-thought wrapped in ``<think>...</think>`` (or
``<thinking>``, ``<reasoning>``, ``<scratchpad>``). If we let those tags
land in ``sessions.db`` or ``trajectories/*.jsonl``, downstream
``recall_past`` queries and offline replay will be polluted by
private-by-intent reasoning, which both hurts retrieval quality and
risks feeding the model its own hallucinations on the next turn.
"""

from __future__ import annotations

import re

# Tags emitted by common thinking models. Order doesn't matter — the regex
# alternation is case-insensitive and tolerates whitespace inside the tag.
_THINK_TAGS = ("think", "thinking", "reasoning", "scratchpad")

_THINK_PATTERN = re.compile(
    r"<\s*(?:" + "|".join(_THINK_TAGS) + r")\s*>.*?<\s*/\s*(?:"
    + "|".join(_THINK_TAGS) + r")\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Some providers emit only an opening tag and never close it (e.g. when
# the response is truncated). Cut from the opening tag to the next blank
# line (or end-of-string) as a best-effort recovery so the visible reply
# isn't swallowed entirely.
_THINK_PATTERN_OPEN_ONLY = re.compile(
    r"<\s*(?:" + "|".join(_THINK_TAGS) + r")\s*>.*?(?:\n\n|$)",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>``-style reasoning blocks.

    Safe to call on any string; returns the input unchanged when no tags
    are present. ``None`` and empty input collapse to an empty string so
    callers (e.g. SQLite adapters) never have to special-case ``None``.
    """
    if not text:
        return ""
    cleaned = _THINK_PATTERN.sub("", text)
    cleaned = _THINK_PATTERN_OPEN_ONLY.sub("", cleaned)
    return cleaned.strip()


__all__ = ["strip_thinking"]
