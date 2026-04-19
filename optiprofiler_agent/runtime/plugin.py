"""Developer / advanced-user plugin loader (opt-in).

Reads ``OPAGENT_HOME/config.yaml`` once and exposes:

* ``external_wiki_dirs()``  — extra Markdown roots fed into the RAG index.
* ``external_skill_dirs()`` — extra skill packages (used by future skill
  loaders; not yet wired into a tool).

The loader is intentionally tolerant: a missing config file, missing
``yaml`` dependency, or unknown keys all degrade gracefully to "no plugins".
This keeps the user-facing default experience zero-friction.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from optiprofiler_agent.runtime import paths


def _safe_load_yaml(text: str) -> dict:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(text) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _config() -> dict:
    cp = paths.config_path()
    if not cp.exists():
        return {}
    try:
        text = cp.read_text(encoding="utf-8")
    except OSError:
        return {}
    return _safe_load_yaml(text)


def reload() -> None:
    """Force re-read of ``config.yaml`` (mostly for tests)."""
    _config.cache_clear()


def _expanded_dirs(key: str) -> list[Path]:
    section = _config().get("plugin", {}) or {}
    raw = section.get(key, []) or []
    out: list[Path] = []
    for item in raw:
        if not item:
            continue
        p = Path(str(item)).expanduser()
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def external_wiki_dirs() -> list[Path]:
    return _expanded_dirs("external_wiki_dirs")


def external_skill_dirs() -> list[Path]:
    return _expanded_dirs("external_skill_dirs")
