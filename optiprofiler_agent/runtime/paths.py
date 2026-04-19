"""Canonical filesystem layout for the OptiProfiler Agent runtime.

All user-writable runtime data lives under ``OPAGENT_HOME``. The default is
``~/.opagent/``; users can override by setting the ``OPAGENT_HOME``
environment variable (useful for CI, sandboxes, or per-project isolation).

The pip-installed package itself ships **only read-only seeds**
(``optiprofiler_agent/runtime/_seed/``); the bootstrap step copies them into
``OPAGENT_HOME`` on first run, so ``pip install --upgrade`` never overwrites
user notes / sessions / custom skills.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "OPAGENT_HOME"
DEFAULT_HOME = Path.home() / ".opagent"

_BUNDLED_PACKAGE_ROOT = Path(__file__).resolve().parent
_BUNDLED_SEED_DIR = _BUNDLED_PACKAGE_ROOT / "_seed"


def home() -> Path:
    """Return the active OPAGENT_HOME (resolved, expanded, never None)."""
    raw = os.environ.get(ENV_HOME)
    base = Path(raw).expanduser() if raw else DEFAULT_HOME
    return base


def memory_path() -> Path:
    return home() / "MEMORY.md"


def user_path() -> Path:
    return home() / "USER.md"


def auto_wiki_dir() -> Path:
    return home() / "wiki" / "auto"


def skills_dir() -> Path:
    return home() / "skills"


def session_db_path() -> Path:
    return home() / "sessions.db"


def trajectory_dir() -> Path:
    """Default trajectory output dir (only used if trajectory is enabled)."""
    return home() / "trajectories"


def config_path() -> Path:
    return home() / "config.yaml"


def manifest_path() -> Path:
    return home() / ".bootstrapped.json"


def bundled_seed_dir() -> Path:
    """Read-only seed directory shipped inside the pip package."""
    return _BUNDLED_SEED_DIR


def all_writable_paths() -> dict[str, Path]:
    """Convenience: full map of user-facing path names → resolved path.

    Used by ``opagent home path`` for diagnostics.
    """
    return {
        "home": home(),
        "memory": memory_path(),
        "user": user_path(),
        "auto_wiki": auto_wiki_dir(),
        "skills": skills_dir(),
        "session_db": session_db_path(),
        "trajectory": trajectory_dir(),
        "config": config_path(),
        "manifest": manifest_path(),
    }
