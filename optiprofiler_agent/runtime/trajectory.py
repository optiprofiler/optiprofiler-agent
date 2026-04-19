"""Opt-in ShareGPT-style trajectory dumper for offline replay / RL.

Disabled by default. Two ways to enable:

1. Set ``OPAGENT_TRAJECTORY_DIR=/some/path`` in the environment. The path is
   created on demand. This wins over the config file.
2. Set ``trajectory.enabled: true`` in ``OPAGENT_HOME/config.yaml`` (and
   optionally ``trajectory.dir: <path>``).

When enabled, every chat turn is appended to a per-session JSONL file:

    {"role": "user",      "content": "...", "ts": 1.234}
    {"role": "assistant", "content": "...", "ts": 1.567}

This is a strict superset of session_log (which is for *agent recall*) — we
keep them separate because trajectories are bulk training data while the
session log must stay small enough to FTS-scan on every ``recall_past``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from optiprofiler_agent.runtime import paths

ENV_DIR = "OPAGENT_TRAJECTORY_DIR"


def _config_section() -> dict:
    try:
        from optiprofiler_agent.runtime import plugin
    except ImportError:
        return {}
    section = plugin._config().get("trajectory") if hasattr(plugin, "_config") else None
    return section or {}


def enabled() -> bool:
    if os.environ.get(ENV_DIR):
        return True
    section = _config_section()
    return bool(section.get("enabled"))


def output_dir() -> Path:
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env).expanduser()
    section = _config_section()
    custom = section.get("dir")
    if custom:
        return Path(str(custom)).expanduser()
    return paths.trajectory_dir()


def append(session_id: str, role: str, content: str) -> None:
    """Append one turn to ``<dir>/<session_id>.jsonl``. Best-effort."""
    if not enabled() or not content:
        return
    try:
        out = output_dir()
        out.mkdir(parents=True, exist_ok=True)
        rec = {"role": role, "content": content, "ts": time.time()}
        with (out / f"{session_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
