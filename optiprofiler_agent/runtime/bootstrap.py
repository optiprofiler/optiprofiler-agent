"""First-run bootstrap for the OptiProfiler Agent runtime.

``ensure()`` is **idempotent and silent** when the environment is already
provisioned. The contract:

1. Create ``OPAGENT_HOME`` and its standard sub-directories (``wiki/auto``,
   ``skills``, ``trajectories``).
2. Copy each *missing* file from ``runtime/_seed/`` into the home dir. Existing
   user files are NEVER overwritten — even if the bundled seed has changed,
   the manifest only adds new files (so users keep their edits across upgrades).
3. Write / update ``.bootstrapped.json`` with the current package version and
   the list of seed files that have been provisioned.

The function is cheap on subsequent invocations (one stat per seed file) so
the CLI can call it on every launch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from optiprofiler_agent.runtime import paths

try:
    from importlib.metadata import version as _pkg_version

    _PKG_VERSION = _pkg_version("optiprofiler-agent")
except Exception:
    _PKG_VERSION = "unknown"


def _seed_files() -> list[Path]:
    seed_root = paths.bundled_seed_dir()
    if not seed_root.exists():
        return []
    return sorted(p for p in seed_root.rglob("*") if p.is_file())


def _load_manifest() -> dict:
    mp = paths.manifest_path()
    if not mp.exists():
        return {"version": None, "seeded": []}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": None, "seeded": []}


def _save_manifest(manifest: dict) -> None:
    mp = paths.manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure() -> dict:
    """Create dirs and copy missing seed files. Returns the updated manifest.

    Safe to call on every CLI invocation — the work after first run is one
    stat per seed file plus one manifest read.
    """
    home = paths.home()
    home.mkdir(parents=True, exist_ok=True)
    for sub in (paths.auto_wiki_dir(), paths.skills_dir()):
        sub.mkdir(parents=True, exist_ok=True)

    seed_root = paths.bundled_seed_dir()
    manifest = _load_manifest()
    seeded = set(manifest.get("seeded", []))
    changed = False

    for seed_file in _seed_files():
        rel = seed_file.relative_to(seed_root)
        target = home / rel
        if target.exists():
            seeded.add(str(rel))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_file, target)
        seeded.add(str(rel))
        changed = True

    if changed or manifest.get("version") != _PKG_VERSION:
        manifest = {
            "version": _PKG_VERSION,
            "seeded": sorted(seeded),
        }
        _save_manifest(manifest)

    return manifest
