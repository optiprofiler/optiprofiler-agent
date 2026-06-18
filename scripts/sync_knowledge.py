#!/usr/bin/env python
"""One-command knowledge sync for OptiProfiler Agent.

This refreshes package API sources, platform sources, generated wiki reference
pages, and the mechanical coverage/lint checks used by CI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OPTIPROFILER_ROOT = REPO_ROOT.parent / "optiprofiler"
DEFAULT_PLATFORM_ROOT = REPO_ROOT.parent / "optiprofiler-platform"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _package_env(optiprofiler_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    package_path = str((optiprofiler_root / "python").resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_path if not existing else f"{package_path}{os.pathsep}{existing}"
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh source-backed agent knowledge")
    parser.add_argument("--optiprofiler-root", type=Path, default=DEFAULT_OPTIPROFILER_ROOT)
    parser.add_argument("--platform-root", type=Path, default=DEFAULT_PLATFORM_ROOT)
    parser.add_argument("--skip-package", action="store_true", help="Do not refresh OptiProfiler package API sources")
    parser.add_argument("--skip-platform", action="store_true", help="Do not refresh platform source docs")
    parser.add_argument("--skip-wiki-lint", action="store_true", help="Skip `opagent wiki lint`")
    args = parser.parse_args()

    if not args.skip_package:
        optiprofiler_root = args.optiprofiler_root.expanduser().resolve()
        _run(
            [
                sys.executable,
                "scripts/extract_knowledge.py",
                "--optiprofiler-root",
                str(optiprofiler_root),
            ],
            env=_package_env(optiprofiler_root),
        )

    if not args.skip_platform:
        _run(
            [
                sys.executable,
                "scripts/extract_platform_knowledge.py",
                "--platform-root",
                str(args.platform_root.expanduser().resolve()),
            ]
        )

    _run([sys.executable, "scripts/sync_wiki_reference.py"])
    _run([sys.executable, "scripts/audit_wiki_coverage.py"])

    if not args.skip_wiki_lint:
        _run([sys.executable, "-m", "optiprofiler_agent.cli", "wiki", "lint"])

    print("Knowledge sync completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
