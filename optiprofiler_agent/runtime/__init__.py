"""Runtime layer for the OptiProfiler Agent.

This sub-package provides the *persistent, user-facing* runtime services that
survive across CLI invocations:

* ``paths``        — canonical locations under ``OPAGENT_HOME`` (default
                     ``~/.opagent/``).
* ``bootstrap``    — idempotent first-run setup: creates directories, copies
                     bundled seed files, writes a manifest.
* ``memory``       — two-level memory (``USER.md`` + ``MEMORY.md``) with a
                     frozen-snapshot serializer for system-prompt injection.
* ``session_log``  — SQLite + FTS5 store of every chat turn, exposed as
                     ``recall_past``.
* ``wiki_local``   — agent-writable wiki pages under ``wiki/auto/``.
* ``trajectory``   — opt-in ShareGPT-style turn dumper for RL / replay.
* ``plugin``       — reads ``config.yaml`` for advanced extension dirs.

Design notes
------------
* The pip-installed package ships *only read-only seeds*; everything writable
  lives under ``OPAGENT_HOME`` so that ``pip install --upgrade`` never
  overwrites user data.
* User-facing modules (memory, session_log, wiki_local) are wired in by
  default. Developer-facing modules (trajectory, plugin) are gated by env
  vars / config flags and stay quiet otherwise.
* The whole layer is zero-extra-dependency: stdlib ``sqlite3``, ``json``,
  ``shutil`` only.
"""

from __future__ import annotations
