"""Local script runner with timeout and output capture.

Runs a Python script in a subprocess, captures stdout/stderr, and
enforces a wall-clock timeout. Used by Agent B's ``--run`` mode for
an automated diagnose-fix-re-run loop.

Safety notes:
- Scripts run as the current user (no sandbox).
- A temp copy of the script is used; the original file is never modified.
- On timeout, the entire process tree is killed via killpg + descendant scan.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    """Result of running a Python script."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def traceback(self) -> str:
        """Extract the traceback portion from stderr."""
        lines = self.stderr.splitlines()
        tb_start = None
        for i, line in enumerate(lines):
            if line.startswith("Traceback (most recent call last)"):
                tb_start = i
        if tb_start is not None:
            return "\n".join(lines[tb_start:])
        if self.stderr.strip():
            return self.stderr.strip()
        return ""


def _kill_tree(root_pid: int) -> None:
    """Kill root_pid and ALL descendants, using multiple strategies."""
    if root_pid <= 0:
        return

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(root_pid)],
            capture_output=True, check=False,
        )
        return

    # Strategy 1: killpg (covers processes in the same session group)
    try:
        pgid = os.getpgid(root_pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError, PermissionError):
        pass

    time.sleep(0.3)

    # Strategy 2: walk the process tree via `ps` and kill every descendant
    try:
        ps = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid="],
            capture_output=True, text=True, timeout=5, check=False,
        )
        children_map: dict[int, list[int]] = {}
        for line in ps.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    children_map.setdefault(int(parts[1]), []).append(int(parts[0]))
                except ValueError:
                    pass

        to_kill: list[int] = []
        stack = [root_pid]
        visited: set[int] = set()
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            to_kill.append(pid)
            stack.extend(children_map.get(pid, []))

        for pid in to_kill:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError, PermissionError):
                pass
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Strategy 3: final killpg SIGKILL
    try:
        pgid = os.getpgid(root_pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError, PermissionError):
        pass


def run_script(
    code: str,
    timeout: int = 120,
    cwd: str | Path | None = None,
) -> RunResult:
    """Run a Python script in a subprocess.

    Uses a new session so that on timeout the entire process tree
    (including multiprocessing workers, matplotlib subprocesses, etc.)
    can be terminated.

    Parameters
    ----------
    code : str
        Python source code to execute.
    timeout : int
        Wall-clock timeout in seconds.
    cwd : str or Path, optional
        Working directory for the script.

    Returns
    -------
    RunResult
        Captured output and exit status.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(code)
        tmp_path = f.name

    is_posix = sys.platform != "win32"
    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": cwd,
    }
    if is_posix:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen([sys.executable, tmp_path], **popen_kwargs)

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return RunResult(
            exit_code=proc.returncode or 0,
            stdout=stdout or "",
            stderr=stderr or "",
        )
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        return RunResult(
            exit_code=-1,
            stdout=stdout or "",
            stderr=f"Script timed out after {timeout} seconds.\n{stderr or ''}",
            timed_out=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
