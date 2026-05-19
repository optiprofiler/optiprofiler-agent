"""MATLAB script runner with timeout, output capture, and traceback extraction.

Mirrors the contract of ``local_runner.run_script``: same ``RunResult``
shape, same kill-tree-on-timeout semantics. Used by Agent B's
``--run`` mode when ``language="matlab"``.

Activation
----------
This runner only works when a local MATLAB install is reachable. Resolution
order:

1. Explicit ``matlab_bin`` argument to :func:`run_matlab_script`.
2. ``MATOP_MATLAB_BIN`` environment variable (preferred).
3. ``matlab`` on ``PATH``.

If none of the above resolve to an executable, :func:`run_matlab_script`
raises :class:`MatlabNotAvailable` so callers can fall back to a static
diagnostic instead of failing silently.

Sandbox notes
-------------
* Each run uses a fresh temp directory as ``cwd``. The script is written
  there as ``opagent_script.m`` and called with a deterministic wrapper
  that forces a structured traceback on error and an explicit ``exit``
  on success.
* MATLAB is launched with ``-batch`` (R2019a+) which already implies
  ``-nodesktop -nosplash`` and disables the user's ``startup.m``.
* The whole process group is killed via :func:`local_runner._kill_tree`
  on timeout, matching the Python runner.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from optiprofiler_agent.debugger.local_runner import RunResult, _kill_tree


class MatlabNotAvailable(RuntimeError):
    """Raised when no MATLAB binary can be resolved on this host."""


_DEFAULT_TIMEOUT_SEC = 120

_TRACEBACK_HEAD_RE = re.compile(
    r"^(?:Error using\b|Error in\b|Unrecognized function or variable\b"
    r"|Undefined function or variable\b|Index exceeds\b"
    r"|Dimensions of arrays\b|Not enough input arguments\b"
    r"|Too many input arguments\b)",
    re.MULTILINE,
)


def resolve_matlab_bin(matlab_bin: str | None = None) -> str | None:
    """Return the absolute path to a MATLAB executable, or None.

    Tries (in order): explicit arg, ``MATOP_MATLAB_BIN``, ``PATH``.
    """
    candidate = matlab_bin or os.environ.get("MATOP_MATLAB_BIN") or shutil.which("matlab")
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    if path.is_dir():
        nested = path / "bin" / "matlab"
        if nested.is_file() and os.access(nested, os.X_OK):
            return str(nested)
    return None


def is_matlab_available(matlab_bin: str | None = None) -> bool:
    """Quick predicate for pytest markers / CLI gating."""
    return resolve_matlab_bin(matlab_bin) is not None


_WRAPPER_FN_NAME = "opagent_runner"


def _build_wrapper(script_name: str) -> str:
    """Return MATLAB source for a wrapper function that runs ``script_name``.

    ``matlab -batch`` reads a single statement (typically a function name),
    so we ship the wrapper as a real ``.m`` file alongside the user script.
    Both files live in the same temp directory which gets ``addpath``'d at
    startup.

    The wrapper:
    * ``addpath(pwd)`` so the inner script is on the MATLAB path.
    * ``run('<script>')`` executes it in the base workspace, the same way
      the platform sandbox runs ``scratch.m``.
    * On error, prints the full report via ``getReport(ME, 'extended',
      'hyperlinks', 'off')`` to stderr then exits with code 1.
      ``getReport`` is MATLAB's traceback analogue.
    """
    return (
        f"function {_WRAPPER_FN_NAME}()\n"
        "try\n"
        "    addpath(pwd);\n"
        f"    run('{script_name}.m');\n"
        "    exit(0);\n"
        "catch ME\n"
        "    fprintf(2, '%s\\n', getReport(ME, 'extended', 'hyperlinks', 'off'));\n"
        "    exit(1);\n"
        "end\n"
        "end\n"
    )


def _extract_traceback(stderr: str) -> str:
    """MATLAB analogue of ``RunResult.traceback`` for Python."""
    stderr = stderr or ""
    if not stderr.strip():
        return ""
    m = _TRACEBACK_HEAD_RE.search(stderr)
    if m:
        return stderr[m.start():].rstrip()
    return stderr.rstrip()


def run_matlab_script(
    code: str,
    timeout: int = _DEFAULT_TIMEOUT_SEC,
    cwd: str | Path | None = None,
    matlab_bin: str | None = None,
) -> RunResult:
    """Run a MATLAB script via ``matlab -batch`` and capture the result.

    Raises :class:`MatlabNotAvailable` if no MATLAB binary can be resolved.
    Otherwise returns a :class:`RunResult` with the same shape as the
    Python runner so :func:`debugger.run_and_debug` can be language-agnostic.
    """
    resolved = resolve_matlab_bin(matlab_bin)
    if not resolved:
        raise MatlabNotAvailable(
            "No MATLAB binary found. Set MATOP_MATLAB_BIN or add `matlab` to PATH."
        )

    # MATLAB requires the script to live as a real .m file because `run()`
    # uses the filename as the function name. Use a fresh tmp dir so a
    # second call cannot clash with the first. Both the user script and
    # the wrapper function get written there; -batch invokes the wrapper
    # by name (multi-line code does NOT round-trip through -batch reliably).
    work_dir = Path(tempfile.mkdtemp(prefix="opagent_matlab_"))
    script_name = "opagent_script"
    (work_dir / f"{script_name}.m").write_text(code, encoding="utf-8")
    (work_dir / f"{_WRAPPER_FN_NAME}.m").write_text(
        _build_wrapper(script_name), encoding="utf-8",
    )

    is_posix = sys.platform != "win32"
    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": str(cwd) if cwd else str(work_dir),
    }
    if is_posix:
        popen_kwargs["start_new_session"] = True

    # ``-batch`` (R2019a+) implies -nodesktop -nosplash and skips startup.m.
    # Invoking the wrapper *by name* (not as inline code) sidesteps MATLAB's
    # one-statement-per-batch limitation.
    cmd = [resolved, "-batch", _WRAPPER_FN_NAME]
    proc = subprocess.Popen(cmd, **popen_kwargs)

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
            stderr=f"MATLAB script timed out after {timeout} seconds.\n{stderr or ''}",
            timed_out=True,
        )
    finally:
        # Best-effort cleanup; never raise from finally.
        try:
            for p in work_dir.iterdir():
                p.unlink(missing_ok=True)
            work_dir.rmdir()
        except OSError:
            pass


def run_matlab_script_traceback(result: RunResult) -> str:
    """Public helper: pull a Python-style traceback out of a RunResult."""
    return _extract_traceback(result.stderr)
