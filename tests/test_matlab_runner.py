"""Tests for the MATLAB sandbox runner.

The fast checks (no MATLAB binary needed) verify pure-Python helpers:
binary resolution, wrapper script, traceback extraction, and the
``MatlabNotAvailable`` path.

The end-to-end tests carry the ``requires_matlab`` marker — they run only
when ``MATOP_MATLAB_BIN`` is set or ``matlab`` is on ``PATH``. They are
the equivalent of ``tests/test_local_runner.py`` on the MATLAB side.
"""

from __future__ import annotations

import pytest

from optiprofiler_agent.debugger.matlab_runner import (
    MatlabNotAvailable,
    _build_wrapper,
    _extract_traceback,
    is_matlab_available,
    resolve_matlab_bin,
    run_matlab_script,
)


class TestResolveMatlabBin:

    def test_explicit_arg_takes_precedence(self, tmp_path, monkeypatch):
        bin_path = tmp_path / "matlab"
        bin_path.write_text("#!/bin/sh\nexit 0\n")
        bin_path.chmod(0o755)
        monkeypatch.delenv("MATOP_MATLAB_BIN", raising=False)
        assert resolve_matlab_bin(str(bin_path)) == str(bin_path)

    def test_env_var_resolves(self, tmp_path, monkeypatch):
        bin_path = tmp_path / "matlab"
        bin_path.write_text("#!/bin/sh\nexit 0\n")
        bin_path.chmod(0o755)
        monkeypatch.setenv("MATOP_MATLAB_BIN", str(bin_path))
        assert resolve_matlab_bin() == str(bin_path)

    def test_app_bundle_dir_resolves_to_inner_binary(self, tmp_path, monkeypatch):
        # Simulate ``/Applications/MATLAB_R2023b.app`` style layout.
        app = tmp_path / "MATLAB.app"
        (app / "bin").mkdir(parents=True)
        bin_path = app / "bin" / "matlab"
        bin_path.write_text("#!/bin/sh\nexit 0\n")
        bin_path.chmod(0o755)
        monkeypatch.delenv("MATOP_MATLAB_BIN", raising=False)
        assert resolve_matlab_bin(str(app)) == str(bin_path)

    def test_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("MATOP_MATLAB_BIN", raising=False)
        monkeypatch.setattr(
            "optiprofiler_agent.debugger.matlab_runner.shutil.which",
            lambda _: None,
        )
        assert resolve_matlab_bin() is None
        assert is_matlab_available() is False

    def test_raises_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("MATOP_MATLAB_BIN", raising=False)
        monkeypatch.setattr(
            "optiprofiler_agent.debugger.matlab_runner.shutil.which",
            lambda _: None,
        )
        with pytest.raises(MatlabNotAvailable):
            run_matlab_script("disp('hi')")


class TestWrapperAndTraceback:

    def test_wrapper_uses_run_and_exit(self):
        w = _build_wrapper("opagent_script")
        assert w.startswith("function opagent_runner()")
        assert "run('opagent_script.m')" in w
        assert "exit(0)" in w
        assert "exit(1)" in w
        assert "getReport" in w
        assert "addpath(pwd)" in w
        assert w.rstrip().endswith("end")

    def test_traceback_extracts_error_using(self):
        stderr = (
            "some chatter\n"
            "Error using my_solver\n"
            "Too many input arguments.\n"
        )
        out = _extract_traceback(stderr)
        assert out.startswith("Error using my_solver")
        assert "Too many input arguments" in out

    def test_traceback_extracts_undefined(self):
        stderr = "Undefined function or variable 'foo'.\n"
        out = _extract_traceback(stderr)
        assert "Undefined function" in out

    def test_traceback_empty_when_no_stderr(self):
        assert _extract_traceback("") == ""
        assert _extract_traceback("   \n") == ""

    def test_traceback_falls_back_to_full_stderr(self):
        stderr = "weird message that matches nothing in particular"
        assert _extract_traceback(stderr) == stderr


# ---------------------------------------------------------------------------
# End-to-end — only runs when MATLAB is available.
# Mirrors test_local_runner.py: success / fail / timeout / syntax.
# ---------------------------------------------------------------------------

# Keep startup cost in mind: each ``matlab -batch`` invocation is ~25s on macOS.
# We deliberately keep this suite small so a developer can run it on demand.

DEFAULT_TIMEOUT = 90


@pytest.mark.requires_matlab
@pytest.mark.integration
class TestMatlabRunnerE2E:

    def test_successful_script(self):
        result = run_matlab_script(
            "disp('hello matlab');\nfprintf('%d\\n', 1+1);\n",
            timeout=DEFAULT_TIMEOUT,
        )
        assert result.success
        assert "hello matlab" in result.stdout
        assert result.exit_code == 0

    def test_runtime_error_propagates(self):
        # Index past the end of an array. MATLAB returns non-zero and writes
        # a structured ``getReport`` traceback to stderr.
        result = run_matlab_script(
            "a = [1, 2]; disp(a(3));",
            timeout=DEFAULT_TIMEOUT,
        )
        assert not result.success
        assert result.exit_code != 0
        assert "Index" in result.stderr or "exceeds" in result.stderr.lower()
        assert result.traceback  # should not be empty

    def test_undefined_variable(self):
        result = run_matlab_script(
            "x = foo_that_does_not_exist(1, 2);",
            timeout=DEFAULT_TIMEOUT,
        )
        assert not result.success
        assert (
            "Undefined function" in result.stderr
            or "Unrecognized function" in result.stderr
        )

    def test_timeout_is_enforced(self):
        # Sleep longer than our short timeout. We expect timed_out=True
        # and a forced kill.
        result = run_matlab_script(
            "pause(120);",
            timeout=8,
        )
        assert result.timed_out
        assert not result.success
        assert "timed out" in result.stderr.lower()
