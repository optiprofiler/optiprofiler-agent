"""Tests for Agent B's local runner."""

import pytest


class TestLocalRunner:

    def test_successful_script(self):
        from optiprofiler_agent.agent_b.local_runner import run_script

        result = run_script("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout
        assert result.exit_code == 0

    def test_failing_script(self):
        from optiprofiler_agent.agent_b.local_runner import run_script

        result = run_script("raise ValueError('test error')")
        assert not result.success
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    def test_traceback_extraction(self):
        from optiprofiler_agent.agent_b.local_runner import run_script

        result = run_script("1/0")
        assert not result.success
        tb = result.traceback
        assert "ZeroDivisionError" in tb
        assert "Traceback" in tb

    def test_timeout(self):
        from optiprofiler_agent.agent_b.local_runner import run_script

        result = run_script("import time; time.sleep(60)", timeout=2)
        assert result.timed_out
        assert not result.success

    def test_syntax_error(self):
        from optiprofiler_agent.agent_b.local_runner import run_script

        result = run_script("def foo(:\n  pass")
        assert not result.success
        assert "SyntaxError" in result.stderr

    def test_timeout_kills_child_processes(self):
        """Verify that child processes spawned by the script are also killed."""
        import subprocess
        from optiprofiler_agent.agent_b.local_runner import run_script

        code = (
            "import multiprocessing, time\n"
            "def worker():\n"
            "    time.sleep(300)\n"
            "if __name__ == '__main__':\n"
            "    procs = [multiprocessing.Process(target=worker) for _ in range(3)]\n"
            "    for p in procs: p.start()\n"
            "    time.sleep(300)\n"
        )
        result = run_script(code, timeout=3)
        assert result.timed_out

        import time
        time.sleep(1)

        ps = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, check=False,
        )
        zombie_workers = [
            line for line in ps.stdout.splitlines()
            if "time.sleep(300)" in line and "grep" not in line
        ]
        assert len(zombie_workers) == 0, f"Leftover child processes: {zombie_workers}"
