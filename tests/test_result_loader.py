"""Tests for Agent C's result_loader."""

import pytest
from pathlib import Path

OP_TEST_OUT = Path(__file__).resolve().parent.parent / "op_test" / "out"


def _has_test_data():
    return OP_TEST_OUT.exists() and any(OP_TEST_OUT.iterdir())


@pytest.mark.skipif(not _has_test_data(), reason="op_test/out not available")
class TestResultLoader:

    def test_find_latest_experiment(self):
        from optiprofiler_agent.agent_c.result_loader import find_latest_experiment

        latest = find_latest_experiment(str(OP_TEST_OUT))
        assert latest.exists()

    def test_load_results(self):
        from optiprofiler_agent.agent_c.result_loader import (
            find_latest_experiment,
            load_results,
        )

        latest = find_latest_experiment(str(OP_TEST_OUT))
        results = load_results(str(latest))
        assert results is not None
        assert results.config is not None
        assert len(results.solver_scores) > 0

    def test_detect_language(self):
        from optiprofiler_agent.agent_c.result_loader import (
            find_latest_experiment,
            load_results,
        )

        latest = find_latest_experiment(str(OP_TEST_OUT))
        results = load_results(str(latest))
        assert results.language in ("python", "matlab")
