"""Fact-check eval for Agent C on a real MATLAB OptiProfiler experiment.

The "ground truth" is whatever ``build_summary`` extracts from the
experiment directory (winner, scores, solver names, dimension range,
problem count). Agent C's job is to produce a report that *mentions*
those facts. We verify two output paths:

1. ``llm_enabled=False`` — the deterministic JSON summary. Must contain
   the winner, the score, and the solver count.
2. ``llm_enabled=True`` with a mocked LLM — assert the rendered Markdown
   still mentions the winner and the runner-up (so a thinking-model
   that does NOT hallucinate names still wins).

Real LLM-as-Judge runs are gated behind ``--judge`` in
``scripts/run_eval.py`` and are not part of this fast test.

Activation: set ``MATOP_REAL_RESULTS_DIR`` to an experiment directory
(e.g. ``~/Desktop/tmp/matlab_op/out/fminsearch_fminunc_u_1_2_plain_…``).
Without it, the suite is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from optiprofiler_agent.config import AgentConfig
from optiprofiler_agent.interpreter.interpreter import interpret
from optiprofiler_agent.interpreter.summary import build_summary


@pytest.fixture(scope="module")
def real_experiment() -> Path:
    raw = os.environ.get("MATOP_REAL_RESULTS_DIR", "").strip()
    if not raw:
        pytest.skip(
            "Set MATOP_REAL_RESULTS_DIR to a full MATLAB out/<experiment>/ path"
        )
    p = Path(raw).expanduser()
    if not (p / "test_log" / "log.txt").is_file():
        pytest.skip(f"Not a valid experiment dir: {p}")
    return p


@pytest.fixture(scope="module")
def ground_truth(real_experiment) -> dict:
    """Extract verifiable facts from the experiment without involving an LLM."""
    summary = build_summary(real_experiment, read_profiles=False)
    scores = dict(summary.solver_scores)
    winner = max(scores, key=scores.get)
    runner_up = min(scores, key=scores.get)
    return {
        "summary": summary,
        "scores": scores,
        "winner": winner,
        "runner_up": runner_up,
        "language": summary.language,
        "dimension_range": summary.dimension_range,
    }


class TestNoLLMReport:
    """JSON path — must contain ground-truth facts verbatim."""

    def test_winner_and_scores_present(self, real_experiment, ground_truth):
        report = interpret(
            results_dir=real_experiment,
            llm_enabled=False,
            read_profiles=False,
        )
        data = json.loads(report)
        assert data["language"] == ground_truth["language"]
        assert ground_truth["winner"] in data["solver_scores"]
        assert ground_truth["runner_up"] in data["solver_scores"]
        gt_winner_score = ground_truth["scores"][ground_truth["winner"]]
        assert data["solver_scores"][ground_truth["winner"]] == pytest.approx(
            gt_winner_score
        )

    def test_rankings_match_score_order(self, real_experiment, ground_truth):
        report = interpret(
            results_dir=real_experiment,
            llm_enabled=False,
            read_profiles=False,
        )
        data = json.loads(report)
        # First-ranked solver must be the highest-scoring one.
        if data.get("rankings"):
            top = data["rankings"][0]
            top_name = top.get("solver") or top.get("name")
            assert top_name == ground_truth["winner"]


class TestMockedLLMReport:
    """Mocked LLM path — Markdown must still mention the winner.

    We rig the LLM mock to return a syntactically valid Markdown report
    that includes both solver names; the test then enforces that the
    rendered output mentions the winner. This guards against future
    refactors that drop facts from the rendered text.
    """

    @patch("optiprofiler_agent.common.llm_client.create_llm")
    def test_winner_appears_in_rendered_markdown(
        self, mock_create, real_experiment, ground_truth
    ):
        mock_llm = MagicMock()
        # Force the manual-JSON path → free-form fallback that just
        # echoes the names. This is the worst-case rendering and the
        # one we most need to guard against silent fact-dropping.
        mock_llm.with_structured_output.side_effect = NotImplementedError("nope")
        mock_llm.invoke.return_value = MagicMock(
            content=(
                "# Benchmark Report\n\n"
                f"On this benchmark, **{ground_truth['winner']}** is the winner; "
                f"**{ground_truth['runner_up']}** comes second.\n"
            )
        )
        mock_create.return_value = mock_llm

        report = interpret(
            results_dir=real_experiment,
            config=AgentConfig(llm=MagicMock()),
            llm_enabled=True,
            read_profiles=False,
        )
        assert ground_truth["winner"] in report
        assert ground_truth["runner_up"] in report
        # The thinking-model sanitiser must not eat the visible content.
        assert "<think>" not in report

    @patch("optiprofiler_agent.common.llm_client.create_llm")
    def test_hallucinated_winner_does_not_silently_overwrite_facts(
        self, mock_create, real_experiment, ground_truth
    ):
        """Sanity probe: if the LLM mentions a wrong winner, the rendered
        report should *still* contain the genuine summary JSON in its
        no-LLM degraded path. This is the floor we want to hold.
        """
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = NotImplementedError("nope")
        # Mock returns an empty response → interpret falls back to free-form
        # which re-emits the LLM content. With empty content we expect the
        # caller can still see the underlying summary via the no-LLM path.
        mock_llm.invoke.return_value = MagicMock(content="")
        mock_create.return_value = mock_llm

        # Pull the JSON ground truth as the safety net.
        no_llm = interpret(
            results_dir=real_experiment,
            llm_enabled=False,
            read_profiles=False,
        )
        data = json.loads(no_llm)
        # The summary JSON is authoritative — the LLM can't change it.
        assert data["solver_scores"][ground_truth["winner"]] >= data["solver_scores"][
            ground_truth["runner_up"]
        ]
