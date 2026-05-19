"""Tests for scripts/run_eval.py scoring and LLM-as-Judge plumbing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_eval import (  # noqa: E402
    _extract_json_object,
    _normalise_judge_score,
    extract_language_code_blocks,
    filter_cases_for_mode,
    generate_report,
    score_code_quality,
    score_with_judge,
    select_cases,
    summarize_results,
    run_eval,
    _write_json_file,
    _write_text_file,
)
from optiprofiler_agent.config import AgentConfig, LLMConfig


class _FakeJudge:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return SimpleNamespace(content=self.content)


def test_extract_json_object_from_fenced_block():
    data = _extract_json_object(
        "```json\n{\"accuracy\": 9, \"reason\": \"ok\"}\n```"
    )
    assert data["accuracy"] == 9
    assert data["reason"] == "ok"


def test_extract_json_object_from_chatty_response():
    data = _extract_json_object(
        "Here is my score:\n{\"accuracy\": 8, \"reason\": \"grounded\"}\nThanks."
    )
    assert data == {"accuracy": 8, "reason": "grounded"}


def test_extract_json_object_uses_balanced_object_not_greedy_tail():
    data = _extract_json_object(
        "Score: {\"accuracy\": 8, \"reason\": \"grounded\"}\n"
        "Trailing note with an unmatched parenthesis and brace-ish text: )"
    )
    assert data == {"accuracy": 8, "reason": "grounded"}


def test_extract_json_object_from_python_literal_style_response():
    data = _extract_json_object(
        "Score: {'accuracy': 9, 'completeness': 8, 'reason': 'ok'}"
    )
    assert data["accuracy"] == 9
    assert data["reason"] == "ok"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (10, 1.0),
        (7, 0.7),
        (0.8, 0.8),
        (-1, 0.0),
        (15, 1.0),
        ("bad", 0.0),
    ],
)
def test_normalise_judge_score(raw, expected):
    assert _normalise_judge_score(raw) == expected


def test_score_with_judge_accepts_fenced_json_and_normalises_scores():
    judge = _FakeJudge(
        """```json
{"accuracy": 9, "completeness": 8, "code_quality": 7, "hallucination": 10,
 "instruction_following": 6, "reason": "Mostly correct."}
```"""
    )
    result = score_with_judge("What is ptype?", "ptype selects problem type.", judge)

    assert judge.calls == 1
    assert result["judge_scores"]["accuracy"] == 0.9
    assert result["judge_scores"]["hallucination"] == 1.0
    assert result["judge_avg"] == 0.8
    assert result["judge_reason"] == "Mostly correct."


def test_score_with_judge_includes_case_contract():
    judge = _FakeJudge(
        '{"accuracy": 10, "completeness": 10, "code_quality": 10, '
        '"hallucination": 10, "instruction_following": 10, "reason": "ok"}'
    )
    score_with_judge(
        "Can I use one solver?",
        "No, at least two solvers are required.",
        judge,
        case={
            "id": "a01",
            "expected_keywords": ["at least 2"],
            "must_not_contain": ["yes"],
            "reference_answer": "`benchmark()` requires at least 2 solvers.",
        },
    )
    user_msg = judge.messages[-1].content
    assert "Evaluation case contract" in user_msg
    assert "must_not_contain" in user_msg
    assert "reference_answer" in user_msg


def test_score_with_judge_returns_error_on_invalid_json():
    result = score_with_judge("Q", "A", _FakeJudge("not json"))
    assert result["judge_scores"] is None
    assert result["judge_avg"] is None
    assert "Judge error" in result["judge_reason"]


def test_score_code_quality_uses_matlab_checker_for_matlab_case():
    response = """```matlab
options.ptype = 'u';
scores = benchmark({@solver1, @solver2}, options);
```"""
    result = score_code_quality(response, {"language": "matlab", "expect_code": True})
    assert result["code_score"] == 1.0
    assert any("matlab_ok" in item for item in result["code_details"])


def test_extract_language_code_blocks_gets_matlab_fence():
    response = "```matlab\nx = 1;\n```"
    assert extract_language_code_blocks(response, "matlab") == ["x = 1;"]


def test_select_cases_filters_by_id_and_limit():
    cases = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert select_cases(cases, ids="c,a", limit=1) == [{"id": "a"}]


def test_select_cases_rejects_unknown_id():
    with pytest.raises(ValueError, match="Unknown case"):
        select_cases([{"id": "a"}], ids="missing")


def test_filter_cases_for_advisor_skips_structured_agent_cases():
    cases = [
        {"id": "qa", "question": "What is ptype?"},
        {"id": "dbg", "agent": "debugger", "task": "classify_error"},
    ]
    assert filter_cases_for_mode(cases, "advisor") == [{"id": "qa", "question": "What is ptype?"}]


def test_summarize_results_includes_judge_dimensions():
    summary = summarize_results([
        {
            "combined_score": 0.8,
            "judge_avg": 0.9,
            "judge_scores": {
                "accuracy": 0.8,
                "completeness": 0.9,
                "code_quality": 1.0,
                "hallucination": 1.0,
                "instruction_following": 0.8,
            },
        }
    ])
    assert summary["avg_score"] == 0.8
    assert summary["judge_avg"] == 0.9
    assert summary["judge_hallucination_avg"] == 1.0


def test_generate_report_includes_judge_summary():
    config = AgentConfig(llm=LLMConfig(provider="minimax", api_key="fake"))
    report = generate_report(
        [
            {
                "id": "f01",
                "category": "factual",
                "keyword_score": 1.0,
                "code_score": 1.0,
                "tool_routing_score": None,
                "combined_score": 0.95,
                "elapsed_s": 0.1,
                "judge_avg": 0.9,
                "judge_scores": {
                    "accuracy": 0.9,
                    "completeness": 0.9,
                    "code_quality": 0.8,
                    "hallucination": 1.0,
                    "instruction_following": 0.9,
                },
            }
        ],
        mode="advisor",
        config=config,
    )
    assert "Judge Average" in report
    assert "Judge Hallucination" in report
    assert "| + f01 | factual | 1.00 | 1.00 | — | 0.90 | **0.95** | 0.1s |" in report


def test_output_writers_create_parent_directories(tmp_path):
    json_path = tmp_path / "nested" / "results.json"
    md_path = tmp_path / "nested" / "report.md"
    _write_json_file(json_path, [{"id": "ok"}])
    _write_text_file(md_path, "# ok\n")
    assert json_path.exists()
    assert md_path.read_text(encoding="utf-8") == "# ok\n"


def test_run_eval_records_case_timeout():
    def slow_runner(question, agent):
        import time
        time.sleep(0.2)
        return "too late", []

    results = run_eval(
        [{"id": "slow", "question": "Q", "expected_keywords": ["answer"]}],
        agent=object(),
        runner=slow_runner,
        case_timeout=0.01,
    )
    assert results[0]["response"].startswith("ERROR: case timed out")
    assert results[0]["combined_score"] < 0.5
