"""Regression checks for Advisor prompt rules used by release evals."""

from pathlib import Path


_PROMPTS = Path(__file__).resolve().parents[1] / "optiprofiler_agent" / "advisor" / "prompts"


def test_advisor_prompt_constrains_scipy_minimize_guidance():
    text = (_PROMPTS / "system_prompt.md").read_text(encoding="utf-8")
    assert "scipy.optimize.minimize" in text
    assert "return `result.x`" in text
    assert "Do not provide method-support tables" in text
    assert "Do not add MATLAB examples for SciPy-specific questions" in text


def test_few_shot_uses_valid_python_ptype_and_result_x():
    text = (_PROMPTS / "few_shots.md").read_text(encoding="utf-8")
    assert 'ptype="u"' in text
    assert "ptype=\"unconstrained\"" not in text
    assert "return result.x" in text
