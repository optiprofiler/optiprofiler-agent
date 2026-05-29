"""Tests for the constrained-decoding smoke runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_constrained_decoding_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_constrained_decoding_smoke", SCRIPT_PATH)
smoke = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(smoke)


def test_local_smoke_binds_schema_and_parses_report():
    result = smoke._local_smoke()
    assert result["status"] == "pass"
    assert result["schema_bound"] is True
    assert result["parsed_report"] is True


def test_real_smoke_reports_blocked_without_custom_endpoint(monkeypatch):
    for key in (
        "OPAGENT_CUSTOM_BASE_URL",
        "OPAGENT_CUSTOM_MODEL",
        "OPAGENT_CUSTOM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    result = smoke._real_smoke(Path("/tmp/does-not-matter"))
    assert result["status"] == "blocked"
    assert "OPAGENT_CUSTOM_BASE_URL" in result["missing"]
