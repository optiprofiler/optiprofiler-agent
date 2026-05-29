#!/usr/bin/env python
"""Smoke test Interpreter constrained decoding.

The local smoke verifies the request payload shape and schema parsing without
calling a model. If OPAGENT_CUSTOM_{BASE_URL,MODEL,API_KEY} are configured,
the script can also run one real self-hosted vLLM smoke through
``opagent interpret``'s constrained path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optiprofiler_agent.config import AgentConfig, LLMConfig
from optiprofiler_agent.interpreter.constraint_backend import VLLMJSONSchemaBackend
from optiprofiler_agent.interpreter.interpreter import _try_constrained_output, interpret
from optiprofiler_agent.interpreter.report_schema import BenchmarkReport


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "tests" / "fixtures" / "matlab_results" / "experiment_matlab"


def _valid_report_json() -> str:
    return json.dumps({
        "schema_version": "1.0",
        "key_findings": ["solver_a wins"],
        "overview": {
            "headline": "solver_a leads.",
            "setup": "Two solvers, dim 1-5.",
        },
        "performance_profile": {
            "winner_at_tau1": "solver_a",
            "most_robust": "solver_a",
            "ranking_change": "Stable.",
        },
        "data_profile": {
            "most_efficient": "solver_a",
            "commentary": "solver_a reaches targets faster.",
        },
        "convergence_issues": {
            "entries": [],
            "common_failure_problems": [],
        },
        "anomalies": {"entries": []},
        "recommendations": {"actions": [], "caveats": ""},
    })


class _FakeBoundLLM:
    def invoke(self, _messages):
        return SimpleNamespace(content=_valid_report_json())


class _FakeLLM:
    def __init__(self):
        self.kwargs = None

    def bind(self, **kwargs):
        self.kwargs = kwargs
        return _FakeBoundLLM()


def _local_smoke() -> dict:
    fake = _FakeLLM()
    backend = VLLMJSONSchemaBackend()
    report = _try_constrained_output(fake, [], backend)
    expected_extra_body = {
        "structured_outputs": {
            "json": BenchmarkReport.model_json_schema(),
        },
    }
    actual_extra_body = (fake.kwargs or {}).get("extra_body")
    return {
        "status": "pass" if report and actual_extra_body == expected_extra_body else "fail",
        "backend": backend.name,
        "schema_bound": actual_extra_body == expected_extra_body,
        "parsed_report": isinstance(report, BenchmarkReport),
    }


def _real_smoke(results_dir: Path) -> dict:
    cfg = LLMConfig(provider="custom", constrained_decoding=True)
    missing = []
    if not cfg.base_url:
        missing.append("OPAGENT_CUSTOM_BASE_URL")
    if not cfg.model:
        missing.append("OPAGENT_CUSTOM_MODEL")
    if not cfg.api_key:
        missing.append("OPAGENT_CUSTOM_API_KEY")
    if missing:
        return {
            "status": "blocked",
            "reason": "custom vLLM endpoint is not configured",
            "missing": missing,
        }
    if not results_dir.exists():
        return {
            "status": "blocked",
            "reason": f"results_dir does not exist: {results_dir}",
        }

    t0 = time.perf_counter()
    try:
        rendered = interpret(
            results_dir=results_dir,
            config=AgentConfig(llm=cfg),
            language="English",
            read_profiles=False,
            llm_enabled=True,
            output_format="json",
        )
        BenchmarkReport.model_validate_json(rendered)
    except Exception as exc:  # noqa: BLE001 - smoke records provider/runtime errors.
        return {
            "status": "fail",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "pass",
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output", default="docs/eval/constrained_decoding_smoke.json")
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Only run local payload/schema smoke, even if custom endpoint env is configured.",
    )
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Return non-zero when the real endpoint smoke is blocked or fails.",
    )
    args = parser.parse_args()

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "agent": "interpreter",
        "target": "BenchmarkReport constrained decoding",
        "local": _local_smoke(),
        "real": {"status": "skipped", "reason": "--skip-real"} if args.skip_real else _real_smoke(Path(args.results_dir)),
    }
    payload["passed"] = payload["local"]["status"] == "pass" and payload["real"]["status"] in {"pass", "blocked", "skipped"}

    out = (REPO_ROOT / args.output).resolve()
    _write_json(out, payload)
    print(f"[constrained_smoke] local={payload['local']['status']} real={payload['real']['status']}")
    print(f"[constrained_smoke] wrote {out}")

    if payload["local"]["status"] != "pass":
        return 1
    if args.require_real and payload["real"]["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
