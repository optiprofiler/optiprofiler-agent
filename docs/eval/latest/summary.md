# OptiProfiler Agent Evaluation Suite

- Timestamp: `2026-05-21T04:20:19+00:00`
- Provider: `minimax`
- Judge provider: `minimax`
- Overall status: `PASS`
- Output directory: `/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest`

## Thresholds

| Metric | Required | Actual |
|---|---:|---:|
| `min_pass_rate` | >= 90.0% | 100.0% |
| `min_avg_score` | >= 0.750 | 0.798 |
| `min_judge_coverage` | >= 80.0% | 100.0% |
| `min_hallucination` | >= 0.900 | 0.967 |

## Suites

| Suite | Kind | Cases | Pass Rate | Avg Score | Judge Coverage | Judge Avg | Hallucination | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `advisor_factual` | `advisor_or_unified` | 3/3 | 100.0% | 0.973 | 100.0% | 0.947 | 1.000 | PASS |
| `advisor_adversarial` | `advisor_or_unified` | 3/3 | 100.0% | 0.923 | 100.0% | 0.980 | 1.000 | PASS |
| `advisor_code_generation` | `advisor_or_unified` | 3/3 | 100.0% | 0.987 | 100.0% | 0.973 | 0.967 | PASS |
| `unified_tool_routing` | `advisor_or_unified` | 3/3 | 100.0% | 0.798 | n/a | n/a | n/a | PASS |
| `interpreter_report_factcheck` | `interpreter` | 1/1 | 100.0% | 1.000 | 100.0% | 1.000 | 1.000 | PASS |
| `multinode_deterministic` | `multinode` | 21/21 | 100.0% | n/a | n/a | n/a | n/a | PASS |

## Issues

- `unified_tool_routing` low-score cases: tr03=0.56

## Commands

### `advisor_factual`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_eval.py --mode advisor --provider minimax --cases tests/eval_cases/factual.json --case-timeout 90 --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/advisor_factual.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/advisor_factual.md --judge --judge-provider minimax --limit 3
```

### `advisor_adversarial`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_eval.py --mode advisor --provider minimax --cases tests/eval_cases/adversarial.json --case-timeout 90 --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/advisor_adversarial.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/advisor_adversarial.md --judge --judge-provider minimax --limit 3
```

### `advisor_code_generation`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_eval.py --mode advisor --provider minimax --cases tests/eval_cases/code_generation.json --case-timeout 90 --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/advisor_code_generation.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/advisor_code_generation.md --judge --judge-provider minimax --limit 3
```

### `unified_tool_routing`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_eval.py --mode unified --provider minimax --cases tests/eval_cases/tool_routing.json --case-timeout 90 --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/unified_tool_routing.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/unified_tool_routing.md --limit 3
```

### `interpreter_report_factcheck`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_interpreter_eval.py --strategy deterministic --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/interpreter_report.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/interpreter_report.md --judge --judge-provider minimax
```

### `multinode_deterministic`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_multinode_eval.py --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/multinode.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest/multinode.md
```
