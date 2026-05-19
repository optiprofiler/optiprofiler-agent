# OptiProfiler Agent Evaluation Suite

- Timestamp: `2026-05-19T01:11:33+00:00`
- Provider: `minimax`
- Judge provider: `n/a`
- Overall status: `PASS`
- Output directory: `/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest_no_matlab`

## Thresholds

| Metric | Required | Actual |
|---|---:|---:|
| `min_pass_rate` | >= 90.0% | 100.0% |
| `min_avg_score` | >= 0.750 | 1.000 |
| `min_judge_coverage` | >= 80.0% | n/a |
| `min_hallucination` | >= 0.900 | n/a |

## Suites

| Suite | Kind | Cases | Pass Rate | Avg Score | Judge Coverage | Judge Avg | Hallucination | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `debugger_python_golden` | `debugger` | 15/15 | 100.0% | n/a | n/a | n/a | n/a | PASS |
| `interpreter_report_factcheck` | `interpreter` | 1/1 | 100.0% | 1.000 | n/a | n/a | n/a | PASS |
| `multinode_deterministic` | `multinode` | 21/21 | 100.0% | n/a | n/a | n/a | n/a | PASS |

## Commands

### `debugger_python_golden`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_debugger_eval.py --language python --strategy golden --timeout 30 --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest_no_matlab/debugger_python_golden.json --markdown-output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest_no_matlab/debugger_python_golden.md
```

### `interpreter_report_factcheck`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_interpreter_eval.py --strategy deterministic --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest_no_matlab/interpreter_report.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest_no_matlab/interpreter_report.md
```

### `multinode_deterministic`

```bash
/Users/huangcunxin/Work/Research/OP/optiprofiler-agent/.venv/bin/python scripts/run_multinode_eval.py --output /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest_no_matlab/multinode.json --report /Users/huangcunxin/Work/Research/OP/optiprofiler-agent/docs/eval/latest_no_matlab/multinode.md
```
