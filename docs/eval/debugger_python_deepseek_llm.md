# Debugger Eval Last Run

- Timestamp: `2026-05-28T16:03:31+00:00`
- Agent: `debugger`
- Strategy: `llm`
- Language: `python`
- Provider: `deepseek`
- Model: `deepseek-v4-flash`
- Cases: `15/15`
- Pass@1: `100.0%`

| Case | Status | Classification | Category ok | Fix proposed | Time (s) |
|---|---:|---|---:|---:|---:|
| `attribute_error_result` | PASS | `runtime_error` | yes | yes | 0.06 |
| `bad_bounds_shape` | PASS | `runtime_error` | yes | yes | 0.07 |
| `bad_x0_type` | PASS | `runtime_error` | yes | yes | 0.07 |
| `iface_missing_x0` | PASS | `interface_mismatch` | yes | yes | 0.06 |
| `iface_unexpected_keyword` | PASS | `interface_mismatch` | yes | yes | 0.07 |
| `index_oob` | PASS | `runtime_error` | yes | yes | 0.08 |
| `inf_objective` | PASS | `numerical` | yes | yes | 0.08 |
| `key_error` | PASS | `runtime_error` | yes | yes | 0.08 |
| `missing_dependency` | PASS | `dependency_missing` | yes | yes | 0.08 |
| `name_error` | PASS | `runtime_error` | yes | yes | 0.08 |
| `nan_objective` | PASS | `numerical` | yes | yes | 0.09 |
| `pickle_lambda` | PASS | `runtime_error` | yes | yes | 0.09 |
| `syntax_error` | PASS | `runtime_error` | yes | yes | 0.07 |
| `timeout_loop` | PASS | `timeout` | yes | yes | 31.47 |
| `zero_division_objective` | PASS | `numerical` | yes | yes | 0.07 |
