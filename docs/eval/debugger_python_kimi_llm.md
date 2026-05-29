# Debugger Eval Last Run

- Timestamp: `2026-05-28T16:09:23+00:00`
- Agent: `debugger`
- Strategy: `llm`
- Language: `python`
- Provider: `kimi`
- Model: `kimi-k2.5`
- Cases: `15/15`
- Pass@1: `100.0%`

| Case | Status | Classification | Category ok | Fix proposed | Time (s) |
|---|---:|---|---:|---:|---:|
| `attribute_error_result` | PASS | `runtime_error` | yes | yes | 0.06 |
| `bad_bounds_shape` | PASS | `runtime_error` | yes | yes | 0.07 |
| `bad_x0_type` | PASS | `runtime_error` | yes | yes | 0.08 |
| `iface_missing_x0` | PASS | `interface_mismatch` | yes | yes | 0.08 |
| `iface_unexpected_keyword` | PASS | `interface_mismatch` | yes | yes | 0.08 |
| `index_oob` | PASS | `runtime_error` | yes | yes | 0.08 |
| `inf_objective` | PASS | `numerical` | yes | yes | 0.08 |
| `key_error` | PASS | `runtime_error` | yes | yes | 0.08 |
| `missing_dependency` | PASS | `dependency_missing` | yes | yes | 0.85 |
| `name_error` | PASS | `runtime_error` | yes | yes | 0.08 |
| `nan_objective` | PASS | `numerical` | yes | yes | 0.08 |
| `pickle_lambda` | PASS | `runtime_error` | yes | yes | 0.08 |
| `syntax_error` | PASS | `runtime_error` | yes | yes | 0.08 |
| `timeout_loop` | PASS | `timeout` | yes | yes | 31.47 |
| `zero_division_objective` | PASS | `runtime_error` | yes | yes | 0.14 |
