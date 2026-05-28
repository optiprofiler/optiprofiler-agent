# Debugger Eval Last Run

- Timestamp: `2026-05-22T02:02:44+00:00`
- Agent: `debugger`
- Strategy: `llm`
- Language: `matlab`
- Provider: `minimax`
- Model: `MiniMax-M2.7`
- Cases: `15/15`
- Pass@1: `100.0%`

| Case | Status | Classification | Category ok | Fix proposed | Time (s) |
|---|---:|---|---:|---:|---:|
| `bad_bounds_shape` | PASS | `runtime_error` | yes | yes | 15.46 |
| `bad_field_access` | PASS | `runtime_error` | yes | yes | 14.76 |
| `dimension_mismatch` | PASS | `runtime_error` | yes | yes | 14.68 |
| `iface_reorder` | PASS | `runtime_error` | yes | yes | 18.21 |
| `index_oob` | PASS | `runtime_error` | yes | yes | 14.91 |
| `key_missing_struct` | PASS | `runtime_error` | yes | yes | 14.87 |
| `nan_objective` | PASS | `numerical` | yes | yes | 14.17 |
| `negative_sqrt_objective` | PASS | `numerical` | yes | yes | 16.93 |
| `not_enough_inputs` | PASS | `interface_mismatch` | yes | yes | 17.97 |
| `timeout_pause` | PASS | `timeout` | yes | yes | 52.79 |
| `too_many_inputs` | PASS | `interface_mismatch` | yes | yes | 16.02 |
| `unbalanced_parens` | PASS | `runtime_error` | yes | yes | 15.22 |
| `undefined_function` | PASS | `dependency_missing` | yes | yes | 15.01 |
| `undefined_variable` | PASS | `dependency_missing` | yes | yes | 14.65 |
| `zero_division_objective` | PASS | `numerical` | yes | yes | 16.79 |
