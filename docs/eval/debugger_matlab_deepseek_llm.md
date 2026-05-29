# Debugger Eval Last Run

- Timestamp: `2026-05-28T16:26:30+00:00`
- Agent: `debugger`
- Strategy: `llm`
- Language: `matlab`
- Provider: `deepseek`
- Model: `deepseek-v4-flash`
- Cases: `15/15`
- Pass@1: `100.0%`

| Case | Status | Classification | Category ok | Fix proposed | Time (s) |
|---|---:|---|---:|---:|---:|
| `bad_bounds_shape` | PASS | `runtime_error` | yes | yes | 17.51 |
| `bad_field_access` | PASS | `runtime_error` | yes | yes | 17.01 |
| `dimension_mismatch` | PASS | `runtime_error` | yes | yes | 17.2 |
| `iface_reorder` | PASS | `interface_mismatch` | yes | yes | 18.63 |
| `index_oob` | PASS | `runtime_error` | yes | yes | 17.58 |
| `key_missing_struct` | PASS | `runtime_error` | yes | yes | 17.73 |
| `nan_objective` | PASS | `numerical` | yes | yes | 19.37 |
| `negative_sqrt_objective` | PASS | `numerical` | yes | yes | 19.78 |
| `not_enough_inputs` | PASS | `interface_mismatch` | yes | yes | 17.29 |
| `timeout_pause` | PASS | `timeout` | yes | yes | 55.8 |
| `too_many_inputs` | PASS | `interface_mismatch` | yes | yes | 17.77 |
| `unbalanced_parens` | PASS | `runtime_error` | yes | yes | 17.1 |
| `undefined_function` | PASS | `dependency_missing` | yes | yes | 18.36 |
| `undefined_variable` | PASS | `dependency_missing` | yes | yes | 16.64 |
| `zero_division_objective` | PASS | `numerical` | yes | yes | 19.56 |
