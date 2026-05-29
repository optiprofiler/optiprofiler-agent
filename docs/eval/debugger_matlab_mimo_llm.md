# Debugger Eval Last Run

- Timestamp: `2026-05-29T02:05:31+00:00`
- Agent: `debugger`
- Strategy: `llm`
- Language: `matlab`
- Provider: `mimo`
- Model: `mimo-v2-flash`
- Cases: `15/15`
- Pass@1: `100.0%`

| Case | Status | Classification | Category ok | Fix proposed | Time (s) |
|---|---:|---|---:|---:|---:|
| `bad_bounds_shape` | PASS | `interface_mismatch` | no | yes | 17.54 |
| `bad_field_access` | PASS | `runtime_error` | yes | yes | 15.74 |
| `dimension_mismatch` | PASS | `runtime_error` | yes | yes | 16.91 |
| `iface_reorder` | PASS | `interface_mismatch` | yes | yes | 14.06 |
| `index_oob` | PASS | `runtime_error` | yes | yes | 14.0 |
| `key_missing_struct` | PASS | `runtime_error` | yes | yes | 14.12 |
| `nan_objective` | PASS | `numerical` | yes | yes | 14.99 |
| `negative_sqrt_objective` | PASS | `numerical` | yes | yes | 15.4 |
| `not_enough_inputs` | PASS | `interface_mismatch` | yes | yes | 15.21 |
| `timeout_pause` | PASS | `timeout` | yes | yes | 53.45 |
| `too_many_inputs` | PASS | `interface_mismatch` | yes | yes | 15.51 |
| `unbalanced_parens` | PASS | `interface_mismatch` | no | yes | 15.36 |
| `undefined_function` | PASS | `dependency_missing` | yes | yes | 14.17 |
| `undefined_variable` | PASS | `dependency_missing` | yes | yes | 16.58 |
| `zero_division_objective` | PASS | `numerical` | yes | yes | 16.28 |
