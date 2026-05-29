# Debugger Eval Last Run

- Timestamp: `2026-05-29T01:54:42+00:00`
- Agent: `debugger`
- Strategy: `llm`
- Language: `matlab`
- Provider: `kimi`
- Model: `kimi-k2.5`
- Cases: `15/15`
- Pass@1: `100.0%`

| Case | Status | Classification | Category ok | Fix proposed | Time (s) |
|---|---:|---|---:|---:|---:|
| `bad_bounds_shape` | PASS | `interface_mismatch` | no | yes | 23.84 |
| `bad_field_access` | PASS | `runtime_error` | yes | yes | 16.13 |
| `dimension_mismatch` | PASS | `runtime_error` | yes | yes | 15.22 |
| `iface_reorder` | PASS | `interface_mismatch` | yes | yes | 14.17 |
| `index_oob` | PASS | `runtime_error` | yes | yes | 27.53 |
| `key_missing_struct` | PASS | `runtime_error` | yes | yes | 17.05 |
| `nan_objective` | PASS | `numerical` | yes | yes | 16.47 |
| `negative_sqrt_objective` | PASS | `numerical` | yes | yes | 14.12 |
| `not_enough_inputs` | PASS | `interface_mismatch` | yes | yes | 14.92 |
| `timeout_pause` | PASS | `timeout` | yes | yes | 52.66 |
| `too_many_inputs` | PASS | `interface_mismatch` | yes | yes | 14.61 |
| `unbalanced_parens` | PASS | `runtime_error` | yes | yes | 15.81 |
| `undefined_function` | PASS | `dependency_missing` | yes | yes | 15.1 |
| `undefined_variable` | PASS | `dependency_missing` | yes | yes | 15.68 |
| `zero_division_objective` | PASS | `numerical` | yes | yes | 14.4 |
