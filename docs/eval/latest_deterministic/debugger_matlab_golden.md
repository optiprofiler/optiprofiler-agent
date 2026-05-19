# Debugger Eval Last Run

- Timestamp: `2026-05-19T01:14:55+00:00`
- Agent: `debugger`
- Strategy: `golden`
- Language: `matlab`
- Provider: `n/a`
- Model: `n/a`
- Cases: `15/15`
- Pass@1: `100.0%`

| Case | Status | Classification | Category ok | Fix proposed | Time (s) |
|---|---:|---|---:|---:|---:|
| `bad_bounds_shape` | PASS | `runtime_error` | yes | - | 17.83 |
| `bad_field_access` | PASS | `runtime_error` | yes | - | 16.85 |
| `dimension_mismatch` | PASS | `runtime_error` | yes | - | 17.83 |
| `iface_reorder` | PASS | `runtime_error` | yes | - | 18.97 |
| `index_oob` | PASS | `runtime_error` | yes | - | 15.04 |
| `key_missing_struct` | PASS | `runtime_error` | yes | - | 17.37 |
| `nan_objective` | PASS | `numerical` | yes | - | 15.74 |
| `negative_sqrt_objective` | PASS | `numerical` | yes | - | 22.87 |
| `not_enough_inputs` | PASS | `interface_mismatch` | yes | - | 16.91 |
| `timeout_pause` | PASS | `timeout` | yes | - | 55.45 |
| `too_many_inputs` | PASS | `interface_mismatch` | yes | - | 16.14 |
| `unbalanced_parens` | PASS | `runtime_error` | yes | - | 19.42 |
| `undefined_function` | PASS | `dependency_missing` | yes | - | 16.23 |
| `undefined_variable` | PASS | `dependency_missing` | yes | - | 20.26 |
| `zero_division_objective` | PASS | `numerical` | yes | - | 16.06 |
