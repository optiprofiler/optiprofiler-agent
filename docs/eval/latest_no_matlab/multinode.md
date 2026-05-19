# Multi-Node Agent Evaluation Report

- Timestamp: `2026-05-19T01:11:33+00:00`
- Cases: `21/21`
- Pass rate: `100.0%`

## By Agent

| Agent | Passed | Total | Pass Rate |
|---|---:|---:|---:|
| `debugger` | 15 | 15 | 100.0% |
| `interpreter` | 6 | 6 | 100.0% |

## Cases

| Case | Agent | Task | Status | Time (s) |
|---|---|---|---:|---:|
| `dbg_matlab_dep_undefined` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_dep_unrecognized` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_iface_too_many` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_iface_not_enough` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_runtime_index` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_runtime_dim` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_timeout` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_numerical_nan` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_numerical_inf` | `debugger` | `classify_error` | PASS | 0.0 |
| `dbg_matlab_checker_safe` | `debugger` | `validate_code` | PASS | 0.0 |
| `dbg_matlab_checker_system` | `debugger` | `validate_code` | PASS | 0.0 |
| `dbg_matlab_checker_shell_escape` | `debugger` | `validate_code` | PASS | 0.0 |
| `dbg_matlab_checker_unbalanced` | `debugger` | `validate_code` | PASS | 0.0 |
| `dbg_matlab_iface_reordered` | `debugger` | `analyze_solver` | PASS | 0.0 |
| `dbg_matlab_iface_canonical` | `debugger` | `analyze_solver` | PASS | 0.0 |
| `int_matlab_fixture_load` | `interpreter` | `load_results` | PASS | 0.001 |
| `int_matlab_log_snippet` | `interpreter` | `parse_log_snippet` | PASS | 0.001 |
| `int_matlab_build_summary` | `interpreter` | `build_summary` | PASS | 0.0 |
| `int_matlab_summary_json_roundtrip` | `interpreter` | `summary_json_roundtrip` | PASS | 0.0 |
| `int_matlab_no_pdfs_flag_true` | `interpreter` | `profile_curves_flag` | PASS | 0.0 |
| `int_matlab_dual_run_counter` | `interpreter` | `parse_log_snippet_padded` | PASS | 0.0 |
