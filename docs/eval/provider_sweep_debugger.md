# Debugger Provider Sweep

- Timestamp: `2026-05-29T02:16:08+00:00`
- Strategy: `llm`
- Case set: Python 15 fixtures + MATLAB 15 fixtures
- Overall status: `PASS`

| Language | Provider | Model | Pass@1 | Artifact |
|---|---|---|---:|---|
| `matlab` | `deepseek` | `deepseek-v4-flash` | 15/15 (100.0%) | `docs/eval/debugger_matlab_deepseek_llm.json` |
| `matlab` | `kimi` | `kimi-k2.5` | 15/15 (100.0%) | `docs/eval/debugger_matlab_kimi_llm.json` |
| `matlab` | `mimo` | `mimo-v2-flash` | 15/15 (100.0%) | `docs/eval/debugger_matlab_mimo_llm.json` |
| `matlab` | `minimax` | `MiniMax-M2.7` | 15/15 (100.0%) | `docs/eval/debugger_matlab_minimax_llm.json` |
| `python` | `deepseek` | `deepseek-v4-flash` | 15/15 (100.0%) | `docs/eval/debugger_python_deepseek_llm.json` |
| `python` | `kimi` | `kimi-k2.5` | 15/15 (100.0%) | `docs/eval/debugger_python_kimi_llm.json` |
| `python` | `mimo` | `mimo-v2-flash` | 15/15 (100.0%) | `docs/eval/debugger_python_mimo_llm.json` |
| `python` | `minimax` | `MiniMax-M2.7` | 15/15 (100.0%) | `docs/eval/debugger_python_minimax_llm.json` |
