# OptiProfiler Agent Evaluation Plan

This document is the evaluation contract for `optiprofiler-agent`.
It complements `TASKS.md` and `ROADMAP.md`: those files track what is
planned; this file defines how we prove the agent is reliable.

## Why We Evaluate More Than Final Answers

Modern agent evaluation treats an agent as a workflow, not a single
chat completion. Current practice across OpenAI Evals/graders,
LangChain/LangSmith agent evals, Ragas agent metrics, and public agent
benchmarks such as AgentBench, ToolBench, and SWE-bench converges on
the same shape:

1. **Outcome quality**: the final answer or artifact must solve the
   user's task.
2. **Trajectory quality**: the agent must choose the right tools, in
   the right order, without unnecessary or unsafe calls.
3. **Grounding**: claims should be traceable to docs, retrieved
   knowledge, execution output, or parsed benchmark facts.
4. **Executable artifacts**: generated code, fixes, and reports should
   run through validators or sandboxes.
5. **Regression gates**: deterministic checks run cheaply on every PR;
   expensive LLM/Judge/provider sweeps run on release branches or
   scheduled jobs.

For OptiProfiler, this matters because a polished final paragraph can
still hide a bad route: the agent may skip validation, hallucinate a
MATLAB API shape, or produce a report whose prose contradicts parsed
solver scores. We therefore evaluate every meaningful node in the
workflow.

## Evaluation Layers

| Layer | What It Measures | Current Tooling | Gate |
|---|---|---|---|
| L0 deterministic units | Parsers, classifiers, validators, wrappers | `pytest tests/` | 100% green |
| L1 structured data | Result loaders and summaries on real-shaped data | `tests/test_result_loader*.py`, `tests/test_interpreter_eval_matlab.py` | Required fields non-empty; facts match fixtures |
| L2 routing/trajectory | Tool selection and deterministic agent tasks | `tests/test_deterministic_agent_eval.py`, `scripts/run_eval.py --mode unified` | Routing >= 95%; deterministic task pass = 100% |
| L3 task success | Debugger Pass@1/Pass@3, interpreter fact success | `scripts/run_debugger_eval.py`, `scripts/run_interpreter_eval.py` | Pass@1 >= 70%, Pass@3 >= 85% on >=15 debugger cases/language; Interpreter fact pass = 100% |
| L4 LLM-as-Judge quality | Correctness, completeness, code quality, groundedness, no hallucination, instruction following | `scripts/run_eval.py --judge`, `scripts/run_interpreter_eval.py --judge`, `scripts/run_eval_suite.py --judge` | Mean judge >= 0.8; hallucination >= 0.9 on release sample; judge coverage >= 0.8 |
| L5 production telemetry | Real user trajectories and post-hoc audits | future platform trace export | Drift alerts, human audit samples |

## Agent-Specific Metrics

### Agent A — Advisor

Node-level checks:

- **Knowledge answer**: keyword/must-contain/must-not-contain.
- **Code generation**: syntax, API validation, language-specific code
  block extraction.
- **Grounding**: answer should align with bundled wiki/API knowledge.
- **LLM-as-Judge**: multidimensional rubric over the final answer,
  with the case contract included in the judge prompt.

Commands:

```bash
python scripts/run_eval.py \
  --cases tests/eval_cases/factual.json \
  --judge --judge-provider minimax \
  --output /tmp/advisor_factual.json \
  --report /tmp/advisor_factual.md
```

### Unified Agent

Node-level checks:

- **Tool route**: expected tool is called (`knowledge_search`,
  `validate_script`, `debug_error`, `interpret_results`, etc.).
- **Tool result use**: final answer should reflect the tool output.
- **No premature refusal**: if a tool exists, the agent should call it
  before claiming a capability is unavailable.

Commands:

```bash
python scripts/run_eval.py \
  --mode unified \
  --cases tests/eval_cases/tool_routing.json \
  --output /tmp/unified_routing.json \
  --report /tmp/unified_routing.md
```

### Agent B — Debugger

Node-level checks:

- **Classifier**: error category and dependency/module extraction.
- **Validator**: generated fixed code passes syntax/API/MATLAB safety
  checks.
- **Sandbox**: `broken.<ext>` fails and proposed or golden `fix.<ext>`
  runs cleanly.
- **Task success**: Pass@1/Pass@3 over curated broken scripts.

Commands:

```bash
python scripts/run_debugger_eval.py --language python --strategy golden
python scripts/run_debugger_eval.py --language matlab --strategy golden
python scripts/run_debugger_eval.py --language python --strategy llm --provider minimax
```

### Agent C — Interpreter

Node-level checks:

- **Loader facts**: language, solver names, scores, run results.
- **Summary facts**: rankings, anomalies, profile-curve availability.
- **Report grounding**: report winner/runner-up and caveats match the
  parsed `BenchmarkSummary`.
- **LLM-as-Judge**: correctness/completeness/grounding/no hallucination
  on generated reports.
- **User-report hygiene**: internal execution-language metadata remains
  available to the parser and evaluator, but is not rendered as a
  user-facing report field.

Commands:

```bash
python scripts/run_interpreter_eval.py \
  --strategy deterministic \
  --output /tmp/interpreter_eval.json \
  --report /tmp/interpreter_eval.md

python scripts/run_interpreter_eval.py \
  --strategy llm \
  --provider minimax \
  --judge --judge-provider minimax \
  --output /tmp/interpreter_eval_judge.json \
  --report /tmp/interpreter_eval_judge.md
```

## Current Implementation

`scripts/run_eval.py` is the Advisor/Unified evaluation runner. It now
supports:

- schema-aware case filtering, so Advisor/Unified modes ignore
  non-question structured cases;
- `--case-ids` and `--limit` for smoke runs;
- language-aware code scoring, including MATLAB fenced blocks;
- robust LLM-as-Judge parsing for plain JSON, fenced JSON, and text
  containing a JSON object;
- judge prompts that include the eval case contract
  (`expected_keywords`, `must_contain`, `must_not_contain`,
  `expect_code`, `expect_tool`);
- JSON and Markdown reports with judge dimension summaries.

`scripts/run_debugger_eval.py` is the Debugger task-success runner. It
supports golden and LLM strategies, Python and MATLAB fixtures, and
Markdown/JSON reporting.

`scripts/run_interpreter_eval.py` is the Interpreter report evaluator.
It fact-checks generated Markdown against `BenchmarkSummary` and can
attach a report-specific LLM-as-Judge score over correctness,
completeness, grounding, and hallucination.

`scripts/run_eval_suite.py` is the release orchestrator. It runs lower
level suites in separate subprocesses with hard timeouts, aggregates
pass rate / average score / judge coverage / low-score cases, and writes
`summary.json` plus `summary.md` under an output directory. This is the
default way to run a release-grade sweep because it isolates provider
hangs and malformed judge responses to one suite.

`tests/test_deterministic_agent_eval.py` is the deterministic
multi-node test suite for structured debugger/interpreter cases.

## Full Evaluation Procedure

### PR Gate

Fast and deterministic:

```bash
pytest tests/test_run_eval_scoring.py tests/test_run_eval_suite.py tests/test_interpreter_eval_runner.py -q
pytest tests/test_deterministic_agent_eval.py tests/test_multinode_eval.py -q
pytest tests/test_debugger_eval_python.py -q
pytest tests/ -m "not requires_matlab" -q
```

### Local Release Gate

Deterministic release gate:

```bash
export MATOP_MATLAB_BIN=/Applications/MATLAB_R2023b.app/bin/matlab

python scripts/run_eval_suite.py \
  --skip-advisor \
  --output-dir docs/eval/latest_deterministic
```

Provider/Judge release gate:

```bash
python scripts/run_eval_suite.py \
  --provider minimax \
  --judge --judge-provider minimax \
  --include-unified \
  --case-timeout 90 \
  --suite-timeout 900 \
  --output-dir docs/eval/latest
```

If a provider stalls, rerun only the failed suite/case with a different
provider. Do not rerun a full 40-case judge sweep as one unbounded
process.

### Interpreter Constrained Decoding Smoke

`BenchmarkReport` supports an opt-in decode-time JSON Schema path for a
self-hosted vLLM OpenAI-compatible endpoint. API-only providers keep the
existing provider-structured-output and manual-JSON fallbacks.

```bash
opagent interpret /path/to/results \
  --provider custom \
  --constrained-decoding \
  --format json
```

The `custom` provider should point at the vLLM OpenAI-compatible base URL.
The constrained path sends vLLM's `structured_outputs.json` schema hint;
if that endpoint rejects the hint, Interpreter falls back to the normal
structured-output chain and the same report validators still run.

### Provider Sweep

Run the same judge and task-success samples across configured providers:

```bash
for provider in minimax kimi deepseek mimo; do
  python scripts/run_eval.py \
    --provider "$provider" \
    --judge --judge-provider "$provider" \
    --output "docs/eval/advisor_${provider}.json" \
    --report "docs/eval/advisor_${provider}.md"

  python scripts/run_debugger_eval.py \
    --language python --strategy llm --provider "$provider" \
    --output "docs/eval/debugger_python_${provider}.json"
done
```

## Known Gaps

- Unified-agent trajectory scoring currently checks whether the
  expected tool appears; it should next score ordering, redundant tool
  calls, and whether final answers faithfully use tool outputs.
- Release CI should separate cheap deterministic gates from expensive
  provider/Judge sweeps.
- Historical `results.json` used an older judge parser and should not
  be treated as a current L4 result; it had `judge_avg = null` for all
  cases due to parse/provider errors.

## Source Notes

This plan follows current public practice:

- [OpenAI Evals/graders](https://platform.openai.com/docs/guides/graders/):
  model-graded outputs with explicit rubrics, structured score outputs,
  and separate grader validation.
- [OpenAI Evals API](https://platform.openai.com/docs/api-reference/evals):
  dataset-backed eval runs with item/sample namespaces, tool outputs, and
  grader lists.
- [LangChain Agent Evals](https://docs.langchain.com/oss/python/langchain/test/evals):
  deterministic trajectory matching and LLM-as-Judge evaluation for agent
  message/tool-call traces.
- [Ragas agent metrics](https://docs.ragas.io/en/v0.4.1/concepts/metrics/available_metrics/agents/):
  tool-call accuracy/F1 and goal-accuracy style metrics for multi-turn
  agent workflows.
- AgentBench / ToolBench / SWE-bench: benchmark agents on task success,
  tool use, execution, and end-to-end artifacts rather than prose alone.
