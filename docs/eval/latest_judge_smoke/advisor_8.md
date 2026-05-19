# OptiProfiler Agent Evaluation Report

- **Date**: 2026-05-19 01:31 UTC
- **Mode**: advisor
- **Provider**: minimax
- **Model**: MiniMax-M2.7
- **Cases**: 8

## Summary

| Metric | Value |
|--------|-------|
| Average Score | **0.932** |
| Pass Rate (>=0.5) | **8/8** (100%) |
| Judge Average | **0.970** |
| Judge Accuracy | **0.975** |
| Judge Completeness | **0.963** |
| Judge Code Quality | **0.912** |
| Judge Hallucination | **1.000** |
| Judge Instruction Following | **1.000** |

## Per-Category Breakdown

| Category | Cases | Avg Score | Pass Rate |
|----------|-------|-----------|-----------|
| adversarial | 2 | 0.98 | 2/2 (100%) |
| code_generation | 2 | 0.97 | 2/2 (100%) |
| factual_query | 2 | 0.87 | 2/2 (100%) |
| tool_routing | 2 | 0.90 | 2/2 (100%) |

## Detailed Results

| ID | Category | KW | Code | Tool | Judge | Combined | Time |
|----|----------|----|------|------|-------|----------|------|
| + a01 | adversarial | 1.00 | 1.00 | — | 0.98 | **0.99** | 20.7s |
| + a05 | adversarial | 1.00 | 1.00 | — | 0.96 | **0.98** | 7.6s |
| + cg01 | code_generation | 1.00 | 1.00 | — | 1.00 | **1.00** | 6.6s |
| + cg02 | code_generation | 1.00 | 1.00 | — | 0.90 | **0.95** | 12.5s |
| + f01 | factual_query | 1.00 | 1.00 | — | 0.96 | **0.98** | 29.0s |
| + f05 | factual_query | 0.25 | 1.00 | — | 0.96 | **0.76** | 8.7s |
| + tr02 | tool_routing | 0.67 | 1.00 | — | 1.00 | **0.90** | 5.0s |
| + tr05 | tool_routing | 0.67 | 1.00 | — | 1.00 | **0.90** | 6.1s |
