# OptiProfiler Agent Evaluation Report

- **Date**: 2026-05-21 04:18 UTC
- **Mode**: advisor
- **Provider**: minimax
- **Model**: MiniMax-M2.7
- **Cases**: 3

## Summary

| Metric | Value |
|--------|-------|
| Average Score | **0.923** |
| Pass Rate (>=0.5) | **3/3** (100%) |
| Judge Average | **0.980** |
| Judge Accuracy | **1.000** |
| Judge Completeness | **0.967** |
| Judge Code Quality | **0.933** |
| Judge Hallucination | **1.000** |
| Judge Instruction Following | **1.000** |

## Per-Category Breakdown

| Category | Cases | Avg Score | Pass Rate |
|----------|-------|-----------|-----------|
| adversarial | 3 | 0.92 | 3/3 (100%) |

## Detailed Results

| ID | Category | KW | Code | Tool | Judge | Combined | Time |
|----|----------|----|------|------|-------|----------|------|
| + a01 | adversarial | 0.50 | 1.00 | — | 0.96 | **0.83** | 7.0s |
| + a02 | adversarial | 1.00 | 1.00 | — | 0.98 | **0.99** | 3.0s |
| + a03 | adversarial | 0.83 | 1.00 | — | 1.00 | **0.95** | 15.5s |
