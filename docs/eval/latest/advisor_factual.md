# OptiProfiler Agent Evaluation Report

- **Date**: 2026-05-21 04:17 UTC
- **Mode**: advisor
- **Provider**: minimax
- **Model**: MiniMax-M2.7
- **Cases**: 3

## Summary

| Metric | Value |
|--------|-------|
| Average Score | **0.973** |
| Pass Rate (>=0.5) | **3/3** (100%) |
| Judge Average | **0.947** |
| Judge Accuracy | **0.967** |
| Judge Completeness | **0.967** |
| Judge Code Quality | **0.800** |
| Judge Hallucination | **1.000** |
| Judge Instruction Following | **1.000** |

## Per-Category Breakdown

| Category | Cases | Avg Score | Pass Rate |
|----------|-------|-----------|-----------|
| factual_query | 3 | 0.97 | 3/3 (100%) |

## Detailed Results

| ID | Category | KW | Code | Tool | Judge | Combined | Time |
|----|----------|----|------|------|-------|----------|------|
| + f01 | factual_query | 1.00 | 1.00 | — | 0.96 | **0.98** | 10.2s |
| + f02 | factual_query | 1.00 | 1.00 | — | 0.96 | **0.98** | 3.3s |
| + f03 | factual_query | 1.00 | 1.00 | — | 0.92 | **0.96** | 20.0s |
