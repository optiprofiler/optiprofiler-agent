# OptiProfiler Agent Evaluation Report

- **Date**: 2026-05-21 04:20 UTC
- **Mode**: unified
- **Provider**: minimax
- **Model**: MiniMax-M2.7
- **Cases**: 3

## Summary

| Metric | Value |
|--------|-------|
| Average Score | **0.798** |
| Pass Rate (>=0.5) | **3/3** (100%) |
| Tool Routing Accuracy | **2/3** (67%) |

## Per-Category Breakdown

| Category | Cases | Avg Score | Pass Rate |
|----------|-------|-----------|-----------|
| tool_routing | 3 | 0.80 | 3/3 (100%) |

## Detailed Results

| ID | Category | KW | Code | Tool | Combined | Time |
|----|----------|----|------|------|----------|------|
| + tr01 | tool_routing | 1.00 | 1.00 | PASS | **1.00** | 13.6s |
| + tr02 | tool_routing | 0.67 | 1.00 | PASS | **0.83** | 15.8s |
| ~ tr03 | tool_routing | 1.00 | 0.20 | MISS | **0.56** | 9.0s |
