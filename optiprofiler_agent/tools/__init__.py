"""Optional agent tools that live outside the core ``unified_agent`` module.

Each tool is responsible for graceful degradation when its backing
service is unavailable (missing dependency, missing API key, network
failure) so that the unified agent boots and runs even when the tool is
not configured.
"""
