#!/usr/bin/env python
"""Quick smoke test — verify that we can reach the configured LLM provider.

Usage::

    # Test with Kimi (default)
    python scripts/test_llm_connection.py

    # Test with MiniMax
    python scripts/test_llm_connection.py --provider minimax

    # Test with a specific model
    python scripts/test_llm_connection.py --provider kimi --model moonshot-v1-8k
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optiprofiler_agent.config import LLMConfig
from optiprofiler_agent.common.llm_client import create_llm


def main():
    parser = argparse.ArgumentParser(description="Test LLM connection")
    parser.add_argument("--provider", default="minimax", help="LLM provider name")
    parser.add_argument("--model", default=None, help="Model name (optional)")
    args = parser.parse_args()

    cfg = LLMConfig(provider=args.provider, model=args.model)

    print(f"Provider : {cfg.provider}")
    print(f"Model    : {cfg.model}")
    print(f"Base URL : {cfg.base_url or '(default)'}")
    print(f"API Key  : {'***' + cfg.api_key[-4:] if cfg.api_key else '(NOT SET)'}")
    print()

    if not cfg.api_key:
        print(f"ERROR: API key not found. Set {cfg.provider.upper()}_API_KEY in .env")
        sys.exit(1)

    llm = create_llm(cfg)

    print("Sending test message: 'Hello, please reply with exactly: Connection OK'")
    print("-" * 60)

    t0 = time.time()
    response = llm.invoke("Hello, please reply with exactly: Connection OK")
    elapsed = time.time() - t0

    print(f"Response : {response.content}")
    print(f"Latency  : {elapsed:.2f}s")
    print(f"Tokens   : {response.response_metadata.get('token_usage', 'N/A')}")
    print("-" * 60)
    print("SUCCESS — LLM connection is working.")


if __name__ == "__main__":
    main()
