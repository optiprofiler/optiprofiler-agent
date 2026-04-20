#!/usr/bin/env python
"""Interactive chat with Agent A (Product Advisor).

Usage::

    python3.11 scripts/chat.py
    python3.11 scripts/chat.py --provider kimi
    python3.11 scripts/chat.py --provider minimax --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markdown import Markdown

from optiprofiler_agent.config import AgentConfig, LLMConfig
from optiprofiler_agent.advisor.advisor import AdvisorAgent

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Chat with OptiProfiler Advisor")
    parser.add_argument("--provider", default="minimax", help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name (optional)")
    parser.add_argument("--verbose", action="store_true", help="Show system prompt")
    args = parser.parse_args()

    config = AgentConfig(llm=LLMConfig(provider=args.provider, model=args.model))
    agent = AdvisorAgent(config)

    console.print(
        f"[bold green]OptiProfiler Advisor[/] "
        f"(provider={config.llm.provider}, model={config.llm.model})"
    )
    console.print("Type your question. Commands: /reset, /prompt, /quit\n")

    if args.verbose:
        console.print("[dim]System prompt loaded "
                      f"({len(agent.system_prompt)} chars)[/]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("Bye!")
            break

        if user_input.lower() == "/reset":
            agent.reset()
            console.print("[dim]Conversation history cleared.[/]\n")
            continue

        if user_input.lower() == "/prompt":
            console.print(Markdown(agent.system_prompt))
            continue

        try:
            reply = agent.chat(user_input)
            console.print()
            console.print(Markdown(reply))
            console.print()
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}\n")


if __name__ == "__main__":
    main()
