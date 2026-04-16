"""CLI entry point for optiprofiler-agent (alias: opagent).

Installed via ``pip install optiprofiler-agent``, provides the
``opagent`` / ``optiprofiler-agent`` command with subcommands:

- ``agent`` (default): unified tool-use agent (ReAct)
- ``chat``: interactive conversation with Agent A (Product Advisor)
- ``index``: build/rebuild the RAG vector index
- ``check``: validate a Python script's benchmark() calls
- ``interpret``: analyze benchmark results and generate a report
- ``debug``: diagnose and fix benchmark errors (supports --run mode)

Usage::

    opagent                 # default → agent mode
    opagent chat --provider kimi --rag
    opagent check my_script.py
    opagent interpret /path/to/experiment --latest
    opagent debug script.py --run
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.spinner import SPINNERS

from optiprofiler_agent import __version__
from optiprofiler_agent.config import AgentConfig, LLMConfig, PROVIDER_REGISTRY

console = Console()

# Thinking spinner: one braille cell per frame — Grade-1 o → p → a (⠕ ⠏ ⠁).
SPINNERS["opa"] = {
    "interval": 350,
    "frames": ["⠕", "⠏", "⠁"],
}

# Claude Code–style terminal orange (common CC / Anthropic CLI accent approximation)
_LOGO_OPA_COLOR = "bold #F97316"

_LOGO = (
    f"[{_LOGO_OPA_COLOR}] █▀█ █▀█ █▀█[/]\n"
    f"[{_LOGO_OPA_COLOR}] █ █ █▀▀ █▀█[/]  [bold]Agent for OptiProfiler[/]\n"
    f"[{_LOGO_OPA_COLOR}] ▀▀▀ ▀   ▀ ▀[/]  [dim]v{__version__}[/]"
)

_LLM_DISCLAIMER = (
    "[dim]All responses and actions are LLM-generated and may not match official behavior; "
    "see the documentation at www.optprof.com if in doubt.[/]"
)


def _print_agent_banner(config: AgentConfig) -> None:
    """Startup lines under the ASCII logo: provider + LLM disclaimer."""
    console.print(f"  [dim]LLM Provider:[/] {config.llm.provider} | [dim]Model:[/] {config.llm.model}")
    console.print(f"  {_LLM_DISCLAIMER}")
    console.print("  [dim]Type /help for commands[/]\n")


@click.group(invoke_without_command=True)
@click.version_option(package_name="optiprofiler-agent")
@click.pass_context
def main(ctx):
    """OptiProfiler Agent — AI assistant for optimization benchmarking."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(agent)


@main.command()
@click.option("--provider", default="minimax",
              type=click.Choice(list(PROVIDER_REGISTRY.keys())),
              help="LLM provider to use.")
@click.option("--model", default=None, help="Model name (overrides provider default).")
@click.option("--rag", is_flag=True, default=False, help="Enable RAG retrieval.")
@click.option("--rag-top-k", default=5, type=int, help="Number of RAG chunks to retrieve.")
@click.option("--verbose", is_flag=True, default=False, help="Show system prompt size.")
@click.option("--validate", is_flag=True, default=False,
              help="Validate generated code in responses.")
def chat(provider: str, model: str | None, rag: bool, rag_top_k: int,
         verbose: bool, validate: bool):
    """Interactive chat with the OptiProfiler Product Advisor."""
    from optiprofiler_agent.agent_a.advisor import AdvisorAgent

    config = AgentConfig(
        llm=LLMConfig(provider=provider, model=model),
        rag_enabled=rag,
        rag_top_k=rag_top_k,
        verbose=verbose,
    )

    console.print("[bold green]OptiProfiler Advisor[/]")
    console.print(f"  LLM Provider: {config.llm.provider} | Model: {config.llm.model}")
    console.print(f"  {_LLM_DISCLAIMER}")
    if rag:
        console.print("  RAG: [green]enabled[/]")
    console.print("  Commands: /reset /prompt /quit\n")

    agent = AdvisorAgent(config)

    if verbose:
        console.print(f"[dim]System prompt: {len(agent.system_prompt)} chars[/]\n")

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
            console.print("[dim]History cleared.[/]\n")
            continue

        if user_input.lower() == "/prompt":
            console.print(Markdown(agent.system_prompt))
            continue

        try:
            reply = agent.chat(user_input)
            console.print()
            console.print(Markdown(reply))

            if validate:
                _validate_reply(reply)

            console.print()
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}\n")


def _validate_reply(reply: str):
    """Run syntax + API validation on code blocks in the reply."""
    from optiprofiler_agent.validators.syntax_checker import check_syntax
    from optiprofiler_agent.validators.api_checker import validate_response_code

    syn = check_syntax(reply)
    if syn.has_errors:
        console.print("[bold yellow]⚠ Syntax issues found:[/]")
        for err in syn.errors:
            console.print(f"  Block {err.block_index}, line {err.line}: {err.message}")

    api = validate_response_code(reply)
    if api.has_errors or api.has_warnings:
        console.print("[bold yellow]⚠ API validation issues:[/]")
        for issue in api.issues:
            console.print(f"  [{issue.severity}] {issue.message}")


@main.command()
@click.option("--force", is_flag=True, default=False, help="Force rebuild even if up-to-date.")
@click.option("--no-persist", is_flag=True, default=False, help="Use in-memory index only.")
def index(force: bool, no_persist: bool):
    """Build or rebuild the RAG vector index."""
    from optiprofiler_agent.common.rag import KnowledgeRAG

    config = AgentConfig()
    persist_dir = None if no_persist else config.rag_persist_dir
    console.print(f"Knowledge dir: {config.knowledge_dir}")
    if persist_dir:
        console.print(f"Persist dir:   {persist_dir}")

    rag = KnowledgeRAG(config.knowledge_dir, persist_dir=persist_dir)
    with console.status("Building index...", spinner="opa", spinner_style=_LOGO_OPA_COLOR):
        n = rag.build_index(force=force)

    console.print(f"[green]Done![/] Indexed {n} chunks.")


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--language", default="python", type=click.Choice(["python", "matlab"]))
def check(filepath: str, language: str):
    """Validate benchmark() calls in a Python script."""
    from optiprofiler_agent.validators.syntax_checker import check_code_string
    from optiprofiler_agent.validators.api_checker import validate_benchmark_call

    code = open(filepath, encoding="utf-8").read()

    syn = check_code_string(code)
    if syn.has_errors:
        console.print(f"[bold red]Syntax errors in {filepath}:[/]")
        for err in syn.errors:
            console.print(f"  Line {err.line}: {err.message}")
            console.print(f"    {err.code_snippet}")
        sys.exit(1)

    api = validate_benchmark_call(code, language=language)
    if api.has_errors:
        console.print(f"[bold red]API errors in {filepath}:[/]")
        for issue in api.issues:
            if issue.severity == "error":
                console.print(f"  [red]{issue.message}[/]")
        sys.exit(1)

    if api.has_warnings:
        console.print(f"[bold yellow]Warnings in {filepath}:[/]")
        for issue in api.issues:
            if issue.severity == "warning":
                console.print(f"  [yellow]{issue.message}[/]")

    if api.is_clean:
        console.print(f"[green]✓ {filepath} looks good![/] "
                       f"({api.benchmark_calls_found} benchmark() call(s) validated)")


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--provider", default="minimax",
              type=click.Choice(list(PROVIDER_REGISTRY.keys())),
              help="LLM provider for report generation.")
@click.option("--model", default=None, help="Model name (overrides provider default).")
@click.option("--language", default="English", help="Report language.")
@click.option("--no-llm", is_flag=True, default=False,
              help="Output raw JSON summary instead of LLM-generated report.")
@click.option("--no-profiles", is_flag=True, default=False,
              help="Skip PDF profile reading (faster, less detailed).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write report to file instead of stdout.")
@click.option("--latest", is_flag=True, default=False,
              help="Auto-detect the latest experiment in the given directory.")
def interpret(results_dir: str, provider: str, model: str | None,
              language: str, no_llm: bool, no_profiles: bool,
              output: str | None, latest: bool):
    """Analyze benchmark results and generate a report."""
    from optiprofiler_agent.agent_c.interpreter import interpret as do_interpret
    from optiprofiler_agent.agent_c.result_loader import find_latest_experiment

    if latest:
        try:
            results_dir = str(find_latest_experiment(results_dir))
            console.print(f"[dim]Using latest experiment: {results_dir}[/]")
        except FileNotFoundError as e:
            console.print(f"[bold red]Error:[/] {e}")
            sys.exit(1)

    config = AgentConfig(
        llm=LLMConfig(provider=provider, model=model),
    )

    with console.status("Analyzing benchmark results...", spinner="opa", spinner_style=_LOGO_OPA_COLOR):
        report = do_interpret(
            results_dir=results_dir,
            config=config,
            language=language,
            read_profiles=not no_profiles,
            llm_enabled=not no_llm,
        )

    if output:
        Path(output).write_text(report, encoding="utf-8")
        console.print(f"[green]Report written to {output}[/]")
    else:
        if no_llm:
            console.print(report)
        else:
            console.print(Markdown(report))


@main.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--traceback", "-t", "traceback_file", default=None,
              type=click.Path(exists=True),
              help="File containing the error traceback.")
@click.option("--error", "-e", "error_text", default=None,
              help="Error message text (alternative to --traceback file).")
@click.option("--run", is_flag=True, default=False,
              help="Run the script first, then auto-debug if it fails.")
@click.option("--timeout", default=120, type=int,
              help="Timeout in seconds for each run (with --run).")
@click.option("--save-fixed", default=None, type=click.Path(),
              help="Save the fixed code to this file (with --run).")
@click.option("--provider", default="minimax",
              type=click.Choice(list(PROVIDER_REGISTRY.keys())),
              help="LLM provider for debugging.")
@click.option("--model", default=None, help="Model name (overrides provider default).")
@click.option("--max-retries", default=3, type=int, help="Maximum fix attempts.")
@click.option("--code-limit", default=0, type=int,
              help="Max code chars sent to LLM (0 = no limit, send full code).")
def debug(filepath: str, traceback_file: str | None, error_text: str | None,
          run: bool, timeout: int, save_fixed: str | None,
          provider: str, model: str | None, max_retries: int,
          code_limit: int):
    """Diagnose and suggest fixes for benchmark script errors."""
    code = open(filepath, encoding="utf-8").read()

    config = AgentConfig(
        llm=LLMConfig(provider=provider, model=model),
        max_debug_retries=max_retries,
        code_char_limit=code_limit,
    )

    if run:
        from optiprofiler_agent.agent_b.debugger import run_and_debug

        def _on_progress(msg: str):
            console.print(f"[dim]{msg}[/]")

        console.print(f"[bold]Running {filepath}...[/]\n")
        result = run_and_debug(
            code=code,
            config=config,
            timeout=timeout,
            cwd=str(Path(filepath).parent),
            save_fixed=save_fixed,
            progress_callback=_on_progress,
        )
    else:
        from optiprofiler_agent.agent_b.debugger import debug_script

        if traceback_file:
            error = open(traceback_file, encoding="utf-8").read()
        elif error_text:
            error = error_text
        else:
            console.print("[bold red]Error:[/] Provide --traceback, --error, or use --run")
            sys.exit(1)

        with console.status("Diagnosing error...", spinner="opa", spinner_style=_LOGO_OPA_COLOR):
            result = debug_script(
                code=code,
                error=error,
                config=config,
            )

    console.print(Markdown(result.diagnostic_report))

    if result.fixed_code:
        console.print("\n[bold green]Suggested fix:[/]\n")
        console.print(Markdown(f"```python\n{result.fixed_code}\n```"))
        if save_fixed and not run:
            Path(save_fixed).write_text(result.fixed_code, encoding="utf-8")
            console.print(f"[green]Fixed code saved to {save_fixed}[/]")
    else:
        console.print("\n[bold yellow]No automatic fix available.[/]")

    console.print(f"\n[dim]Attempts: {result.attempts}[/]")


def _print_help(mode: str):
    """Print available slash commands for the current mode."""
    console.print("[bold]Commands:[/]")
    console.print("  [bold]/agent[/]              Switch to unified agent mode")
    console.print("  [bold]/chat[/]               Switch to advisor chat mode")
    if mode == "chat":
        console.print("  [bold]/reset[/]              Clear chat history")
    console.print("  [bold]/debug[/] <file>       Run & debug a script")
    console.print("  [bold]/interpret[/] <dir>    Analyze benchmark results")
    console.print("  [bold]/help[/]               Show this help")
    console.print("  [bold]/quit[/]               Exit\n")


def _slash_debug(args: str, config: AgentConfig):
    """Handle /debug <filepath> within the interactive loop."""
    if not args:
        console.print("[yellow]Usage: /debug <script.py>[/]\n")
        return

    filepath = Path(args)
    if not filepath.exists():
        console.print(f"[red]File not found: {filepath}[/]\n")
        return

    from optiprofiler_agent.agent_b.debugger import run_and_debug

    code = filepath.read_text(encoding="utf-8")
    console.print(f"[bold]Running & debugging {filepath}...[/]\n")

    result = run_and_debug(
        code=code,
        config=config,
        timeout=120,
        cwd=str(filepath.parent),
        progress_callback=lambda msg: console.print(f"  [dim]{msg}[/]"),
    )

    console.print(Markdown(result.diagnostic_report))
    if result.fixed_code:
        save = filepath.parent / f"{filepath.stem}_fixed{filepath.suffix}"
        save.write_text(result.fixed_code, encoding="utf-8")
        console.print(f"\n[green]Fixed code saved to {save}[/]")
    console.print()


def _slash_interpret(args: str, config: AgentConfig):
    """Handle /interpret <dir> [--latest] within the interactive loop."""
    if not args:
        console.print("[yellow]Usage: /interpret <results_dir> [--latest][/]\n")
        return

    latest = "--latest" in args
    path_str = args.replace("--latest", "").strip()
    results_dir = Path(path_str)

    if not results_dir.exists():
        console.print(f"[red]Directory not found: {results_dir}[/]\n")
        return

    from optiprofiler_agent.agent_c.interpreter import interpret as do_interpret
    from optiprofiler_agent.agent_c.result_loader import find_latest_experiment

    if latest:
        try:
            results_dir = Path(str(find_latest_experiment(str(results_dir))))
            console.print(f"[dim]Latest experiment: {results_dir}[/]")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/]\n")
            return

    with console.status("Analyzing...", spinner="opa", spinner_style=_LOGO_OPA_COLOR):
        report = do_interpret(
            results_dir=str(results_dir),
            config=config,
            language="English",
            read_profiles=True,
            llm_enabled=True,
        )

    console.print()
    console.print(Markdown(report))
    console.print()


@main.command()
@click.option("--provider", default="minimax",
              type=click.Choice(list(PROVIDER_REGISTRY.keys())),
              help="LLM provider.")
@click.option("--model", default=None, help="Model name (overrides provider default).")
def agent(provider: str, model: str | None):
    """Interactive unified agent with tool-use capabilities.

    Combines knowledge retrieval, script validation, debugging, and
    result interpretation in a single conversational interface.
    Supports /agent, /chat, /debug, /interpret mode switching.
    """
    from optiprofiler_agent.unified_agent import create_unified_agent

    config = AgentConfig(
        llm=LLMConfig(provider=provider, model=model),
        rag_enabled=True,
    )

    console.print(_LOGO)
    _print_agent_banner(config)

    unified = create_unified_agent(config)
    advisor = None
    mode = "agent"
    messages: list = []

    while True:
        try:
            if mode == "agent":
                user_input = console.input("[bold cyan]You:[/] ").strip()
            else:
                user_input = console.input("[bold green]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            console.print("Bye!")
            break

        if cmd in ("/help", "/h"):
            _print_help(mode)
            continue

        if cmd == "/agent":
            mode = "agent"
            messages.clear()
            console.print("[dim]▸ Agent mode (unified)[/]\n")
            continue

        if cmd == "/chat":
            if advisor is None:
                from optiprofiler_agent.agent_a.advisor import AdvisorAgent
                advisor = AdvisorAgent(config)
            mode = "chat"
            console.print("[dim]▸ Chat mode (advisor)[/]\n")
            continue

        if cmd == "/reset" and mode == "chat":
            if advisor:
                advisor.reset()
            console.print("[dim]History cleared.[/]\n")
            continue

        if cmd.startswith("/debug"):
            _slash_debug(user_input[6:].strip(), config)
            continue

        if cmd.startswith("/interpret"):
            _slash_interpret(user_input[10:].strip(), config)
            continue

        if mode == "agent":
            messages.append(("user", user_input))
            try:
                with console.status("Thinking...", spinner="opa", spinner_style=_LOGO_OPA_COLOR):
                    result = unified.invoke({"messages": messages})

                reply = result["messages"][-1].content
                messages = result["messages"]

                console.print()
                console.print(Markdown(reply))
                console.print()

            except Exception as e:
                console.print(f"[bold red]Error:[/] {e}\n")

        else:
            try:
                with console.status("Thinking...", spinner="opa", spinner_style=_LOGO_OPA_COLOR):
                    reply = advisor.chat(user_input)

                console.print()
                console.print(Markdown(reply))
                console.print()

            except Exception as e:
                console.print(f"[bold red]Error:[/] {e}\n")


@main.group()
def wiki():
    """Wiki knowledge base management commands."""


@wiki.command("lint")
def wiki_lint():
    """Check wiki health: orphan pages, broken links, index consistency."""
    config = AgentConfig()
    wiki_dir = config.wiki_dir
    index_path = wiki_dir / "index.md"

    if not wiki_dir.exists():
        console.print("[bold red]Error:[/] wiki/ directory not found.")
        sys.exit(1)

    all_pages = set()
    for f in wiki_dir.rglob("*.md"):
        rel = str(f.relative_to(wiki_dir))
        if rel not in ("index.md", "log.md"):
            all_pages.add(rel)

    indexed_pages = set()
    if index_path.exists():
        import re
        index_text = index_path.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^)]+\.md)\)", index_text):
            indexed_pages.add(match.group(1))

    issues = []

    missing_from_index = all_pages - indexed_pages
    if missing_from_index:
        for p in sorted(missing_from_index):
            issues.append(f"[yellow]Not in index.md:[/] {p}")

    in_index_not_on_disk = indexed_pages - all_pages
    if in_index_not_on_disk:
        for p in sorted(in_index_not_on_disk):
            issues.append(f"[red]In index but missing file:[/] {p}")

    if issues:
        console.print(f"[bold]Wiki lint: {len(issues)} issue(s) found[/]\n")
        for issue in issues:
            console.print(f"  - {issue}")
    else:
        console.print(f"[green]Wiki lint: all clean![/] {len(all_pages)} pages, "
                       f"{len(indexed_pages)} indexed.")


@wiki.command("rebuild-index")
@click.option("--force", is_flag=True, default=False, help="Force RAG rebuild.")
def wiki_rebuild_index(force: bool):
    """Rebuild the RAG vector index from wiki content."""
    from optiprofiler_agent.common.rag import KnowledgeRAG

    config = AgentConfig()
    console.print(f"Knowledge dir: {config.knowledge_dir}")
    console.print(f"Wiki dir:      {config.wiki_dir}")

    rag = KnowledgeRAG(config.knowledge_dir, persist_dir=config.rag_persist_dir)
    with console.status("Rebuilding wiki index...", spinner="opa", spinner_style=_LOGO_OPA_COLOR):
        n = rag.build_index(force=force)

    console.print(f"[green]Done![/] Indexed {n} chunks from wiki + sources.")


@wiki.command("stats")
def wiki_stats():
    """Show wiki statistics: page count, category breakdown, total size."""
    config = AgentConfig()
    wiki_dir = config.wiki_dir

    if not wiki_dir.exists():
        console.print("[bold red]Error:[/] wiki/ directory not found.")
        sys.exit(1)

    categories: dict[str, int] = {}
    total_size = 0
    total_pages = 0

    for f in sorted(wiki_dir.rglob("*.md")):
        rel = f.relative_to(wiki_dir)
        parts = rel.parts
        cat = parts[0] if len(parts) > 1 else "(root)"
        categories[cat] = categories.get(cat, 0) + 1
        total_size += f.stat().st_size
        total_pages += 1

    console.print("[bold]Wiki Statistics[/]\n")
    console.print(f"  Total pages: {total_pages}")
    console.print(f"  Total size:  {total_size / 1024:.1f} KB\n")
    console.print("  [bold]By category:[/]")
    for cat, count in sorted(categories.items()):
        console.print(f"    {cat}: {count} pages")


if __name__ == "__main__":
    main()
