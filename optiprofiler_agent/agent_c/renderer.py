"""Render a structured ``BenchmarkReport`` to Markdown / HTML.

The renderer is the *only* place that knows about output format. The
LLM produces a ``BenchmarkReport`` (Pydantic), the validator checks
business invariants, and this module turns the validated object into
the surface representation.

Multi-format support is intentional:
- Terminal: Rich renders the Markdown directly.
- File: ``.md`` for git, ``.html`` for sharing, JSON for programmatic.
- Future web platform: same template, swap the Jinja extension.

Jinja2 is already a transitive dependency of LangChain, so adopting it
adds zero new top-level dependency.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from optiprofiler_agent.agent_c.report_schema import BenchmarkReport
from optiprofiler_agent.agent_c.summary import BenchmarkSummary


_TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=1)
def _markdown_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=select_autoescape(disabled_extensions=("md", "j2")),
        trim_blocks=False,
        lstrip_blocks=False,
    )


def render_markdown(report: BenchmarkReport, summary: BenchmarkSummary) -> str:
    """Render a validated report + raw summary to Markdown.

    The summary is required for objective metadata that the LLM never
    sees in its raw form (solver scores table, problem libraries) so
    that those fields cannot be hallucinated and always reflect ground
    truth.
    """
    template = _markdown_env().get_template("report.md.j2")
    return template.render(report=report, summary=summary)


def render_html(report: BenchmarkReport, summary: BenchmarkSummary) -> str:
    """Render the same report to a minimal self-contained HTML document.

    Implemented as a thin wrapper that converts the Markdown output via
    the optional `markdown` package; falls back to <pre>-wrapped Markdown
    when the package is not installed (no new top-level dependency).
    """
    md = render_markdown(report, summary)
    try:
        import markdown as _markdown
    except ImportError:
        body = (
            "<pre style=\"white-space: pre-wrap; "
            "font-family: ui-monospace, monospace;\">"
            f"{_html_escape(md)}</pre>"
        )
    else:
        body = _markdown.markdown(md, extensions=["tables", "fenced_code"])
    return _HTML_SHELL.format(body=body)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_HTML_SHELL = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OptiProfiler Benchmark Report</title>
<style>
  body {{ max-width: 920px; margin: 2rem auto; padding: 0 1rem;
          font-family: -apple-system, system-ui, sans-serif; line-height: 1.55; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; }}
  th {{ background: #f5f5f5; }}
  code {{ background: #f5f5f5; padding: 0.1em 0.3em; border-radius: 3px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


__all__ = ["render_html", "render_markdown"]
