#!/usr/bin/env python3
"""Auto-extract structured knowledge from optiprofiler sources.

Generates a language-partitioned knowledge base from two sources:
  1. Python docstrings (via numpydoc) — for Python API
  2. Sphinx HTML build output (via BeautifulSoup) — for MATLAB API + examples + installation

Output structure:
  knowledge/
  ├── common/          (project-level: concepts, enums, installation)
  ├── python/          (Python API: benchmark, classes, plib tools, examples)
  └── matlab/          (MATLAB API: benchmark, classes, plib tools, examples)

Usage:
  python scripts/extract_knowledge.py [--optiprofiler-root PATH] [--dry-run]

Requirements: numpydoc, beautifulsoup4, optiprofiler (installed)
"""

from __future__ import annotations

import argparse
import enum
import inspect
import json
import re
import textwrap
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from numpydoc.docscrape import NumpyDocString

OPTIPROFILER_ROOT = Path(__file__).resolve().parent.parent.parent / "optiprofiler"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _join(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines).strip()


def _extract_default(desc: str) -> str | None:
    m = re.search(r"[Dd]efault\s+(?:is|setting is|value is)\s+(.+?)(?:\.\s|$)", desc)
    return m.group(1).strip().rstrip(".") if m else None


def _extract_choices(desc: str) -> list[str] | None:
    m = re.search(
        r"(?:should be|can be|available\s+\w+\s+are)\s+(?:either\s+)?(.+?)(?:\.\s|$)",
        desc, re.IGNORECASE,
    )
    if m:
        choices = re.findall(r"'([^']+)'", m.group(1))
        if len(choices) >= 2:
            return choices
    return None


def _write_json(path: Path, data, label: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    items = ""
    if isinstance(data, dict):
        items = f" ({len(data)} keys)"
    print(f"  Written: {path}{items}  {label}")


def _write_md(path: Path, content: str, label: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  Written: {path} ({len(content)} chars)  {label}")


def _load_html(path: Path) -> BeautifulSoup | None:
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    return BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")


def _html_main_text(soup: BeautifulSoup) -> str:
    """Extract clean text from the main content area of a Sphinx page."""
    article = soup.find("article") or soup.find("div", role="main")
    if not article:
        return ""
    for tag in article.find_all(["script", "style", "nav"]):
        tag.decompose()
    return article.get_text(separator="\n", strip=False)


def _convert_tag_to_md(el: Tag) -> list[str]:
    """Convert a single HTML tag to markdown lines (recursive for sections)."""
    lines: list[str] = []
    classes = el.get("class", [])
    class_str = " ".join(classes)

    if el.name in ("h1", "h2", "h3", "h4"):
        level = int(el.name[1])
        text = el.get_text(strip=True).rstrip("#").strip()
        if text:
            lines.append(f"{'#' * level} {text}\n")
    elif el.name == "p":
        text = _get_spaced_text(el)
        if text:
            lines.append(text + "\n")
    elif el.name == "div" and any(c.startswith("highlight") for c in classes):
        pre = el.find("pre")
        code = pre.get_text() if pre else el.get_text()
        code = code.strip()
        lang = ""
        for c in classes:
            if c.startswith("highlight-"):
                lang = c.replace("highlight-", "")
        lines.append(f"```{lang}\n{code}\n```\n")
    elif el.name in ("ul", "ol"):
        for li in el.find_all("li", recursive=False):
            lines.append(f"- {_get_spaced_text(li)}\n")
    elif el.name == "blockquote":
        for child in el.children:
            if isinstance(child, Tag):
                child_lines = _convert_tag_to_md(child)
                lines.extend(
                    f"> {blk}" if not blk.startswith(">") else blk for blk in child_lines
                )
    elif "admonition" in class_str:
        title = el.find("p", class_="admonition-title")
        body = el.find_all("p")
        if title:
            lines.append(f"> **{title.get_text(strip=True)}**")
        for p in body:
            if p != title:
                lines.append(f"> {_get_spaced_text(p)}")
        lines.append("")
    elif el.name == "section":
        for child in el.children:
            if isinstance(child, Tag):
                lines.extend(_convert_tag_to_md(child))
    elif el.name == "dl":
        for dt in el.find_all("dt", recursive=False):
            dd = dt.find_next_sibling("dd")
            lines.append(f"**{dt.get_text(strip=True)}**")
            if dd:
                lines.append(f": {_get_spaced_text(dd)[:300]}\n")
    return lines


def _get_spaced_text(el: Tag) -> str:
    """Get text from an element, inserting spaces between inline tags.

    Sphinx HTML often omits whitespace between <strong>Name</strong>text,
    so we insert a space before/after each inline tag's text.
    """
    parts: list[str] = []
    for child in el.descendants:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, Tag) and child.name in ("br",):
            parts.append(" ")
    raw = "".join(parts)
    return " ".join(raw.split())


def _html_to_md(soup: BeautifulSoup) -> str:
    """Convert Sphinx HTML main content to readable markdown."""
    article = soup.find("article") or soup.find("div", role="main")
    if not article:
        return ""

    lines: list[str] = []
    for el in article.children:
        if isinstance(el, Tag):
            lines.extend(_convert_tag_to_md(el))

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Python API extraction (numpydoc)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_numpydoc(obj) -> dict:
    """Parse a callable or class docstring into structured dict."""
    docstr = inspect.getdoc(obj)
    if not docstr:
        return {"description": "", "parameters": {}}

    doc = NumpyDocString(docstr)
    result: dict = {
        "description": _join(doc["Summary"] + doc["Extended Summary"]),
    }

    sig = None
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        pass
    if sig:
        result["signature"] = sig

    # Parameters + Other Parameters
    params = {}
    cat_map = {
        "options for features": "feature_options",
        "options for profiles": "profile_options",
        "options for problems": "problem_options",
    }
    cur_cat = "parameters"

    for p in list(doc["Parameters"]) + list(doc["Other Parameters"]):
        name = p.name.strip()
        if name.startswith("*") and name.endswith("*"):
            header = name.strip("*").strip().lower().rstrip(":")
            for key, cat in cat_map.items():
                if key in header:
                    cur_cat = cat
                    break
            continue
        if not name or not p.type:
            continue
        desc = _join(p.desc)
        entry = {"type": p.type.replace(", optional", "").strip(), "description": desc}
        d = _extract_default(desc)
        if d:
            entry["default"] = d
        ch = _extract_choices(desc)
        if ch:
            entry["choices"] = ch
        params.setdefault(cur_cat, {})[name] = entry

    if len(params) == 1 and "parameters" in params:
        result["parameters"] = params["parameters"]
    else:
        result.update(params)

    # Returns
    rets = {}
    for r in doc["Returns"]:
        rets[r.name] = {"type": r.type, "description": _join(r.desc)}
    if rets:
        result["returns"] = rets

    # Raises
    raises = [{"exception": e.type, "description": _join(e.desc)} for e in doc["Raises"]]
    if raises:
        result["raises"] = raises

    # Notes
    notes = _join(doc["Notes"])
    if notes:
        result["notes"] = notes

    # See Also
    see_also = []
    for ref in doc["See Also"]:
        fn = ref[0][0] if ref[0] and ref[0][0] else ""
        desc = _join(ref[1]) if len(ref) > 1 else ""
        if fn:
            see_also.append({"name": fn, "description": desc})
    if see_also:
        result["see_also"] = see_also

    return result


def _parse_class_doc(cls) -> dict:
    """Parse a class: constructor + properties + methods."""
    result = _parse_numpydoc(cls)
    result["name"] = cls.__name__

    # Properties
    props = {}
    for name, obj in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if isinstance(obj, property) or (hasattr(cls, name) and isinstance(getattr(cls, name), property)):
            prop_doc = ""
            prop_obj = getattr(cls, name, None)
            if prop_obj and hasattr(prop_obj, "fget") and prop_obj.fget:
                prop_doc = inspect.getdoc(prop_obj.fget) or ""
            if not prop_doc:
                prop_doc = inspect.getdoc(obj) if inspect.getdoc(obj) else ""
            props[name] = {"description": prop_doc.split("\n")[0].strip() if prop_doc else ""}

    if props:
        result["properties"] = props

    # Public methods
    methods = {}
    for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        mdoc = inspect.getdoc(obj)
        methods[name] = {"description": mdoc.split("\n")[0].strip() if mdoc else ""}
        try:
            methods[name]["signature"] = str(inspect.signature(obj))
        except (ValueError, TypeError):
            pass

    if methods:
        result["methods"] = methods

    return result


def extract_python_benchmark() -> dict:
    from optiprofiler.profiles import benchmark
    data = _parse_numpydoc(benchmark)
    data["name"] = "benchmark"

    data["calling_convention"] = {
        "syntax": "scores = benchmark([solver1, solver2], ptype='u', mindim=2, maxdim=20)",
        "solvers": "list of callables: [solver1, solver2]",
        "options": "keyword arguments to benchmark(). Example: benchmark(solvers, ptype='u', mindim=2)",
    }
    data["solver_signatures"] = {
        "unconstrained": "solver(fun, x0) -> numpy.ndarray",
        "bound_constrained": "solver(fun, x0, xl, xu) -> numpy.ndarray",
        "linearly_constrained": "solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray",
        "nonlinearly_constrained": "solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray",
    }
    data["solver_notes"] = [
        "fun(x) -> float: provides ONLY function values — no gradient/Hessian (DFO).",
        "Must return numpy.ndarray of shape (n,).",
        "At least 2 solvers required.",
    ]
    return data


def extract_python_classes() -> dict:
    from optiprofiler import Problem, Feature, FeaturedProblem
    return {
        "Problem": _parse_class_doc(Problem),
        "Feature": _parse_class_doc(Feature),
        "FeaturedProblem": _parse_class_doc(FeaturedProblem),
    }


def extract_python_plib_tools() -> dict:
    from optiprofiler.problem_libs.s2mpj.s2mpj_tools import s2mpj_load, s2mpj_select
    from optiprofiler.plib_config import get_plib_config, set_plib_config

    result = {
        "s2mpj_load": _parse_numpydoc(s2mpj_load),
        "s2mpj_select": _parse_numpydoc(s2mpj_select),
        "get_plib_config": _parse_numpydoc(get_plib_config),
        "set_plib_config": _parse_numpydoc(set_plib_config),
    }

    try:
        from optiprofiler.problem_libs.pycutest.pycutest_tools import pycutest_load, pycutest_select
        result["pycutest_load"] = _parse_numpydoc(pycutest_load)
        result["pycutest_select"] = _parse_numpydoc(pycutest_select)
    except ImportError:
        result["pycutest_load"] = {"description": "Load a PyCUTEst problem. Requires pycutest package (Linux/macOS)."}
        result["pycutest_select"] = {"description": "Select PyCUTEst problems matching criteria. Requires pycutest package."}

    return result


def extract_python_api_notes() -> dict:
    return {
        "language": "Python",
        "solver_format": "list of callables: [solver1, solver2]",
        "options_format": "keyword arguments to benchmark()",
        "vector_convention": "1-D numpy arrays, shape (n,)",
        "problem_libs": ["s2mpj", "pycutest", "custom"],
        "python_only_options": ["custom_problem_libs_path"],
        "pycutest_note": "Requires separate installation; Linux and macOS only",
        "lambda_warning": "Lambda functions are not picklable — use named functions (def) for parallel execution (n_jobs > 1)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MATLAB API extraction (HTML)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_matlab_benchmark_html(soup: BeautifulSoup) -> dict:
    """Parse MATLAB benchmark HTML page into structured dict."""
    article = soup.find("article") or soup.find("div", role="main")
    if not article:
        return {"name": "benchmark"}

    result: dict = {
        "name": "benchmark",
        "description": "Benchmark optimization solvers on a set of problems with specified features.",
    }

    result["calling_convention"] = {
        "syntax": "[solver_scores, profile_scores, curves] = benchmark(solvers, options)",
        "solvers": "cell array of function handles: {@solver1, @solver2}",
        "options": "struct with fields (NOT name-value pairs). Example: options.ptype = 'u'; options.mindim = 2; benchmark(solvers, options);",
    }
    result["solver_signatures"] = {
        "unconstrained": "x = solver(fun, x0)",
        "bound_constrained": "x = solver(fun, x0, xl, xu)",
        "linearly_constrained": "x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)",
        "nonlinearly_constrained": "x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)",
    }
    result["solver_notes"] = [
        "fun is a function handle: fun(x) -> scalar. Provides ONLY function values (DFO).",
        "x0 is a column vector.",
        "All constraint vectors are column vectors.",
        "Must return column vector x.",
        "At least 2 solvers required (cell array of function handles).",
    ]

    # Options are in <li> elements with <strong> for the name.
    # Category headers appear in <p> tags like "Options for profiles and plots"
    cat_markers = {
        "options for profiles": "profile_options",
        "options for features": "feature_options",
        "options for problems": "problem_options",
    }
    cur_cat = "profile_options"
    skip_names = {"solver", "solvers", "x", "fun", "x0", "xl", "xu",
                  "aub", "bub", "aeq", "beq", "cub", "ceq"}

    # Walk all elements in order to detect category headers
    all_elements = article.find_all(["p", "li"])
    for el in all_elements:
        if el.name == "p":
            p_text = el.get_text(strip=True).lower()
            for marker, cat in cat_markers.items():
                if marker in p_text:
                    cur_cat = cat
            continue

        # <li> with option
        strong = el.find("strong")
        if not strong:
            continue
        opt_name = strong.get_text(strip=True).rstrip(":")
        if not opt_name or opt_name in skip_names or opt_name[0].isupper():
            continue

        desc = el.get_text(strip=True)
        # Remove the leading "opt_name:" prefix from the description
        desc = re.sub(r"^\w+:\s*", "", desc, count=1)
        entry = {"description": desc[:500]}
        d = _extract_default(desc)
        if d:
            entry["default"] = d
        ch = _extract_choices(desc)
        if ch:
            entry["choices"] = ch

        result.setdefault(cur_cat, {})[opt_name] = entry

    # Returns
    result["returns"] = {
        "solver_scores": {"type": "vector", "description": "Scores of the solvers based on the profiles."},
        "profile_scores": {"type": "4D tensor", "description": "Scores for all profiles (solver × tolerance × hist/output × profile_type)."},
        "curves": {"type": "cell array", "description": "Curves of all the profiles."},
    }

    return result


def _parse_matlab_class_html(soup: BeautifulSoup, class_name: str) -> dict:
    """Parse a MATLAB class HTML page.

    MATLAB class pages use <p> headers followed by <blockquote> for
    properties/methods, not tables.
    """
    article = soup.find("article") or soup.find("div", role="main")
    if not article:
        return {"name": class_name}

    result: dict = {"name": class_name}

    # Get description from the <dl> definition
    dl = article.find("dl")
    if dl:
        dd = dl.find("dd")
        if dd:
            first_p = dd.find("p")
            result["description"] = first_p.get_text(strip=True) if first_p else ""

    # Walk <p> + <blockquote> siblings in order.
    # <p> sets the section context, <blockquote> contains the data.
    properties = {}
    methods = {}
    current_section = ""

    section = article.find("section") or article
    for el in section.children:
        if not isinstance(el, Tag):
            continue

        if el.name == "p":
            text_lower = el.get_text(strip=True).lower()
            if "properties" in text_lower or "compulsory" in text_lower or "optional" in text_lower:
                current_section = "properties"
            elif "methods" in text_lower:
                current_section = "methods"
            continue

        if el.name == "blockquote" and current_section:
            bq_text = _get_spaced_text(el)
            # Try "name: description" entries (split on pattern "word:")
            entries = re.split(r"(?<=[.!])\s+(?=[a-z_]\w*\s*[\(:])", bq_text)
            parsed_any = False
            for entry in entries:
                m = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", entry.strip(), re.DOTALL)
                if m:
                    name = m.group(1)
                    desc = " ".join(m.group(2).split())[:300]
                    target = properties if current_section == "properties" else methods
                    target[name] = {"description": desc}
                    parsed_any = True

            if not parsed_any:
                # Simple comma-separated list: "name, x0, xl, xu"
                names = [n.strip() for n in re.split(r"[,\s]+", bq_text) if n.strip()]
                for n in names:
                    if re.match(r"^[a-z_]\w*$", n):
                        target = properties if current_section == "properties" else methods
                        target.setdefault(n, {"description": ""})

    if properties:
        result["properties"] = properties
    if methods:
        result["methods"] = methods

    return result


def _parse_matlab_tool_html(soup: BeautifulSoup, func_name: str) -> dict:
    """Parse a MATLAB tool function HTML page (s2mpj_load, etc.)."""
    article = soup.find("article") or soup.find("div", role="main")
    if not article:
        return {"name": func_name}

    result: dict = {"name": func_name}

    dl = article.find("dl")
    if dl:
        dt = dl.find("dt")
        dd = dl.find("dd")
        if dt:
            result["signature"] = dt.get_text(strip=True)
        if dd:
            first_p = dd.find("p")
            result["description"] = first_p.get_text(strip=True)[:500] if first_p else ""

            # Extract parameters from <li> items in dd
            params = {}
            for li in dd.find_all("li"):
                strong = li.find("strong")
                if strong:
                    pname = strong.get_text(strip=True).rstrip(":")
                    pdesc = li.get_text(strip=True)
                    pdesc = re.sub(r"^\w+:\s*", "", pdesc, count=1)
                    params[pname] = {"description": pdesc[:300]}
            if params:
                result["parameters"] = params

    return result


def extract_matlab_api(html_dir: Path) -> tuple[dict, dict, dict]:
    """Extract all MATLAB API from HTML pages."""
    mat_gen = html_dir / "matlab" / "matlab_generated"

    # benchmark
    soup = _load_html(mat_gen / "benchmark.html")
    benchmark = _parse_matlab_benchmark_html(soup) if soup else {"name": "benchmark"}

    # classes
    classes = {}
    for cls_name in ("Problem", "Feature", "FeaturedProblem"):
        soup = _load_html(mat_gen / f"{cls_name}.html")
        if soup:
            classes[cls_name] = _parse_matlab_class_html(soup, cls_name)

    # plib tools
    plib_tools = {}
    for tool in ("s2mpj_load", "s2mpj_select", "matcutest_load", "matcutest_select"):
        soup = _load_html(mat_gen / f"{tool}.html")
        if soup:
            plib_tools[tool] = _parse_matlab_tool_html(soup, tool)

    return benchmark, classes, plib_tools


def extract_matlab_api_notes() -> dict:
    return {
        "language": "MATLAB",
        "solver_format": "cell array of function handles: {@solver1, @solver2}",
        "options_format": "struct with fields: options.ptype = 'u'",
        "vector_convention": "column vectors (n×1 matrices)",
        "problem_libs": ["s2mpj", "matcutest"],
        "matcutest_note": "matcutest is only available on Linux",
        "differences_from_python": {
            "maxdim_default": "mindim + 10 (Python: mindim + 1)",
            "draw_hist_plots_default": "'sequential' (Python: 'parallel')",
            "solvers_to_load": "1-indexed (Python: 0-indexed)",
            "line_colors_default": "MATLAB 'gem' colororder (Python: matplotlib tab10)",
            "no_custom_problem_libs_path": "MATLAB uses folder structure instead",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Examples + Installation (HTML)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_examples_html(html_dir: Path, language: str) -> str:
    """Extract usage examples from Sphinx HTML."""
    if language == "python":
        path = html_dir / "user" / "usage_python.html"
    else:
        path = html_dir / "user" / "usage.html"

    soup = _load_html(path)
    if not soup:
        return f"# {language.title()} Examples\n\n(HTML not found)\n"

    header = (
        f"# {language.title()} Examples\n\n"
        f"> **OptiProfiler benchmarks derivative-free optimization (DFO) solvers.**\n"
        f"> `fun` provides ONLY function values — no gradient or Hessian.\n"
        f"> `benchmark()` requires at least 2 solvers.\n\n---\n\n"
    )
    return header + _html_to_md(soup)


def extract_installation_html(html_dir: Path) -> tuple[str, str]:
    """Extract installation guide from Sphinx HTML, split by language.

    Returns (python_install, matlab_install).
    """
    soup = _load_html(html_dir / "user" / "installation.html")
    if not soup:
        return ("# Python Installation\n\n(HTML not found)\n",
                "# MATLAB Installation\n\n(HTML not found)\n")

    article = soup.find("article") or soup.find("div", role="main")
    if not article:
        return ("# Python Installation\n\n(empty)\n",
                "# MATLAB Installation\n\n(empty)\n")

    # Find the Python and MATLAB sections
    py_lines: list[str] = ["# Python Installation\n"]
    mat_lines: list[str] = ["# MATLAB Installation\n"]
    current = None

    sections = article.find_all("section", recursive=True)
    for sec in sections:
        h = sec.find(["h1", "h2", "h3", "h4"], recursive=False)
        if not h:
            continue
        title = h.get_text(strip=True).lower().rstrip("#").strip()
        if "python" in title:
            current = "python"
        elif "matlab" in title:
            current = "matlab"
        else:
            continue

        target = py_lines if current == "python" else mat_lines
        for child in sec.children:
            if isinstance(child, Tag) and child != h:
                target.extend(_convert_tag_to_md(child))

    return ("\n".join(py_lines), "\n".join(mat_lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Common knowledge
# ═══════════════════════════════════════════════════════════════════════════════

def extract_enums() -> dict:
    from optiprofiler import utils
    result = {}
    for name, obj in vars(utils).items():
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj is not enum.Enum:
            result[name] = {m.name: m.value for m in obj}
    return result


def generate_concepts_md() -> str:
    """Generate language-neutral core concepts (no Python/MATLAB specifics)."""
    return textwrap.dedent("""\
    # OptiProfiler Core Concepts

    ## What is OptiProfiler?

    OptiProfiler is a platform for benchmarking optimization solvers.
    It supports both Python and MATLAB, with nearly identical APIs.
    It generates performance profiles, data profiles, and log-ratio profiles
    to compare solver effectiveness across standardized test problem sets.

    ## Derivative-Free Optimization (DFO)

    OptiProfiler is designed for **derivative-free optimization** benchmarking.
    The objective function `fun` returns **only a scalar function value**.
    No gradient, Jacobian, or Hessian information is available.

    Every call to `fun` is counted internally and used for performance scoring.
    Methods that internally approximate gradients via finite differences
    consume extra `fun` evaluations, making them generally unsuitable
    for DFO benchmarking.

    **Recommended DFO methods** (see language-specific guides for details).

    ## Solver Requirements

    - `benchmark()` requires **at least 2 solvers** for comparison.
    - Each solver must follow a specific signature depending on the problem type.
    - Solvers must return the solution vector (not a dict or result object).

    ## Four Problem Types

    | Type | Signature Pattern |
    |------|------------------|
    | Unconstrained | `solver(fun, x0)` |
    | Bound-constrained | `solver(fun, x0, xl, xu)` |
    | Linearly constrained | `solver(fun, x0, xl, xu, aub, bub, aeq, beq)` |
    | Nonlinearly constrained | `solver(fun, x0, ..., cub, ceq)` |

    ## Profiles and Scoring

    - **Performance profiles**: fraction of problems solved within a factor of the best solver
    - **Data profiles**: fraction of problems solved as a function of computational budget
    - **Log-ratio profiles**: pairwise comparison (exactly 2 solvers only)
    - **Scores**: by default, average of history-based performance profiles across tolerances

    ## Output Structure

    Running `benchmark()` creates:
    - `<benchmark_id>/<feature_stamp>/` directory with per-problem results
    - `summary.pdf` with all profile plots
    - Return values: `solver_scores`, `profile_scores` (4D), `curves`

    ## Additional Notes

    - **Log-ratio profiles** are available only when there are exactly 2 solvers.
    - The `load` option allows reloading a previous experiment to redraw profiles with different options.
    - More information: https://www.optprof.com
    """)


def generate_solver_interface_md() -> str:
    return textwrap.dedent("""\
    # Solver Function Interface Specification

    OptiProfiler benchmarks **derivative-free optimization (DFO)** solvers.
    `benchmark()` requires **at least 2 solvers**.

    ## Python Signatures

    ```python
    # Unconstrained
    def solver(fun, x0) -> numpy.ndarray: ...

    # Bound-constrained
    def solver(fun, x0, xl, xu) -> numpy.ndarray: ...

    # Linearly constrained
    def solver(fun, x0, xl, xu, aub, bub, aeq, beq) -> numpy.ndarray: ...

    # Nonlinearly constrained
    def solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq) -> numpy.ndarray: ...

    # Universal wrapper (handles all types)
    def solver(fun, x0, xl=None, xu=None, aub=None, bub=None,
               aeq=None, beq=None, cub=None, ceq=None) -> numpy.ndarray: ...
    ```

    ## MATLAB Signatures

    ```matlab
    % Unconstrained
    x = solver(fun, x0)

    % Bound-constrained
    x = solver(fun, x0, xl, xu)

    % Linearly constrained
    x = solver(fun, x0, xl, xu, aub, bub, aeq, beq)

    % Nonlinearly constrained
    x = solver(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq)
    ```

    ## Key Differences

    | Aspect | Python | MATLAB |
    |--------|--------|--------|
    | Solvers arg | list of callables | cell array of function handles |
    | Options | keyword arguments | struct |
    | Vectors | numpy 1-D arrays (n,) | column vectors (n×1) |
    | Return | numpy.ndarray | column vector |
    """)


def generate_problem_libs_md(language: str) -> str:
    if language == "python":
        return textwrap.dedent("""\
        # Python Problem Libraries

        ## Built-in Libraries

        - **s2mpj**: Default. Pure Python, no extra installation.
        - **pycutest**: Requires separate installation. Linux and macOS only.
          See https://jfowkes.github.io/pycutest/

        ## Custom Libraries

        Use `custom_problem_libs_path` to add your own:

        ```
        /path/to/my_libs/
        └── myproblems/
            └── myproblems_tools.py  (implements myproblems_load + myproblems_select)
        ```

        ```python
        benchmark(
            [solver1, solver2],
            plibs=['s2mpj', 'myproblems'],
            custom_problem_libs_path='/path/to/my_libs',
        )
        ```
        """)
    else:
        return textwrap.dedent("""\
        # MATLAB Problem Libraries

        ## Built-in Libraries

        - **s2mpj**: Default. Bundled with OptiProfiler.
        - **matcutest**: Requires setup. **Linux only.**
          See https://github.com/matcutest

        ## Custom Libraries

        Create a subfolder in the `problems` directory:

        ```
        problems/
        └── myproblems/
            ├── myproblems_load.m
            └── myproblems_select.m
        ```

        ```matlab
        options.plibs = {'s2mpj', 'myproblems'};
        scores = benchmark({@solver1, @solver2}, options);
        ```
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optiprofiler-root", type=Path, default=OPTIPROFILER_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "optiprofiler_agent" / "knowledge")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    op_root = args.optiprofiler_root
    html_dir = op_root / "doc" / "build" / "html"
    out = args.out_dir

    print(f"OptiProfiler root: {op_root}")
    print(f"HTML doc dir:      {html_dir} (exists={html_dir.exists()})")
    print(f"Output dir:        {out}\n")

    if not html_dir.exists():
        print("ERROR: HTML doc directory not found.")
        print("       Run 'make html' in the optiprofiler/doc/ directory first.")
        return 1

    # ── Python API (numpydoc) ──
    print("=== Python API (numpydoc) ===")
    py_benchmark = extract_python_benchmark()
    py_opt_count = sum(len(v) for k, v in py_benchmark.items() if k.endswith("_options"))
    print(f"  benchmark: {py_opt_count} options, "
          f"{len(py_benchmark.get('returns', {}))} returns")

    py_classes = extract_python_classes()
    for cname, cdata in py_classes.items():
        print(f"  {cname}: {len(cdata.get('properties', {}))} props, "
              f"{len(cdata.get('methods', {}))} methods")

    py_plib = extract_python_plib_tools()
    print(f"  plib tools: {len(py_plib)} functions")

    py_notes = extract_python_api_notes()

    # ── MATLAB API (HTML) ──
    print("\n=== MATLAB API (HTML) ===")
    mat_benchmark, mat_classes, mat_plib = extract_matlab_api(html_dir)
    mat_opt_count = sum(len(v) for k, v in mat_benchmark.items() if k.endswith("_options"))
    print(f"  benchmark: {mat_opt_count} options")
    for cname, cdata in mat_classes.items():
        print(f"  {cname}: {len(cdata.get('properties', {}))} props, "
              f"{len(cdata.get('methods', {}))} methods")
    print(f"  plib tools: {len(mat_plib)} functions")

    mat_notes = extract_matlab_api_notes()

    # ── Examples + Installation (HTML) ──
    print("\n=== Examples & Installation (HTML) ===")
    py_examples = extract_examples_html(html_dir, "python")
    mat_examples = extract_examples_html(html_dir, "matlab")
    py_install, mat_install = extract_installation_html(html_dir)
    print(f"  Python examples: {len(py_examples)} chars")
    print(f"  MATLAB examples: {len(mat_examples)} chars")
    print(f"  Python install: {len(py_install)} chars")
    print(f"  MATLAB install: {len(mat_install)} chars")

    # ── Common ──
    print("\n=== Common knowledge ===")
    enums = extract_enums()
    for cls, members in enums.items():
        print(f"  {cls}: {len(members)} members")

    concepts = generate_concepts_md()
    solver_iface = generate_solver_interface_md()

    if args.dry_run:
        print("\n[DRY RUN] Would write all files. Exiting.")
        return 0

    # ── Write files ──
    print("\n=== Writing files ===")

    # common/ — only language-neutral knowledge
    _write_json(out / "common" / "enums.json", enums)
    _write_md(out / "common" / "concepts.md", concepts)
    _write_md(out / "common" / "solver_interface.md", solver_iface)

    # python/
    _write_json(out / "python" / "benchmark.json", py_benchmark)
    _write_json(out / "python" / "classes.json", py_classes)
    _write_json(out / "python" / "plib_tools.json", py_plib)
    _write_json(out / "python" / "api_notes.json", py_notes)
    _write_md(out / "python" / "examples.md", py_examples)
    _write_md(out / "python" / "problem_libs.md", generate_problem_libs_md("python"))
    _write_md(out / "python" / "installation.md", py_install)

    # matlab/
    _write_json(out / "matlab" / "benchmark.json", mat_benchmark)
    _write_json(out / "matlab" / "classes.json", mat_classes)
    _write_json(out / "matlab" / "plib_tools.json", mat_plib)
    _write_json(out / "matlab" / "api_notes.json", mat_notes)
    _write_md(out / "matlab" / "examples.md", mat_examples)
    _write_md(out / "matlab" / "problem_libs.md", generate_problem_libs_md("matlab"))
    _write_md(out / "matlab" / "installation.md", mat_install)

    # Clean up old files
    old_files = [
        out / "common" / "api_params.json",
        out / "common" / "installation.md",
        out / "api_params.json", out / "enums.json", out / "examples.md",
        out / "matlab_guide.md", out / "problem_libs_guide.md",
        out / "solver_interface_spec.md",
    ]
    for f in old_files:
        if f.exists():
            f.unlink()

    print(f"\nDone. {16} files written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
