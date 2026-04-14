"""Extract profile curve data from OptiProfiler-generated PDF files.

Uses PyMuPDF (fitz) to read vector paths from matplotlib-generated PDFs.
Supports three profile types:
- **Performance profiles** (step functions, log-scale x-axis)
- **Data profiles** (step functions, log-scale x-axis)
- **Log-ratio profiles** (bar charts, one bar per problem)

Each profile PDF may be a single-page (detailed) or multi-page (summary)
document, with one tolerance level per page.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CurveData:
    """A single solver's curve extracted from a profile page."""

    solver_name: str
    color_rgb: tuple[float, float, float]
    points: list[tuple[float, float]]  # (x_data, y_data) in data coordinates


@dataclass
class ProfilePage:
    """Data extracted from one page of a profile PDF."""

    profile_type: str   # "perf", "data", "log-ratio"
    basis: str          # "history-based" or "output-based"
    tolerance: str      # e.g. "1e-1", "1e-10"
    title: str
    x_label: str
    y_label: str
    curves: list[CurveData]


@dataclass
class BarChartData:
    """Data from a log-ratio profile (bar chart)."""

    solver_name: str
    color_rgb: tuple[float, float, float]
    bars: list[float]  # y-values for each problem bar


@dataclass
class LogRatioPage:
    """Data extracted from a log-ratio profile page."""

    basis: str
    tolerance: str
    title: str
    solver_names: list[str]
    bar_data: list[BarChartData]


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def _build_axis_mapping(
    tick_positions: list[float],
    tick_values: list[float],
    log_scale: bool = False,
) -> tuple[float, float, float, float]:
    """Build a linear mapping from pixel coords to data coords.

    Returns (pixel_min, pixel_max, data_min, data_max) for linear interpolation.
    For log-scale axes, tick_values should be the actual values (not log).
    """
    if len(tick_positions) < 2 or len(tick_values) < 2:
        return (0.0, 1.0, 0.0, 1.0)

    if log_scale:
        log_vals = [math.log2(v) if v > 0 else 0 for v in tick_values]
    else:
        log_vals = tick_values

    p0, p1 = tick_positions[0], tick_positions[-1]
    v0, v1 = log_vals[0], log_vals[-1]
    return (p0, p1, v0, v1)


def _pixel_to_data(
    pixel: float,
    p_min: float, p_max: float,
    d_min: float, d_max: float,
    log_scale: bool = False,
) -> float:
    """Convert a pixel coordinate to a data coordinate."""
    if abs(p_max - p_min) < 1e-9:
        return d_min
    ratio = (pixel - p_min) / (p_max - p_min)
    d = d_min + ratio * (d_max - d_min)
    if log_scale:
        return 2.0 ** d
    return d


# ---------------------------------------------------------------------------
# Text and axis extraction
# ---------------------------------------------------------------------------

_TOL_RE = re.compile(r"tol\s*=\s*10[−\-](\d+)")


def _extract_page_metadata(page) -> tuple[str, str, list[str]]:
    """Extract title, tolerance, and solver names from page text.

    Returns (title, tolerance_str, solver_names).
    """
    text = page.get_text()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    title = ""
    tolerance = ""
    solver_names: list[str] = []

    for line in lines:
        if "profile" in line.lower() or "proﬁle" in line.lower():
            title = line
            m = _TOL_RE.search(line)
            if m:
                tolerance = f"1e-{m.group(1)}"

    # Solver names appear as legend entries — typically at the end of text,
    # after axis labels. They are multi-word names like "scipy cobyla".
    # We detect them by looking for text blocks in the legend region.
    blocks = page.get_text("dict")["blocks"]
    legend_texts = []
    page_w = page.rect.width
    for b in blocks:
        if "lines" not in b:
            continue
        for line_obj in b["lines"]:
            for span in line_obj["spans"]:
                bbox = span["bbox"]
                txt = span["text"].strip()
                # Legend is typically in the right portion of the page
                if bbox[0] > page_w * 0.6 and txt:
                    legend_texts.append((bbox[1], txt))

    # Group legend text by y-position to reconstruct multi-word names
    if legend_texts:
        legend_texts.sort(key=lambda t: t[0])
        current_y = legend_texts[0][0]
        current_name_parts: list[str] = []
        for y, txt in legend_texts:
            if abs(y - current_y) < 5:
                current_name_parts.append(txt)
            else:
                if current_name_parts:
                    name = " ".join(current_name_parts).replace(" ", "_")
                    solver_names.append(name)
                current_name_parts = [txt]
                current_y = y
        if current_name_parts:
            name = " ".join(current_name_parts).replace(" ", "_")
            solver_names.append(name)

    return title, tolerance, solver_names


def _extract_axis_ticks(page) -> tuple[
    list[tuple[float, float]],  # x-axis: (pixel_pos, value)
    list[tuple[float, float]],  # y-axis: (pixel_pos, value)
]:
    """Extract axis tick positions and values from text blocks.

    Returns two lists of (pixel_position, data_value) pairs.
    """
    blocks = page.get_text("dict")["blocks"]

    x_ticks: list[tuple[float, float]] = []
    y_ticks: list[tuple[float, float]] = []

    # Collect all numeric text spans
    num_spans: list[tuple[float, float, float, float, str]] = []
    for b in blocks:
        if "lines" not in b:
            continue
        for line_obj in b["lines"]:
            line_text = "".join(s["text"] for s in line_obj["spans"]).strip()
            if not line_text:
                continue
            # Get bounding box of the whole line
            x0 = min(s["bbox"][0] for s in line_obj["spans"])
            y0 = min(s["bbox"][1] for s in line_obj["spans"])
            x1 = max(s["bbox"][2] for s in line_obj["spans"])
            y1 = max(s["bbox"][3] for s in line_obj["spans"])
            num_spans.append((x0, y0, x1, y1, line_text))

    # Find the plot area boundaries from axis lines
    plot_left, plot_right, plot_top, plot_bottom = _find_plot_area(page)

    for x0, y0, x1, y1, txt in num_spans:
        try:
            val = float(txt)
        except ValueError:
            continue

        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2

        # X-axis labels are below the plot area
        if y0 > plot_bottom - 5:
            x_ticks.append((mid_x, val))
        # Y-axis labels are to the left of the plot area
        elif x1 < plot_left + 5:
            y_ticks.append((mid_y, val))

    x_ticks.sort(key=lambda t: t[0])
    y_ticks.sort(key=lambda t: t[0])

    return x_ticks, y_ticks


def _find_plot_area(page) -> tuple[float, float, float, float]:
    """Find the plot area boundaries from axis frame lines.

    Returns (left, right, top, bottom) in pixel coordinates.
    """
    drawings = page.get_drawings()

    # The plot frame is drawn as 4 black lines forming a rectangle.
    # Look for the longest horizontal and vertical black lines.
    h_lines: list[tuple[float, float, float]] = []  # (y, x_start, x_end)
    v_lines: list[tuple[float, float, float]] = []  # (x, y_start, y_end)

    for d in drawings:
        c = d.get("color")
        w = d.get("width") or 0
        if not c or not (abs(c[0]) < 0.01 and abs(c[1]) < 0.01 and abs(c[2]) < 0.01):
            continue
        if w < 0.5:
            continue

        for item in d["items"]:
            if item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            if abs(p1.y - p2.y) < 1:
                length = abs(p2.x - p1.x)
                if length > 50:
                    h_lines.append((p1.y, min(p1.x, p2.x), max(p1.x, p2.x)))
            elif abs(p1.x - p2.x) < 1:
                length = abs(p2.y - p1.y)
                if length > 50:
                    v_lines.append((p1.x, min(p1.y, p2.y), max(p1.y, p2.y)))

    if not h_lines or not v_lines:
        return (50, 400, 10, 280)

    h_lines.sort(key=lambda seg: seg[0])
    v_lines.sort(key=lambda seg: seg[0])

    top = h_lines[0][0] if len(h_lines) > 1 else h_lines[0][0]
    bottom = h_lines[-1][0]
    left = v_lines[0][0]
    right = v_lines[-1][0]

    if top > bottom:
        top, bottom = bottom, top

    return (left, right, top, bottom)


# ---------------------------------------------------------------------------
# Curve extraction
# ---------------------------------------------------------------------------

# Default matplotlib tab10 colors (first 10)
_TAB10_COLORS = [
    (0.122, 0.467, 0.706),  # blue
    (1.000, 0.498, 0.055),  # orange
    (0.173, 0.627, 0.173),  # green
    (0.839, 0.153, 0.157),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # gray
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.812),  # cyan
]


def _color_distance(c1: tuple, c2: tuple) -> float:
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _match_color_to_index(color: tuple[float, ...]) -> int:
    """Match a drawing color to the closest matplotlib tab10 index."""
    dists = [_color_distance(color, tc) for tc in _TAB10_COLORS]
    return min(range(len(dists)), key=lambda i: dists[i])


def _extract_step_curves(
    page,
    plot_left: float,
    plot_right: float,
    plot_top: float,
    plot_bottom: float,
) -> list[tuple[tuple[float, float, float], list[tuple[float, float]]]]:
    """Extract step-function curves from a profile page.

    Returns list of (color_rgb, [(px_x, px_y), ...]) for each curve.
    Only considers drawings with width >= 1.4 and more than 2 items
    (to exclude legend line segments).
    """
    drawings = page.get_drawings()
    curves: list[tuple[tuple[float, float, float], list[tuple[float, float]]]] = []

    for d in drawings:
        c = d.get("color")
        w = d.get("width") or 0
        if not c or w < 1.4:
            continue

        items = d["items"]
        if len(items) <= 2:
            continue

        # Collect all line segment endpoints
        points: list[tuple[float, float]] = []
        for item in items:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if not points:
                    points.append((p1.x, p1.y))
                points.append((p2.x, p2.y))

        # Filter to points within the plot area (with margin)
        margin = 2
        filtered = [
            (x, y) for x, y in points
            if plot_left - margin <= x <= plot_right + margin
            and plot_top - margin <= y <= plot_bottom + margin
        ]

        if len(filtered) > 2:
            color_rgb = (round(c[0], 3), round(c[1], 3), round(c[2], 3))
            curves.append((color_rgb, filtered))

    return curves


def _deduplicate_step_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Remove redundant intermediate points from a step function.

    A step function alternates between horizontal and vertical segments.
    We keep only the corner points where direction changes.
    """
    if len(points) <= 2:
        return points

    result = [points[0]]
    for i in range(1, len(points) - 1):
        prev = result[-1]
        curr = points[i]
        nxt = points[i + 1]

        dy1 = curr[1] - prev[1]
        dy2 = nxt[1] - curr[1]

        # Keep point if direction changes (horizontal -> vertical or vice versa)
        is_h1 = abs(dy1) < 0.5
        is_h2 = abs(dy2) < 0.5
        if is_h1 != is_h2:
            result.append(curr)

    result.append(points[-1])
    return result


# ---------------------------------------------------------------------------
# Bar chart extraction (log-ratio profiles)
# ---------------------------------------------------------------------------

def _extract_bar_data(
    page,
    plot_left: float,
    plot_right: float,
    plot_top: float,
    plot_bottom: float,
) -> list[tuple[tuple[float, float, float], list[float]]]:
    """Extract bar chart data from a log-ratio profile page.

    For log-ratio profiles, bars extend from a baseline (y=0) either
    upward (positive values) or downward (negative values).  We return
    the pixel y-coordinate of the bar's *data end* (the end away from
    the baseline).

    Returns list of (color_rgb, [y_pixel_of_data_end, ...]) for each solver.
    """
    drawings = page.get_drawings()

    from collections import defaultdict
    bars_by_color: dict[tuple, list[tuple[float, float]]] = defaultdict(list)

    for d in drawings:
        fill = d.get("fill")
        if not fill:
            continue
        if all(abs(v - 1.0) < 0.01 for v in fill):
            continue
        if all(abs(v) < 0.01 for v in fill):
            continue

        color = (round(fill[0], 3), round(fill[1], 3), round(fill[2], 3))

        for item in d["items"]:
            if item[0] == "re":
                rect = item[1]
                if (rect.x0 >= plot_left - 2 and rect.x1 <= plot_right + 2
                        and rect.y0 >= plot_top - 2):
                    # Determine which end is the data end vs baseline.
                    # Bars going up: y0 is data end (small pixel), y1 is baseline
                    # Bars going down: y1 is data end (large pixel), y0 is baseline
                    mid_plot = (plot_top + plot_bottom) / 2
                    if abs(rect.y0 - mid_plot) < abs(rect.y1 - mid_plot):
                        data_y = rect.y1
                    else:
                        data_y = rect.y0
                    bars_by_color[color].append((rect.x0, data_y))

    result = []
    for color, bar_list in bars_by_color.items():
        bar_list.sort(key=lambda b: b[0])
        y_vals = [y for _, y in bar_list]
        result.append((color, y_vals))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_profile_pdf(
    pdf_path: str | Path,
    profile_type: str | None = None,
    basis: str | None = None,
) -> list[ProfilePage | LogRatioPage]:
    """Read a profile PDF and extract curve data from all pages.

    Parameters
    ----------
    pdf_path : str or Path
        Path to a profile PDF file.
    profile_type : str, optional
        One of "perf", "data", "log-ratio". Auto-detected from filename if not given.
    basis : str, optional
        "history-based" or "output-based". Auto-detected from filename if not given.

    Returns
    -------
    list[ProfilePage | LogRatioPage]
        One entry per page in the PDF.
    """
    try:
        import fitz  # noqa: F811
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF profile reading. "
            "Install it with: pip install PyMuPDF"
        )

    pdf_path = Path(pdf_path)

    # Auto-detect profile type and basis from filename/path
    fname = pdf_path.stem.lower()
    parent = pdf_path.parent.name.lower()

    if profile_type is None:
        if "perf" in fname or "perf" in parent:
            profile_type = "perf"
        elif "data" in fname or "data" in parent:
            profile_type = "data"
        elif "log-ratio" in fname or "log-ratio" in parent:
            profile_type = "log-ratio"
        else:
            profile_type = "unknown"

    if basis is None:
        combined = fname + " " + parent
        if "hist" in combined and "out" not in combined.replace("hist", ""):
            basis = "history-based"
        elif "out" in combined:
            basis = "output-based"
        else:
            basis = "unknown"

    doc = fitz.open(str(pdf_path))
    pages: list[ProfilePage | LogRatioPage] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        title, tolerance, solver_names = _extract_page_metadata(page)

        if not tolerance:
            tolerance = f"tol_{page_idx + 1}"

        plot_left, plot_right, plot_top, plot_bottom = _find_plot_area(page)

        x_ticks, y_ticks = _extract_axis_ticks(page)

        if profile_type == "log-ratio":
            bar_data_raw = _extract_bar_data(
                page, plot_left, plot_right, plot_top, plot_bottom
            )

            # Convert pixel y to data y using y-axis ticks
            if y_ticks:
                y_pixels = [t[0] for t in y_ticks]
                y_values = [t[1] for t in y_ticks]
                y_map = _build_axis_mapping(y_pixels, y_values, log_scale=False)
            else:
                y_map = (plot_bottom, plot_top, 0.0, 1.0)

            bar_charts = []
            for i, (color, y_tops) in enumerate(bar_data_raw):
                name = solver_names[i] if i < len(solver_names) else f"solver_{i}"
                data_vals = [
                    _pixel_to_data(y, y_map[0], y_map[1], y_map[2], y_map[3])
                    for y in y_tops
                ]
                bar_charts.append(BarChartData(
                    solver_name=name,
                    color_rgb=color,
                    bars=data_vals,
                ))

            pages.append(LogRatioPage(
                basis=basis,
                tolerance=tolerance,
                title=title,
                solver_names=solver_names,
                bar_data=bar_charts,
            ))
        else:
            # Step-function profiles (perf or data)
            is_log_x = profile_type == "perf"

            raw_curves = _extract_step_curves(
                page, plot_left, plot_right, plot_top, plot_bottom
            )

            # Build axis mappings
            if x_ticks:
                x_pixels = [t[0] for t in x_ticks]
                x_values = [t[1] for t in x_ticks]
                x_map = _build_axis_mapping(x_pixels, x_values, log_scale=is_log_x)
            else:
                x_map = (plot_left, plot_right, 0.0, 1.0)

            if y_ticks:
                # Y-axis: pixel increases downward, value increases upward
                y_pixels = [t[0] for t in y_ticks]
                y_values = [t[1] for t in y_ticks]
                # Reverse: largest pixel = smallest value
                y_map = _build_axis_mapping(y_pixels, y_values, log_scale=False)
            else:
                y_map = (plot_bottom, plot_top, 0.0, 1.0)

            curves: list[CurveData] = []
            for i, (color, pixel_points) in enumerate(raw_curves):
                name = solver_names[i] if i < len(solver_names) else f"solver_{i}"

                deduped = _deduplicate_step_points(pixel_points)

                data_points = []
                for px, py in deduped:
                    dx = _pixel_to_data(
                        px, x_map[0], x_map[1], x_map[2], x_map[3],
                        log_scale=is_log_x,
                    )
                    dy = _pixel_to_data(
                        py, y_map[0], y_map[1], y_map[2], y_map[3],
                    )
                    data_points.append((round(dx, 6), round(dy, 6)))

                curves.append(CurveData(
                    solver_name=name,
                    color_rgb=color,
                    points=data_points,
                ))

            text = page.get_text()
            x_label = ""
            y_label = ""
            for line in text.split("\n"):
                s = line.strip().lower()
                if "ratio" in s and "performance" not in s and "log" not in s:
                    x_label = line.strip()
                elif "simplex" in s or "gradient" in s:
                    x_label = line.strip()
                elif "number" in s:
                    x_label = line.strip()

            pages.append(ProfilePage(
                profile_type=profile_type,
                basis=basis,
                tolerance=tolerance,
                title=title,
                x_label=x_label,
                y_label=y_label,
                curves=curves,
            ))

    doc.close()
    return pages


def read_all_profiles(
    profile_paths,
) -> dict[str, list[ProfilePage | LogRatioPage]]:
    """Read all discovered profile PDFs.

    Parameters
    ----------
    profile_paths : ProfilePaths
        From ``result_loader.load_results().profile_paths``.

    Returns
    -------
    dict[str, list[ProfilePage | LogRatioPage]]
        Keyed by profile identifier (e.g. "perf_hist", "data_out").
    """
    result: dict[str, list[ProfilePage | LogRatioPage]] = {}

    for attr, ptype, basis in [
        ("perf_hist", "perf", "history-based"),
        ("perf_out", "perf", "output-based"),
        ("data_hist", "data", "history-based"),
        ("data_out", "data", "output-based"),
        ("log_ratio_hist", "log-ratio", "history-based"),
        ("log_ratio_out", "log-ratio", "output-based"),
    ]:
        path = getattr(profile_paths, attr, None)
        if path and path.exists():
            try:
                pages = read_profile_pdf(path, profile_type=ptype, basis=basis)
                result[attr] = pages
            except Exception:
                pass

    return result
