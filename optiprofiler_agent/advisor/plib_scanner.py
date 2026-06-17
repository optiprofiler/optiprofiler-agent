"""Local discovery for custom problem-library scaffolding."""

from __future__ import annotations

import ast
import csv
import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TEXT_EXTENSIONS = {
    ".py", ".m", ".f", ".f90", ".c", ".cpp", ".h", ".hpp",
    ".csv", ".json", ".txt", ".md", ".rst", ".toml", ".cfg",
    ".ini", ".yaml", ".yml",
}
DATA_EXTENSIONS = {".csv", ".json", ".txt", ".dat", ".mat", ".npy", ".npz"}
SOURCE_EXTENSIONS = {".py", ".m", ".f", ".f90", ".c", ".cpp", ".h", ".hpp"}
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", "node_modules", "dist", "build",
}
MAX_FILES = 400
MAX_TEXT_BYTES = 256_000


@dataclass(frozen=True)
class PlibFileEvidence:
    """Evidence extracted from one local file."""

    path: str
    kind: str
    size_bytes: int
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlibScanEvidence:
    """Structured local evidence for a custom problem-library wrapper."""

    source_dir: str
    library_name: str
    files_scanned: int
    files_considered: int
    skipped_files: list[str]
    languages: list[str]
    dependencies: list[str]
    loader_hints: list[str]
    selector_hints: list[str]
    data_files: list[str]
    pickle_risk_hints: list[str]
    recommended_adapter_shape: str
    files: list[PlibFileEvidence]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _safe_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", path.name.lower()).strip("_") or "custom"


def _iter_candidate_files(src_dir: Path) -> tuple[list[Path], list[str]]:
    candidates: list[Path] = []
    skipped: list[str] = []
    for path in sorted(src_dir.rglob("*")):
        rel = path.relative_to(src_dir)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
            if path.is_file():
                skipped.append(str(rel))
            continue
        if not path.is_file():
            continue
        if len(candidates) >= MAX_FILES:
            skipped.append(str(rel))
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS | DATA_EXTENSIONS:
            skipped.append(str(rel))
            continue
        candidates.append(path)
    return candidates, skipped


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_TEXT_BYTES:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_python(path: Path, text: str) -> tuple[list[str], list[str], list[str]]:
    symbols: list[str] = []
    imports: list[str] = []
    hints: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return symbols, imports, ["python_syntax_unreadable"]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
        elif isinstance(node, ast.Lambda):
            hints.append("lambda")
        elif isinstance(node, ast.With):
            hints.append("context_manager")

    lowered = text.lower()
    if "problem(" in lowered or "from optiprofiler" in lowered or "opclasses" in lowered:
        hints.append("optiprofiler_problem_contract")
    if any(name in symbols for name in ("load", "get_problem", "import_problem", "make_problem")):
        hints.append("single_problem_loader")
    if any("select" in name or "find" in name or "filter" in name for name in symbols):
        hints.append("selector_function")
    if "pickle" in lowered or "multiprocessing" in lowered:
        hints.append("parallelism_related")
    return sorted(set(symbols)), sorted(set(imports)), sorted(set(hints))


def _extract_table_columns(path: Path, text: str) -> list[str]:
    if path.suffix.lower() == ".csv":
        try:
            sample = text.splitlines()[:5]
            rows = list(csv.reader(sample))
        except csv.Error:
            return []
        return [cell.strip() for cell in rows[0]] if rows else []
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return sorted(str(key) for key in data[0].keys())
        if isinstance(data, dict):
            return sorted(str(key) for key in data.keys())
    return []


def _extract_pyproject_dependencies(path: Path, text: str) -> list[str]:
    if path.name != "pyproject.toml":
        return []
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    deps = data.get("project", {}).get("dependencies", []) or []
    optional = data.get("project", {}).get("optional-dependencies", {}) or {}
    for values in optional.values():
        deps.extend(values or [])
    names: list[str] = []
    for dep in deps:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", str(dep))
        if match:
            names.append(match.group(1).lower())
    return sorted(set(names))


def _language_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".m":
        return "matlab"
    if ext in {".f", ".f90"}:
        return "fortran"
    if ext in {".c", ".cpp", ".h", ".hpp"}:
        return "c/c++"
    return None


def _kind_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SOURCE_EXTENSIONS:
        return "source"
    if ext in DATA_EXTENSIONS:
        return "data"
    if path.name.lower().startswith("readme") or ext in {".md", ".rst"}:
        return "docs"
    if ext in {".toml", ".cfg", ".ini", ".yaml", ".yml"}:
        return "config"
    return "other"


def _recommend_shape(files: list[PlibFileEvidence], selector_hints: list[str], data_files: list[str]) -> str:
    if selector_hints:
        return "reuse_upstream_selector"
    if any("problem_name" in f.columns for f in files):
        return "csv_metadata"
    if data_files:
        return "data_file_loader_plus_csv"
    return "loader_plus_generated_csv"


def scan_local_plib(src_dir: str | Path, library_name: str | None = None) -> PlibScanEvidence:
    """Scan a local problem-library directory and return wrapper evidence."""
    root = Path(src_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    name = library_name or _safe_name(root)
    candidates, skipped = _iter_candidate_files(root)

    files: list[PlibFileEvidence] = []
    languages: set[str] = set()
    dependencies: set[str] = set()
    loader_hints: set[str] = set()
    selector_hints: set[str] = set()
    data_files: list[str] = []
    pickle_risks: set[str] = set()

    for path in candidates:
        rel = str(path.relative_to(root))
        text = _read_text(path)
        lang = _language_for(path)
        if lang:
            languages.add(lang)
        kind = _kind_for(path)
        if kind == "data":
            data_files.append(rel)

        symbols: list[str] = []
        imports: list[str] = []
        hints: list[str] = []
        if path.suffix.lower() == ".py" and text:
            symbols, imports, hints = _extract_python(path, text)
            dependencies.update(imports)
        dependencies.update(_extract_pyproject_dependencies(path, text))
        columns = _extract_table_columns(path, text) if text else []

        lowered = text.lower()
        if any(token in lowered for token in ("import_problem", "get_problem", "load_problem", "_load", "mylib_load")):
            loader_hints.add(rel)
        if any(token in lowered for token in ("select", "find_problems", "filter", "secup")):
            selector_hints.add(rel)
        if "lambda" in hints or "file handle" in lowered or "open(" in lowered:
            pickle_risks.add(rel)

        files.append(
            PlibFileEvidence(
                path=rel,
                kind=kind,
                size_bytes=path.stat().st_size,
                symbols=symbols[:40],
                imports=imports[:40],
                columns=columns[:40],
                hints=hints[:20],
            )
        )

    return PlibScanEvidence(
        source_dir=str(root),
        library_name=name,
        files_scanned=len(candidates),
        files_considered=len(files),
        skipped_files=skipped[:80],
        languages=sorted(languages),
        dependencies=sorted(dependencies),
        loader_hints=sorted(loader_hints),
        selector_hints=sorted(selector_hints),
        data_files=sorted(data_files),
        pickle_risk_hints=sorted(pickle_risks),
        recommended_adapter_shape=_recommend_shape(files, sorted(selector_hints), data_files),
        files=files,
    )
