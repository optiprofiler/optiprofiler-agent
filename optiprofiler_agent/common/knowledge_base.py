"""Language-aware knowledge loader for OptiProfiler Agent.

Loads structured knowledge from the wiki-based knowledge directory:

  knowledge/
  ├── enums.json         (enum constants)
  ├── _sources/          (raw JSON extractions — immutable)
  │   ├── python/
  │   └── matlab/
  ├── wiki/              (compiled, interlinked markdown pages)
  │   ├── index.md
  │   ├── concepts/  api/  guides/  profiles/  solvers/  troubleshooting/
  ├── common/            (legacy, kept for backward compat)
  ├── python/            (legacy, kept for backward compat)
  └── matlab/            (legacy, kept for backward compat)

Usage:
  kb = KnowledgeBase()
  kb.to_prompt_text("python")   # common + python knowledge
  kb.to_prompt_text("matlab")   # common + matlab knowledge
  kb.to_prompt_text()           # common only (language-agnostic)
"""

from __future__ import annotations

import json
from pathlib import Path

OPTION_CATEGORIES = ("parameters", "feature_options", "profile_options", "problem_options")


class KnowledgeBase:
    """Load and query OptiProfiler's structured knowledge assets."""

    def __init__(self, knowledge_dir: Path | str | None = None):
        if knowledge_dir is None:
            knowledge_dir = Path(__file__).parent.parent / "knowledge"
        self._dir = Path(knowledge_dir)

        self._enums: dict = {}
        self._common_guides: dict[str, str] = {}
        self._lang_data: dict[str, dict] = {"python": {}, "matlab": {}}

        self._load()

    def _load(self):
        enums_path = self._dir / "enums.json"
        if not enums_path.exists():
            enums_path = self._dir / "common" / "enums.json"
        if enums_path.exists():
            with open(enums_path, encoding="utf-8") as f:
                self._enums = json.load(f)

        common_dir = self._dir / "common"
        if common_dir.exists():
            for md_file in sorted(common_dir.glob("*.md")):
                self._common_guides[md_file.stem] = md_file.read_text(encoding="utf-8")

        for lang in ("python", "matlab"):
            sources_dir = self._dir / "_sources" / lang
            legacy_dir = self._dir / lang

            data: dict = {}

            target_dir = sources_dir if sources_dir.exists() else legacy_dir
            if target_dir.exists():
                for json_file in sorted(target_dir.glob("*.json")):
                    with open(json_file, encoding="utf-8") as f:
                        data[json_file.stem] = json.load(f)

            if legacy_dir.exists():
                for md_file in sorted(legacy_dir.glob("*.md")):
                    data[md_file.stem] = md_file.read_text(encoding="utf-8")

            self._lang_data[lang] = data

    # ── query methods ──

    def get_benchmark(self, lang: str) -> dict:
        return self._lang_data.get(lang, {}).get("benchmark", {})

    def get_param(self, lang: str, name: str) -> dict | None:
        bm = self.get_benchmark(lang)
        for cat in OPTION_CATEGORIES:
            if cat in bm and name in bm[cat]:
                return bm[cat][name]
        return None

    def get_enum(self, enum_class: str) -> dict | None:
        return self._enums.get(enum_class)

    def get_classes(self, lang: str) -> dict:
        return self._lang_data.get(lang, {}).get("classes", {})

    def get_plib_tools(self, lang: str) -> dict:
        return self._lang_data.get(lang, {}).get("plib_tools", {})

    def get_api_notes(self, lang: str) -> dict:
        return self._lang_data.get(lang, {}).get("api_notes", {})

    def get_common_guide(self, name: str) -> str | None:
        return self._common_guides.get(name)

    def get_lang_guide(self, lang: str, name: str) -> str | None:
        return self._lang_data.get(lang, {}).get(name)

    # ── prompt assembly ──

    def to_prompt_text(self, language: str | None = None, max_chars: int = 12000) -> str:
        """Build a compact knowledge block for system prompt injection.

        Args:
            language: "python", "matlab", or None (common only).
            max_chars: rough character budget.
        """
        lines: list[str] = []

        # DFO header
        lines.append("## OptiProfiler — Derivative-Free Optimization Benchmarking\n")
        lines.append("The `fun` provides **ONLY function values** — no gradient or Hessian.")
        lines.append("`benchmark()` requires **at least 2 solvers**.\n")

        # Calling convention + Solver signatures
        lang = language or "python"
        bm = self.get_benchmark(lang)

        cc = bm.get("calling_convention", {})
        if cc:
            lines.append("### Calling Convention")
            if cc.get("syntax"):
                lines.append(f"- Syntax: `{cc['syntax']}`")
            if cc.get("solvers"):
                lines.append(f"- Solvers: {cc['solvers']}")
            if cc.get("options"):
                lines.append(f"- Options: {cc['options']}")
            lines.append("")

        sigs = bm.get("solver_signatures", {})
        if sigs:
            lines.append("### Solver Signatures")
            for ptype, sig in sigs.items():
                lines.append(f"- {ptype}: `{sig}`")
            for note in bm.get("solver_notes", []):
                lines.append(f"- {note}")
            lines.append("")

        # Key parameters
        important = [
            "feature_name", "n_runs", "n_jobs", "ptype", "plibs", "mindim",
            "maxdim", "max_eval_factor", "max_tol_order", "seed", "savepath",
            "normalized_scores", "score_only", "silent", "load",
            "custom_problem_libs_path", "noise_level", "noise_type",
            "perturbation_level", "draw_hist_plots",
        ]
        param_lines = []
        for name in important:
            p = self.get_param(lang, name)
            if p:
                default = p.get("default", "")
                desc = p.get("description", "")[:200]
                if default:
                    param_lines.append(f"- `{name}` ({p.get('type', '')}) = {default} — {desc}")
                else:
                    param_lines.append(f"- `{name}` ({p.get('type', '')}) — {desc}")
        if param_lines:
            lines.append("### Key Parameters")
            lines.extend(param_lines)
            lines.append("")

        # Returns
        rets = bm.get("returns", {})
        if rets:
            lines.append("### Return Values")
            for rname, rinfo in rets.items():
                lines.append(f"- `{rname}` ({rinfo.get('type', '')}): {rinfo.get('description', '')}")
            lines.append("")

        # Enums
        if self._enums:
            lines.append("### Enum Values")
            for cls_name, members in self._enums.items():
                vals = ", ".join(f"`{v}`" for v in members.values())
                lines.append(f"- **{cls_name}**: {vals}")
            lines.append("")

        # Classes
        classes = self.get_classes(lang)
        if classes:
            lines.append("### Classes")
            for cls_name, cls_data in classes.items():
                desc = cls_data.get("description", "")[:150]
                props = cls_data.get("properties", {})
                methods = cls_data.get("methods", {})
                lines.append(f"- **{cls_name}**: {desc}")
                if props:
                    lines.append(f"  Properties: {', '.join(f'`{p}`' for p in list(props)[:10])}")
                if methods:
                    lines.append(f"  Methods: {', '.join(f'`{m}`' for m in list(methods)[:10])}")
            lines.append("")

        # Language-specific sections (only inject content for the detected language)
        if language and language in self._lang_data:
            notes = self.get_api_notes(language)
            if notes:
                lines.append(f"### {language.title()}-Specific Notes")
                if notes.get("solver_format"):
                    lines.append(f"- Solver format: {notes['solver_format']}")
                if notes.get("options_format"):
                    lines.append(f"- Options format: {notes['options_format']}")
                if notes.get("vector_convention"):
                    lines.append(f"- Vectors: {notes['vector_convention']}")
                if notes.get("problem_libs"):
                    lines.append(f"- Problem libraries: {', '.join(notes['problem_libs'])}")
                # Only inject notes relevant to this language
                if language == "python" and notes.get("lambda_warning"):
                    lines.append(f"- Warning: {notes['lambda_warning']}")
                if language == "python" and notes.get("pycutest_note"):
                    lines.append(f"- Note: {notes['pycutest_note']}")
                if language == "matlab" and notes.get("matcutest_note"):
                    lines.append(f"- Note: {notes['matcutest_note']}")

                diffs = notes.get("differences_from_python", {})
                if diffs:
                    lines.append("\n**Differences from Python:**")
                    for k, v in diffs.items():
                        lines.append(f"- `{k}`: {v}")
                lines.append("")

            # Language-specific installation
            install = self.get_lang_guide(language, "installation")
            if install:
                lines.append(install[:1500])
                lines.append("")

            # Problem libs guide
            plib_guide = self.get_lang_guide(language, "problem_libs")
            if plib_guide:
                lines.append(plib_guide[:1000])
                lines.append("")

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"
        return text
