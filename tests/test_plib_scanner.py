"""Tests for local problem-library discovery."""

from __future__ import annotations

import json

from optiprofiler_agent.advisor.plib_scanner import scan_local_plib


def _make_toy_plib(root):
    (root / "toyprob.py").write_text(
        """\
import numpy as np
from optiprofiler import Problem


class Rosen:
    def __init__(self):
        self.x0 = np.zeros(2)

    def fun(self, x):
        return float(np.sum(x * x))


def load_problem(name):
    return Rosen()


def find_problems(options):
    return ["ROSEN"]
""",
        encoding="utf-8",
    )
    (root / "probinfo_toy.csv").write_text(
        "problem_name,ptype,dim,mb,mcon,mlcon,mnlcon\nROSEN,u,2,0,0,0,0\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'toyprob'\ndependencies = ['numpy>=1.26']\n",
        encoding="utf-8",
    )


def test_scan_local_plib_extracts_core_evidence(tmp_path):
    _make_toy_plib(tmp_path)

    evidence = scan_local_plib(tmp_path, library_name="toy")

    assert evidence.library_name == "toy"
    assert evidence.files_considered == 3
    assert "python" in evidence.languages
    assert "numpy" in evidence.dependencies
    assert evidence.loader_hints == ["toyprob.py"]
    assert evidence.selector_hints == ["toyprob.py"]
    assert evidence.recommended_adapter_shape == "reuse_upstream_selector"
    csv_file = next(item for item in evidence.files if item.path == "probinfo_toy.csv")
    assert "problem_name" in csv_file.columns


def test_scan_local_plib_json_roundtrip(tmp_path):
    _make_toy_plib(tmp_path)

    data = json.loads(scan_local_plib(tmp_path).to_json())

    assert data["library_name"] == tmp_path.name.lower()
    assert data["files_scanned"] == 3
    assert any(item["path"] == "toyprob.py" for item in data["files"])


def test_scan_local_plib_detects_pickle_risk(tmp_path):
    (tmp_path / "bad.py").write_text(
        """\
def make_problem():
    handle = open('data.txt')
    return lambda x: x
""",
        encoding="utf-8",
    )

    evidence = scan_local_plib(tmp_path)

    assert evidence.pickle_risk_hints == ["bad.py"]
    file_evidence = evidence.files[0]
    assert "lambda" in file_evidence.hints
