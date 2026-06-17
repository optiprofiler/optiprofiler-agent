"""Tests for generated custom problem-library wrappers."""

from __future__ import annotations

import importlib.util
import sys

from optiprofiler_agent.advisor.plib_wrapper import (
    scaffold_plib_wrapper,
    smoke_test_plib_wrapper,
)


def _make_toy_source(root):
    (root / "toy_source.py").write_text(
        """\
import numpy as np


class ToyProblem:
    def __init__(self, name):
        self.name = name
        self.x0 = np.array([1.0, -1.0])
        self.xl = np.array([-5.0, -5.0])
        self.xu = np.array([5.0, 5.0])

    def fun(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * x))


def load_problem(name):
    return ToyProblem(name)


def find_problems(options):
    return ["TOY_A", "TOY_B"]
""",
        encoding="utf-8",
    )
    (root / "probinfo_toyplib.csv").write_text(
        "problem_name,ptype,dim,mb,mcon,mlcon,mnlcon\nTOY_A,b,2,2,0,0,0\nTOY_B,b,2,2,0,0,0\n",
        encoding="utf-8",
    )


def _write_optiprofiler_stub(root):
    (root / "optiprofiler.py").write_text(
        """\
class Problem:
    def __init__(self, fun, x0, name="", xl=None, xu=None):
        self.fun = fun
        self.x0 = x0
        self.name = name
        self.xl = xl
        self.xu = xu
""",
        encoding="utf-8",
    )


def test_scaffold_plib_wrapper_generates_tools_and_copies_inputs(tmp_path):
    _make_toy_source(tmp_path)
    stage = tmp_path / "stage"

    result = scaffold_plib_wrapper(tmp_path, "toyplib", staging_dir=stage)

    assert result.library_name == "toyplib"
    assert result.tools_path.exists()
    assert (stage / "toy_source.py").exists()
    assert (stage / "probinfo_toyplib.csv").exists()
    assert result.warnings == []


def test_generated_plib_wrapper_selects_and_loads(tmp_path, monkeypatch):
    _make_toy_source(tmp_path)
    result = scaffold_plib_wrapper(tmp_path, "toyplib", staging_dir=tmp_path / "stage")
    _write_optiprofiler_stub(result.staging_dir)
    monkeypatch.syspath_prepend(str(result.staging_dir))

    spec = importlib.util.spec_from_file_location("toyplib_tools", result.tools_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["toyplib_tools"] = module
    spec.loader.exec_module(module)

    names = module.toyplib_select({"mindim": 1, "maxdim": 20})
    problem = module.toyplib_load(names[0])

    assert names == ["TOY_A", "TOY_B"]
    assert problem.name == "TOY_A"
    assert problem.fun(problem.x0) == 2.0


def test_smoke_test_plib_wrapper_passes(tmp_path):
    _make_toy_source(tmp_path)
    result = scaffold_plib_wrapper(tmp_path, "toyplib", staging_dir=tmp_path / "stage")
    _write_optiprofiler_stub(result.staging_dir)

    smoke = smoke_test_plib_wrapper(result.staging_dir, result.library_name)

    assert smoke.ok, smoke.stderr
    assert smoke.tested_problem_names == ["TOY_A", "TOY_B"]
