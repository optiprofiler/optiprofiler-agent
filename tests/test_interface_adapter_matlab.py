"""Tests for MATLAB support in ``common.interface_adapter``.

Mirrors the breadth of the Python interface-adapter coverage that's
embedded across ``test_debugger.py``.
"""

from optiprofiler_agent.common.interface_adapter import (
    EXPECTED_SIGNATURES,
    analyze_solver,
    generate_wrapper,
    generate_wrapper_with_context,
)


class TestMatlabAnalyze:

    def test_reordered_params_needs_wrapper(self):
        code = (
            "function x = my_solver(x0, fun)\n"
            "    x = fun(x0);\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab")
        assert analysis.func_name == "my_solver"
        assert analysis.needs_wrapper
        assert analysis.reorder_needed

    def test_canonical_params_no_wrapper(self):
        code = (
            "function x = my_solver(fun, x0)\n"
            "    x = fminsearch(fun, x0);\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab")
        assert not analysis.needs_wrapper

    def test_aliased_params_recognised(self):
        # Aliases in the canonical order are treated as already-correct;
        # no wrapper is generated. The mapping is recorded in `notes`.
        code = (
            "function x = my_solver(objective, x_init)\n"
            "    x = fminsearch(objective, x_init);\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab")
        assert analysis.matched_params.get("fun") == "objective"
        assert analysis.matched_params.get("x0") == "x_init"
        assert not analysis.needs_wrapper
        assert any("alias" in n for n in analysis.notes)

    def test_aliased_params_reordered_needs_wrapper(self):
        code = (
            "function x = my_solver(x_init, objective)\n"
            "    x = fminsearch(objective, x_init);\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab")
        assert analysis.matched_params.get("fun") == "objective"
        assert analysis.needs_wrapper
        assert analysis.reorder_needed

    def test_bound_constrained_signature(self):
        code = (
            "function x = my_bnd(fun, x0, xl, xu)\n"
            "    x = x0;\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab", problem_type="bound_constrained")
        assert not analysis.needs_wrapper

    def test_missing_required_param(self):
        code = (
            "function x = my_solver(fun)\n"
            "    x = 0;\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab")
        assert "x0" in analysis.missing_params
        assert analysis.needs_wrapper

    def test_extra_unknown_param(self):
        code = (
            "function x = my_solver(fun, x0, options, verbose)\n"
            "    x = x0;\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab")
        assert "options" in analysis.extra_params or "verbose" in analysis.extra_params
        assert analysis.needs_wrapper

    def test_parse_error_returns_safe_default(self):
        analysis = analyze_solver("this is not matlab code", language="matlab")
        assert analysis.func_name == "<parse_error>"
        assert analysis.needs_wrapper


class TestMatlabWrapper:

    def test_wrapper_signature_uses_canonical_order(self):
        code = "function x = my_solver(x0, fun)\n    x = fun(x0);\nend\n"
        analysis = analyze_solver(code, language="matlab")
        wrapper = generate_wrapper(analysis, language="matlab")
        assert "function x = my_solver_wrapper(fun, x0)" in wrapper
        assert wrapper.strip().endswith("end")

    def test_wrapper_calls_inner_with_matched_args(self):
        code = (
            "function x = my_solver(objective, x_init)\n"
            "    x = fminsearch(objective, x_init);\n"
            "end\n"
        )
        analysis = analyze_solver(code, language="matlab")
        wrapper = generate_wrapper(analysis, language="matlab")
        assert "my_solver(fun, x0)" in wrapper

    def test_wrapper_column_vector_idiom(self):
        code = "function x = my_solver(x0, fun)\n    x = fun(x0);\nend\n"
        analysis = analyze_solver(code, language="matlab")
        wrapper = generate_wrapper(analysis, language="matlab")
        assert "x = x(:);" in wrapper

    def test_wrapper_notes_missing_params_in_comment(self):
        code = "function x = my_solver(fun)\n    x = 0;\nend\n"
        analysis = analyze_solver(code, language="matlab")
        wrapper = generate_wrapper(analysis, language="matlab")
        assert "Unused parameters" in wrapper

    def test_bound_constrained_wrapper(self):
        code = "function x = my_bnd(x0, fun, xl, xu)\n    x = x0;\nend\n"
        analysis = analyze_solver(
            code, language="matlab", problem_type="bound_constrained"
        )
        wrapper = generate_wrapper(analysis, language="matlab", problem_type="bound_constrained")
        expected_sig = ", ".join(EXPECTED_SIGNATURES["bound_constrained"])
        assert f"function x = my_bnd_wrapper({expected_sig})" in wrapper


class TestMatlabWrapperContext:

    def test_with_context_no_wrapper_when_canonical(self):
        code = "function x = my_solver(fun, x0)\n    x = x0;\nend\n"
        analysis, wrapper = generate_wrapper_with_context(code, language="matlab")
        assert not analysis.needs_wrapper
        assert wrapper == ""

    def test_with_context_generates_wrapper_when_needed(self):
        code = "function x = my_solver(x0, fun)\n    x = fun(x0);\nend\n"
        analysis, wrapper = generate_wrapper_with_context(code, language="matlab")
        assert analysis.needs_wrapper
        assert "my_solver_wrapper" in wrapper
