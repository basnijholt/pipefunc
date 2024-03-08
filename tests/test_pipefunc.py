"""Tests for pipefunc.py."""

from __future__ import annotations

import inspect

import pytest

from pipefunc import (
    Pipeline,
    PipelineFunction,
    Sweep,
    count_sweep,
    get_precalculation_order,
    pipefunc,
)
from pipefunc._pipefunc import _combine_nodes, _get_signature


def test_pipeline_and_all_arg_combinations() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True)

    fc = pipeline.func("c")
    fd = pipeline.func("d")
    c = f1(a=2, b=3)
    assert fc(a=2, b=3) == c == fc(b=3, a=2) == 5
    assert fd(a=2, b=3) == f2(b=3, c=c) == fd(b=3, c=c) == 15

    fe = pipeline.func("e")
    assert fe(a=2, b=3, x=1) == fe(a=2, b=3, d=15, x=1) == f3(c=c, d=15, x=1) == 75

    all_args = pipeline.all_arg_combinations()
    assert all_args == {
        "c": {("a", "b")},
        "d": {("a", "b", "x"), ("b", "c", "x")},
        "e": {("a", "b", "d", "x"), ("a", "b", "x"), ("b", "c", "x"), ("c", "d", "x")},
    }
    assert pipeline.all_arg_combinations(root_args_only=True) == {
        "c": {("a", "b")},
        "d": {("a", "b", "x")},
        "e": {("a", "b", "x")},
    }

    kw = {"a": 2, "b": 3, "x": 1}
    kw["c"] = f1(a=kw["a"], b=kw["b"])
    kw["d"] = f2(b=kw["b"], c=kw["c"])
    kw["e"] = f3(c=kw["c"], d=kw["d"], x=kw["x"])
    for params in all_args["e"]:
        _kw = {k: kw[k] for k in params}
        assert fe(**_kw) == kw["e"]


@pytest.mark.parametrize(
    "f2",
    [
        PipelineFunction(
            lambda b, c, x: b * c * x,
            output_name="d",
            renames={"x": "xx"},
        ),
        PipelineFunction(lambda b, c, xx: b * c * xx, output_name="d"),
    ],
)
def test_pipeline_and_all_arg_combinations_rename(f2):
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True)

    fc = pipeline.func("c")
    fd = pipeline.func("d")
    c = f1(a=2, b=3)
    assert fc(a=2, b=3) == c == fc(b=3, a=2) == 5
    assert fd(a=2, b=3, xx=1) == f2(b=3, c=c, xx=1) == fd(b=3, c=c, xx=1) == 15

    fe = pipeline.func("e")
    assert (
        fe(a=2, b=3, x=1, xx=1)
        == fe(a=2, b=3, d=15, x=1, xx=1)
        == f3(c=c, d=15, x=1)
        == 75
    )

    all_args = pipeline.all_arg_combinations()
    assert all_args == {
        "c": {("a", "b")},
        "d": {("a", "b", "xx"), ("b", "c", "xx")},
        "e": {
            ("a", "b", "d", "x"),
            ("a", "b", "x", "xx"),
            ("b", "c", "x", "xx"),
            ("c", "d", "x"),
        },
    }

    assert pipeline.all_arg_combinations(root_args_only=True) == {
        "c": {("a", "b")},
        "d": {("a", "b", "xx")},
        "e": {("a", "b", "x", "xx")},
    }


def test_disjoint_pipelines() -> None:
    @pipefunc(output_name="x")
    def f(a, b):
        return a + b

    @pipefunc(output_name="y")
    def g(c, d):
        return c * d

    p = Pipeline([f, g])
    assert p("x", a=1, b=2) == 3
    assert p("y", c=3, d=4) == 12


def test_different_defaults() -> None:
    @pipefunc(output_name="c")
    def f(a, b=1):
        return a + b

    @pipefunc(output_name="y")
    def g(c, b=2):
        return c * b

    p = Pipeline([f, g])
    with pytest.raises(ValueError, match="Inconsistent default values"):
        _ = p.graph


def test_output_name_in_kwargs():
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    assert Pipeline([f])("a", a=1) == 1


def test_profiling():
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c, b=2):
        return c * b

    p = Pipeline([f, g], debug=True, profile=True)
    p("d", a=1, b=2)
    p.resources_report()
    for f in p.functions:
        f.set_profiling(enable=False)
    with pytest.raises(ValueError, match="Profiling is not enabled"):
        p.resources_report()


def test_pipeline_function_and_execution():
    def func1(a, b=2):
        return a + b

    def func2(x):
        return 2 * x

    def func3(y, z=3):
        return y - z

    pipe_func1 = PipelineFunction(func1, "out1", renames={"a": "a1"})
    pipe_func2 = PipelineFunction(func2, "out2", renames={"x": "x2"})
    pipe_func3 = PipelineFunction(func3, "out3", renames={"y": "y3", "z": "z3"})

    pipeline = Pipeline([pipe_func1, pipe_func2, pipe_func3], debug=True, profile=True)

    # Create _Function instances
    function1 = pipeline.func("out1")
    function2 = pipeline.func("out2")
    function3 = pipeline.func("out3")

    # Test calling the functions with keyword arguments
    assert function1(a1=3, b=2) == 5
    assert function2(x2=4) == 8
    assert function3(y3=9, z3=3) == 6

    # Test calling the functions with dict arguments
    assert function1.call_with_dict({"a1": 3, "b": 2}) == 5
    assert function2.call_with_dict({"x2": 4}) == 8
    assert function3.call_with_dict({"y3": 9, "z3": 3}) == 6

    # Test calling the functions with `execute` method
    assert function1.call_with_root_args(3, 2) == 5
    assert function1.call_with_root_args(a1=3, b=2) == 5
    assert function2.call_with_root_args(4) == 8
    assert function3.call_with_root_args(9, 3) == 6

    # Test the pipeline object itself
    assert pipeline("out1", a1=3, b=2) == 5
    assert pipeline("out2", x2=4) == 8
    assert pipeline("out3", y3=9, z3=3) == 6


def test_pipeline_function_profile():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipeline_function = PipelineFunction(f1, output_name="c", profile=True)
    assert pipeline_function.profile
    assert pipeline_function.profiling_stats is not None
    pipeline_function.profile = False
    assert not pipeline_function.profile
    assert pipeline_function.profiling_stats is None


def test_pipeline_function_str():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipeline_function = PipelineFunction(f1, output_name="c")
    assert str(pipeline_function) == "f1(a, b) → c"


def test_pipeline_function_getstate_setstate():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipeline_function = PipelineFunction(f1, output_name="c")
    state = pipeline_function.__getstate__()

    # We'll validate getstate by asserting that 'func' in the state
    # is a bytes object (dumped by cloudpickle) and other attributes
    # are as expected
    assert isinstance(state["func"], bytes)
    assert state["output_name"] == "c"

    # Now we'll test setstate by creating a new instance, applying setstate and
    # verifying that the object attributes match the original
    new_pipeline_function = PipelineFunction.__new__(PipelineFunction)
    new_pipeline_function.__setstate__(state)

    assert new_pipeline_function.output_name == pipeline_function.output_name
    assert new_pipeline_function.parameters == pipeline_function.parameters
    assert new_pipeline_function.func(2, 3) == pipeline_function.func(
        2,
        3,
    )  # the functions behave the same


def test_complex_pipeline():
    def f1(a, b, c, d):
        return a + b + c + d

    def f2(a, b, e):
        return a + b + e

    def f3(a, b, f1):
        return a + b + f1

    def f4(f1, f2, f3):
        return f1 + f2 + f3

    def f5(f1, f4):
        return f1 + f4

    def f6(b, f5):
        return b + f5

    def f7(a, f2, f6):
        return a + f2 + f6

    pipeline = Pipeline([f1, f2, f3, f4, f5, f6, f7])

    pipeline("f7", a=1, b=2, c=3, d=4, e=5)


def test_tuple_outputs():
    cache = True

    @pipefunc(
        output_name=("c", "_throw"),
        profile=True,
        debug=True,
        cache=cache,
        output_picker=dict.__getitem__,
    )
    def f_c(a, b):
        return {"c": a + b, "_throw": 1}

    @pipefunc(output_name=("d", "e"), cache=cache)
    def f_d(b, c, x=1):  # noqa: ARG001
        return b * c, 1

    @pipefunc(
        output_name=("g", "h"),
        output_picker=getattr,
        cache=cache,
    )
    def f_e(c, e, x=1):  # noqa: ARG001
        from types import SimpleNamespace

        print(f"Called f_e with c={c} and e={e}")
        return SimpleNamespace(g=c + e, h=c - e)

    @pipefunc(output_name="i", cache=cache)
    def f_i(h, g):
        return h + g

    pipeline = Pipeline([f_c, f_d, f_e, f_i], debug=True, profile=True, cache="shared")
    f = pipeline.func("i")
    assert f.call_full_output(a=1, b=2, x=3)["i"] == f(a=1, b=2, x=3)
    assert (
        pipeline.arg_combinations("g", root_args_only=True)
        == pipeline.arg_combinations("h", root_args_only=True)
        == pipeline.arg_combinations(("g", "h"), root_args_only=True)
        == ("a", "b", "x")
    )
    assert pipeline.cache.cache[(("d", "e"), (("a", 1), ("b", 2), ("x", 3)))] == (6, 1)
    assert pipeline.func(("g", "h"))(a=1, b=2, x=3).g == 4
    assert pipeline.func_dependencies("i") == [("c", "_throw"), ("d", "e"), ("g", "h")]

    assert (
        pipeline.func_dependencies("g")
        == pipeline.func_dependencies("h")
        == pipeline.func_dependencies(("g", "h"))
        == [("c", "_throw"), ("d", "e")]
    )

    f = pipeline.func(("g", "h"))
    r = f(a=1, b=2, x=3)
    assert r.g == 4
    assert r.h == 2

    edges = {
        (f_c, f_d): {"arg": "c"},
        (f_c, f_e): {"arg": "c"},
        ("a", f_c): {"arg": "a"},
        ("b", f_c): {"arg": "b"},
        ("b", f_d): {"arg": "b"},
        (f_d, f_e): {"arg": "e"},
        ("x", f_d): {"arg": "x"},
        ("x", f_e): {"arg": "x"},
        (f_e, f_i): {"arg": ("h", "g")},
    }
    assert edges == dict(pipeline.graph.edges)

    assert dict(pipeline.graph.nodes) == {
        f_c: {},
        "a": {"default_value": inspect._empty},
        "b": {"default_value": inspect._empty},
        f_d: {},
        "x": {"default_value": 1},
        f_e: {},
        f_i: {},
    }


def test_execution_order():
    @pipefunc(output_name=("d", "e"))
    def f_d(b, g, x=1):  # noqa: ARG001
        pass

    @pipefunc(output_name=("g", "h"))
    def f_e(a, x=1):  # noqa: ARG001
        pass

    @pipefunc(output_name="gg")
    def f_gg(g):  # noqa: ARG001
        pass

    @pipefunc(output_name="i")
    def f_i(gg, b, e):  # noqa: ARG001
        pass

    pipeline = Pipeline([f_d, f_e, f_i, f_gg])
    sweep = Sweep({"a": [1, 2], "b": [3, 4], "x": [5, 6]})
    cnt = count_sweep("i", sweep, pipeline)
    # f_d is skipped because max(cnt) is 1
    assert get_precalculation_order(pipeline, cnt) == [f_e, f_gg]


def test_identify_combinable_nodes():
    @pipefunc(output_name=("d", "e"))
    def f_d(b, g, x=1):  # noqa: ARG001
        pass

    @pipefunc(output_name=("g", "h"))
    def f_e(a, x=1):  # noqa: ARG001
        pass

    @pipefunc(output_name="gg")
    def f_gg(g):  # noqa: ARG001
        pass

    @pipefunc(output_name="i")
    def f_i(gg, b, e):  # noqa: ARG001
        pass

    pipeline = Pipeline([f_d, f_e, f_i, f_gg])
    combinable_nodes = pipeline._identify_combinable_nodes(
        "i",
        conservatively_combine=True,
    )
    assert combinable_nodes == {f_gg: {f_e}}
    sig_in, sig_out = _get_signature(combinable_nodes, pipeline.graph)
    assert sig_in == {f_gg: {"a", "x"}}
    assert sig_out == {f_gg: {"gg", "h", "g"}}
    combinable_nodes = pipeline._identify_combinable_nodes(
        "i",
        conservatively_combine=False,
    )
    assert combinable_nodes == {f_gg: {f_e}, f_i: {f_d}}
    sig_in, sig_out = _get_signature(combinable_nodes, pipeline.graph)
    assert sig_in == {f_gg: {"a", "x"}, f_i: {"g", "b", "gg", "x"}}
    assert sig_out == {f_gg: {"gg", "h", "g"}, f_i: {"d", "i"}}


def test_conservatively_combine():
    @pipefunc(output_name="x")
    def f1(a):
        return a

    @pipefunc(output_name="y")
    def f2(b, x):
        return x * b

    @pipefunc(output_name="z")
    def f3(b, x, y):
        return x * y * b

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True)

    root_args = pipeline.all_arg_combinations(root_args_only=True)
    assert root_args == {"x": {("a",)}, "y": {("a", "b")}, "z": {("a", "b")}}

    # Test with conservatively_combine=True
    combinable_nodes_true = pipeline._identify_combinable_nodes(
        "z",
        conservatively_combine=True,
    )
    assert combinable_nodes_true == {}

    # Test simplified_pipeline with conservatively_combine=True
    simplified_pipeline_true = pipeline.simplified_pipeline(
        "z",
        conservatively_combine=True,
    )
    simplified_functions_true = simplified_pipeline_true.functions
    assert len(simplified_functions_true) == 3
    function_names_true = [f.__name__ for f in simplified_functions_true]
    assert "f1" in function_names_true
    assert "f2" in function_names_true
    assert "f3" in function_names_true

    # Test with conservatively_combine=False
    combinable_nodes_false = pipeline._identify_combinable_nodes(
        "z",
        conservatively_combine=False,
    )
    assert combinable_nodes_false == {f3: {f2}}

    # Test simplified_pipeline with conservatively_combine=False
    simplified_pipeline_false = pipeline.simplified_pipeline(
        "z",
        conservatively_combine=False,
    )
    simplified_functions_false = simplified_pipeline_false.functions
    assert len(simplified_functions_false) == 2
    function_names_false = [f.__name__ for f in simplified_functions_false]
    assert "f1" in function_names_false
    assert "combined_f3" in function_names_false

    # Check that the combined function has the expected input and output arguments
    combined_f3 = next(
        f for f in simplified_functions_false if f.__name__ == "combined_f3"
    )
    assert combined_f3.parameters == ["b", "x"]
    assert combined_f3.output_name == "z"

    # Check that the simplified pipeline produces the same output as the original pipeline
    input_data = {"a": 2, "b": 3}
    original_output = pipeline("z", **input_data)
    simplified_output_true = simplified_pipeline_true("z", **input_data)
    simplified_output_false = simplified_pipeline_false("z", **input_data)
    assert original_output == simplified_output_true == simplified_output_false

    assert pipeline._func_node_colors() == ["C1", "C0", "C0"]


def test_identify_combinable_nodes2():
    def f1(a, b, c, d):
        return a + b + c + d

    def f2(a, b, e):
        return a + b + e

    def f3(a, b, f1):
        return a + b + f1

    def f4(f1, f3):
        return f1 + f3

    def f5(f1, f4):
        return f1 + f4

    def f6(b, f5):
        return b + f5

    def f7(a, f2, f6):
        return a + f2 + f6

    pipeline = Pipeline([f1, f2, f3, f4, f5, f6, f7])
    m = pipeline.node_mapping

    expected = {m["f6"]: {m["f1"], m["f3"], m["f4"], m["f5"]}}
    combinable_nodes = pipeline._identify_combinable_nodes("f7")
    simplified_combinable_nodes = _combine_nodes(combinable_nodes)
    assert simplified_combinable_nodes == expected

    sig_in, sig_out = _get_signature(simplified_combinable_nodes, pipeline.graph)
    assert sig_in == {m["f6"]: {"a", "b", "c", "d"}}
    assert sig_out == {m["f6"]: {"f6"}}

    # Test simplified_pipeline
    simplified_pipeline = pipeline.simplified_pipeline()
    assert (
        pipeline._unique_leaf_node().output_name
        == simplified_pipeline._unique_leaf_node().output_name
        == "f7"
    )
    simplified_functions = simplified_pipeline.functions

    # Check that the simplified pipeline has the expected number of functions
    assert len(simplified_functions) == 3

    # Check that the simplified pipeline has the expected function names
    function_names = [f.__name__ for f in simplified_functions]
    assert "f2" in function_names
    assert "combined_f6" in function_names
    assert "f7" in function_names

    # Check that the combined function has the expected input and output arguments
    combined_f6 = next(f for f in simplified_functions if f.__name__ == "combined_f6")
    assert combined_f6.parameters == ["a", "b", "c", "d"]
    assert combined_f6.output_name == "f6"

    # Check that the simplified pipeline produces the same output as the original pipeline
    input_data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    original_output = pipeline("f7", **input_data)
    simplified_output = simplified_pipeline("f7", **input_data)
    assert original_output == simplified_output


@pytest.mark.parametrize("cache", [True, False])
def test_full_output(cache):
    from pipefunc import Pipeline

    def f1(a, b):
        return a + b

    def f2(f1):
        return 2 * f1

    def f3(a, f2):
        return a + f2

    pipeline = Pipeline([f1, f2, f3])
    for f in pipeline.functions:
        f.cache = cache
    pipeline("f3", a=1, b=2)
    func = pipeline.func("f3")
    assert func.call_full_output(a=1, b=2) == {
        "a": 1,
        "b": 2,
        "f1": 3,
        "f2": 6,
        "f3": 7,
    }


def test_pipeline_function_annotations():
    def add_numbers(a: int, b: float) -> tuple[int, float]:
        return a + 1, b + 1.0

    add_func = PipelineFunction(
        add_numbers,
        output_name=("a_plus_one", "b_plus_one"),
        renames={"a": "x", "b": "y"},
    )

    assert add_func.parameter_annotations == {"x": int, "y": float}
    assert add_func.output_annotation == tuple[int, float]

    result = add_func(x=1, y=2.0)
    assert result == (2, 3.0)

    assert str(add_func) == "add_numbers(x, y) → a_plus_one, b_plus_one"
    assert repr(add_func) == "PipelineFunction(add_numbers)"
