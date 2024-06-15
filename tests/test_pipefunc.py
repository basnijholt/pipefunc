"""Tests for pipefunc.py."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from pipefunc import NestedPipeFunc, PipeFunc, Pipeline, pipefunc
from pipefunc._cache import LRUCache
from pipefunc.exceptions import UnusedParametersError
from pipefunc.resources import Resources

if TYPE_CHECKING:
    from pathlib import Path


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

    all_args = pipeline.all_arg_combinations
    assert all_args == {
        "c": {("a", "b")},
        "d": {("a", "b", "x"), ("b", "c", "x")},
        "e": {("a", "b", "d", "x"), ("a", "b", "x"), ("b", "c", "x"), ("c", "d", "x")},
    }
    assert pipeline.all_root_args == {
        "c": ("a", "b"),
        "d": ("a", "b", "x"),
        "e": ("a", "b", "x"),
    }

    kw = {"a": 2, "b": 3, "x": 1}
    kw["c"] = f1(a=kw["a"], b=kw["b"])
    kw["d"] = f2(b=kw["b"], c=kw["c"])
    kw["e"] = f3(c=kw["c"], d=kw["d"], x=kw["x"])
    for params in all_args["e"]:
        _kw = {k: kw[k] for k in params}
        assert fe(**_kw) == kw["e"]

    # Test NestedPipeFunc
    f_nested = NestedPipeFunc([f1, f2])
    assert f_nested.output_name == ("c", "d")
    assert f_nested.parameters == ("a", "b", "x")
    assert f_nested.defaults == {"x": 1}
    assert f_nested(a=2, b=3) == (5, 15)
    assert f_nested.renames == {}
    f_nested.update_renames({"a": "a1", "b": "b1"})
    assert f_nested.renames == {"a": "a1", "b": "b1"}
    f_nested.update_renames({"a": "a2", "b": "b2"}, overwrite=True, update_from="original")
    assert f_nested.renames == {"a": "a2", "b": "b2"}
    assert f_nested.parameters == ("a2", "b2", "x")
    f_nested_copy = f_nested.copy()
    assert f_nested_copy.renames == f_nested.renames
    assert f_nested_copy(a2=2, b2=3) == (5, 15)
    f_nested_copy.update_renames({}, overwrite=True)
    pipeline = Pipeline([f_nested_copy, f3])
    assert pipeline("e", a=2, b=3, x=1) == 75

    assert str(pipeline).startswith("Pipeline:")


def test_pipeline_and_all_arg_combinations_lazy() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True, lazy=True)

    fc = pipeline.func("c")
    fd = pipeline.func("d")
    c = f1(a=2, b=3)
    assert fc(a=2, b=3).evaluate() == c == fc(b=3, a=2).evaluate() == 5
    assert fd(a=2, b=3).evaluate() == f2(b=3, c=c) == fd(b=3, c=c).evaluate() == 15

    fe = pipeline.func("e")
    assert (
        fe(a=2, b=3, x=1).evaluate()
        == fe(a=2, b=3, d=15, x=1).evaluate()
        == f3(c=c, d=15, x=1)
        == 75
    )

    all_args = pipeline.all_arg_combinations

    kw = {"a": 2, "b": 3, "x": 1}
    kw["c"] = f1(a=kw["a"], b=kw["b"])
    kw["d"] = f2(b=kw["b"], c=kw["c"])
    kw["e"] = f3(c=kw["c"], d=kw["d"], x=kw["x"])
    for params in all_args["e"]:
        _kw = {k: kw[k] for k in params}
        assert fe(**_kw).evaluate() == kw["e"]


@pytest.mark.parametrize(
    "f2",
    [
        PipeFunc(
            lambda b, c, x: b * c * x,
            output_name="d",
            renames={"x": "xx"},
        ),
        PipeFunc(lambda b, c, xx: b * c * xx, output_name="d"),
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
    assert fe(a=2, b=3, x=1, xx=1) == fe(a=2, b=3, d=15, x=1) == f3(c=c, d=15, x=1) == 75

    all_args = pipeline.all_arg_combinations
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

    assert pipeline.all_root_args == {
        "c": ("a", "b"),
        "d": ("a", "b", "xx"),
        "e": ("a", "b", "x", "xx"),
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

    with pytest.raises(ValueError, match="Inconsistent default values"):
        Pipeline([f, g])


def test_output_name_in_kwargs():
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    p = Pipeline([f])
    with pytest.raises(ValueError, match="cannot be provided in"):
        assert p("a", a=1)


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


def test_pipe_func_and_execution():
    def func1(a, b=2):
        return a + b

    def func2(x):
        return 2 * x

    def func3(y, z=3):
        return y - z

    pipe_func1 = PipeFunc(func1, "out1", renames={"a": "a1"})
    pipe_func2 = PipeFunc(func2, "out2", renames={"x": "x2"})
    pipe_func3 = PipeFunc(func3, "out3", renames={"y": "y3", "z": "z3"})

    pipeline = Pipeline([pipe_func1, pipe_func2, pipe_func3], debug=True, profile=True)

    # Create _PipelineAsFunc instances
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


def test_pipe_func_profile():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipe_func = PipeFunc(f1, output_name="c", profile=True)
    assert pipe_func.profile
    assert pipe_func.profiling_stats is not None
    pipe_func.profile = False
    assert not pipe_func.profile
    assert pipe_func.profiling_stats is None


def test_pipe_func_str():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipe_func = PipeFunc(f1, output_name="c")
    assert str(pipe_func) == "f1(...) → c"


def test_pipe_func_getstate_setstate():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipe_func = PipeFunc(f1, output_name="c")
    state = pipe_func.__getstate__()

    # We'll validate getstate by asserting that 'func' in the state
    # is a bytes object (dumped by cloudpickle) and other attributes
    # are as expected
    assert isinstance(state["func"], bytes)
    assert state["output_name"] == "c"

    # Now we'll test setstate by creating a new instance, applying setstate and
    # verifying that the object attributes match the original
    new_pipe_func = PipeFunc.__new__(PipeFunc)
    new_pipe_func.__setstate__(state)

    assert new_pipe_func.output_name == pipe_func.output_name
    assert new_pipe_func.parameters == pipe_func.parameters
    assert new_pipe_func.func(2, 3) == pipe_func.func(
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

    pipeline = Pipeline([f1, f2, f3, f4, f5, f6, f7], lazy=True)

    r = pipeline("f7", a=1, b=2, c=3, d=4, e=5)
    assert r.evaluate() == 52


def test_tuple_outputs(tmp_path: Path):
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

    def save_function(fname, result):
        p = tmp_path / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result, f)

    @pipefunc(
        output_name=("d", "e"),
        cache=cache,
        save_function=save_function,
    )
    def f_d(b, c, x=1):  # noqa: ARG001
        return b * c, 1

    @pipefunc(
        output_name=("g", "h"),
        output_picker=getattr,
        cache=cache,
        save_function=save_function,
    )
    def f_e(c, e, x=1):  # noqa: ARG001
        from types import SimpleNamespace

        print(f"Called f_e with c={c} and e={e}")
        return SimpleNamespace(g=c + e, h=c - e)

    @pipefunc(output_name="i", cache=cache)
    def f_i(h, g):
        return h + g

    pipeline = Pipeline(
        [f_c, f_d, f_e, f_i],
        debug=True,
        profile=True,
        cache_type="lru",
        lazy=True,
        cache_kwargs={"shared": False},
    )
    f = pipeline.func("i")
    r = f.call_full_output(a=1, b=2, x=3)["i"].evaluate()
    assert r == f(a=1, b=2, x=3).evaluate()
    assert (
        pipeline.root_args("g")
        == pipeline.root_args("h")
        == pipeline.root_args(("g", "h"))
        == ("a", "b", "x")
    )
    key = (("d", "e"), (("a", 1), ("b", 2), ("x", 3)))
    assert pipeline.cache is not None
    assert pipeline.cache.cache[key].evaluate() == (6, 1)
    assert pipeline.func(("g", "h"))(a=1, b=2, x=3).evaluate().g == 4
    assert pipeline.func_dependencies("i") == [("c", "_throw"), ("d", "e"), ("g", "h")]
    assert pipeline.func_dependents("c") == [("d", "e"), ("g", "h"), "i"]

    assert (
        pipeline.func_dependencies("g")
        == pipeline.func_dependencies("h")
        == pipeline.func_dependencies(("g", "h"))
        == [("c", "_throw"), ("d", "e")]
    )

    f = pipeline.func(("g", "h"))
    r = f(a=1, b=2, x=3).evaluate()
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
        "a": {},
        "b": {},
        f_d: {},
        "x": {},
        f_e: {},
        f_i: {},
    }


@pytest.mark.parametrize("cache", [True, False])
def test_full_output(cache, tmp_path: Path):
    from pipefunc import Pipeline

    def save_function(fname, result):
        p = tmp_path / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result, f)

    @pipefunc(output_name="f1", save_function=save_function)
    def f1(a, b):
        return a + b

    @pipefunc(output_name=("f2i", "f2j"), save_function=save_function)
    def f2(f1):
        return 2 * f1, 1

    @pipefunc(output_name="f3", save_function=save_function)
    def f3(a, f2i):
        return a + f2i

    if cache:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_kwargs = {"cache_type": "disk", "cache_kwargs": {"cache_dir": cache_dir}}
    else:
        cache_kwargs = {}
    pipeline = Pipeline([f1, f2, f3], **cache_kwargs)  # type: ignore[arg-type]
    for f in pipeline.functions:
        f.cache = cache
    pipeline("f3", a=1, b=2)
    func = pipeline.func("f3")
    assert func.call_full_output(a=1, b=2) == {
        "a": 1,
        "b": 2,
        "f1": 3,
        "f2i": 6,
        "f2j": 1,
        "f3": 7,
    }
    if cache:
        assert len(list(cache_dir.glob("*.pkl"))) == 3


def test_lazy_pipeline():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], lazy=True)

    f = pipeline.func("e")
    r = f(a=1, b=2, x=3).evaluate()
    assert r == 162
    r = f.call_full_output(a=1, b=2, x=3)["e"].evaluate()
    assert r == 162


@pipefunc(output_name="test_function")
def test_function(arg1: str, arg2: str) -> str:
    return f"{arg1} {arg2}"


pipeline = Pipeline([test_function])


def test_function_pickling():
    # Get the _PipelineAsFunc instance from the pipeline
    func = pipeline.func("test_function")

    # Pickle the _PipelineAsFunc instance
    pickled_func = pickle.dumps(func)

    # Unpickle the _PipelineAsFunc instance
    unpickled_func = pickle.loads(pickled_func)  # noqa: S301

    # Assert that the unpickled instance has the same attributes
    assert unpickled_func.output_name == "test_function"
    assert unpickled_func.root_args == ("arg1", "arg2")

    # Assert that the unpickled instance behaves the same as the original
    result = unpickled_func(arg1="hello", arg2="world")
    assert result == "hello world"

    # Assert that the call_with_root_args method is recreated after unpickling
    assert unpickled_func.call_with_root_args is not None
    assert unpickled_func.call_with_root_args.__signature__.parameters.keys() == {
        "arg1",
        "arg2",
    }


def test_drop_from_pipeline():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3])
    assert "d" in pipeline.output_to_func
    pipeline.drop(output_name="d")
    assert "d" not in pipeline.output_to_func

    pipeline = Pipeline([f1, f2, f3])
    assert "d" in pipeline.output_to_func
    pipeline.drop(f=f2)

    pipeline = Pipeline([f1, f2, f3])

    @pipefunc(output_name="e")
    def f4(c, d, x=1):
        return c * d * x

    pipeline.replace(f4)
    assert len(pipeline.functions) == 3
    assert pipeline.output_to_func == {"c": f1, "d": f2, "e": f4}

    pipeline.replace(f3, f4)
    assert len(pipeline.functions) == 3
    assert pipeline.output_to_func == {"c": f1, "d": f2, "e": f3}

    with pytest.raises(ValueError, match="Either `f` or `output_name` should be provided"):
        pipeline.drop()


def test_used_variable():
    @pipefunc(output_name="c", cache=True)
    def f1(a, b):
        return a + b

    pipeline = Pipeline([f1])  # automatically sets cache_type="lru" because of cache=True
    assert isinstance(pipeline.cache, LRUCache)
    with pytest.raises(UnusedParametersError, match="Unused keyword arguments"):
        pipeline("c", a=1, b=2, doesnotexist=3)

    pipeline("c", a=1, b=2)

    # Test regression with cache:
    def f(a):
        return a

    pipeline = Pipeline([PipeFunc(f, output_name="c", cache=True)], cache_type="lru")
    f = pipeline.func("c")
    assert f(a=1) == 1
    assert f(a=1) == 1  # should not raise an error


def test_handle_error():
    @pipefunc(output_name="c")
    def f1(a, b):  # noqa: ARG001
        msg = "Test error"
        raise ValueError(msg)

    pipeline = Pipeline([f1])
    try:
        pipeline("c", a=1, b=2)
    except ValueError as e:
        msg = "Error occurred while executing function `f1(a=1, b=2)`"
        assert msg in str(e) or msg in str(e.__notes__)  # noqa: PT017
        # NOTE: with pytest.raises match="..." does not work
        # with add_note for some reason on my Mac, however,
        # on CI it works fine (Linux)...


def test_full_output_cache():
    ran_f1 = False
    ran_f2 = False

    @pipefunc(output_name="c", cache=True)
    def f1(a, b):
        nonlocal ran_f1
        if ran_f2:
            raise RuntimeError
        ran_f1 = True
        return a + b

    @pipefunc(output_name="d", cache=True)
    def f2(b, c, x=1):
        nonlocal ran_f2
        if ran_f2:
            raise RuntimeError
        ran_f2 = True
        return b * c * x

    pipeline = Pipeline([f1, f2], cache_type="hybrid")
    f = pipeline.func("d")
    r = f.call_full_output(a=1, b=2, x=3)
    expected = {"a": 1, "b": 2, "c": 3, "d": 18, "x": 3}
    assert r == expected
    assert len(pipeline.cache) == 2
    r = f.call_full_output(a=1, b=2, x=3)
    assert r == expected
    r = f(a=1, b=2, x=3)
    assert r == 18


def test_output_picker_single_output():
    @pipefunc(output_name=("y",), output_picker=dict.__getitem__)
    def f(a, b):
        return {"y": a + b, "_throw": 1}

    pipeline = Pipeline([f])
    assert pipeline("y", a=1, b=2) == 3


def f(a, b):
    return a + b


@dataclass
class DataClass:
    a: int


def test_pickle_pipefunc():
    func = PipeFunc(f, output_name="c")
    p = pickle.dumps(func)
    func2 = pickle.loads(p)  # noqa: S301
    assert func(1, 2) == func2(1, 2)

    func = PipeFunc(DataClass, output_name="c")
    p = pickle.dumps(func)
    func2 = pickle.loads(p)  # noqa: S301
    assert func(a=1) == func2(a=1)


def test_independent_axes_in_mapspecs_with_disconnected_chains():
    @pipefunc(output_name=("c", "d"), mapspec="a[i] -> c[i], d[i]")
    def f(a: int, b: int):
        return a + b, 1

    @pipefunc(output_name="z", mapspec="x[i], y[i] -> z[i]")
    def g(x, y):
        return x + y

    pipeline = Pipeline([f, g])
    assert pipeline.mapspecs_as_strings == [
        "a[i] -> c[i], d[i]",
        "x[i], y[i] -> z[i]",
    ]
    assert pipeline.independent_axes_in_mapspecs("c") == {"i"}
    assert pipeline.independent_axes_in_mapspecs("d") == {"i"}
    assert pipeline.independent_axes_in_mapspecs(("c", "d")) == {"i"}
    assert pipeline.independent_axes_in_mapspecs("z") == {"i"}

    pipeline.add_mapspec_axis("b", axis="j")
    assert pipeline.mapspecs_as_strings == [
        "a[i], b[j] -> c[i, j], d[i, j]",
        "x[i], y[i] -> z[i]",
    ]
    assert pipeline.independent_axes_in_mapspecs("c") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("d") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs(("c", "d")) == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("z") == {"i"}

    pipeline.add_mapspec_axis("x", axis="j")
    pipeline.add_mapspec_axis("y", axis="j")
    assert pipeline.mapspecs_as_strings == [
        "a[i], b[j] -> c[i, j], d[i, j]",
        "x[i, j], y[i, j] -> z[i, j]",
    ]
    assert pipeline.independent_axes_in_mapspecs("c") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("d") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs(("c", "d")) == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("z") == {"i", "j"}

    with pytest.raises(
        ValueError,
        match="The provided `pipefuncs` should have only one leaf node, not 2.",
    ):
        NestedPipeFunc([f, g])


def test_max_single_execution_per_call() -> None:
    counter = {"f_c": 0, "f_d": 0, "f_e": 0}

    @pipefunc(output_name="c")
    def f_c(a, b):
        assert counter["f_c"] == 0
        counter["f_c"] += 1
        print("c")
        return a + b

    @pipefunc(output_name="d")
    def f_d(b, c, x=1):
        assert counter["f_d"] == 0
        counter["f_d"] += 1
        return b * c * x

    @pipefunc(output_name="e")
    def f_e(c, d, x=1):
        assert counter["f_e"] == 0
        counter["f_e"] += 1
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline("e", a=1, b=2, x=3)
    assert counter == {"f_c": 1, "f_d": 1, "f_e": 1}


def test_setting_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 2}, renames={"a": "a1"})
    def f(a, b=1):
        return a + b

    assert f.parameters == ("a1", "b")
    assert f.defaults == {"b": 2}
    with pytest.raises(ValueError, match="Unexpected keyword arguments"):
        f(a=0)

    assert f(a1=0) == 2

    pipeline = Pipeline([f])
    assert pipeline("c", a1=0) == 2
    assert pipeline("c", a1="a1", b="b") == "a1b"

    with pytest.raises(ValueError, match="Unexpected `defaults` arguments"):

        @pipefunc(output_name="b", defaults={"a": 2}, renames={"a": "a1"})
        def g(a):
            return a

    @pipefunc(output_name="c", defaults={"a": "a_new", "b": "b_new"}, renames={"a": "b", "b": "a"})
    def h(a="a", b="b"):
        return a, b

    assert h() == ("b_new", "a_new")
    assert h(a="aa", b="bb") == ("bb", "aa")


def test_update_defaults_and_renames_and_bound() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    # Test initial parameters and defaults
    assert f.parameters == ("a1", "b")
    assert f.defaults == {"a1": 42, "b": 1}

    # Update defaults
    f.update_defaults({"b": 2})
    assert f.defaults == {"a1": 42, "b": 2}

    # Call function with updated defaults
    assert f(a1=3) == 5

    # Overwrite defaults
    f.update_defaults({"a1": 1, "b": 3}, overwrite=True)
    assert f.defaults == {"a1": 1, "b": 3}
    assert f.parameters == ("a1", "b")

    # Call function with new defaults
    assert f(a1=2) == 5
    assert f() == 4
    assert f(a1=2, b=3) == 5

    # Update renames
    f.update_renames({"a": "a2"}, update_from="original")
    assert f.renames == {"a": "a2"}
    assert f.parameters == ("a2", "b")

    # Call function with updated renames
    assert f(a2=4) == 7
    assert f(b=0) == 1

    # Overwrite renames
    f.update_renames({"a": "a3"}, overwrite=True, update_from="original")
    assert f.parameters == ("a3", "b")

    # Call function with new renames
    assert f(a3=1) == 4

    pipeline = Pipeline([f])
    assert pipeline("c", a3=1) == 4
    assert pipeline("c", a3=2, b=3) == 5

    f.update_bound({"a3": "yolo", "b": "swag"})
    assert f(a3=88, b=1) == "yoloswag"
    assert f.bound == {"a3": "yolo", "b": "swag"}
    f.update_renames({"a": "a4"}, update_from="original")
    assert f.bound == {"a4": "yolo", "b": "swag"}
    f.update_bound({}, overwrite=True)
    assert f(a4=88, b=1) == 89

    f.update_renames({"a4": "a5"}, update_from="current")
    assert f(a5=88, b=1) == 89
    f.update_renames({"b": "b1"}, update_from="current")
    assert f.renames == {"a": "a5", "b": "b1"}

    f.update_renames({}, overwrite=True)
    assert f.parameters == ("a", "b")
    assert f.renames == {}


def test_update_renames_with_mapspec() -> None:
    @pipefunc(output_name="c", renames={"a": "a1"}, mapspec="a1[i], b[j] -> c[i, j]")
    def f(a=42, b=69):
        return a + b

    # Test initial parameters and defaults
    assert f.parameters == ("a1", "b")
    assert str(f.mapspec) == "a1[i], b[j] -> c[i, j]"

    f.update_renames({"a": "a2"}, update_from="original")
    assert f.renames == {"a": "a2"}
    assert f.parameters == ("a2", "b")
    assert str(f.mapspec) == "a2[i], b[j] -> c[i, j]"
    f.update_renames({"a": "a3"}, overwrite=True, update_from="original")
    assert f.parameters == ("a3", "b")
    assert str(f.mapspec) == "a3[i], b[j] -> c[i, j]"
    f.update_renames({"a": "a4"}, update_from="original")
    assert str(f.mapspec) == "a4[i], b[j] -> c[i, j]"
    f.update_renames({"a4": "a5"}, update_from="current")
    assert str(f.mapspec) == "a5[i], b[j] -> c[i, j]"
    f.update_renames({"b": "b1"}, update_from="current")
    assert str(f.mapspec) == "a5[i], b1[j] -> c[i, j]"
    f.update_renames({}, overwrite=True)
    assert str(f.mapspec) == "a[i], b[j] -> c[i, j]"

    # Test updating output_name
    f.update_renames({"c": "c1"}, update_from="original")
    assert str(f.mapspec) == "a[i], b[j] -> c1[i, j]"
    assert f.output_name == "c1"
    f.update_renames({"c1": "c2"}, update_from="current")
    assert str(f.mapspec) == "a[i], b[j] -> c2[i, j]"
    assert f.output_name == "c2"
    f.update_renames({"c": "c3"}, update_from="original")
    assert str(f.mapspec) == "a[i], b[j] -> c3[i, j]"
    assert f.output_name == "c3"
    f.update_renames({}, overwrite=True)
    assert str(f.mapspec) == "a[i], b[j] -> c[i, j]"
    assert f.output_name == "c"


def test_renaming_output_name() -> None:
    @pipefunc(output_name=("c", "d"), renames={"a": "a1"})
    def f(a, b):
        return a + b, 1

    pipeline = Pipeline([f])
    f.update_renames({"c": "c1"}, update_from="current")
    assert f.output_name == ("c1", "d")
    pipeline.update_renames({"c1": "c2"}, update_from="current")
    assert f.output_name == ("c2", "d")
    pipeline.update_renames({"d": "d1"}, overwrite=True, update_from="current")
    assert f.output_name == ("c", "d1")
    pipeline.update_renames({"c": "c1"}, overwrite=True, update_from="original")
    assert f.output_name == ("c1", "d")
    f2 = f.copy()
    assert f2.output_name == ("c1", "d")
    f2.update_renames({"c1": "c2"}, update_from="current")
    assert f2.output_name == ("c2", "d")
    f2.update_renames({}, overwrite=True)
    assert f2.output_name == ("c", "d")


def test_update_pipeline_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    pipeline = Pipeline([f])
    fp = pipeline.functions[0]

    # Test initial parameters and defaults
    assert fp.parameters == ("a1", "b")
    assert fp.defaults == {"a1": 42, "b": 1}

    # Update defaults
    pipeline.update_defaults({"b": 2})
    assert fp.defaults == {"a1": 42, "b": 2}

    # Call function with updated defaults
    assert pipeline(a1=3) == 5

    # Overwrite defaults
    pipeline.update_defaults({"a1": 1, "b": 3}, overwrite=True)
    assert fp.defaults == {"a1": 1, "b": 3}
    assert fp.parameters == ("a1", "b")

    # Call function with new defaults
    assert fp(a1=2) == 5
    assert fp() == 4
    assert fp(a1=2, b=3) == 5

    with pytest.raises(ValueError, match="Unused keyword arguments"):
        pipeline.update_defaults({"does_not_exist": 1})


def test_validate_update_defaults_and_renames_and_bound() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    with pytest.raises(ValueError, match="The allowed arguments are"):
        f.update_defaults({"does_not_exist": 1})
    with pytest.raises(ValueError, match="The allowed arguments are"):
        f.update_renames({"does_not_exist": "1"}, update_from="original")
    with pytest.raises(ValueError, match="The allowed arguments are"):
        f.update_bound({"does_not_exist": 1})


def test_update_defaults_and_renames_with_pipeline() -> None:
    @pipefunc(output_name="x", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    @pipefunc(output_name="y", defaults={"c": 2}, renames={"d": "d1"})
    def g(c=999, d=666):
        return c * d

    pipeline = Pipeline([f, g])

    # Test initial pipeline parameters and defaults
    assert f.parameters == ("a1", "b")
    assert f.defaults == {"a1": 42, "b": 1}
    assert g.parameters == ("c", "d1")
    assert g.defaults == {"c": 2, "d1": 666}

    # Update defaults and renames within pipeline
    f.update_defaults({"b": 3})
    f.update_renames({"a": "a2"}, update_from="original")
    g.update_defaults({"c": 4})
    g.update_renames({"d": "d2"}, update_from="original")

    # Test updated pipeline parameters and defaults
    assert f.parameters == ("a2", "b")
    assert f.defaults == {"a2": 42, "b": 3}
    assert g.parameters == ("c", "d2")
    assert g.defaults == {"c": 4, "d2": 666}

    # Call functions within pipeline with updated defaults and renames
    assert pipeline("x", a2=3) == 6
    assert pipeline("y", c=2, d2=3) == 6
    assert pipeline("y") == 4 * 666


@pytest.mark.parametrize("output_name", [("a.1", "b"), "#a", "1"])
def test_invalid_output_name_identifier(output_name):
    with pytest.raises(
        ValueError,
        match="The `output_name` should contain/be valid Python identifier",
    ):

        @pipefunc(output_name=output_name)
        def f(): ...


def test_invalid_output_name():
    with pytest.raises(
        TypeError,
        match="The output name should be a string or a tuple of strings",
    ):

        @pipefunc(output_name=["a"])
        def f(): ...


def test_subpipeline():
    @pipefunc(output_name=("c", "d"))
    def f(a: int, b: int):
        return a + b, 1

    @pipefunc(output_name="z")
    def g(x, y):
        return x + y

    pipeline = Pipeline([f, g])
    partial = pipeline.subpipeline(inputs=["a", "b"])
    assert [f.output_name for f in partial.functions] == [("c", "d")]

    partial = pipeline.subpipeline(inputs=["a", "b", "x", "y"])
    assert [f.output_name for f in partial.functions] == [("c", "d"), "z"]

    partial = pipeline.subpipeline(output_names=[("c", "d")])
    assert [f.output_name for f in partial.functions] == [("c", "d")]

    with pytest.raises(ValueError, match="Cannot construct a partial pipeline"):
        partial = pipeline.subpipeline(inputs=["a"])

    @pipefunc(output_name="h")
    def h(c):
        return c

    pipeline = Pipeline([f, g, h])
    partial = pipeline.subpipeline(inputs=["a", "b"])
    assert [f.output_name for f in partial.functions] == [("c", "d"), "h"]

    partial = pipeline.subpipeline(output_names=["h"])
    assert partial.topological_generations.root_args == ["a", "b"]
    assert partial.root_nodes == ["a", "b"]

    partial = pipeline.subpipeline(output_names=["h"], inputs=["c"])
    assert partial.topological_generations.root_args == ["c"]
    assert [f.output_name for f in partial.functions] == ["h"]

    with pytest.raises(
        ValueError,
        match="At least one of `inputs` or `output_names` should be provided",
    ):
        pipeline.subpipeline()


def test_nested_func() -> None:
    def f(a, b):
        return a + b

    def g(f):
        return f

    def h(g, x):  # noqa: ARG001
        return g

    nf = NestedPipeFunc([PipeFunc(f, "f"), PipeFunc(g, "g")])
    assert str(nf) == "NestedPipeFunc_f_g(...) → f, g"
    assert repr(nf) == "NestedPipeFunc(pipefuncs=[PipeFunc(f), PipeFunc(g)])"
    assert nf(a=1, b=2) == (3, 3)

    nf = NestedPipeFunc(
        [
            PipeFunc(f, "f", mapspec="a[i], b[i] -> f[i]"),
            PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
        ],
    )
    assert str(nf.mapspec) == "a[i], b[i] -> f[i], g[i]"
    nf_copy = nf.copy()
    assert str(nf.mapspec) == str(nf_copy.mapspec)

    # Test not returning all outputs by providing a output_name
    nf = NestedPipeFunc(
        [
            PipeFunc(f, "f", mapspec="a[i], b[i] -> f[i]"),
            PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
        ],
        output_name="g",
    )
    assert str(nf.mapspec) == "a[i], b[i] -> g[i]"
    assert nf(a=1, b=2) == 3

    # Check all exceptions
    with pytest.raises(ValueError, match="The provided `output_name='not_exist'` should"):
        nf = NestedPipeFunc(
            [PipeFunc(f, "f"), PipeFunc(g, "g")],
            output_name="not_exist",
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine MapSpecs with different input and output mappings",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="... -> f[i]"),
                PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine a mix of None and MapSpec instances",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="... -> f[i]"),
                PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
                PipeFunc(h, "z", mapspec=None),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine MapSpecs with different input mappings",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="a[i], b[j] -> f[i, j]"),
                PipeFunc(g, "g", mapspec="f[i, :] -> g[i]"),
            ],
        )

    with pytest.raises(
        ValueError,
        match="Cannot combine MapSpecs with different output mappings",
    ):
        NestedPipeFunc(
            [
                PipeFunc(f, "f", mapspec="a[i], b[j] -> f[i, j]"),
                PipeFunc(g, "g", mapspec="f[i, j] -> g[j, i]"),
            ],
        )

    with pytest.raises(ValueError, match="should have at least two"):
        NestedPipeFunc([PipeFunc(f, "f")])

    with pytest.raises(
        TypeError,
        match="All elements in `pipefuncs` should be instances of `PipeFunc`.",
    ):
        NestedPipeFunc([f, PipeFunc(g, "g")])  # type: ignore[list-item]


def test_nested_func_renames_defaults_and_bound() -> None:
    def f(a, b=99):
        return a + b

    def g(f):
        return f

    # Test renaming
    nf = NestedPipeFunc(
        [
            PipeFunc(f, "f", mapspec="a[i], b[i] -> f[i]"),
            PipeFunc(g, "g", mapspec="f[i] -> g[i]"),
        ],
        output_name="g",
    )

    assert nf.renames == {}
    nf.update_renames({"a": "a1", "b": "b1"}, update_from="original")
    assert nf.renames == {"a": "a1", "b": "b1"}
    assert nf(a1=1, b1=2) == 3
    assert nf(a1=1) == 100
    nf.update_defaults({"b1": 2, "a1": 2})
    assert nf() == 4
    nf.update_bound({"a1": "a", "b1": "b"})
    assert nf(a1=3, b1=4) == "ab"  # will ignore the input values now


def test_nested_pipefunc_with_resources() -> None:
    def f(a, b=99):
        return a + b

    def g(f):
        return f

    # Test the resources are combined correctly
    nf = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "num_cpus": 2}),
            PipeFunc(g, "g", resources={"memory": "2GB", "num_cpus": 1}),
        ],
        output_name="g",
    )
    assert isinstance(nf.resources, Resources)
    assert nf.resources.num_cpus == 2
    assert nf.resources.memory == "2GB"

    # Test that the resources specified in NestedPipeFunc are used
    nf2 = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "num_cpus": 2}),
            PipeFunc(g, "g", resources={"memory": "2GB", "num_cpus": 1}),
        ],
        output_name="g",
        resources={"memory": "3GB", "num_cpus": 3},
    )
    assert isinstance(nf2.resources, Resources)
    assert nf2.resources.num_cpus == 3
    assert nf2.resources.memory == "3GB"

    # Test that the resources specified in PipeFunc are used, with the other None
    nf3 = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "num_cpus": 2}),
            PipeFunc(g, "g", resources=None),
        ],
        output_name="g",
    )
    assert isinstance(nf3.resources, Resources)
    assert nf3.resources.num_cpus == 2
    assert nf3.resources.memory == "1GB"

    # Test that Resources instance in NestedPipeFunc is used
    nf3 = NestedPipeFunc(
        [
            PipeFunc(f, "f", resources={"memory": "1GB", "num_cpus": 2}),
            PipeFunc(g, "g", resources=None),
        ],
        output_name="g",
        resources=Resources(num_cpus=3, memory="3GB"),
    )
    assert isinstance(nf3.resources, Resources)
    assert nf3.resources.num_cpus == 3
    assert nf3.resources.memory == "3GB"


def test_nest_all() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c + 1

    pipeline = Pipeline([f, g])
    assert pipeline("d", a=1, b=2) == 4
    nested = pipeline.nest_funcs("*")
    assert nested.output_name == ("c", "d")
    assert nested(a=1, b=2) == (3, 4)
    assert pipeline("d", a=1, b=2) == 4
    assert len(pipeline.functions) == 1


def test_missing_kw():
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    with pytest.raises(
        ValueError,
        match=re.escape("Missing value for argument `b` in `f(...) → c`."),
    ):
        pipeline("c", a=1)


def test_join_pipelines() -> None:
    def f(a, b):
        return a + b

    def g(a, b):
        return a * b

    pipeline1 = Pipeline([PipeFunc(f, "f")], debug=True)
    pipeline2 = Pipeline([PipeFunc(g, "g")], debug=False)
    pipeline = pipeline1.join(pipeline2)
    assert pipeline("f", a=1, b=2) == 3
    assert pipeline("g", a=1, b=2) == 2
    assert pipeline.debug

    pipeline = pipeline1 | pipeline2
    assert pipeline("f", a=1, b=2) == 3
    assert pipeline("g", a=1, b=2) == 2
    assert pipeline.debug

    pipeline = pipeline1 | PipeFunc(g, "g")
    assert pipeline("f", a=1, b=2) == 3
    assert pipeline("g", a=1, b=2) == 2
    assert pipeline.debug

    with pytest.raises(
        TypeError,
        match="Only `Pipeline` or `PipeFunc` instances can be joined",
    ):
        pipeline1 | g  # type: ignore[operator]


def test_empty_pipeline() -> None:
    pipeline = Pipeline([])
    assert pipeline.output_to_func == {}
    assert pipeline.topological_generations.root_args == []
    assert pipeline.topological_generations.function_lists == []

    with pytest.raises(TypeError, match="must be a `PipeFunc` or callable"):
        pipeline.add(1)  # type: ignore[arg-type]


def test_unhashable_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": []})
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", defaults={"b": {}})
    def g(a, b):
        return a + b

    pipeline = Pipeline([f])
    assert pipeline.defaults == {"b": []}

    # The problem should occur when using the default twice
    with pytest.raises(ValueError, match="Inconsistent default"):
        Pipeline([f, g])


def test_update_renames_pipeline() -> None:
    @pipefunc(output_name="c", renames={"a": "a1"})
    def f(a, b):
        return a, b

    pipeline = Pipeline([f])
    assert pipeline("c", a1="a1", b="b") == ("a1", "b")

    pipeline.update_renames({"a1": "a2"}, update_from="current")
    assert pipeline("c", a2="a2", b="b") == ("a2", "b")

    pipeline.update_renames({"a2": "a3"}, update_from="current")
    assert pipeline("c", a3="a3", b="b") == ("a3", "b")

    pipeline.update_renames({"b": "b1"}, update_from="current")
    assert pipeline("c", a3="a3", b1="b1") == ("a3", "b1")

    with pytest.raises(
        ValueError,
        match="Unused keyword arguments: `a3`. These are not settable renames",
    ):
        pipeline.update_renames({"a3": "foo"}, update_from="original")

    with pytest.raises(
        ValueError,
        match="Unused keyword arguments: `a`. These are not settable renames",
    ):
        pipeline.update_renames({"a": "foo"}, update_from="current")

    pipeline.update_renames({"a": "a5"}, update_from="original")
    assert pipeline("c", a5="a5", b1="b1") == ("a5", "b1")

    pipeline.update_renames({"a": "a6"}, update_from="original", overwrite=True)
    assert pipeline("c", a6="a6", b="b") == ("a6", "b")


def test_set_debug_and_profile():
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    assert not f.debug
    assert not f.profile
    pipeline.debug = True
    pipeline.profile = True
    assert f.debug
    assert f.profile


def test_simple_cache():
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a, b

    with pytest.raises(ValueError, match="Invalid cache type"):
        Pipeline([f], cache_type="not_exist")
    pipeline = Pipeline([f], cache_type="simple")
    assert pipeline("c", a=1, b=2) == (1, 2)
    assert pipeline.cache.cache == {("c", (("a", 1), ("b", 2))): (1, 2)}
    pipeline.cache.clear()
    assert pipeline("c", a={"a": 1}, b=[2]) == ({"a": 1}, [2])
    assert pipeline.cache.cache == {("c", (("a", (("a", 1),)), ("b", (2,)))): ({"a": 1}, [2])}
    pipeline.cache.clear()
    assert pipeline("c", a={"a"}, b=[2]) == ({"a"}, [2])
    assert pipeline.cache.cache == {("c", (("a", ("a",)), ("b", (2,)))): ({"a"}, [2])}


def test_hybrid_cache_lazy_warning():
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a, b

    with pytest.warns(UserWarning, match="Hybrid cache uses function evaluation"):
        Pipeline([f], cache_type="hybrid", lazy=True)


def test_cache_non_root_args():
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", cache=True)
    def g(c, b):
        return c + b

    pipeline = Pipeline([f, g], cache_type="simple")
    # Won't populate cache because `c` is not a root argument
    assert pipeline("d", c=1, b=2) == 3
    assert pipeline.cache.cache == {}


def test_axis_in_root_args():
    # Test reaches the `output_name in visited` condition
    @pipefunc(output_name="c", mapspec="a[i] -> c[i]")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", mapspec="c[i] -> d[i]")
    def g(a, c):
        return a + c

    @pipefunc(output_name="e", mapspec="c[i], d[i] -> e[i]")
    def h(c, d):
        return c + d

    pipeline = Pipeline([f, g, h])
    assert pipeline.independent_axes_in_mapspecs("e") == {"i"}


def test_nesting_funcs_with_bound():
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", bound={"b": "b2"})
    def g(c, b):
        return c + b

    pipeline = Pipeline([f, g])
    assert pipeline("d", a="a", b="b") == "abb2"
    pipeline.nest_funcs(["c", "d"], "d")
    assert pipeline("d", a="a", b="b") == "abb2"
    assert len(pipeline.functions) == 1


def test_duplicate_output_name():
    @pipefunc(output_name="c")
    def f(x):
        return x

    pipeline1 = Pipeline([f])

    @pipefunc(output_name="c")
    def g(foo):
        return foo

    pipeline2 = Pipeline([g])

    with pytest.raises(ValueError, match="The provided `output_name='c'` cannot be identical"):
        pipeline1 | pipeline2
