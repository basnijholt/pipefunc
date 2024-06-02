"""Tests for pipefunc.py."""

from __future__ import annotations

import inspect
import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.exceptions import UnusedParametersError
from pipefunc.sweep import Sweep, count_sweep, get_precalculation_order

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


def test_pipeline_function_annotations():
    def add_numbers(a: int, b: float) -> tuple[int, float]:
        return a + 1, b + 1.0

    add_func = PipeFunc(
        add_numbers,
        output_name=("a_plus_one", "b_plus_one"),
        renames={"a": "x", "b": "y"},
    )

    assert add_func.parameter_annotations == {"x": int, "y": float}
    assert add_func.output_annotation == tuple[int, float]

    result = add_func(x=1, y=2.0)
    assert result == (2, 3.0)

    assert str(add_func) == "add_numbers(...) → a_plus_one, b_plus_one"
    assert repr(add_func) == "PipeFunc(add_numbers)"


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
    # Get the _Function instance from the pipeline
    func = pipeline.func("test_function")

    # Pickle the _Function instance
    pickled_func = pickle.dumps(func)

    # Unpickle the _Function instance
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


def test_used_variable():
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipeline = Pipeline([f1], cache_type="lru")
    pipeline("c", a=1, b=2)
    with pytest.raises(UnusedParametersError, match="Unused keyword arguments"):
        pipeline("c", a=1, b=2, doesnotexist=3)

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

    @pipefunc(output_name="b", defaults={"a": 2}, renames={"a": "a1"})
    def g(a):
        return a

    with pytest.raises(ValueError, match="Unexpected default arguments"):
        _ = g.defaults

    @pipefunc(output_name="c", defaults={"a": "a_new", "b": "b_new"}, renames={"a": "b", "b": "a"})
    def h(a="a", b="b"):
        return a, b

    assert h() == ("b_new", "a_new")
    assert h(a="aa", b="bb") == ("bb", "aa")


def test_update_defaults_and_renames() -> None:
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
    f.update_renames({"a": "a2"})
    assert f.renames == {"a": "a2"}
    assert f.parameters == ("a2", "b")

    # Call function with updated renames
    assert f(a2=4) == 7
    assert f(b=0) == 1

    # Overwrite renames
    f.update_renames({"a": "a3"}, overwrite=True)
    assert f.parameters == ("a3", "b")

    # Call function with new renames
    assert f(a3=1) == 4

    pipeline = Pipeline([f])
    assert pipeline("c", a3=1) == 4
    assert pipeline("c", a3=2, b=3) == 5


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
    f.update_renames({"a": "a2"})
    g.update_defaults({"c": 4})
    g.update_renames({"d": "d2"})

    # Test updated pipeline parameters and defaults
    assert f.parameters == ("a2", "b")
    assert f.defaults == {"a2": 42, "b": 3}
    assert g.parameters == ("c", "d2")
    assert g.defaults == {"c": 4, "d2": 666}

    # Call functions within pipeline with updated defaults and renames
    assert pipeline("x", a2=3) == 6
    assert pipeline("y", c=2, d2=3) == 6
    assert pipeline("y") == 4 * 666
