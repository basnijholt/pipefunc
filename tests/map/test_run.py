from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._utils import prod
from pipefunc.map._mapspec import trace_dependencies
from pipefunc.map._run import _reduced_axes, load_outputs, load_xarray_dataset, run
from pipefunc.map._run_info import RunInfo, map_shapes
from pipefunc.map._storage_base import storage_registry

if TYPE_CHECKING:
    from pathlib import Path

storage_options = list(storage_registry)


@pytest.fixture(params=storage_options)
def storage(request):
    return request.param


def test_simple(storage, tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray[Any, np.dtype[np.int_]]) -> int:
        assert isinstance(y, np.ndarray)
        return sum(y)

    pipeline = Pipeline(
        [
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )

    inputs = {"x": [0, 1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["sum"].output == 12
    assert results["sum"].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 12
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (4,), "y": (4,)}
    # Test `map` and a tmp run_folder
    results2 = pipeline.map(inputs, run_folder=None, parallel=False, storage=storage)
    assert results2["sum"].output == 12

    axes = pipeline.mapspec_axes
    assert axes == {"x": ("i",), "y": ("i",)}
    dimensions = pipeline.mapspec_dimensions
    assert dimensions.keys() == axes.keys()
    assert all(dimensions[k] == len(v) for k, v in axes.items())
    ds = load_xarray_dataset(run_folder=tmp_path)
    assert ds["y"].data.tolist() == [0, 2, 4, 6]

    run_info = RunInfo.load(tmp_path)
    run_info.dump(tmp_path)
    run_info2 = RunInfo.load(tmp_path)
    assert run_info2 == run_info

    with pytest.raises(ValueError, match="Pipeline is fully connected"):
        pipeline.split_disconnected()
    assert results["y"].store is not None
    assert isinstance(results["y"].store.parallelizable, bool)


def test_simple_2_dim_array(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, np.int_)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return np.sum(y, axis=0)

    pipeline = Pipeline(
        [
            (double_it, "x[i, j] -> y[i, j]"),
            take_sum,
        ],
    )

    inputs = {"x": np.arange(12).reshape(3, 4)}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["sum"].output_name == "sum"
    assert results["sum"].output.tolist() == [24, 30, 36, 42]
    assert load_outputs("sum", run_folder=tmp_path).tolist() == [24, 30, 36, 42]
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (3, 4), "y": (3, 4)}
    results2 = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    assert results2["sum"].output.tolist() == [24, 30, 36, 42]
    # Load the results as xarray
    load_xarray_dataset(run_folder=tmp_path)


def test_simple_2_dim_array_to_1_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, np.int_)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return sum(y)

    pipeline = Pipeline(
        [
            (double_it, "x[i, j] -> y[i, j]"),
            (take_sum, "y[i, :] -> sum[i]"),
        ],
    )

    inputs = {"x": np.arange(12).reshape(3, 4)}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["sum"].output_name == "sum"
    assert results["sum"].output.tolist() == [12, 44, 76]
    assert load_outputs("sum", run_folder=tmp_path).tolist() == [12, 44, 76]
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
        "x": (3, 4),
        "y": (3, 4),
        "sum": (3,),
    }
    load_xarray_dataset(run_folder=tmp_path)


def test_simple_2_dim_array_to_1_dim_to_0_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, np.int_)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return sum(y)

    @pipefunc(output_name="prod")
    def take_prod(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return np.prod(y)

    pipeline = Pipeline(
        [
            (double_it, "x[i, j] -> y[i, j]"),
            (take_sum, "y[i, :] -> sum[i]"),
            take_prod,
        ],
    )

    inputs = {"x": np.arange(1, 13).reshape(3, 4)}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["prod"].output_name == "prod"
    assert isinstance(results["prod"].output, np.int_)
    assert results["prod"].output == 1961990553600
    assert load_outputs("prod", run_folder=tmp_path) == 1961990553600
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
        "x": (3, 4),
        "y": (3, 4),
        "sum": (3,),
    }
    load_xarray_dataset(run_folder=tmp_path)


def run_outer_product(pipeline: Pipeline, tmp_path: Path) -> None:
    """Run the outer product test for the given pipeline."""
    # Used in the next three tests where we use alternative ways of defining the same pipeline
    inputs = {"x": [1, 2, 3], "y": [1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["z"].output_name == "z"
    expected = [[2, 3, 4], [3, 4, 5], [4, 5, 6]]
    assert results["z"].output.tolist() == expected
    assert load_outputs("z", run_folder=tmp_path).tolist() == expected
    assert results["sum"].output_name == "sum"
    assert results["sum"].output == 36
    assert load_outputs("sum", run_folder=tmp_path) == 36
    assert len(results) == 2
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"y": (3,), "x": (3,), "z": (3, 3)}
    load_xarray_dataset(run_folder=tmp_path)


def test_outer_product(tmp_path: Path) -> None:
    @pipefunc(output_name="z")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="sum")
    def total_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.sum(z)

    pipeline = Pipeline(
        [
            (add, "x[i], y[j] -> z[i, j]"),
            total_sum,
        ],
    )
    run_outer_product(pipeline, tmp_path)


def test_outer_product_decorator(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i], y[j] -> z[i, j]")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="sum")
    def total_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.sum(z)

    pipeline = Pipeline([add, total_sum])
    run_outer_product(pipeline, tmp_path)


def test_outer_product_functions(tmp_path: Path) -> None:
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    def total_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.sum(z)

    pipeline = Pipeline(
        [
            PipeFunc(add, "z", mapspec="x[i], y[j] -> z[i, j]"),
            PipeFunc(total_sum, "sum"),
        ],
    )
    run_outer_product(pipeline, tmp_path)


def test_simple_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline(
        [
            generate_ints,  # will autogen "... -> x[i]"
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )
    assert pipeline.mapspecs_as_strings == ["... -> x[i]", "x[i] -> y[i]"]
    inputs = {"n": 4}
    results = run(
        pipeline,
        inputs,
        run_folder=tmp_path,
        internal_shapes={"x": (4,)},
        parallel=False,
    )
    assert results["sum"].output == 12
    assert results["sum"].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 12

    shapes, masks = map_shapes(pipeline, inputs, {"x": (4,)})
    assert masks == {"x": (False,), "y": (True,)}
    assert shapes == {"x": (4,), "y": (4,)}

    with pytest.raises(ValueError, match="Internal shape for 'x' is missing."):
        map_shapes(pipeline, inputs)

    with pytest.raises(
        RuntimeError,
        match="Use `Pipeline.map` instead",
    ):
        pipeline("sum", n=4)
    assert pipeline("x", n=4) == list(range(4))
    ds = load_xarray_dataset("y", run_folder=tmp_path)
    assert "x" in ds.coords
    ds = load_xarray_dataset("y", run_folder=tmp_path, load_intermediate=False)
    assert "x" not in ds.coords


@pytest.mark.parametrize("output_picker", [None, dict.__getitem__])
def test_simple_multi_output(tmp_path: Path, output_picker) -> None:
    @pipefunc(output_name=("single", "double"), output_picker=output_picker)
    def double_it(x: int) -> tuple[int, int] | dict[str, int]:
        assert isinstance(x, int)
        return (x, 2 * x) if output_picker is None else {"single": x, "double": 2 * x}

    @pipefunc(output_name="sum")
    def take_sum(single: np.ndarray[Any, np.dtype[np.int_]]) -> int:
        return sum(single)

    pipeline = Pipeline(
        [
            (double_it, "x[i] -> single[i], double[i]"),
            take_sum,
        ],
    )

    inputs = {"x": [0, 1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["sum"].output == 6
    assert results["sum"].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 6
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
        "x": (4,),
        ("single", "double"): (4,),
        "single": (4,),
        "double": (4,),
    }
    load_xarray_dataset(run_folder=tmp_path)


def test_simple_from_step_nd(tmp_path: Path) -> None:
    @pipefunc(output_name="array")
    def generate_array(shape: tuple[int, ...]) -> np.ndarray[Any, np.dtype[np.int_]]:
        return np.arange(1, prod(shape) + 1).reshape(shape)

    @pipefunc(output_name="vector")
    def double_it(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        assert isinstance(array, np.ndarray)
        assert array.shape == shape[1:]
        return array.sum(axis=0).sum(axis=0)

    @pipefunc(output_name="sum")
    def norm(vector: np.ndarray) -> np.float64:
        return np.linalg.norm(vector)

    pipeline = Pipeline(
        [
            generate_array,  # will autogen "... -> array[i, unnamed_0, unnamed_1]"
            (double_it, "array[i, :, :] -> vector[i]"),
            norm,
        ],
    )
    assert pipeline.mapspecs_as_strings == [
        "... -> array[i, unnamed_0, unnamed_1]",
        "array[i, :, :] -> vector[i]",
    ]
    inputs = {"shape": (1, 2, 3)}
    internal_shapes: dict[str, int | tuple[int, ...]] = {"array": (1, 2, 3)}
    results = run(
        pipeline,
        inputs,
        run_folder=tmp_path,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        parallel=False,
    )
    assert results["sum"].output == 21.0
    assert results["sum"].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 21.0
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)
    assert shapes == {"array": (1, 2, 3), "vector": (1,)}
    assert masks == {"array": (False, False, False), "vector": (True,)}
    load_xarray_dataset(run_folder=tmp_path)


@dataclass(frozen=True)
class Geometry:
    x: float
    y: float


@dataclass(frozen=True)
class Mesh:
    geometry: Geometry
    mesh_size: float


@dataclass(frozen=True)
class Materials:
    geometry: Geometry
    materials: list[str]


@dataclass(frozen=True)
class Electrostatics:
    mesh: Mesh
    materials: Materials
    voltages: list[float]


@pytest.mark.parametrize("with_multiple_outputs", [False, True])
def test_pyiida_example(with_multiple_outputs: bool, tmp_path: Path) -> None:  # noqa: FBT001
    @pipefunc(output_name="geo")
    def make_geometry(x: float, y: float) -> Geometry:
        return Geometry(x, y)

    if with_multiple_outputs:

        @pipefunc(output_name=("mesh", "coarse_mesh"))
        def make_mesh(
            geo: Geometry,
            mesh_size: float,
            coarse_mesh_size: float,
        ) -> tuple[Mesh, Mesh]:
            return Mesh(geo, mesh_size), Mesh(geo, coarse_mesh_size)
    else:

        @pipefunc(output_name="mesh")
        def make_mesh(
            geo: Geometry,
            mesh_size: float,
            coarse_mesh_size: float,  # noqa: ARG001
        ) -> Mesh:
            return Mesh(geo, mesh_size)

    @pipefunc(output_name="materials")
    def make_materials(geo: Geometry) -> Materials:
        return Materials(geo, ["a", "b", "c"])

    @pipefunc(output_name="electrostatics")
    def run_electrostatics(
        mesh: Mesh,
        materials: Materials,
        V_left: float,  # noqa: N803
        V_right: float,  # noqa: N803
    ) -> Electrostatics:
        return Electrostatics(mesh, materials, [V_left, V_right])

    @pipefunc(output_name="charge")
    def get_charge(electrostatics: Electrostatics) -> float:
        # obviously not actually the charge; but we should return _some_ number that
        # is "derived" from the electrostatics.
        return sum(electrostatics.voltages)

    @pipefunc(output_name="average_charge")
    def average_charge(charge: np.ndarray) -> float:
        return np.mean(charge)

    pipeline = Pipeline(
        [
            make_geometry,
            make_mesh,
            make_materials,
            (run_electrostatics, "V_left[a], V_right[b] -> electrostatics[a, b]"),
            (get_charge, "electrostatics[a, b] -> charge[a, b]"),
            average_charge,
        ],
    )

    inputs = {
        "mesh_size": 0.01,
        "V_left": np.linspace(0, 2, 3),
        "V_right": np.linspace(-0.5, 0.5, 2),
        "x": 0.1,
        "y": 0.2,
        "coarse_mesh_size": 0.05,
    }
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
        "V_right": (2,),
        "V_left": (3,),
        "electrostatics": (3, 2),
        "charge": (3, 2),
    }
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["average_charge"].output == 1.0
    assert results["average_charge"].output_name == "average_charge"
    assert load_outputs("average_charge", run_folder=tmp_path) == 1.0
    load_xarray_dataset(run_folder=tmp_path)

    assert _reduced_axes(pipeline) == {"charge": {"b", "a"}}
    pipeline.add_mapspec_axis("x", axis="i")
    assert _reduced_axes(pipeline) == {"charge": {"b", "a"}}


def test_pipeline_with_defaults(tmp_path: Path) -> None:
    @pipefunc(output_name="z")
    def f(x: int, y: int = 1) -> int:
        return x + y

    @pipefunc(output_name="sum")
    def g(z: np.ndarray) -> int:
        return sum(z)

    pipeline = Pipeline([(f, "x[i] -> z[i]"), g])

    inputs = {"x": [0, 1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results["sum"].output == 10
    assert results["sum"].output_name == "sum"
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (4,), "z": (4,)}
    sum_result = load_outputs("sum", run_folder=tmp_path)
    assert sum_result == 10
    sum_result = load_outputs("z", run_folder=tmp_path)
    assert sum_result.tolist() == [1, 2, 3, 4]  # type: ignore[union-attr]

    inputs = {"x": [0, 1, 2, 3], "y": 2}  # type: ignore[dict-item]
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    assert results["sum"].output == 14
    load_xarray_dataset(run_folder=tmp_path)


def test_pipeline_loading_existing_results(tmp_path: Path) -> None:
    counters = {"f": 0, "g": 0}

    @pipefunc(output_name="z")
    def f(x: int, y: int = 1) -> int:
        counters["f"] += 1
        return x + y

    @pipefunc(output_name="sum")
    def g(z: np.ndarray) -> int:
        counters["g"] += 1
        return sum(z)

    pipeline = Pipeline([(f, "x[i] -> z[i]"), g])
    inputs = {"x": [1, 2, 3]}

    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=True)
    assert results["sum"].output == 9
    assert results["sum"].output_name == "sum"
    assert counters["f"] == 3
    assert counters["g"] == 1

    results2 = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=False)
    assert results2["sum"].output == 9
    assert results2["sum"].output_name == "sum"
    assert counters["f"] == 3
    assert counters["g"] == 1

    results3 = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=True)
    assert results3["sum"].output == 9
    assert results3["sum"].output_name == "sum"
    assert counters["f"] == 6
    assert counters["g"] == 2
    load_xarray_dataset(run_folder=tmp_path)


def test_run_info_compare(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i] -> z[i]")
    def f(x: int, y: int = 1) -> int:
        return x + y

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3]}

    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=True)
    assert results["z"].output.tolist() == [2, 3, 4]
    assert results["z"].output_name == "z"
    load_xarray_dataset(run_folder=tmp_path)

    inputs = {"x": [1, 2, 3, 4]}
    with pytest.raises(ValueError, match="Shapes do not match previous run"):
        pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=False)


def test_nd_input_list(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([(double_it, "x[i, j] -> y[i, j]")])

    inputs_list = {"x": [[1, 2], [3, 4]]}
    with pytest.raises(ValueError, match="Expected 2D"):
        pipeline.map(inputs_list, tmp_path, parallel=False)

    inputs_arr = {k: np.array(v) for k, v in inputs_list.items()}
    shapes, masks = map_shapes(pipeline, inputs_arr)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (2, 2), "y": (2, 2)}
    results = pipeline.map(inputs_arr, tmp_path, parallel=False)
    assert results["y"].output.tolist() == [[2, 4], [6, 8]]

    pipeline.add_mapspec_axis("x", axis="k")
    inputs = {"x": np.arange(2**3).reshape(2, 2, 2)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (2, 2, 2), "y": (2, 2, 2)}
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results["y"].output.tolist() == [[[0, 2], [4, 6]], [[8, 10], [12, 14]]]
    load_xarray_dataset(run_folder=tmp_path)


def test_add_mapspec_axis(tmp_path: Path) -> None:
    @pipefunc(output_name="one", mapspec="a[i], b[j] -> one[i, j]")
    def one(a, b):
        assert isinstance(a, (int, np.float64))
        assert isinstance(b, (int, np.float64))
        return a * b

    @pipefunc(output_name="two")
    def two(one, d):
        assert isinstance(one, np.ndarray)
        return np.sum(one) / d

    @pipefunc(output_name="three")
    def three(two, d):
        return two / d

    pipeline = Pipeline([one, two, three])
    inputs = {"a": np.ones((2,)), "b": [1, 1], "d": 1}
    expected = {"b": (2,), "a": (2,), "one": (2, 2)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == expected
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results["three"].output == 4.0

    # Adding another axis to "one"
    pipeline.add_mapspec_axis("a", axis="k")
    assert str(one.mapspec) == "a[i, k], b[j] -> one[i, j, k]"
    assert str(two.mapspec) == "one[:, :, k] -> two[k]"
    assert str(three.mapspec) == "two[k] -> three[k]"

    # Run the pipeline
    inputs = {"a": np.ones((2, 3)), "b": [1, 1], "d": 1}
    expected = {"b": (2,), "a": (2, 3), "one": (2, 2, 3), "two": (3,), "three": (3,)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == expected
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results["three"].output.tolist() == [4.0, 4.0, 4.0]

    # Adding another axis to "d"
    pipeline.add_mapspec_axis("d", axis="l")
    assert str(one.mapspec) == "a[i, k], b[j] -> one[i, j, k]"
    assert str(two.mapspec) == "one[:, :, k], d[l] -> two[k, l]"
    assert str(three.mapspec) == "two[k, l], d[l] -> three[k, l]"

    # Run the pipeline
    inputs = {"a": np.ones((2, 3)), "b": [1, 1], "d": [1, 1]}
    assert pipeline.mapspec_names == {"one", "a", "three", "two", "b", "d"}
    expected = {"b": (2,), "a": (2, 3), "one": (2, 2, 3), "two": (3, 2), "three": (3, 2), "d": (2,)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == expected
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results["three"].output.tolist() == [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]
    load_xarray_dataset(run_folder=tmp_path)

    assert pipeline.independent_axes_in_mapspecs("three") == {"k", "l"}


def test_mapspec_internal_shapes(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="... -> x[i]")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def add(x: int, z: int) -> int:
        assert isinstance(x, int)
        return x + z

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline([generate_ints, add, take_sum])

    pipeline.add_mapspec_axis("z", axis="k")
    assert str(generate_ints.mapspec) == "... -> x[i]"
    assert str(add.mapspec) == "x[i], z[k] -> y[i, k]"
    assert str(take_sum.mapspec) == "y[:, k] -> sum[k]"

    inputs = {"n": 4, "z": [1, 2]}
    internal_shapes = {"x": 4}
    results = pipeline.map(inputs, tmp_path, internal_shapes, parallel=False)  # type: ignore[arg-type]
    assert load_outputs("x", run_folder=tmp_path) == list(range(4))
    assert load_outputs("y", run_folder=tmp_path).tolist() == [[1, 2], [2, 3], [3, 4], [4, 5]]
    assert load_outputs("sum", run_folder=tmp_path).tolist() == [10, 14]
    assert results["sum"].output.tolist() == [10, 14]
    expected = {"z": (2,), "x": (4,), "y": (4, 2), "sum": (2,)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert masks == {"z": (True,), "x": (False,), "y": (True, True), "sum": (True,)}
    assert shapes == expected  # type: ignore[arg-type]
    deps = trace_dependencies(pipeline.mapspecs())  # type: ignore[arg-type]
    assert deps == {"y": {"x": ("i",), "z": ("k",)}, "sum": {"z": ("k",)}}
    load_xarray_dataset(run_folder=tmp_path)


def test_disconnected_independent_axes() -> None:
    @pipefunc(output_name="c", mapspec="a[i], b[i] -> c[i]")
    def f(a: int, b: int):
        return a + b

    @pipefunc(output_name="z", mapspec="x[i], y[i] -> z[i]")
    def g(x, y):
        return x + y

    pipeline = Pipeline([f, g])
    assert pipeline.independent_axes_in_mapspecs("z") == {"i"}
    assert pipeline.independent_axes_in_mapspecs("c") == {"i"}

    pipeline1, pipeline2 = pipeline.split_disconnected()
    assert len(pipeline1.functions) == 1
    assert len(pipeline2.functions) == 1


def test_reusing_axis_names_and_double_map_reduce(tmp_path: Path) -> None:
    pipeline = Pipeline(
        [
            PipeFunc(lambda x: x, "y", mapspec="x[i] -> y[i]"),
            PipeFunc(lambda y, z: y + z, "yz", mapspec="y[i] -> yz[i]"),
            PipeFunc(lambda yz: sum(yz), "sum_"),  # first map-reduce
            PipeFunc(lambda sum_: [sum_, sum_], "duplication", mapspec="... -> duplication[i]"),
            PipeFunc(lambda duplication: sum(duplication), "sum_final"),  # second map-reduce
        ],
    )
    internal_shapes = {"duplication": (2,)}
    results = pipeline.map(
        {"x": [1, 2, 3], "z": 1},
        run_folder=tmp_path,
        parallel=False,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
    )
    assert results["y"].output.tolist() == [1, 2, 3]
    assert results["yz"].output.tolist() == [2, 3, 4]
    assert results["sum_"].output == 9
    assert results["sum_final"].output == 18
    assert pipeline.independent_axes_in_mapspecs("sum_final") == set()


def test_from_step_2_dim_array(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    pipeline = Pipeline([(generate_ints, "... -> x[i]")])
    inputs = {"n": 4}
    internal_shapes = {"x": (4,)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert shapes == {"x": (4,)}
    assert masks == {"x": (False,)}
    results = pipeline.map(inputs, tmp_path, internal_shapes, parallel=False)  # type: ignore[arg-type]
    assert results["x"].output == list(range(4))
    assert load_outputs("x", run_folder=tmp_path) == list(range(4))
    load_xarray_dataset(run_folder=tmp_path)


def test_from_step_2_dim_array_2(storage: str, tmp_path: Path) -> None:
    @pipefunc(output_name="c")
    def f(a: int, b: int) -> list[int]:
        return [a + b, a - b]

    pipeline = Pipeline([(f, "b[i] -> c[i, j]")])
    inputs = {"a": 1, "b": [1, 2]}
    internal_shapes = {"c": (2,)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert shapes == {"b": (2,), "c": (2, 2)}
    assert masks == {"b": (True,), "c": (True, False)}
    results = pipeline.map(
        inputs,
        tmp_path,
        internal_shapes,  # type: ignore[arg-type]
        storage=storage,
        parallel=False,
    )
    assert results["c"].output.shape == (2, 2)
    assert results["c"].output.tolist() == [[2, 0], [3, -1]]
    assert load_outputs("c", run_folder=tmp_path).tolist() == [[2, 0], [3, -1]]
    load_xarray_dataset(run_folder=tmp_path)


def test_add_mapspec_axis_from_step(storage: str, tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="side")
    def side_chain(z: int) -> int:
        return z

    @pipefunc(output_name="sum")
    def take_sum(y: list[int], z: int) -> int:
        return sum(y) + z

    pipeline = Pipeline(
        [
            (generate_ints, "... -> x[i]"),
            (double_it, "x[i] -> y[i]"),
            side_chain,
            take_sum,
        ],
    )

    inputs = {"n": 4, "z": 1}
    internal_shapes = {"x": (4,)}
    assert pipeline.mapspec_axes == {"x": ("i",), "y": ("i",)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert masks == {"x": (False,), "y": (True,)}
    assert shapes == {"x": (4,), "y": (4,)}
    results = pipeline.map(
        inputs,
        tmp_path,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        parallel=False,
        storage=storage,
    )
    assert results["sum"].output == 13

    # Add an axis `j` to `x`
    pipeline_map = Pipeline(
        [
            (generate_ints, "n[j] -> x[i, j]"),
            (double_it, "x[i, j] -> y[i, j]"),
            side_chain,
            (take_sum, "y[:, j] -> sum[j]"),
        ],
    )
    inputs_map = {"n": [4], "z": 1}
    internal_shapes_map = {"x": (4,)}
    shapes, masks = map_shapes(pipeline_map, inputs_map, internal_shapes_map)  # type: ignore[arg-type]
    assert masks == {"n": (True,), "x": (False, True), "y": (True, True), "sum": (True,)}
    assert shapes == {"n": (1,), "x": (4, 1), "y": (4, 1), "sum": (1,)}
    results = pipeline_map.map(
        inputs_map,
        tmp_path,
        internal_shapes=internal_shapes_map,  # type: ignore[arg-type]
        parallel=False,
        storage=storage,
    )
    assert results["sum"].output.tolist() == [13]

    # Do the same but with `add_mapspec_axis` on the first pipeline
    assert pipeline.mapspecs_as_strings == ["... -> x[i]", "x[i] -> y[i]"]
    pipeline.add_mapspec_axis("n", axis="j")
    assert pipeline.mapspecs_as_strings == [
        "n[j] -> x[i, j]",
        "x[i, j] -> y[i, j]",
        "y[:, j] -> sum[j]",
    ]
    results = pipeline.map(
        inputs_map,
        tmp_path,
        internal_shapes=internal_shapes_map,  # type: ignore[arg-type]
        parallel=False,
        storage=storage,
    )
    assert results["sum"].output.tolist() == [13]
    load_xarray_dataset(run_folder=tmp_path)


def test_return_2d_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> np.ndarray:
        return np.ones((n, n))

    @pipefunc(output_name="y", mapspec="x[i, :] -> y[i]")
    def double_it(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return 2 * sum(x)

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline([generate_ints, double_it, take_sum])
    r = pipeline.map({"n": 4}, tmp_path, internal_shapes={"x": (4, 4)}, parallel=False)
    assert r["x"].output.tolist() == np.ones((4, 4)).tolist()
    assert r["y"].output.tolist() == [8, 8, 8, 8]
    assert r["sum"].output == 32
    load_xarray_dataset(run_folder=tmp_path)


def test_multi_output_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name=("x", "y"))
    def generate_ints(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.ones((n, n)), np.ones((n, n))

    @pipefunc(output_name="z", mapspec="x[i, :], y[i, :] -> z[i]")
    def double_it(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return 2 * sum(x) + 0 * sum(y)

    @pipefunc(output_name="sum")
    def take_sum(z: list[int]) -> int:
        return sum(z)

    pipeline = Pipeline([generate_ints, double_it, take_sum])
    assert pipeline.mapspecs_as_strings == [
        "... -> x[i, unnamed_0], y[i, unnamed_0]",
        "x[i, :], y[i, :] -> z[i]",
    ]
    shapes, masks = map_shapes(pipeline, {"n": 4}, {"x": (4, 4)})
    assert shapes == {("x", "y"): (4, 4), "x": (4, 4), "y": (4, 4), "z": (4,)}
    assert masks == {
        ("x", "y"): (False, False),
        "x": (False, False),
        "y": (False, False),
        "z": (True,),
    }
    r = pipeline.map({"n": 4}, tmp_path, internal_shapes={"x": (4, 4)}, parallel=False)
    assert r["x"].output_name == "x"
    assert r["x"].output.tolist() == np.ones((4, 4)).tolist()
    assert r["y"].output_name == "y"
    assert r["y"].output.tolist() == np.ones((4, 4)).tolist()
    assert r["z"].output_name == "z"
    assert r["z"].output.tolist() == [8, 8, 8, 8]
    assert r["sum"].output_name == "sum"
    assert r["sum"].output.tolist() == 32
    load_xarray_dataset(run_folder=tmp_path)


@pytest.mark.xfail(reason="jagged/ragged arrays are not supported (yet?)")
def test_growing_axis(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="n[j] -> x[i, j]")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i, j] -> y[i, j]")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="sum", mapspec="y[i, j] -> sum[j]")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline([generate_ints, double_it, take_sum])
    inputs = {"n": [4, 5]}  # TODO: how to deal with this?
    # The internal_shapes becomes dynamic...
    internal_shapes = {"x": (4,)}
    pipeline.map(
        inputs,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        run_folder=tmp_path,
        parallel=False,
    )


def test_storage_options():
    f = PipeFunc(lambda x: x, "y")
    with pytest.raises(ValueError, match="Storage class `invalid` not found"):
        Pipeline([f]).map({"x": 1}, None, storage="invalid")

    pipeline = Pipeline([PipeFunc(lambda x: x, "y", mapspec="x[i] -> y[i]")])
    inputs = {"x": [1, 2, 3]}
    with pytest.raises(
        ValueError,
        match="Parallel execution is not supported with `zarr_memory` storage",
    ):
        pipeline.map(inputs, None, storage="zarr_memory", parallel=True)


def test_custom_executor():
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x):
        return x

    pipeline = Pipeline([f])
    results = pipeline.map({"x": [1, 2]}, None, executor=ThreadPoolExecutor())
    assert results["y"].output.tolist() == [1, 2]


def test_independent_axes_1():
    @pipefunc(output_name="c", mapspec="a[i], b[i] -> c[i]")
    def f(a: int, b: int):
        return a + b

    @pipefunc(output_name="z", mapspec="x[i], y[i] -> z[i, k]")
    def g(x, y):
        return x + y

    output_name = "c"
    self = Pipeline([f, g])
    assert self.independent_axes_in_mapspecs(output_name) == {"i"}


def test_independent_axes_2():
    @pipefunc(output_name="y", mapspec="... -> y[i]")
    def f(x):
        return x

    @pipefunc(output_name="r", mapspec="z[i], y[i] -> r[i]")
    def g(y, z):
        return y + z

    pipeline = Pipeline([f, g])
    inputs = {"x": [1, 2, 3], "z": [3, 4, 5]}
    internal_shapes = {"y": (3,)}
    r = pipeline.map(inputs, None, internal_shapes=internal_shapes, parallel=False)
    assert r["y"].output == [1, 2, 3]
    assert r["r"].output.tolist() == [4, 6, 8]
    assert pipeline.independent_axes_in_mapspecs("r") == set()

    pipeline.add_mapspec_axis("x", axis="k")
    assert pipeline.mapspecs_as_strings == ["x[k] -> y[i, k]", "z[i], y[i, k] -> r[i, k]"]
    assert pipeline.independent_axes_in_mapspecs("r") == {"k"}


def test_parallel():
    @pipefunc(output_name="double", mapspec="x[i] -> double[i]")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="half", mapspec="x[i] -> half[i]")
    def half_it(x: int) -> int:
        return x // 2

    @pipefunc(output_name="sum")
    def take_sum(half: np.ndarray, double: np.ndarray) -> int:
        return sum(half + double)

    pipeline = Pipeline([double_it, half_it, take_sum])
    inputs = {"x": [0, 1, 2, 3]}
    run_folder = "my_run_folder"
    executor = ProcessPoolExecutor(max_workers=2)  # Use 2 processes
    results = pipeline.map(
        inputs,
        run_folder=run_folder,
        parallel=True,
        executor=executor,
        storage="shared_memory_dict",
    )
    assert results["sum"].output == 14


def test_fixed_indices(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i], y[i] -> z[i]")
    def f(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3], "y": [4, 5, 6]}
    results = pipeline.map(inputs, tmp_path, fixed_indices={"i": slice(1, None)}, parallel=False)
    assert results["z"].output.tolist() == [None, 7, 9]
    assert results["z"].store is not None
    assert results["z"].store.mask.mask.tolist() == [True, False, False]

    @pipefunc(output_name="z", mapspec="x[i], y[i, j] -> z[i, j]")
    def g(x: int, y: int) -> tuple[int, int]:
        return (x, y)

    pipeline = Pipeline([g])
    y = np.array([[4, 5], [6, 7], [8, 9]])
    assert y.shape == (3, 2)
    inputs = {"x": [1, 2, 3], "y": y}  # type: ignore[dict-item]

    results = pipeline.map(
        inputs,
        tmp_path,
        fixed_indices={"i": slice(1, None), "j": 0},
        parallel=False,
    )
    assert y[slice(1, None), 0].tolist() == [6, 8]
    assert results["z"].output.tolist() == [
        [None, None],
        [(2, 6), None],
        [(3, 8), None],
    ]

    results = pipeline.map(
        inputs,
        tmp_path,
        fixed_indices={"i": slice(2, 0, -1), "j": slice(1, None)},
        parallel=False,
    )
    assert y[slice(2, 0, -1), slice(1, None)].tolist() == [[9], [7]]
    assert results["z"].output.tolist() == [
        [None, None],
        [None, (2, 7)],
        [None, (3, 9)],
    ]

    results = pipeline.map(
        inputs,
        tmp_path,
        fixed_indices={"i": 2, "j": 0},
        parallel=False,
    )
    assert results["z"].output.tolist() == [[None, None], [None, None], [(3, 8), None]]

    with pytest.raises(IndexError, match="Fixed index `2000` for parameter `x` is out of bounds"):
        pipeline.map(
            inputs,
            tmp_path,
            fixed_indices={"i": 2000, "j": 1},
            parallel=False,
        )

    with pytest.raises(
        ValueError,
        match="Got extra `fixed_indices`: `{'not_an_index'}` that are not",
    ):
        pipeline.map(
            inputs,
            tmp_path,
            fixed_indices={"not_an_index": 0},
            parallel=False,
        )


def test_fixed_indices_with_reduction(tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x: int) -> int:
        return x

    @pipefunc(output_name="z")
    def g(y: np.ndarray) -> int:
        return sum(y)

    pipeline = Pipeline([f, g])
    inputs = {"x": [1, 2, 3]}
    with pytest.raises(ValueError, match="Axis `i` in `y` is reduced"):
        pipeline.map(inputs, tmp_path, fixed_indices={"i": 1}, parallel=False)


def test_missing_inputs():
    @pipefunc(output_name="y")
    def f(x: int) -> int:
        return x

    pipeline = Pipeline([f])
    inputs = {}
    with pytest.raises(ValueError, match="Missing inputs"):
        pipeline.map(inputs, None, parallel=False)

    with pytest.raises(
        ValueError,
        match="Got extra inputs: `not_used` that are not accepted by this pipeline",
    ):
        pipeline.map({"x": 1, "not_used": 1}, None, parallel=False)


def test_map_without_mapspec(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def f(x: int) -> int:
        return x

    pipeline = Pipeline([f])
    inputs = {"x": 1}
    results = pipeline.map(inputs, tmp_path)
    assert results["y"].output == 1


def test_add_double_axis(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def f(x):
        return x

    @pipefunc(output_name="r", mapspec="... -> r[k]")
    def g(y, z):  # noqa: ARG001
        return 1

    pipeline = Pipeline([f, g])
    pipeline.add_mapspec_axis("y", axis="i")
    pipeline.add_mapspec_axis("y", axis="j")
    assert pipeline.mapspecs_as_strings == ["... -> y[i, j]", "y[i, j] -> r[k, i, j]"]

    inputs = {"x": np.array([[0, 1, 2, 3], [0, 1, 2, 3]]), "z": np.arange(5)}
    internal_shapes = {"z": 5}
    results = pipeline.map(
        inputs,
        tmp_path,
        internal_shapes,  # type: ignore[arg-type]
        parallel=False,
    )
    assert results["r"].output == 1


def test_map_with_partial(tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x: int) -> int:
        return x

    @pipefunc(output_name="z")
    def g(y: np.ndarray) -> int:
        return sum(y)

    pipeline = Pipeline([f, g])
    inputs = {"x": [1, 2, 3]}
    results = pipeline.map(inputs, tmp_path)
    assert results["y"].output.tolist() == [1, 2, 3]
    assert results["z"].output == 6

    # Run via subpipeline and map with partial inputs
    partial = pipeline.subpipeline({"y"})
    inputs1 = {"y": results["y"].output}
    r = partial.map(inputs1, tmp_path)
    assert len(r) == 1
    assert r["z"].output == 6

    r = pipeline.map(inputs1, tmp_path, auto_subpipeline=True)
    assert len(r) == 1
    assert r["z"].output == 6

    partial = pipeline.subpipeline(output_names={"y"})
    r = partial.map(inputs, tmp_path)
    assert len(r) == 1
    assert r["y"].output.tolist() == [1, 2, 3]

    r = pipeline.map(inputs, tmp_path, output_names={"y"})
    assert len(r) == 1
    assert r["y"].output.tolist() == [1, 2, 3]
