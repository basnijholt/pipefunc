from __future__ import annotations

from typing import TYPE_CHECKING

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs
from pipefunc.map.xarray import to_xarray

if TYPE_CHECKING:
    from pathlib import Path


def test_to_xarray_1_dim(tmp_path: Path):
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([double_it])
    inputs = {"x": [1, 2, 3]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    data = results[-1].output
    output_name = results[-1].output_name
    mapspecs = pipeline.mapspecs()

    da = to_xarray(data, output_name, mapspecs, inputs)

    expected_coords = {"x": inputs["x"]}
    expected_dims = ["x"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.to_numpy().tolist() == [2, 4, 6]


def test_to_xarray_2_dim(tmp_path: Path):
    @pipefunc(output_name="z", mapspec="x[i], y[j] -> z[i, j]")
    def f(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3], "y": [4, 5]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    data = results[-1].output
    output_name = results[-1].output_name
    mapspecs = pipeline.mapspecs()

    da = to_xarray(data, output_name, mapspecs, inputs)

    expected_coords = {"x": inputs["x"], "y": inputs["y"]}
    expected_dims = ["x", "y"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.coords["y"].to_numpy().tolist() == expected_coords["y"]
    assert da.to_numpy().tolist() == [[5, 6], [6, 7], [7, 8]]


def test_to_xarray_2_dim_zipped(tmp_path: Path):
    @pipefunc(output_name="r", mapspec="x[i], y[i], z[j] -> r[i, j]")
    def f(x: int, y: int, z: int) -> int:
        return x + y + z

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    data = results[-1].output
    output_name = results[-1].output_name
    mapspecs = pipeline.mapspecs()

    da = to_xarray(data, output_name, mapspecs, inputs)

    expected_coords = {"x": inputs["x"], "y": inputs["y"], "z": inputs["z"]}
    expected_dims = ["x:y", "z"]
    assert list(da.dims) == expected_dims
    assert da.coords["x:y"].to_numpy().tolist() == [(1, 4), (2, 5), (3, 6)]
    assert da.coords["z"].to_numpy().tolist() == expected_coords["z"]
    assert da.to_numpy().tolist() == [[12, 13], [14, 15], [16, 17]]


def test_to_xarray_1_dim_2_funcs(tmp_path: Path):
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def g(y: int) -> int:
        return y + 1

    pipeline = Pipeline([f, g])
    inputs = {"x": [1, 2, 3]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    data = results[-1].output
    output_name = results[-1].output_name
    mapspecs = pipeline.mapspecs()

    da = to_xarray(
        data,
        output_name,
        mapspecs,
        inputs,
        use_intermediate=True,
        run_folder=tmp_path,
    )
    expected_coords = {"x": inputs["x"]}
    expected_dims = ["x"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.to_numpy().tolist() == [3, 5, 7]


def test_to_xarray_from_step(tmp_path: Path):
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        """Generate a list of integers from 0 to n-1."""
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([generate_ints, double_it])
    inputs = {"n": 4}
    internal_shapes = {"x": (4,)}
    results = pipeline.map(
        inputs,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        run_folder=tmp_path,
        parallel=False,
    )
    data = results[-1].output
    mapspecs = pipeline.mapspecs()
    output_name = results[-1].output_name

    # First *with* intermediate results
    da = to_xarray(
        data,
        output_name,
        mapspecs,
        inputs,
        use_intermediate=True,
        run_folder=tmp_path,
    )
    x = load_outputs("x", run_folder=tmp_path)
    expected_coords = {"x": x}
    expected_dims = ["x"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]

    # Now *without* intermediate results
    # TODO: Fix this case!
