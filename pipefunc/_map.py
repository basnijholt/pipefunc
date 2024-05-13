from __future__ import annotations

import functools
import json
import shutil
from pathlib import Path
from typing import Any, Literal, NamedTuple, Tuple, Union

import cloudpickle
import networkx as nx
import numpy as np

from pipefunc import PipeFunc, Pipeline
from pipefunc._filearray import FileArray
from pipefunc._mapspec import MapSpec, array_shape
from pipefunc._utils import at_least_tuple

_OUTPUT_TYPE = Union[str, Tuple[str, ...]]


def _func_path(func: PipeFunc, folder: Path) -> Path:
    return folder / f"{func.__module__}.{func.__name__}.cloudpickle"


def _load(path: Path) -> Any:
    with path.open("rb") as f:
        return cloudpickle.load(f)


def _dump(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        cloudpickle.dump(obj, f)


_load_cached = functools.lru_cache(maxsize=None)(_load)


def _dump_functions(pipeline: Pipeline, run_folder: Path) -> list[Path]:
    folder = run_folder / "functions"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for func in pipeline.functions:
        path = _func_path(func, folder)
        _dump(func, path)
        paths.append(path)
    return paths


def _dump_inputs(inputs: dict[str, Any], run_folder: Path) -> dict[str, Path]:
    folder = run_folder / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, v in inputs.items():
        path = folder / f"{k}.cloudpickle"
        _dump(v, path)
        paths[k] = path
    return paths


def _load_input(name: str, input_paths: dict[str, Path]) -> Any:
    path = input_paths[name]
    return _load_cached(path)


def _output_path(output_name: str, folder: Path) -> Path:
    return folder / f"{output_name}.cloudpickle"


def _dump_output(
    func: PipeFunc,
    output: Any,
    run_folder: Path,
) -> Any:
    folder = run_folder / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(func.output_name, tuple):
        new_output = []
        for output_name in func.output_name:
            assert func.output_picker is not None
            _output = func.output_picker(output, output_name)
            new_output.append(_output)
            path = _output_path(output_name, folder)
            _dump(output, path)
        output = new_output
    else:
        path = _output_path(func.output_name, folder)
        _dump(output, path)
    return output


def _load_output(output_name: str, run_folder: Path) -> Any:
    folder = run_folder / "outputs"
    path = _output_path(output_name, folder)
    return _load_cached(path)


def _clean_run_folder(run_folder: Path) -> None:
    for folder in ["functions", "inputs", "outputs"]:
        shutil.rmtree(run_folder / folder, ignore_errors=True)


def _json_serializable(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    msg = f"Object {obj} is not JSON serializable"
    raise ValueError(msg)


def _dump_run_info(
    function_paths: list[Path],
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    run_folder: Path,
) -> None:
    path = run_folder / "run_info.json"
    info = {
        "functions": _json_serializable(function_paths),
        "inputs": _json_serializable(input_paths),
        "shapes": _json_serializable(list(shapes.items())),
    }
    with path.open("w") as f:
        json.dump(info, f, indent=4)


def _file_array_path(output_name: str, run_folder: Path) -> Path:
    assert isinstance(output_name, str)
    return run_folder / "outputs" / output_name


def _output_types(
    pipeline: Pipeline,
) -> dict[str, Literal["single", "single_indexable", "file_array"]]:
    generations = list(nx.topological_generations(pipeline.graph))
    assert all(isinstance(x, str) for x in generations[0])
    otypes: dict[str, Literal["single", "single_indexable", "file_array"]] = {}
    for gen in generations:
        for f in gen:
            if isinstance(f, str):
                otypes[f] = "single"
                continue
            assert isinstance(f, PipeFunc)
            if f.mapspec is None:
                for output_name in at_least_tuple(f.output_name):
                    otypes[output_name] = "single"
            else:
                for output_name in at_least_tuple(f.output_name):
                    otypes[output_name] = "file_array"
            for p in f.parameters:
                if (
                    otypes.get(p) == "single"
                    and f.mapspec is not None
                    and p in f.mapspec.parameters  # type: ignore[union-attr]
                ):
                    otypes[p] = "single_indexable"
    assert all(isinstance(x, str) for x in otypes)
    return otypes


def _func_kwargs(
    func: PipeFunc,
    pipeline: Pipeline,
    output_types: dict[
        str,
        Literal["single", "single_indexable", "file_array"],
    ],
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    run_folder: Path,
) -> dict[str, Any]:
    kwargs = {}
    for parameter in func.parameters:
        output_name = (
            parameter
            if parameter in output_types
            else pipeline.output_to_func[parameter].output_name
        )
        assert isinstance(output_name, str)
        output_type = output_types[output_name]
        if parameter in input_paths:
            assert output_type in ("single", "single_indexable")
            value = _load_input(parameter, input_paths)
            kwargs[parameter] = value
        elif output_type in ("single_indexable", "single"):
            value = _load_output(output_name, run_folder)
            kwargs[output_name] = value
        else:
            assert output_type == "file_array"
            file_array_path = _file_array_path(output_name, run_folder)
            shape = shapes[output_name]
            file_array = FileArray(file_array_path, shape)
            kwargs[output_name] = file_array.to_array()
    return kwargs


def _select_kwargs(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    index: int,
) -> dict[str, Any]:
    assert func.mapspec is not None
    input_keys = {
        k: v[0] if len(v) == 1 else v
        for k, v in func.mapspec.input_keys(shape, index).items()
    }
    return {k: v[input_keys[k]] if k in input_keys else v for k, v in kwargs.items()}


def _execute_map_spec(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    run_folder: Path,
) -> np.ndarray | list[np.ndarray]:
    assert isinstance(func.mapspec, MapSpec)
    shape = shapes[func.output_name]
    n = np.prod(shape)
    file_arrays = []
    output_arrays: list[np.ndarray] = []
    output_names = at_least_tuple(func.output_name)
    for output_name in output_names:
        file_array_path = _file_array_path(output_name, run_folder)
        file_array = FileArray(file_array_path, shape)
        file_arrays.append(file_array)
        output_arrays.append(np.empty(n, dtype=object))

    for index in range(n):
        selected = _select_kwargs(func, kwargs, shape, index)
        try:
            output = func(**selected)
        except Exception as e:
            msg = f"Error in {func.__name__} at {index=}, {kwargs=}, {selected=}"
            raise ValueError(msg) from e

        output_key = func.mapspec.output_key(shape, index)
        for output_name, file_array, output_array in zip(
            output_names,
            file_arrays,
            output_arrays,
        ):
            _output = (
                func.output_picker(output, output_name)
                if func.output_picker is not None
                else output
            )
            file_array.dump(output_key, _output)
            output_array[index] = _output
    output_arrays = [x.reshape(shape) for x in output_arrays]
    return output_arrays if isinstance(func.output_name, tuple) else output_arrays[0]


def map_shapes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    manual_shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[_OUTPUT_TYPE, tuple[int, ...]]:
    map_parameters: set[str] = set()
    for func in pipeline.functions:
        if func.mapspec:
            map_parameters.update(func.mapspec.parameters)
            for output in func.mapspec.outputs:
                map_parameters.add(output.name)

    generations = list(nx.topological_generations(pipeline.graph))
    input_parameters = set(generations[0])

    shapes = {
        p: array_shape(inputs[p]) for p in input_parameters if p in map_parameters
    }
    if manual_shapes is not None:
        shapes.update(manual_shapes)

    for gen in generations[1:]:
        for func in gen:
            if func.mapspec:
                input_shapes = {}
                for p in func.mapspec.parameters:
                    if shape := shapes.get(p):
                        input_shapes[p] = shape
                    else:
                        msg = (
                            f"Parameter `{p}` is used in map but its shape"
                            " cannot be inferred from the inputs."
                            " Provide the shape manually in `manual_shapes`."
                        )
                        raise ValueError(msg)
                output_shape = func.mapspec.shape(input_shapes)
                shapes[func.output_name] = output_shape
                if isinstance(func.output_name, tuple):
                    for output_name in func.output_name:
                        shapes[output_name] = output_shape

    assert all(k in shapes for k in map_parameters)
    return shapes


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any


def run_pipeline(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path,
    manual_shapes: dict[str, tuple[int, ...]] | None = None,
) -> list[Result]:
    run_folder = Path(run_folder)
    _clean_run_folder(run_folder)
    function_paths = _dump_functions(pipeline, run_folder)
    input_paths = _dump_inputs(inputs, run_folder)
    shapes = map_shapes(pipeline, inputs, manual_shapes)
    _dump_run_info(function_paths, input_paths, shapes, run_folder)
    output_types = _output_types(pipeline)
    if manual_shapes is not None:
        assert all(
            k in manual_shapes
            for k, v in output_types.items()
            if v == "single_indexable"
        )

    generations = list(nx.topological_generations(pipeline.graph))
    assert all(isinstance(x, str) for x in generations[0])
    outputs = []
    for gen in generations[1:]:
        # These evaluations can happen in parallel
        for func in gen:
            kwargs = _func_kwargs(
                func,
                pipeline,
                output_types,
                input_paths,
                shapes,
                run_folder,
            )
            if func.mapspec:
                output = _execute_map_spec(func, kwargs, shapes, run_folder)
            else:
                try:
                    output = func(**kwargs)
                except Exception as e:
                    msg = f"Error in {func.__name__} with {kwargs=}"
                    raise ValueError(msg) from e
                output = _dump_output(func, output, run_folder)

            if isinstance(func.output_name, str):
                outputs.append(
                    Result(
                        function=func.__name__,
                        kwargs=kwargs,
                        output_name=func.output_name,
                        output=output,
                    ),
                )
            else:
                for output_name, _output in zip(func.output_name, output):
                    outputs.append(
                        Result(
                            function=func.__name__,
                            kwargs=kwargs,
                            output_name=output_name,
                            output=_output,
                        ),
                    )
    return outputs
