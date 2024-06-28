from __future__ import annotations

import functools
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, Union

from pipefunc._utils import at_least_tuple, dump, equal_dicts, load
from pipefunc._version import __version__
from pipefunc.map._mapspec import MapSpec, array_shape
from pipefunc.map._storage_base import StorageBase, storage_registry

if TYPE_CHECKING:
    from pipefunc import Pipeline

_OUTPUT_TYPE: TypeAlias = Union[str, tuple[str, ...]]


class Shapes(NamedTuple):
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]]
    masks: dict[_OUTPUT_TYPE, tuple[bool, ...]]


def map_shapes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> Shapes:
    if internal_shapes is None:
        internal_shapes = {}
    internal = {k: at_least_tuple(v) for k, v in internal_shapes.items()}

    input_parameters = set(pipeline.topological_generations.root_args)

    inputs_with_defaults = pipeline.defaults | inputs
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]] = {
        p: array_shape(inputs_with_defaults[p], p)
        for p in input_parameters
        if p in pipeline.mapspec_names
    }
    masks = {name: len(shape) * (True,) for name, shape in shapes.items()}
    mapspec_funcs = [f for f in pipeline.sorted_functions if f.mapspec]
    for func in mapspec_funcs:
        assert func.mapspec is not None  # mypy
        input_shapes = {p: shapes[p] for p in func.mapspec.input_names if p in shapes}
        output_shapes = {p: internal[p] for p in func.mapspec.output_names if p in internal}
        output_shape, mask = func.mapspec.shape(input_shapes, output_shapes)  # type: ignore[arg-type]
        shapes[func.output_name] = output_shape
        masks[func.output_name] = mask
        if isinstance(func.output_name, tuple):
            for output_name in func.output_name:
                shapes[output_name] = output_shape
                masks[output_name] = mask

    assert all(k in shapes for k in pipeline.mapspec_names if k not in internal)
    return Shapes(shapes, masks)


@dataclass(frozen=True, eq=True)
class RunInfo:
    input_paths: dict[str, Path]
    defaults_path: Path
    all_output_names: set[str]
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]]
    internal_shapes: dict[str, int | tuple[int, ...]] | None
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]]
    run_folder: Path
    mapspecs_as_strings: list[str]
    storage: str
    pipefunc_version: str = __version__

    @classmethod
    def create(
        cls: type[RunInfo],
        run_folder: str | Path,
        pipeline: Pipeline,
        inputs: dict[str, Any],
        internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
        *,
        storage: str,
        cleanup: bool = True,
    ) -> RunInfo:
        run_folder = Path(run_folder)
        if cleanup:
            _cleanup_run_folder(run_folder)
        else:
            _compare_to_previous_run_info(pipeline, run_folder, inputs, internal_shapes)
        _check_inputs(pipeline, inputs)
        input_paths = _dump_inputs(inputs, run_folder)
        defaults_path = _dump_defaults(pipeline.defaults, run_folder)
        shapes, masks = map_shapes(pipeline, inputs, internal_shapes or {})
        return cls(
            input_paths=input_paths,
            defaults_path=defaults_path,
            all_output_names=pipeline.all_output_names,
            shapes=shapes,
            internal_shapes=internal_shapes,
            shape_masks=masks,
            mapspecs_as_strings=pipeline.mapspecs_as_strings,
            run_folder=run_folder,
            storage=storage,
        )

    @property
    def storage_class(self) -> type[StorageBase]:
        if self.storage not in storage_registry:
            available = ", ".join(storage_registry.keys())
            msg = f"Storage class `{self.storage}` not found, only `{available}` available."
            raise ValueError(msg)
        return storage_registry[self.storage]

    def init_store(self) -> dict[str, StorageBase]:
        return _init_storage(
            self.mapspecs,
            self.storage_class,
            self.shapes,
            self.shape_masks,
            self.run_folder,
        )

    @functools.cached_property
    def inputs(self) -> dict[str, Any]:
        return {k: _load_input(k, self.input_paths, cache=False) for k in self.input_paths}

    @functools.cached_property
    def defaults(self) -> Any:
        return load(self.defaults_path, cache=False)

    @functools.cached_property
    def mapspecs(self) -> list[MapSpec]:
        return [MapSpec.from_string(ms) for ms in self.mapspecs_as_strings]

    def dump(self, run_folder: str | Path) -> None:
        path = self.path(run_folder)
        data = asdict(self)
        data["input_paths"] = {k: str(v) for k, v in data["input_paths"].items()}
        data["all_output_names"] = sorted(data["all_output_names"])
        for key in ["shapes", "shape_masks"]:
            data[key] = {",".join(at_least_tuple(k)): v for k, v in data[key].items()}
        data["run_folder"] = str(data["run_folder"])
        data["defaults_path"] = str(data["defaults_path"])
        with path.open("w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls: type[RunInfo], run_folder: str | Path) -> RunInfo:
        path = cls.path(run_folder)
        with path.open() as f:
            data = json.load(f)
        data["input_paths"] = {k: Path(v) for k, v in data["input_paths"].items()}
        data["all_output_names"] = set(data["all_output_names"])
        for key in ["shapes", "shape_masks"]:
            data[key] = {_maybe_tuple(k): tuple(v) for k, v in data[key].items()}
        if data["internal_shapes"] is not None:
            data["internal_shapes"] = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in data["internal_shapes"].items()
            }
        data["run_folder"] = Path(data["run_folder"])
        data["defaults_path"] = Path(data["defaults_path"])
        return cls(**data)

    @staticmethod
    def path(run_folder: str | Path) -> Path:
        return Path(run_folder) / "run_info.json"

    def copy(self, **update: Any) -> RunInfo:
        data = asdict(self)
        data.update(update)
        return RunInfo(**data)


def _cleanup_run_folder(run_folder: str | Path) -> None:
    """Remove the run folder and its contents."""
    run_folder = Path(run_folder)
    shutil.rmtree(run_folder, ignore_errors=True)


def _compare_to_previous_run_info(
    pipeline: Pipeline,
    run_folder: Path,
    inputs: dict[str, Any],
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
) -> None:  # pragma: no cover
    if not RunInfo.path(run_folder).is_file():
        return
    try:
        old = RunInfo.load(run_folder)
    except Exception as e:  # noqa: BLE001
        msg = f"Could not load previous run info: {e}, cannot use `cleanup=False`."
        raise ValueError(msg) from None
    if internal_shapes != old.internal_shapes:
        msg = "Internal shapes do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    if pipeline.mapspecs_as_strings != old.mapspecs_as_strings:
        msg = "Mapspecs do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)
    if shapes != old.shapes:
        msg = "Shapes do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    equal_inputs = equal_dicts(inputs, old.inputs, verbose=True)
    if equal_inputs is None:
        print(
            "Could not compare new `inputs` to `inputs` from previous run."
            " Proceeding *without* `cleanup`, hoping for the best.",
        )
        return
    if not equal_inputs:
        msg = f"Inputs `{inputs=}` / `{old.inputs=}` do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    equal_defaults = equal_dicts(pipeline.defaults, old.defaults)
    if equal_defaults is None:
        print(
            "Could not compare new `defaults` to `defaults` from previous run."
            " Proceeding *without* `cleanup`, hoping for the best.",
        )
        return
    if not equal_defaults:
        msg = f"Defaults `{pipeline.defaults=}` / `{old.defaults=}` do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)


def _check_inputs(pipeline: Pipeline, inputs: dict[str, Any]) -> None:
    input_dimensions = pipeline.mapspec_dimensions
    for name, value in inputs.items():
        if (dim := input_dimensions.get(name, 0)) > 1 and isinstance(value, (list, tuple)):
            msg = f"Expected {dim}D `numpy.ndarray` for input `{name}`, got {type(value)}."
            raise ValueError(msg)


def _maybe_tuple(x: str) -> tuple[str, ...] | str:
    if "," in x:
        return tuple(x.split(","))
    return x


def _dump_inputs(
    inputs: dict[str, Any],
    run_folder: Path,
) -> dict[str, Path]:
    folder = run_folder / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, v in inputs.items():
        path = folder / f"{k}.cloudpickle"
        dump(v, path)
        paths[k] = path
    return paths


def _dump_defaults(defaults: dict[str, Any], run_folder: Path) -> Path:
    folder = run_folder / "defaults"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "defaults.cloudpickle"
    dump(defaults, path)
    return path


def _load_input(name: str, input_paths: dict[str, Path], *, cache: bool = True) -> Any:
    path = input_paths[name]
    return load(path, cache=cache)


def _init_storage(
    mapspecs: list[MapSpec],
    storage_class: type[StorageBase],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]],
    run_folder: Path,
) -> dict[str, StorageBase]:
    store: dict[str, StorageBase] = {}
    for mapspec in mapspecs:
        output_names = mapspec.output_names
        shape = shapes[output_names[0]]
        mask = shape_masks[output_names[0]]
        arrays = _init_file_arrays(output_names, shape, mask, storage_class, run_folder)
        for output_name, arr in zip(output_names, arrays):
            store[output_name] = arr
    return store


def _init_file_arrays(
    output_name: _OUTPUT_TYPE,
    shape: tuple[int, ...],
    mask: tuple[bool, ...],
    storage_class: type[StorageBase],
    run_folder: Path,
) -> list[StorageBase]:
    external_shape = _external_shape(shape, mask)
    internal_shape = _internal_shape(shape, mask)
    output_names = at_least_tuple(output_name)
    paths = [_file_array_path(output_name, run_folder) for output_name in output_names]  # type: ignore[misc]
    return [storage_class(path, external_shape, internal_shape, mask) for path in paths]


def _file_array_path(output_name: str, run_folder: Path) -> Path:
    assert isinstance(output_name, str)
    return run_folder / "outputs" / output_name


def _internal_shape(shape: tuple[int, ...], mask: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(s for s, m in zip(shape, mask) if not m)


def _external_shape(shape: tuple[int, ...], mask: tuple[bool, ...]) -> tuple[int, ...]:
    return tuple(s for s, m in zip(shape, mask) if m)
