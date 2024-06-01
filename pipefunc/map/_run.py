from __future__ import annotations

import functools
import tempfile
from collections import OrderedDict
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, NamedTuple, Sequence, Tuple, Union

import numpy as np

from pipefunc._utils import at_least_tuple, dump, handle_error, load, prod
from pipefunc.map._mapspec import (
    MapSpec,
    _shape_to_key,
    mapspec_dimensions,
    validate_consistent_axes,
)
from pipefunc.map._run_info import RunInfo, _external_shape, _internal_shape, _load_input
from pipefunc.map._storage_base import (
    StorageBase,
    _iterate_shape_indices,
    _select_by_mask,
)

if TYPE_CHECKING:
    import sys

    import xarray as xr

    from pipefunc import PipeFunc, Pipeline
    from pipefunc._pipeline import _Generations

    if sys.version_info < (3, 10):  # pragma: no cover
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias

_OUTPUT_TYPE: TypeAlias = Union[str, Tuple[str, ...]]


@dataclass
class _MockPipeline:
    """An object that contains all information required to run a pipeline.

    Ensures that we're not pickling the entire pipeline object when not needed.
    """

    defaults: dict[str, Any]
    mapspec_names: set[str]
    topological_generations: _Generations

    @classmethod
    def from_pipeline(cls: type[_MockPipeline], pipeline: Pipeline) -> _MockPipeline:  # noqa: PYI019
        return cls(
            defaults=pipeline.defaults,
            mapspec_names=pipeline.mapspec_names,
            topological_generations=pipeline.topological_generations,
        )

    @property
    def functions(self) -> list[PipeFunc]:
        # Return all functions in topological order
        return [f for gen in self.topological_generations[1] for f in gen]

    def mapspecs(self, *, ordered: bool = True) -> list[MapSpec]:  # noqa: ARG002
        """Return the MapSpecs for all functions in the pipeline."""
        functions = self.functions  # topologically ordered
        return [f.mapspec for f in functions if f.mapspec]

    def mapspecs_as_strings(self) -> list[str]:
        """Return the MapSpecs for all functions in the pipeline as strings."""
        return [str(ms) for ms in self.mapspecs()]

    @property
    def sorted_functions(self) -> list[PipeFunc]:
        """Return the functions in the pipeline in topological order."""
        return self.functions

    def mapspec_dimensions(self) -> dict[str, int]:
        """Return the number of dimensions for each array parameter in the pipeline."""
        return mapspec_dimensions(self.mapspecs())


def _output_path(output_name: str, run_folder: Path) -> Path:
    return run_folder / "outputs" / f"{output_name}.cloudpickle"


def _dump_output(func: PipeFunc, output: Any, run_folder: Path) -> tuple[Any, ...]:
    folder = run_folder / "outputs"
    folder.mkdir(parents=True, exist_ok=True)

    if isinstance(func.output_name, tuple):
        new_output = []  # output in same order as func.output_name
        for output_name in func.output_name:
            assert func.output_picker is not None
            _output = func.output_picker(output, output_name)
            new_output.append(_output)
            path = _output_path(output_name, run_folder)
            dump(_output, path)
        return tuple(new_output)
    path = _output_path(func.output_name, run_folder)
    dump(output, path)
    return (output,)


def _load_output(output_name: str, run_folder: Path) -> Any:
    path = _output_path(output_name, run_folder)
    return load(path)


def _load_parameter(
    parameter: str,
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]],
    store: dict[str, StorageBase],
    run_folder: Path,
) -> Any:
    if parameter in input_paths:
        return _load_input(parameter, input_paths)
    if parameter not in shapes or not any(shape_masks[parameter]):
        return _load_output(parameter, run_folder)
    return store[parameter]


def _func_kwargs(
    func: PipeFunc,
    input_paths: dict[str, Path],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]],
    store: dict[str, StorageBase],
    run_folder: Path,
) -> dict[str, Any]:
    return {
        p: _load_parameter(p, input_paths, shapes, shape_masks, store, run_folder)
        for p in func.parameters
    }


def _select_kwargs(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    index: int,
) -> dict[str, Any]:
    assert func.mapspec is not None
    external_shape = _external_shape(shape, shape_mask)
    input_keys = func.mapspec.input_keys(external_shape, index)
    normalized_keys = {k: v[0] if len(v) == 1 else v for k, v in input_keys.items()}
    selected = {k: v[normalized_keys[k]] if k in normalized_keys else v for k, v in kwargs.items()}
    _load_file_array(selected)
    return selected


def _init_result_arrays(output_name: _OUTPUT_TYPE, shape: tuple[int, ...]) -> list[np.ndarray]:
    return [np.empty(prod(shape), dtype=object) for _ in at_least_tuple(output_name)]


def _pick_output(func: PipeFunc, output: Any) -> tuple[Any, ...]:
    return tuple(
        (func.output_picker(output, output_name) if func.output_picker is not None else output)
        for output_name in at_least_tuple(func.output_name)
    )


def _run_iteration(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    index: int,
) -> Any:
    selected = _select_kwargs(func, kwargs, shape, shape_mask, index)
    try:
        return func(**selected)
    except Exception as e:
        handle_error(e, func, selected)
        raise  # handle_error raises but mypy doesn't know that


def _run_iteration_and_process(
    index: int,
    func: PipeFunc,
    kwargs: dict[str, Any],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    file_arrays: Sequence[StorageBase],
) -> tuple[Any, ...]:
    output = _run_iteration(func, kwargs, shape, shape_mask, index)
    outputs = _pick_output(func, output)
    _update_file_array(func, file_arrays, shape, shape_mask, index, outputs)
    return outputs


def _update_file_array(
    func: PipeFunc,
    file_arrays: Sequence[StorageBase],
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    index: int,
    outputs: tuple[Any, ...],
) -> None:
    assert isinstance(func.mapspec, MapSpec)
    external_shape = _external_shape(shape, shape_mask)
    output_key = func.mapspec.output_key(external_shape, index)
    for file_array, _output in zip(file_arrays, outputs):
        file_array.dump(output_key, _output)


def _indices_to_flat_index(
    shape: tuple[int, ...],
    internal_shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
    external_index: tuple[int, ...],
    internal_index: tuple[int, ...],
) -> np.int_:
    full_index = _select_by_mask(shape_mask, external_index, internal_index)
    full_shape = _select_by_mask(shape_mask, shape, internal_shape)
    return np.ravel_multi_index(full_index, full_shape)


def _set_output(
    arr: np.ndarray,
    output: np.ndarray,
    linear_index: int,
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
) -> None:
    external_shape = _external_shape(shape, shape_mask)
    internal_shape = _internal_shape(shape, shape_mask)
    external_index = _shape_to_key(external_shape, linear_index)
    for internal_index in _iterate_shape_indices(internal_shape):
        flat_index = _indices_to_flat_index(
            external_shape,
            internal_shape,
            shape_mask,
            external_index,
            internal_index,
        )
        arr[flat_index] = output[internal_index]


def _update_result_array(
    result_arrays: list[np.ndarray],
    index: int,
    output: list[Any],
    shape: tuple[int, ...],
    mask: tuple[bool, ...],
) -> None:
    for result_array, _output in zip(result_arrays, output):
        if not all(mask):
            _output = np.asarray(_output)  # In case _output is a list
            _set_output(result_array, _output, index, shape, mask)
        else:
            result_array[index] = _output


def _existing_and_missing_indices(file_arrays: list[StorageBase]) -> tuple[list[int], list[int]]:
    masks = (arr.mask_linear() for arr in file_arrays)
    existing_indices = []
    missing_indices = []
    for i, mask_values in enumerate(zip(*masks)):
        if any(mask_values):  # rerun if any of the outputs are missing
            missing_indices.append(i)
        else:
            existing_indices.append(i)
    return existing_indices, missing_indices


@contextmanager
def _maybe_executor(
    executor: Executor | None,
    parallel: bool,  # noqa: FBT001
) -> Generator[Executor | None, None, None]:
    if executor is None and parallel:
        with ProcessPoolExecutor() as new_executor:  # shuts down the executor after use
            yield new_executor
    else:
        yield executor


class _MapSpecArgs(NamedTuple):
    process_index: functools.partial[tuple[Any, ...]]
    existing: list[int]
    missing: list[int]
    result_arrays: list[np.ndarray]
    shape: tuple[int, ...]
    mask: tuple[bool, ...]
    file_arrays: list[StorageBase]


def _prepare_submit_map_spec(
    func: PipeFunc,
    kwargs: dict[str, Any],
    shapes: dict[_OUTPUT_TYPE, tuple[int, ...]],
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]],
    store: dict[str, StorageBase],
) -> _MapSpecArgs:
    assert isinstance(func.mapspec, MapSpec)
    shape = shapes[func.output_name]
    mask = shape_masks[func.output_name]
    file_arrays = [store[name] for name in at_least_tuple(func.output_name)]
    result_arrays = _init_result_arrays(func.output_name, shape)
    process_index = functools.partial(
        _run_iteration_and_process,
        func=func,
        kwargs=kwargs,
        shape=shape,
        shape_mask=mask,
        file_arrays=file_arrays,
    )
    existing, missing = _existing_and_missing_indices(file_arrays)  # type: ignore[arg-type]
    return _MapSpecArgs(process_index, existing, missing, result_arrays, shape, mask, file_arrays)


def _maybe_parallel_map(func: Callable[..., Any], seq: Sequence, executor: Executor | None) -> Any:
    if executor is not None:
        return executor.map(func, seq)
    return map(func, seq)


def _maybe_submit(func: Callable[..., Any], executor: Executor | None, *args: Any) -> Any:
    if executor:
        return executor.submit(func, *args)
    return func(*args)


def _maybe_load_single_output(
    func: PipeFunc,
    run_folder: Path,
    *,
    return_output: bool = True,
) -> tuple[Any, bool]:
    """Load the output if it exists.

    Returns the output and a boolean indicating whether the output exists.
    """
    output_paths = [_output_path(p, run_folder) for p in at_least_tuple(func.output_name)]
    if all(p.is_file() for p in output_paths):
        if not return_output:
            return None, True
        outputs = [load(p) for p in output_paths]
        if isinstance(func.output_name, tuple):
            return outputs, True
        return outputs[0], True
    return None, False


def _submit_single(func: PipeFunc, kwargs: dict[str, Any], run_folder: Path) -> Any:
    # Load the output if it exists
    output, exists = _maybe_load_single_output(func, run_folder)
    if exists:
        return output

    # Otherwise, run the function
    _load_file_array(kwargs)
    try:
        return func(**kwargs)
    except Exception as e:
        handle_error(e, func, kwargs)
        raise  # handle_error raises but mypy doesn't know that


def _maybe_load_file_array(x: Any) -> Any:
    if isinstance(x, StorageBase):
        return x.to_array()
    return x


def _load_file_array(kwargs: dict[str, Any]) -> None:
    for k, v in kwargs.items():
        kwargs[k] = _maybe_load_file_array(v)


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any
    store: StorageBase | None


def _ensure_run_folder(run_folder: str | Path | None) -> Path:
    if run_folder is None:
        tmp_dir = tempfile.mkdtemp()
        run_folder = Path(tmp_dir)
    return Path(run_folder)


def run(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    parallel: bool = True,
    executor: Executor | None = None,
    storage: str = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
) -> dict[str, Result]:
    """Run a pipeline with `MapSpec` functions for given `inputs`.

    Parameters
    ----------
    pipeline
        The pipeline to run.
    inputs
        The inputs to the pipeline. The keys should be the names of the input
        parameters of the pipeline functions and the values should be the
        corresponding input data, these are either single values for functions without `mapspec`
        or lists of values or `numpy.ndarray`s for functions with `mapspec`.
    run_folder
        The folder to store the run information. If `None`, a temporary folder
        is created.
    internal_shapes
        The shapes for intermediary outputs that cannot be inferred from the inputs.
        You will receive an exception if the shapes cannot be inferred and need to be provided.
    parallel
        Whether to run the functions in parallel.
    executor
        The executor to use for parallel execution. If `None`, a `ProcessPoolExecutor`
        is used. Only relevant if `parallel=True`.
    storage
        The storage class to use for the file arrays. The default is `file_array`.
    persist_memory
        Whether to write results to disk when memory based storage is used.
        Does not have any effect when file based storage is used.
        Can use any registered storage class. See `pipefunc.map.storage_registry`.
    cleanup
        Whether to clean up the `run_folder` before running the pipeline.

    """
    # TODO: implement setting `output_name`, see #127
    _validate_complete_inputs(pipeline, inputs, output_name=None)
    validate_consistent_axes(pipeline.mapspecs(ordered=False))
    run_folder = _ensure_run_folder(run_folder)
    run_info = RunInfo.create(
        run_folder,
        pipeline,
        inputs,
        internal_shapes,
        storage=storage,
        cleanup=cleanup,
    )
    run_info.dump(run_folder)
    outputs: dict[str, Result] = OrderedDict()
    store = run_info.init_store()
    _check_parallel(parallel, store)

    with _maybe_executor(executor, parallel) as ex:
        for gen in pipeline.topological_generations.function_lists:
            _run_and_process_generation(gen, run_info, run_folder, store, outputs, ex)

    if persist_memory:  # Only relevant for memory based storage
        for arr in store.values():
            arr.persist()

    return outputs


def _run_and_process_generation(
    generation: list[PipeFunc],
    run_info: RunInfo,
    run_folder: Path,
    store: dict[str, StorageBase],
    outputs: dict[str, Result],
    executor: Executor | None,
) -> None:
    tasks: dict[PipeFunc, Any] = {}

    # First submit all calls
    for func in generation:
        kwargs = _func_kwargs(
            func,
            run_info.input_paths,
            run_info.shapes,
            run_info.shape_masks,
            store,
            run_folder,
        )
        if func.mapspec and func.mapspec.inputs:
            args = _prepare_submit_map_spec(
                func,
                kwargs,
                run_info.shapes,
                run_info.shape_masks,
                store,
            )
            r = _maybe_parallel_map(args.process_index, args.missing, executor)
            tasks[func] = r, args
        else:
            tasks[func] = _maybe_submit(_submit_single, executor, func, kwargs, run_folder)

    # Then process the results
    for func in generation:
        _outputs = _process_task(func, tasks[func], run_folder, store, kwargs, executor)
        outputs.update(_outputs)


def _process_task(
    func: PipeFunc,
    task: Any,
    run_folder: Path,
    store: dict[str, StorageBase],
    kwargs: dict[str, Any],
    executor: Executor | None = None,
) -> dict[str, Result]:
    if func.mapspec and func.mapspec.inputs:
        r, args = task
        outputs_list = list(r)

        for index, outputs in zip(args.missing, outputs_list):
            _update_result_array(args.result_arrays, index, outputs, args.shape, args.mask)

        for index in args.existing:
            outputs = [file_array.get_from_index(index) for file_array in args.file_arrays]
            _update_result_array(args.result_arrays, index, outputs, args.shape, args.mask)

        output = tuple(x.reshape(args.shape) for x in args.result_arrays)
    else:
        r = task.result() if executor else task
        output = _dump_output(func, r, run_folder)

    # Note that the kwargs still contain the StorageBase objects if _submit_map_spec
    # was used.
    return {
        output_name: Result(
            function=func.__name__,
            kwargs=kwargs,
            output_name=output_name,
            output=_output,
            store=store.get(output_name),
        )
        for output_name, _output in zip(at_least_tuple(func.output_name), output)
    }


def _check_parallel(parallel: bool, store: dict[str, StorageBase]) -> None:  # noqa: FBT001
    if not parallel:
        return
    # Assumes all storage classes are the same! Might change in the future.
    storage = next(iter(store.values()))
    if not storage.parallelizable:
        msg = (
            f"Parallel execution is not supported with `{storage.storage_id}` storage."
            " Use a file based storage or `shared_memory` / `zarr_shared_memory`."
        )
        raise ValueError(msg)


def load_outputs(*output_names: str, run_folder: str | Path) -> Any:
    """Load the outputs of a run."""
    run_folder = Path(run_folder)
    run_info = RunInfo.load(run_folder)
    outputs = [
        _load_parameter(
            output_name,
            run_info.input_paths,
            run_info.shapes,
            run_info.shape_masks,
            run_info.init_store(),
            run_folder,
        )
        for output_name in output_names
    ]
    outputs = [_maybe_load_file_array(o) for o in outputs]
    return outputs[0] if len(output_names) == 1 else outputs


def load_xarray_dataset(
    *output_name: str,
    run_folder: str | Path,
    load_intermediate: bool = True,
) -> xr.Dataset:
    """Load the output(s) of a `pipeline.map` as an `xarray.Dataset`.

    Parameters
    ----------
    output_name
        The names of the outputs to load. If empty, all outputs are loaded.
    run_folder
        The folder where the pipeline run was stored.
    load_intermediate
        Whether to load intermediate outputs as coordinates.

    Returns
    -------
        An `xarray.Dataset` containing the outputs of the pipeline run.

    """
    from pipefunc.map.xarray import load_xarray_dataset

    run_info = RunInfo.load(run_folder)
    return load_xarray_dataset(
        run_info.mapspecs,
        run_info.inputs,
        run_folder=run_folder,
        output_names=output_name,  # type: ignore[arg-type]
        load_intermediate=load_intermediate,
    )


def _validate_complete_inputs(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    output_name: _OUTPUT_TYPE | None = None,
) -> None:
    """Validate that all required inputs are provided.

    Note that `output_name is None` means that all outputs are required!
    This is in contrast to some other functions, where `None` means that the `pipeline.unique_leaf_node`
    is used.
    """
    if output_name is None:
        root_args = set(pipeline.topological_generations.root_args)
    else:  # pragma: no cover
        # TODO: this case becomes relevant when #127 is implemented
        root_args = set(pipeline.root_args(output_name))
    inputs_with_defaults = set(inputs) | set(pipeline.defaults)
    if missing := root_args - set(inputs_with_defaults):
        missing_args = ", ".join(missing)
        msg = f"Missing inputs: {missing_args}"
        raise ValueError(msg)
