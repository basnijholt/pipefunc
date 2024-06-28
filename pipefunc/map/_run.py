from __future__ import annotations

import functools
import itertools
import tempfile
from collections import OrderedDict, defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, Union

import numpy as np
import numpy.typing as npt

from pipefunc._utils import at_least_tuple, dump, handle_error, load, prod
from pipefunc.map._mapspec import (
    MapSpec,
    _shape_to_key,
    validate_consistent_axes,
)
from pipefunc.map._run_info import RunInfo, _external_shape, _internal_shape, _load_input
from pipefunc.map._storage_base import StorageBase, _iterate_shape_indices, _select_by_mask

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import xarray as xr

    from pipefunc import PipeFunc, Pipeline


_OUTPUT_TYPE: TypeAlias = Union[str, tuple[str, ...]]


def run(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: str | Path | None = None,
    internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
    *,
    output_names: set[_OUTPUT_TYPE] | None = None,
    parallel: bool = True,
    executor: Executor | None = None,
    storage: str = "file_array",
    persist_memory: bool = True,
    cleanup: bool = True,
    fixed_indices: dict[str, int | slice] | None = None,
    auto_subpipeline: bool = False,
) -> dict[str, Result]:
    """Run a pipeline with `MapSpec` functions for given ``inputs``.

    Parameters
    ----------
    pipeline
        The pipeline to run.
    inputs
        The inputs to the pipeline. The keys should be the names of the input
        parameters of the pipeline functions and the values should be the
        corresponding input data, these are either single values for functions without ``mapspec``
        or lists of values or `numpy.ndarray`s for functions with ``mapspec``.
    run_folder
        The folder to store the run information. If ``None``, a temporary folder
        is created.
    internal_shapes
        The shapes for intermediary outputs that cannot be inferred from the inputs.
        You will receive an exception if the shapes cannot be inferred and need to be provided.
    output_names
        The output(s) to calculate. If ``None``, the entire pipeline is run and all outputs are computed.
    parallel
        Whether to run the functions in parallel.
    executor
        The executor to use for parallel execution. If ``None``, a `ProcessPoolExecutor`
        is used. Only relevant if ``parallel=True``.
    storage
        The storage class to use for the file arrays. Can use any registered storage class.
    persist_memory
        Whether to write results to disk when memory based storage is used.
        Does not have any effect when file based storage is used.
        Can use any registered storage class. See `pipefunc.map.storage_registry`.
    cleanup
        Whether to clean up the ``run_folder`` before running the pipeline.
    fixed_indices
        A dictionary mapping axes names to indices that should be fixed for the run.
        If not provided, all indices are iterated over.
    auto_subpipeline
        If ``True``, a subpipeline is created with the specified ``inputs``, using
        `Pipeline.subpipeline`. This allows to provide intermediate results in the ``inputs`` instead
        of providing the root arguments. If ``False``, all root arguments must be provided,
        and an exception is raised if any are missing.

    """
    inputs = pipeline._flatten_scopes(inputs)
    if auto_subpipeline or output_names is not None:
        pipeline = pipeline.subpipeline(set(inputs), output_names)
    _validate_complete_inputs(pipeline, inputs)
    validate_consistent_axes(pipeline.mapspecs(ordered=False))
    _validate_fixed_indices(fixed_indices, inputs, pipeline)
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
            _run_and_process_generation(
                generation=gen,
                run_info=run_info,
                store=store,
                outputs=outputs,
                fixed_indices=fixed_indices,
                executor=ex,
            )

    if persist_memory:  # Only relevant for memory based storage
        for arr in store.values():
            arr.persist()

    return outputs


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any
    store: StorageBase | None
    run_folder: Path


def load_outputs(*output_names: str, run_folder: str | Path) -> Any:
    """Load the outputs of a run."""
    run_folder = Path(run_folder)
    run_info = RunInfo.load(run_folder)
    store = run_info.init_store()
    outputs = [_load_parameter(output_name, run_info, store) for output_name in output_names]
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


def _load_parameter(parameter: str, run_info: RunInfo, store: dict[str, StorageBase]) -> Any:
    if parameter in run_info.input_paths:
        return _load_input(parameter, run_info.input_paths)
    if parameter not in run_info.shapes or not any(run_info.shape_masks[parameter]):
        return _load_output(parameter, run_info.run_folder)
    return store[parameter]


def _func_kwargs(
    func: PipeFunc,
    run_info: RunInfo,
    store: dict[str, StorageBase],
) -> dict[str, Any]:
    kwargs = {}
    for p in func.parameters:
        if p in func._bound:
            kwargs[p] = func._bound[p]
        elif p in run_info.input_paths or p in run_info.all_output_names:
            kwargs[p] = _load_parameter(p, run_info, store)
        elif p in run_info.defaults and p not in run_info.all_output_names:
            kwargs[p] = run_info.defaults[p]
        else:  # pragma: no cover
            # In principle it should not be possible to reach this point because of
            # the checks in `run` and `_validate_complete_inputs`.
            msg = f"Parameter `{p}` not found in inputs, outputs, bound or defaults."
            raise ValueError(msg)
    return kwargs


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
        # handle_error raises but mypy doesn't know that
        raise  # pragma: no cover


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
    assert np.shape(output) == internal_shape
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


def _existing_and_missing_indices(
    file_arrays: list[StorageBase],
    fixed_mask: np.flatiter[npt.NDArray[np.bool_]] | None,
) -> tuple[list[int], list[int]]:
    # TODO: when `fixed_indices` are used we could be more efficient by not
    # computing the full mask.
    masks = (arr.mask_linear() for arr in file_arrays)
    if fixed_mask is None:
        fixed_mask = itertools.repeat(object=True)  # type: ignore[assignment]

    existing_indices = []
    missing_indices = []
    for i, (*mask_values, select) in enumerate(zip(*masks, fixed_mask)):  # type: ignore[arg-type]
        if not select:
            continue
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
    fixed_indices: dict[str, int | slice] | None,
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
    fixed_mask = _mask_fixed_axes(fixed_indices, func.mapspec, shape, mask)
    existing, missing = _existing_and_missing_indices(file_arrays, fixed_mask)  # type: ignore[arg-type]
    return _MapSpecArgs(process_index, existing, missing, result_arrays, shape, mask, file_arrays)


def _mask_fixed_axes(
    fixed_indices: dict[str, int | slice] | None,
    mapspec: MapSpec,
    shape: tuple[int, ...],
    shape_mask: tuple[bool, ...],
) -> np.flatiter[npt.NDArray[np.bool_]] | None:
    if fixed_indices is None:
        return None
    key = tuple(fixed_indices.get(axis, slice(None)) for axis in mapspec.output_indices)
    external_key = _external_shape(key, shape_mask)  # type: ignore[arg-type]
    external_shape = _external_shape(shape, shape_mask)
    select: npt.NDArray[np.bool_] = np.zeros(external_shape, dtype=bool)
    select[external_key] = True
    return select.flat


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
        # handle_error raises but mypy doesn't know that
        raise  # pragma: no cover


def _maybe_load_file_array(x: Any) -> Any:
    if isinstance(x, StorageBase):
        return x.to_array()
    return x


def _load_file_array(kwargs: dict[str, Any]) -> None:
    for k, v in kwargs.items():
        kwargs[k] = _maybe_load_file_array(v)


def _ensure_run_folder(run_folder: str | Path | None) -> Path:
    if run_folder is None:
        tmp_dir = tempfile.mkdtemp()
        run_folder = Path(tmp_dir)
    return Path(run_folder)


class _KwargsTask(NamedTuple):
    kwargs: dict[str, Any]
    task: tuple[Any, _MapSpecArgs] | Any


def _run_and_process_generation(
    generation: list[PipeFunc],
    run_info: RunInfo,
    store: dict[str, StorageBase],
    outputs: dict[str, Result],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | None,
) -> None:
    tasks: dict[PipeFunc, _KwargsTask] = {}
    for func in generation:
        tasks[func] = _submit_func(func, run_info, store, fixed_indices, executor)
    for func in generation:
        _outputs = _process_task(func, tasks[func], run_info.run_folder, store, executor)
        outputs.update(_outputs)


def _submit_func(
    func: PipeFunc,
    run_info: RunInfo,
    store: dict[str, StorageBase],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | None,
) -> _KwargsTask:
    kwargs = _func_kwargs(func, run_info, store)
    return _submit_task(func, kwargs, run_info, store, fixed_indices, executor)


def _submit_task(
    func: PipeFunc,
    kwargs: dict[str, Any],
    run_info: RunInfo,
    store: dict[str, StorageBase],
    fixed_indices: dict[str, int | slice] | None,
    executor: Executor | None,
) -> _KwargsTask:
    if func.mapspec and func.mapspec.inputs:
        args = _prepare_submit_map_spec(
            func,
            kwargs,
            run_info.shapes,
            run_info.shape_masks,
            store,
            fixed_indices,
        )
        r = _maybe_parallel_map(args.process_index, args.missing, executor)
        task = r, args
    else:
        task = _maybe_submit(_submit_single, executor, func, kwargs, run_info.run_folder)
    return _KwargsTask(kwargs, task)


def _process_task(
    func: PipeFunc,
    kwargs_task: _KwargsTask,
    run_folder: Path,
    store: dict[str, StorageBase],
    executor: Executor | None = None,
) -> dict[str, Result]:
    kwargs, task = kwargs_task
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
        r = task.result() if executor else task  # type: ignore[union-attr]
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
            run_folder=run_folder,
        )
        for output_name, _output in zip(at_least_tuple(func.output_name), output)
    }


def _check_parallel(parallel: bool, store: dict[str, StorageBase]) -> None:  # noqa: FBT001
    if not parallel or not store:
        return
    # Assumes all storage classes are the same! Might change in the future.
    storage = next(iter(store.values()))
    if not storage.parallelizable:
        msg = (
            f"Parallel execution is not supported with `{storage.storage_id}` storage."
            " Use a file based storage or `shared_memory` / `zarr_shared_memory`."
        )
        raise ValueError(msg)


def _validate_complete_inputs(pipeline: Pipeline, inputs: dict[str, Any]) -> None:
    """Validate that all required inputs are provided.

    Note that `output_name is None` means that all outputs are required!
    This is in contrast to some other functions, where ``None`` means that the
    `pipeline.unique_leaf_node` is used.
    """
    root_args = set(pipeline.topological_generations.root_args)
    inputs_with_defaults = set(inputs) | set(pipeline.defaults)
    if missing := root_args - set(inputs_with_defaults):
        missing_args = ", ".join(missing)
        msg = f"Missing inputs: `{missing_args}`."
        raise ValueError(msg)
    if extra := set(inputs_with_defaults) - root_args:
        extra_args = ", ".join(extra)
        msg = f"Got extra inputs: `{extra_args}` that are not accepted by this pipeline."
        raise ValueError(msg)


def _validate_fixed_indices(
    fixed_indices: dict[str, int | slice] | None,
    inputs: dict[str, Any],
    pipeline: Pipeline,
) -> None:
    if fixed_indices is None:
        return
    extra = set(fixed_indices)
    axes = pipeline.mapspec_axes
    for parameter, axes_ in axes.items():
        for axis in axes_:
            if axis in fixed_indices:
                extra.discard(axis)
        if parameter in inputs:
            key = tuple(fixed_indices.get(axis, slice(None)) for axis in axes_)
            if len(key) == 1:
                key = key[0]  # type: ignore[assignment]
            try:
                inputs[parameter][key]
            except IndexError as e:
                msg = f"Fixed index `{key}` for parameter `{parameter}` is out of bounds."
                raise IndexError(msg) from e
    if extra:
        msg = f"Got extra `fixed_indices`: `{extra}` that are not accepted by this map."
        raise ValueError(msg)

    reduced_axes = _reduced_axes(pipeline)
    for name, axes_set in reduced_axes.items():
        if reduced := set(axes_set) & set(fixed_indices):
            reduced_str = ", ".join(reduced)
            msg = f"Axis `{reduced_str}` in `{name}` is reduced and cannot be in `fixed_indices`."
            raise ValueError(msg)


def _reduced_axes(pipeline: Pipeline) -> dict[str, set[str]]:
    # TODO: check the overlap between this an `independent_axes_in_mapspecs`.
    # It might be that this function could be used instead.
    reduced_axes: dict[str, set[str]] = defaultdict(set)
    axes = pipeline.mapspec_axes
    for name in pipeline.mapspec_names:
        for func in pipeline.functions:
            if _is_parameter_reduced_by_function(func, name):
                reduced_axes[name].update(axes[name])
            elif _is_parameter_partially_reduced_by_function(func, name):
                _axes = _get_partially_reduced_axes(func, name, axes)
                reduced_axes[name].update(_axes)
    return dict(reduced_axes)


def _is_parameter_reduced_by_function(func: PipeFunc, name: str) -> bool:
    return name in func.parameters and (
        func.mapspec is None or name not in func.mapspec.input_names
    )


def _is_parameter_partially_reduced_by_function(func: PipeFunc, name: str) -> bool:
    if func.mapspec is None or name not in func.mapspec.input_names:
        return False
    spec = next(spec for spec in func.mapspec.inputs if spec.name == name)
    return None in spec.axes


def _get_partially_reduced_axes(
    func: PipeFunc,
    name: str,
    axes: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    assert func.mapspec is not None
    spec = next(spec for spec in func.mapspec.inputs if spec.name == name)
    return tuple(ax for ax, spec_ax in zip(axes[name], spec.axes) if spec_ax is None)


class TestResult(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    expected_output: Any
    actual_output: Any


def compare(
    pipeline: Pipeline,
    run_folder: str | Path,
    output_names: set[str] | None = None,
) -> dict[str, TestResult]:
    """Rerun the pipeline and compare the outputs to the stored outputs.

    Parameters
    ----------
    pipeline
        The pipeline that was run.
    run_folder
        The folder where the pipeline run was stored.
    output_names
        The output(s) to test. If ``None``, all outputs are tested.

    """
    assert output_names is None or all(isinstance(o, str) for o in output_names)
    run_info = RunInfo.load(run_folder)
    store = run_info.init_store()
    results = {}
    new_run_folder = Path(tempfile.mkdtemp())
    new_run_info = run_info.copy(run_folder=new_run_folder)
    for func in pipeline.functions:
        if output_names is not None and set(at_least_tuple(func.output_name)) not in output_names:
            continue
        kwargs = _func_kwargs(func, run_info, store)
        task = _submit_task(func, kwargs, new_run_info, store, None, None)
        output = _process_task(func, task, new_run_folder, store, None)
        for output_name, result in output.items():
            expected_output = load_outputs(output_name, run_folder=run_info.run_folder)
            actual_output = result.output
            results[output_name] = TestResult(
                function=result.function,
                kwargs=result.kwargs,
                expected_output=expected_output,
                actual_output=actual_output,
            )
    return results
