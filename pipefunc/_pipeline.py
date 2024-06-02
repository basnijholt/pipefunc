"""PipeFunc: A Python library for defining, managing, and executing function pipelines.

This module provides an implementation of a Pipeline class, which allows you
to easily define, manage, and execute a sequence of functions, where each
function may depend on the outputs of other functions. The pipeline is
represented as a directed graph, where nodes represent functions and edges
represent dependencies between functions. The Pipeline class provides methods
for adding functions to the pipeline, executing the pipeline for specific
output values, visualizing the pipeline as a directed graph, and profiling
the resource usage of the pipeline functions.
"""

from __future__ import annotations

import functools
import inspect
import time
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias, Union

import networkx as nx

from pipefunc._cache import DiskCache, HybridCache, LRUCache, SimpleCache
from pipefunc._perf import resources_report
from pipefunc._pipefunc import PipeFunc
from pipefunc._plotting import visualize, visualize_holoviews
from pipefunc._simplify import _func_node_colors, _identify_combinable_nodes, simplified_pipeline
from pipefunc._utils import (
    at_least_tuple,
    clear_cached_properties,
    generate_filename_from_dict,
    handle_error,
)
from pipefunc.exceptions import UnusedParametersError
from pipefunc.lazy import _LazyFunction, task_graph
from pipefunc.map._mapspec import (
    ArraySpec,
    MapSpec,
    mapspec_axes,
    mapspec_dimensions,
    validate_consistent_axes,
)
from pipefunc.map._run import run

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from concurrent.futures import Executor
    from pathlib import Path

    import holoviews as hv

    from pipefunc._perf import ProfilingStats
    from pipefunc.map._run import Result


_OUTPUT_TYPE: TypeAlias = Union[str, tuple[str, ...]]
_CACHE_KEY_TYPE: TypeAlias = tuple[_OUTPUT_TYPE, tuple[tuple[str, Any], ...]]


class _Function:
    """Wrapper class for a pipeline function.

    Parameters
    ----------
    pipeline
        The pipeline to which the function belongs.
    output_name
        The identifier for the return value of the pipeline function.
    root_args
        The names of the pipeline function's root inputs.

    """

    __slots__ = ["pipeline", "output_name", "root_args", "_call_with_root_args"]

    def __init__(
        self,
        pipeline: Pipeline,
        output_name: _OUTPUT_TYPE,
        root_args: tuple[str, ...],
    ) -> None:
        """Initialize the function wrapper."""
        self.pipeline = pipeline
        self.output_name = output_name
        self.root_args = root_args
        self._call_with_root_args: Callable[..., Any] | None = None

    @property
    def call_with_root_args(self) -> Callable[..., Any]:
        if self._call_with_root_args is None:
            self._call_with_root_args = self._create_call_with_root_args_method()
        return self._call_with_root_args

    def __call__(self, **kwargs: Any) -> Any:
        """Call the pipeline function with the given arguments.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
            The return value of the pipeline function.

        """
        return self.pipeline.run(output_name=self.output_name, kwargs=kwargs)

    def call_full_output(self, **kwargs: Any) -> dict[str, Any]:
        """Call the pipeline function with the given arguments and return all outputs.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
            The return value of the pipeline function.

        """
        return self.pipeline.run(self.output_name, full_output=True, kwargs=kwargs)

    def call_with_dict(self, kwargs: dict[str, Any]) -> Any:
        """Call the pipeline function with the given arguments.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
            The return value of the pipeline function.

        """
        return self(**kwargs)

    def __getstate__(self) -> dict:
        """Prepare the state of the current object for pickling."""
        state = {slot: getattr(self, slot) for slot in self.__slots__}
        state["_call_with_root_args"] = None  # don't pickle the execute method
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the current object from the provided state."""
        for slot in self.__slots__:
            setattr(self, slot, state[slot])
        # Initialize _call_with_root_args if necessary
        self._call_with_root_args = None

    def _create_call_with_parameters_method(
        self,
        parameters: tuple[str, ...],
    ) -> Callable[..., Any]:
        sig = inspect.signature(self.__call__)
        new_params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in parameters
        ]
        new_sig = sig.replace(parameters=new_params)

        def call(*args: Any, **kwargs: Any) -> Any:
            """Call the pipeline function with the root arguments."""
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return self(**bound.arguments)

        call.__signature__ = new_sig  # type: ignore[attr-defined]
        return call

    def _create_call_with_root_args_method(self) -> Callable[..., Any]:
        return self._create_call_with_parameters_method(self.root_args)


class Pipeline:
    """Pipeline class for managing and executing a sequence of functions.

    Parameters
    ----------
    functions
        A list of functions that form the pipeline.
    lazy
        Flag indicating whether the pipeline should be lazy.
    debug
        Flag indicating whether debug information should be printed.
        If None, the value of each PipeFunc's debug attribute is used.
    profile
        Flag indicating whether profiling information should be collected.
        If None, the value of each PipeFunc's profile attribute is used.
    cache_type
        The type of cache to use.
    cache_kwargs
        Keyword arguments passed to

    """

    def __init__(
        self,
        functions: list[PipeFunc | tuple[PipeFunc, str | MapSpec]],
        *,
        lazy: bool = False,
        debug: bool | None = None,
        profile: bool | None = None,
        cache_type: Literal["lru", "hybrid", "disk", "simple"] | None = None,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Pipeline class for managing and executing a sequence of functions."""
        self.functions: list[PipeFunc] = []
        self.lazy = lazy
        self._debug = debug
        self._profile = profile
        for f in functions:
            if isinstance(f, tuple):
                f, mapspec = f  # noqa: PLW2901
            else:
                mapspec = None
            self.add(f, mapspec=mapspec)
        self._init_internal_cache()
        self._cache_type = cache_type
        self._cache_kwargs = cache_kwargs
        if cache_type is None and any(f.cache for f in self.functions):
            cache_type = "lru"
        self.cache = _create_cache(cache_type, lazy, cache_kwargs)
        self._validate_mapspec()

    def _init_internal_cache(self) -> None:
        # Internal Pipeline cache
        self._arg_combinations: dict[_OUTPUT_TYPE, set[tuple[str, ...]]] = {}
        self._root_args: dict[_OUTPUT_TYPE, tuple[str, ...]] = {}
        self._func: dict[_OUTPUT_TYPE, _Function] = {}
        clear_cached_properties(self)

    def _validate_mapspec(self) -> None:
        """Validate the MapSpecs for all functions in the pipeline."""
        for f in self.functions:
            if f.mapspec and at_least_tuple(f.output_name) != f.mapspec.output_names:
                msg = (
                    f"The output_name of the function `{f}` does not match the output_names"
                    f" in the MapSpec: `{f.output_name}` != `{f.mapspec.output_names}`."
                )
                raise ValueError(msg)
        validate_consistent_axes(self.mapspecs(ordered=False))
        self._autogen_mapspec_axes()

    def _current_cache(self) -> LRUCache | HybridCache | DiskCache | SimpleCache | None:
        """Return the cache used by the pipeline."""
        if not isinstance(self.cache, SimpleCache) and (tg := task_graph()) is not None:
            return tg.cache
        return self.cache

    @property
    def profile(self) -> bool | None:
        """Flag indicating whether profiling information should be collected."""
        return self._profile

    @profile.setter
    def profile(self, value: bool | None) -> None:
        """Set the profiling flag for the pipeline and all functions."""
        self._profile = value
        if value is not None:
            for f in self.functions:
                f.profile = value

    @property
    def debug(self) -> bool | None:
        """Flag indicating whether debug information should be printed."""
        return self._debug

    @debug.setter
    def debug(self, value: bool | None) -> None:
        """Set the debug flag for the pipeline and all functions."""
        self._debug = value
        if value is not None:
            for f in self.functions:
                f.debug = value

    def add(self, f: PipeFunc | Callable, mapspec: str | MapSpec | None = None) -> PipeFunc:
        """Add a function to the pipeline.

        Parameters
        ----------
        f
            The function to add to the pipeline.
        profile
            Flag indicating whether profiling information should be collected.
        mapspec
            This is a specification for mapping that dictates how input values should
            be merged together. If None, the default behavior is that the input directly
            maps to the output.

        """
        if not isinstance(f, PipeFunc):
            f = PipeFunc(f, output_name=f.__name__, mapspec=mapspec)
        elif mapspec is not None:
            msg = (
                "Initializing the `Pipeline` using tuples of `PipeFunc`s and `MapSpec`s"
                " will create a copy of the `PipeFunc`."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            f: PipeFunc = f.copy()  # type: ignore[no-redef]
            if isinstance(mapspec, str):
                mapspec = MapSpec.from_string(mapspec)
            f.mapspec = mapspec
            f._validate_mapspec()

        self.functions.append(f)

        if self.profile is not None:
            f.set_profiling(enable=self.profile)

        if self.debug is not None:
            f.debug = self.debug

        self._init_internal_cache()  # reset cache
        return f

    def drop(self, *, f: PipeFunc | None = None, output_name: _OUTPUT_TYPE | None = None) -> None:
        """Drop a function from the pipeline.

        Parameters
        ----------
        f
            The function to drop from the pipeline.
        output_name
            The name of the output to drop from the pipeline.

        """
        if (f is not None and output_name is not None) or (f is None and output_name is None):
            msg = "One of `f` or `output_name` should be provided."
            raise ValueError(msg)
        if f is not None:
            self.functions.remove(f)
        elif output_name is not None:
            f = self.output_to_func[output_name]
            self.drop(f=f)
        self._init_internal_cache()

    def replace(self, new: PipeFunc, old: PipeFunc | None = None) -> None:
        """Replace a function in the pipeline with another function.

        Parameters
        ----------
        new
            The function to add to the pipeline.
        old
            The function to replace in the pipeline. If None, `old` is
            assumed to be the function with the same output name as `new`.

        """
        if old is None:
            self.drop(output_name=new.output_name)
        else:
            self.drop(f=old)
        self.add(new)
        self._init_internal_cache()

    @functools.cached_property
    def output_to_func(self) -> dict[_OUTPUT_TYPE, PipeFunc]:
        output_to_func: dict[_OUTPUT_TYPE, PipeFunc] = {}
        for f in self.functions:
            output_to_func[f.output_name] = f
            if isinstance(f.output_name, tuple):
                for name in f.output_name:
                    output_to_func[name] = f
        return output_to_func

    @functools.cached_property
    def node_mapping(self) -> dict[_OUTPUT_TYPE, PipeFunc | str]:
        """Return a mapping from node names to nodes.

        Returns
        -------
            A mapping from node names to nodes.

        """
        mapping: dict[_OUTPUT_TYPE, PipeFunc | str] = {}
        for node in self.graph.nodes:
            if isinstance(node, PipeFunc):
                if isinstance(node.output_name, tuple):
                    for name in node.output_name:
                        mapping[name] = node
                mapping[node.output_name] = node
            else:
                assert isinstance(node, str)
                mapping[node] = node
        return mapping

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Create a directed graph representing the pipeline.

        Returns
        -------
            A directed graph with nodes representing functions and edges
            representing dependencies between functions.

        """
        _check_consistent_defaults(self.functions)
        _check_consistent_type_annotations(self.functions)
        g = nx.DiGraph()
        for f in self.functions:
            g.add_node(f)
            assert f.parameters is not None
            for arg in f.parameters:
                if arg in self.output_to_func:  # is function output
                    edge = (self.output_to_func[arg], f)
                    if edge not in g.edges:
                        g.add_edge(*edge, arg=arg)
                    else:
                        # tuple output of function, and the edge already exists
                        assert isinstance(edge[0].output_name, tuple)
                        current = g.edges[edge]["arg"]
                        g.edges[edge]["arg"] = (*at_least_tuple(current), arg)

                else:
                    if arg not in g:  # Add the node only if it doesn't exist
                        default_value = f.defaults.get(arg, inspect.Parameter.empty)
                        g.add_node(arg, default_value=default_value)
                    g.add_edge(arg, f, arg=arg)
        return g

    def func(self, output_name: _OUTPUT_TYPE) -> _Function:
        """Create a composed function that can be called with keyword arguments.

        Parameters
        ----------
        output_name
            The identifier for the return value of the composed function.

        Returns
        -------
            The composed function that can be called with keyword arguments.

        """
        if f := self._func.get(output_name):
            return f
        root_args = self.root_args(output_name)
        assert isinstance(root_args, tuple)
        f = _Function(self, output_name, root_args=root_args)
        self._func[output_name] = f
        return f

    def __call__(self, __output_name__: _OUTPUT_TYPE | None = None, /, **kwargs: Any) -> Any:
        """Call the pipeline for a specific return value.

        Parameters
        ----------
        __output_name__
            The identifier for the return value of the pipeline.
            Is None by default, in which case the unique leaf node is used.
            This parameter is positional-only and the strange name is used
            to avoid conflicts with the `output_name` argument that might be
            passed via `kwargs`.
        kwargs
            Keyword arguments to be passed to the pipeline functions.

        Returns
        -------
            The return value of the pipeline.

        """
        if __output_name__ is None:
            __output_name__ = self.unique_leaf_node.output_name
        return self.func(__output_name__)(**kwargs)

    def _get_func_args(
        self,
        func: PipeFunc,
        kwargs: dict[str, Any],
        all_results: dict[_OUTPUT_TYPE, Any],
        full_output: bool,  # noqa: FBT001
        used_parameters: set[str | None],
    ) -> dict[str, Any]:
        # Used in _run
        func_args = {}
        for arg in func.parameters:
            if arg in kwargs:
                value = kwargs[arg]
            elif arg in func.defaults:
                value = func.defaults[arg]
            else:
                value = self._run(
                    output_name=arg,
                    kwargs=kwargs,
                    all_results=all_results,
                    full_output=full_output,
                    used_parameters=used_parameters,
                )
            func_args[arg] = value
            used_parameters.add(arg)
        return func_args

    def _run(
        self,
        *,
        output_name: _OUTPUT_TYPE,
        kwargs: Any,
        all_results: dict[_OUTPUT_TYPE, Any],
        full_output: bool,
        used_parameters: set[str | None],
    ) -> Any:
        if output_name in all_results:
            return all_results[output_name]
        func = self.output_to_func[output_name]
        assert func.parameters is not None

        cache = self._current_cache()
        use_cache = (func.cache and cache is not None) or task_graph() is not None

        root_args = self.root_args(output_name)
        result_from_cache = False
        if use_cache:
            assert cache is not None
            cache_key = _compute_cache_key(func.output_name, kwargs, root_args)
            return_now, result_from_cache = _get_result_from_cache(
                func,
                cache,
                cache_key,
                output_name,
                all_results,
                full_output,
                used_parameters,
                self.lazy,
            )
            if return_now:
                return all_results[output_name]

        func_args = self._get_func_args(
            func,
            kwargs,
            all_results,
            full_output,
            used_parameters,
        )

        if result_from_cache:
            assert full_output
            return all_results[output_name]

        start_time = time.perf_counter()
        r = _execute_func(func, func_args, self.lazy)
        if use_cache and cache_key is not None:
            assert cache is not None
            _update_cache(cache, cache_key, r, start_time)
        _update_all_results(func, r, output_name, all_results, self.lazy)
        _save_results(func, r, output_name, all_results, root_args, self.lazy)
        return all_results[output_name]

    def run(
        self,
        output_name: _OUTPUT_TYPE,
        *,
        full_output: bool = False,
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute the pipeline for a specific return value.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline.
        full_output
            Whether to return the outputs of all function executions
            as a dictionary mapping function names to their return values.
        kwargs
            Keyword arguments to be passed to the pipeline functions.

        Returns
        -------
            The return value of the pipeline or a dictionary mapping function
            names to their return values if `full_output` is True.

        """
        if p := self.mapspec_names & set(self.func_dependencies(output_name)):
            inputs = self.mapspec_names & set(self.root_args(output_name))
            msg = (
                f"Cannot execute pipeline to get `{output_name}` because `{p}`"
                f" (depends on `{inputs=}`) have `MapSpec`(s). Use `Pipeline.map` instead."
            )
            raise RuntimeError(msg)

        if output_name in kwargs:
            msg = f"The `output_name='{output_name}'` argument cannot be provided in `kwargs={kwargs}`."
            raise ValueError(msg)

        all_results: dict[_OUTPUT_TYPE, Any] = kwargs.copy()  # type: ignore[assignment]
        used_parameters: set[str | None] = set()

        self._run(
            output_name=output_name,
            kwargs=kwargs,
            all_results=all_results,
            full_output=full_output,
            used_parameters=used_parameters,
        )

        # if has None, result was from cache, so we don't know which parameters were used
        if None not in used_parameters and (unused := kwargs.keys() - set(used_parameters)):
            unused_str = ", ".join(sorted(unused))
            msg = f"Unused keyword arguments: `{unused_str}`. {kwargs=}, {used_parameters=}"
            raise UnusedParametersError(msg)

        return all_results if full_output else all_results[output_name]

    def map(
        self,
        inputs: dict[str, Any],
        run_folder: str | Path | None,
        internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
        *,
        parallel: bool = True,
        executor: Executor | None = None,
        storage: str = "file_array",
        persist_memory: bool = True,
        cleanup: bool = True,
        fixed_indices: dict[str, int | slice] | None = None,
    ) -> dict[str, Result]:
        """Run a pipeline with `MapSpec` functions for given `inputs`.

        Parameters
        ----------
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
            Can use any registered storage class. See `pipefunc.map.storage_registry`.
        persist_memory
            Whether to write results to disk when memory based storage is used.
            Does not have any effect when file based storage is used.
        cleanup
            Whether to clean up the `run_folder` before running the pipeline.
        fixed_indices
            A dictionary mapping axes names to indices that should be fixed for the run.
            If not provided, all indices are iterated over.

        """
        return run(
            self,
            inputs,
            run_folder,
            internal_shapes,
            parallel=parallel,
            executor=executor,
            storage=storage,
            persist_memory=persist_memory,
            cleanup=cleanup,
            fixed_indices=fixed_indices,
        )

    def arg_combinations(self, output_name: _OUTPUT_TYPE) -> set[tuple[str, ...]]:
        """Return the arguments required to compute a specific output.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline.

        Returns
        -------
            A set of tuples containing possible argument combinations.
            The tuples are sorted in lexicographical order.

        """
        if r := self._arg_combinations.get(output_name):
            return r
        head = self.node_mapping[output_name]
        arg_set: set[tuple[str, ...]] = set()
        _compute_arg_mapping(self.graph, head, head, [], [], arg_set)  # type: ignore[arg-type]
        self._arg_combinations[output_name] = arg_set
        return arg_set

    def root_args(self, output_name: _OUTPUT_TYPE) -> tuple[str, ...]:
        """Return the root arguments required to compute a specific output."""
        if r := self._root_args.get(output_name):
            return r
        arg_combos = self.arg_combinations(output_name)
        root_args = next(
            args for args in arg_combos if all(isinstance(self.node_mapping[n], str) for n in args)
        )
        self._root_args[output_name] = root_args
        return root_args

    def func_dependencies(self, output_name: _OUTPUT_TYPE | PipeFunc) -> list[_OUTPUT_TYPE]:
        """Return the functions required to compute a specific output.

        See Also
        --------
        func_predecessors

        """
        return _traverse_graph(output_name, "predecessors", self.graph, self.node_mapping)

    def func_dependents(self, name: _OUTPUT_TYPE | PipeFunc) -> list[_OUTPUT_TYPE]:
        """Return the functions that depend on a specific input/output.

        See Also
        --------
        func_successors

        """
        return _traverse_graph(name, "successors", self.graph, self.node_mapping)

    @functools.cached_property
    def all_arg_combinations(self) -> dict[_OUTPUT_TYPE, set[tuple[str, ...]]]:
        """Compute all possible argument mappings for the pipeline.

        Returns
        -------
            A dictionary mapping function names to sets of tuples containing
            possible argument combinations.

        """
        return {
            node.output_name: self.arg_combinations(node.output_name)
            for node in self.graph.nodes
            if isinstance(node, PipeFunc)
        }

    @functools.cached_property
    def all_root_args(self) -> dict[_OUTPUT_TYPE, tuple[str, ...]]:
        """Return the root arguments required to compute all outputs."""
        return {
            node.output_name: self.root_args(node.output_name)
            for node in self.graph.nodes
            if isinstance(node, PipeFunc)
        }

    @functools.cached_property
    def mapspec_names(self) -> set[str]:
        return {
            name
            for mapspec in self.mapspecs()
            for name in mapspec.input_names + mapspec.output_names
        }

    @functools.cached_property
    def defaults(self) -> dict[str, Any]:
        defaults = {}
        for func in self.functions:
            defaults.update(func.defaults)
        return defaults

    def mapspecs(self, *, ordered: bool = True) -> list[MapSpec]:
        """Return the MapSpecs for all functions in the pipeline."""
        functions = self.sorted_functions if ordered else self.functions
        return [f.mapspec for f in functions if f.mapspec]

    @functools.cached_property
    def mapspecs_as_strings(self) -> list[str]:
        """Return the MapSpecs for all functions in the pipeline as strings."""
        return [str(mapspec) for mapspec in self.mapspecs(ordered=True)]

    @functools.cached_property
    def mapspec_dimensions(self: Pipeline) -> dict[str, int]:
        """Return the number of dimensions for each array parameter in the pipeline."""
        return mapspec_dimensions(self.mapspecs())

    @functools.cached_property
    def mapspec_axes(self: Pipeline) -> dict[str, tuple[str, ...]]:
        """Return the axes for each array parameter in the pipeline."""
        return mapspec_axes(self.mapspecs())

    @functools.cached_property
    def unique_leaf_node(self) -> PipeFunc:
        """Return the unique leaf node of the pipeline graph."""
        leaf_nodes = self.leaf_nodes
        if len(leaf_nodes) != 1:  # pragma: no cover
            msg = (
                "The pipeline has multiple leaf nodes. Please specify the output_name"
                " argument to disambiguate.",
            )
            raise ValueError(msg)
        return leaf_nodes[0]

    @functools.cached_property
    def topological_generations(self) -> _Generations:
        """Return the functions in the pipeline grouped by topological generation.

        Simply calls `nx.topological_generations` on the `pipeline.graph`. Then
        groups the functions in the pipeline by generation. The first generation
        contains the root arguments, while the subsequent generations contain
        the functions in topological order.
        """
        generations = list(nx.topological_generations(self.graph))
        assert all(isinstance(x, str) for x in generations[0])
        assert all(isinstance(x, PipeFunc) for gen in generations[1:] for x in gen)
        return _Generations(generations[0], generations[1:])

    @functools.cached_property
    def sorted_functions(self) -> list[PipeFunc]:
        """Return the functions in the pipeline in topological order."""
        return [f for gen in self.topological_generations.function_lists for f in gen]

    def _autogen_mapspec_axes(self) -> set[PipeFunc]:
        """Generate `MapSpec`s for functions that return arrays with `internal_shapes`."""
        root_args = self.topological_generations.root_args
        mapspecs = self.mapspecs(ordered=False)
        non_root_inputs = _find_non_root_axes(mapspecs, root_args)
        output_names = {at_least_tuple(f.output_name) for f in self.functions}
        multi_output_mapping = {n: names for names in output_names for n in names if len(names) > 1}
        _replace_none_in_axes(mapspecs, non_root_inputs, multi_output_mapping)  # type: ignore[arg-type]
        return _create_missing_mapspecs(self.functions, non_root_inputs)  # type: ignore[arg-type]

    def add_mapspec_axis(self, *parameter: str, axis: str) -> None:
        """Add a new axis to `parameter`'s MapSpec.

        Parameters
        ----------
        parameter
            The parameter to add an axis to.
        axis
            The axis to add to the `MapSpec` of all functions that depends on
            `parameter`. Provide a new axis name to add a new axis or an
            existing axis name to zip the parameter with the existing axis.

        """
        self._autogen_mapspec_axes()
        for p in parameter:
            _add_mapspec_axis(p, dims={}, axis=axis, functions=self.sorted_functions)
        self._init_internal_cache()  # reset cache because mapspecs have changed

    def _func_node_colors(
        self,
        *,
        conservatively_combine: bool = False,
        output_name: _OUTPUT_TYPE | None = None,
    ) -> list[str]:
        if output_name is None:
            output_name = self.unique_leaf_node.output_name
        combinable_nodes = _identify_combinable_nodes(
            self.output_to_func[output_name],
            self.graph,
            self.all_root_args,
            conservatively_combine=conservatively_combine,
        )
        return _func_node_colors(self.functions, combinable_nodes)

    def visualize(
        self,
        figsize: tuple[int, int] = (10, 10),
        filename: str | Path | None = None,
        *,
        color_combinable: bool = False,
        conservatively_combine: bool = False,
        output_name: _OUTPUT_TYPE | None = None,
    ) -> None:
        """Visualize the pipeline as a directed graph.

        Parameters
        ----------
        figsize
            The width and height of the figure in inches.
        filename
            The filename to save the figure to.
        color_combinable
            Whether to color combinable nodes differently.
        conservatively_combine
            Argument as passed to `Pipeline.simply_pipeline`.
        output_name
            Argument as passed to `Pipeline.simply_pipeline`.

        """
        if color_combinable:
            func_node_colors = self._func_node_colors(
                conservatively_combine=conservatively_combine,
                output_name=output_name,
            )
        else:
            func_node_colors = None
        visualize(
            self.graph,
            figsize=figsize,
            filename=filename,
            func_node_colors=func_node_colors,
        )

    def visualize_holoviews(self) -> hv.Graph:
        """Visualize the pipeline as a directed graph using HoloViews."""
        return visualize_holoviews(self.graph)

    def resources_report(self) -> None:
        """Display the resource usage report for each function in the pipeline."""
        if not self.profiling_stats:
            msg = "Profiling is not enabled."
            raise ValueError(msg)
        resources_report(self.profiling_stats)

    def simplified_pipeline(
        self,
        output_name: _OUTPUT_TYPE | None = None,
        *,
        conservatively_combine: bool = False,
    ) -> Pipeline:
        """Simplify pipeline with combined function nodes.

        Generate a simplified version of the pipeline where combinable function
        nodes have been merged into single function nodes.

        This method identifies combinable nodes in the pipeline's execution
        graph (i.e., functions that share the same root arguments) and merges
        them into single function nodes. This results in a simplified pipeline
        where each key function only depends on nodes that cannot be further
        combined.

        Parameters
        ----------
        output_name
            The name of the output from the pipeline function we are starting
            the simplification from. If None, the unique tip of the pipeline
            graph is used (if there is one).
        conservatively_combine
            If True, only combine a function node with its predecessors if all
            of its predecessors have the same root arguments as the function
            node itself. If False, combine a function node with its predecessors
            if any of its predecessors have the same root arguments as the
            function node.

        Returns
        -------
            The simplified version of the pipeline.

        """
        if output_name is None:
            output_name = self.unique_leaf_node.output_name
        return simplified_pipeline(
            functions=self.functions,
            graph=self.graph,
            all_root_args=self.all_root_args,
            node_mapping=self.node_mapping,
            func_dependencies=self.func_dependencies(output_name),
            output_name=output_name,
            conservatively_combine=conservatively_combine,
        )

    @functools.cached_property
    def leaf_nodes(self) -> list[PipeFunc]:
        """Return the leaf nodes in the pipeline's execution graph."""
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

    @functools.cached_property
    def root_nodes(self) -> list[PipeFunc]:
        """Return the root nodes in the pipeline's execution graph."""
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

    @property
    def profiling_stats(self) -> dict[str, ProfilingStats]:
        """Return the profiling data for each function in the pipeline."""
        return {f.__name__: f.profiling_stats for f in self.functions if f.profiling_stats}

    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        pipeline_str = "Pipeline:\n"
        for node in self.graph.nodes:
            if isinstance(node, PipeFunc):
                fn = node
                input_args = self.all_arg_combinations[fn.output_name]
                pipeline_str += f"  {fn.output_name} = {fn.__name__}({', '.join(fn.parameters)})\n"
                pipeline_str += f"    Possible input arguments: {input_args}\n"
        return pipeline_str

    def copy(self) -> Pipeline:
        """Return a copy of the pipeline."""
        return Pipeline(
            [f.copy() for f in self.functions],  # type: ignore[arg-type]
            lazy=self.lazy,
            debug=self._debug,
            profile=self._profile,
            cache_type=self._cache_type,
            cache_kwargs=self._cache_kwargs,
        )

    def _connected_components(self) -> list[set[PipeFunc | str]]:
        """Return the connected components of the pipeline graph."""
        return list(nx.connected_components(self.graph.to_undirected()))

    def split_disconnected(self: Pipeline, **pipeline_kwargs: Any) -> tuple[Pipeline, ...]:
        """Split disconnected components of the pipeline into separate pipelines.

        Parameters
        ----------
        pipeline_kwargs
            Keyword arguments to pass to the `Pipeline` constructor.

        Returns
        -------
            Tuple of fully connected `Pipeline` objects.

        """
        connected_components = self._connected_components()
        pipefunc_lists = [
            [x.copy() for x in xs if isinstance(x, PipeFunc)] for xs in connected_components
        ]
        if len(pipefunc_lists) == 1:
            msg = "Pipeline is fully connected, no need to split."
            raise ValueError(msg)
        return tuple(Pipeline(pfs, **pipeline_kwargs) for pfs in pipefunc_lists)  # type: ignore[arg-type]

    def _axis_in_root_arg(
        self,
        axis: str,
        output_name: _OUTPUT_TYPE,
        root_args: tuple[str, ...] | None = None,
        visited: set[_OUTPUT_TYPE] | None = None,
        result: set[bool] | None = None,
    ) -> bool:
        if root_args is None:
            root_args = self.root_args(output_name)
        if visited is None:
            visited = set()
        if result is None:
            result = set()
        if output_name in visited:
            return None  # type: ignore[return-value]

        visited.add(output_name)
        visited.update(at_least_tuple(output_name))

        func = self.output_to_func[output_name]
        assert func.mapspec is not None
        if axis not in func.mapspec.output_indices:  # pragma: no cover
            msg = f"Axis `{axis}` not in output indices for `{output_name=}`"
            raise ValueError(msg)

        if axis not in func.mapspec.input_indices:
            # Axis was in output but not in input
            result.add(False)  # noqa: FBT003

        axes = self.mapspec_axes
        for name in func.mapspec.input_names:
            if axis not in axes[name]:
                continue
            if name in root_args:
                if axis in axes[name]:
                    result.add(True)  # noqa: FBT003
            else:
                self._axis_in_root_arg(axis, name, root_args, visited, result)

        return all(result)

    def independent_axes_in_mapspecs(self, output_name: _OUTPUT_TYPE) -> set[str]:
        """Return the axes that are both in the output and in the root arguments.

        Identifies axes that are cross-products and can be computed independently.
        """
        func = self.output_to_func[output_name]
        if func.mapspec is None:
            return set()
        return {
            axis
            for axis in func.mapspec.output_indices
            if self._axis_in_root_arg(axis, output_name)
        }


class _Generations(NamedTuple):
    root_args: list[str]
    function_lists: list[list[PipeFunc]]


def _update_all_results(
    func: PipeFunc,
    r: Any,
    output_name: _OUTPUT_TYPE,
    all_results: dict[_OUTPUT_TYPE, Any],
    lazy: bool,  # noqa: FBT001
) -> None:
    if isinstance(func.output_name, tuple) and not isinstance(output_name, tuple):
        # Function produces multiple outputs, but only one is requested
        assert func.output_picker is not None
        for name in func.output_name:
            all_results[name] = (
                _LazyFunction(func.output_picker, args=(r, name))
                if lazy
                else func.output_picker(r, name)
            )
    else:
        all_results[func.output_name] = r


def _valid_key(key: Any) -> Any:
    if isinstance(key, dict):
        return tuple(sorted(key.items()))
    if isinstance(key, list):
        return tuple(key)
    if isinstance(key, set):
        return tuple(sorted(key))
    return key


def _update_cache(
    cache: LRUCache | HybridCache | DiskCache | SimpleCache,
    cache_key: _CACHE_KEY_TYPE,
    r: Any,
    start_time: float,
) -> None:
    # Used in _run
    if isinstance(cache, HybridCache):
        duration = time.perf_counter() - start_time
        cache.put(cache_key, r, duration)
    else:
        cache.put(cache_key, r)


def _get_result_from_cache(
    func: PipeFunc,
    cache: LRUCache | HybridCache | DiskCache | SimpleCache,
    cache_key: _CACHE_KEY_TYPE | None,
    output_name: _OUTPUT_TYPE,
    all_results: dict[_OUTPUT_TYPE, Any],
    full_output: bool,  # noqa: FBT001
    used_parameters: set[str | None],
    lazy: bool = False,  # noqa: FBT002, FBT001
) -> tuple[bool, bool]:
    # Used in _run
    result_from_cache = False
    if cache_key is not None and cache_key in cache:
        r = cache.get(cache_key)
        _update_all_results(func, r, output_name, all_results, lazy)
        result_from_cache = True
        if not full_output:
            used_parameters.add(None)  # indicate that the result was from cache
            return True, result_from_cache
    return False, result_from_cache


def _check_consistent_defaults(functions: list[PipeFunc]) -> None:
    """Check that the default values for shared arguments are consistent."""
    arg_defaults = defaultdict(set)
    for f in functions:
        for arg, default_value in f.defaults.items():
            arg_defaults[arg].add(default_value)
            if len(arg_defaults[arg]) > 1:
                msg = (
                    f"Inconsistent default values for argument '{arg}' in"
                    " functions. Please make sure the shared input arguments have"
                    " the same default value or are set only for one function.",
                )
                raise ValueError(msg)


def _check_consistent_type_annotations(functions: list[PipeFunc]) -> None:
    """Check that the type annotations for shared arguments are consistent."""
    arg_annotations = defaultdict(set)
    for f in functions:
        annotations = f.parameter_annotations.copy()
        if f.output_picker is None:
            assert isinstance(f.output_name, str)
            annotations[f.output_name] = f.output_annotation
        for arg, annotation in annotations.items():
            arg_annotations[arg].add(annotation)
            if len(arg_annotations[arg]) > 1:
                msg = (
                    f"Inconsistent type annotations for argument '{arg}' in"
                    " functions. Please make sure the shared input arguments have"
                    " the same type annotation or are set only for one function.",
                )
                raise ValueError(msg)


def _create_cache(
    cache_type: Literal["lru", "hybrid", "disk", "simple"] | None,
    lazy: bool,  # noqa: FBT001
    cache_kwargs: dict[str, Any] | None,
) -> LRUCache | HybridCache | DiskCache | SimpleCache | None:
    if cache_type is None:
        return None
    if cache_kwargs is None:
        cache_kwargs = {}
    if cache_type == "lru":
        cache_kwargs.setdefault("shared", not lazy)
        return LRUCache(**cache_kwargs)
    if cache_type == "hybrid":
        if lazy:
            warnings.warn(
                "Hybrid cache uses function evaluation duration which"
                " is not measured correctly when using `lazy=True`.",
                UserWarning,
                stacklevel=2,
            )
        cache_kwargs.setdefault("shared", not lazy)
        return HybridCache(**cache_kwargs)
    if cache_type == "disk":
        cache_kwargs.setdefault("lru_shared", not lazy)
        return DiskCache(**cache_kwargs)
    if cache_type == "simple":
        return SimpleCache()

    msg = f"Invalid cache type: {cache_type}."
    raise ValueError(msg)


def _execute_func(func: PipeFunc, func_args: dict[str, Any], lazy: bool) -> Any:  # noqa: FBT001
    if lazy:
        return _LazyFunction(func, kwargs=func_args)
    try:
        return func(**func_args)
    except Exception as e:
        handle_error(e, func, func_args)
        raise  # handle_error raises but mypy doesn't know that


def _compute_cache_key(
    output_name: _OUTPUT_TYPE,
    kwargs: dict[str, Any],
    root_args: tuple[str, ...],
) -> _CACHE_KEY_TYPE | None:
    """Compute the cache key for a specific output name.

    The cache key is a tuple consisting of the output name and a tuple of
    root input keys and their corresponding values. Root inputs are the
    inputs that are not derived from any other function in the pipeline.

    If any of the root inputs required for the output_name are not available
    in kwargs, the cache key computation is skipped, and the method returns
    None. This can happen when a non-root input is directly provided as an
    input to another function, in which case the result should not be
    cached.

    Parameters
    ----------
    output_name
        The identifier for the return value of the pipeline.
    kwargs
        Keyword arguments to be passed to the pipeline functions.
    root_args
        The names of the pipeline function's root inputs.

    Returns
    -------
        A tuple containing the output name and a tuple of root input keys
        and their corresponding values, or None if the cache key computation
        is skipped.

    """
    cache_key_items = []
    for k in root_args:
        if k not in kwargs:
            # This means the computation was run with non-root inputs
            # i.e., the output of a function was directly provided as an input to
            # another function. In this case, we don't want to cache the result.
            return None
        key = _valid_key(kwargs[k])
        cache_key_items.append((k, key))

    return output_name, tuple(cache_key_items)


def _save_results(
    func: PipeFunc,
    r: Any,
    output_name: _OUTPUT_TYPE,
    all_results: dict[_OUTPUT_TYPE, Any],
    root_args: tuple[str, ...],
    lazy: bool,  # noqa: FBT001
) -> None:
    # Used in _run
    if func.save_function is None:
        return
    to_save = {k: all_results[k] for k in root_args}
    filename = generate_filename_from_dict(to_save)  # type: ignore[arg-type]
    filename = func.__name__ / filename
    to_save[output_name] = all_results[output_name]  # type: ignore[index]
    if lazy:
        lazy_save = _LazyFunction(func.save_function, args=(filename, to_save), add_to_graph=False)
        r.add_delayed_callback(lazy_save)
    else:
        func.save_function(filename, to_save)  # type: ignore[arg-type]


def _names(nodes: Iterable[PipeFunc | str]) -> tuple[str, ...]:
    names: list[str] = []
    for n in nodes:
        if isinstance(n, PipeFunc):
            names.extend(at_least_tuple(n.output_name))
        else:
            assert isinstance(n, str)
            names.append(n)
    return tuple(sorted(names))


def _sort_key(node: PipeFunc | str) -> str:
    if isinstance(node, PipeFunc):
        if isinstance(node.output_name, tuple):
            return ",".join(node.output_name)
        return node.output_name
    return node


def _unique(nodes: Iterable[PipeFunc | str]) -> tuple[PipeFunc | str, ...]:
    return tuple(sorted(set(nodes), key=_sort_key))


def _filter_funcs(funcs: Iterable[PipeFunc | str]) -> list[PipeFunc]:
    return [f for f in funcs if isinstance(f, PipeFunc)]


def _compute_arg_mapping(
    graph: nx.DiGraph,
    node: PipeFunc,
    head: PipeFunc,
    args: list[PipeFunc | str],
    replaced: list[PipeFunc | str],
    arg_set: set[tuple[str, ...]],
) -> None:
    preds = [n for n in graph.predecessors(node) if n not in replaced]
    deps = _unique(args + preds)
    deps_names = _names(deps)
    if deps_names in arg_set:
        return
    arg_set.add(deps_names)

    for func in _filter_funcs(deps):
        new_args = [dep for dep in deps if dep != func]
        _compute_arg_mapping(graph, func, head, new_args, [*replaced, node], arg_set)


def _axes_from_dims(p: str, dims: dict[str, int], axis: str) -> tuple[str | None, ...]:
    n = dims.get(p, 1) - 1
    return n * (None,) + (axis,)


def _add_mapspec_axis(p: str, dims: dict[str, int], axis: str, functions: list[PipeFunc]) -> None:
    for f in functions:
        if p not in f.parameters:
            continue
        if f.mapspec is None:
            axes = _axes_from_dims(p, dims, axis)
            input_specs = [ArraySpec(p, axes)]
            output_specs = [ArraySpec(name, (axis,)) for name in at_least_tuple(f.output_name)]
        else:
            existing_inputs = set(f.mapspec.input_names)
            if p in existing_inputs:
                input_specs = [
                    s.add_axes(axis) if s.name == p and axis not in s.axes else s
                    for s in f.mapspec.inputs
                ]
            else:
                axes = _axes_from_dims(p, dims, axis)
                input_specs = [*f.mapspec.inputs, ArraySpec(p, axes)]
            output_specs = [
                s.add_axes(axis) if axis not in s.axes else s for s in f.mapspec.outputs
            ]
        f.mapspec = MapSpec(tuple(input_specs), tuple(output_specs))
        for o in output_specs:
            dims[o.name] = len(o.axes)
            _add_mapspec_axis(o.name, dims, axis, functions)


def _find_non_root_axes(
    mapspecs: list[MapSpec],
    root_args: list[str],
) -> dict[str, list[str | None]]:
    non_root_inputs: dict[str, list[str | None]] = {}
    for mapspec in mapspecs:
        for spec in mapspec.inputs:
            if spec.name not in root_args:
                if spec.name not in non_root_inputs:
                    non_root_inputs[spec.name] = spec.rank * [None]  # type: ignore[assignment]
                for i, axis in enumerate(spec.axes):
                    if axis is not None:
                        non_root_inputs[spec.name][i] = axis
    return non_root_inputs


def _replace_none_in_axes(
    mapspecs: list[MapSpec],
    non_root_inputs: dict[str, list[str]],
    multi_output_mapping: dict[str, tuple[str, ...]],
) -> None:
    all_axes_names = {
        axis.name for mapspec in mapspecs for axis in mapspec.inputs + mapspec.outputs
    }

    i = 0
    axis_template = "unnamed_{}"
    for name, axes in non_root_inputs.items():
        for j, axis in enumerate(axes):
            if axis is None:
                while (new_axis := axis_template.format(i)) in all_axes_names:
                    i += 1
                non_root_inputs[name][j] = new_axis
                all_axes_names.add(new_axis)
                if name in multi_output_mapping:
                    # If output is a tuple, update its axes with the new axis.
                    for output_name in multi_output_mapping[name]:
                        non_root_inputs[output_name][j] = new_axis
    assert not any(None in axes for axes in non_root_inputs.values())


def _create_missing_mapspecs(
    functions: list[PipeFunc],
    non_root_inputs: dict[str, set[str]],
) -> set[PipeFunc]:
    # Mapping from output_name to PipeFunc for functions without a MapSpec
    outputs_without_mapspec: dict[str, PipeFunc] = {
        name: func
        for func in functions
        if func.mapspec is None
        for name in at_least_tuple(func.output_name)
    }

    missing: set[str] = non_root_inputs.keys() & outputs_without_mapspec.keys()
    func_with_new_mapspecs = set()
    for p in missing:
        func = outputs_without_mapspec[p]
        if func in func_with_new_mapspecs:
            continue  # already added a MapSpec because of multiple outputs
        axes = tuple(non_root_inputs[p])
        outputs = tuple(ArraySpec(x, axes) for x in at_least_tuple(func.output_name))
        func.mapspec = MapSpec(inputs=(), outputs=outputs)
        func_with_new_mapspecs.add(func)
        print(f"Autogenerated MapSpec for `{func}`: `{func.mapspec}`")
    return func_with_new_mapspecs


def _traverse_graph(
    start: _OUTPUT_TYPE | PipeFunc,
    direction: Literal["predecessors", "successors"],
    graph: nx.DiGraph,
    node_mapping: dict[_OUTPUT_TYPE, PipeFunc | str],
) -> list[_OUTPUT_TYPE]:
    visited = set()

    def _traverse(x: _OUTPUT_TYPE | PipeFunc) -> list[_OUTPUT_TYPE]:
        results = set()
        if isinstance(x, (str, tuple)):
            x = node_mapping[x]
        for neighbor in getattr(graph, direction)(x):
            if isinstance(neighbor, PipeFunc):
                output_name = neighbor.output_name
                if output_name not in visited:
                    visited.add(output_name)
                    results.add(output_name)
                    results.update(_traverse(neighbor))
        return results  # type: ignore[return-value]

    return sorted(_traverse(start), key=at_least_tuple)
