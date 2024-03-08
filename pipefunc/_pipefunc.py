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

import contextlib
import hashlib
import inspect
import json
import os
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from functools import partial, update_wrapper
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Literal,
    Tuple,
    TypeVar,
    Union,
    get_type_hints,
)

import cloudpickle
import networkx as nx

from pipefunc._cache import HybridCache, LRUCache
from pipefunc._perf import ProfilingStats, ResourceProfiler
from pipefunc._plotting import visualize, visualize_holoviews

if sys.version_info < (3, 9):  # pragma: no cover
    from typing import Callable
else:
    from collections.abc import Callable

if sys.version_info < (3, 10):  # pragma: no cover
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias


if TYPE_CHECKING:
    import holoviews as hv

T = TypeVar("T", bound=Callable[..., Any])
_OUTPUT_TYPE = Union[str, Tuple[str, ...]]
_CACHE_KEY_TYPE: TypeAlias = Tuple[
    _OUTPUT_TYPE,
    Tuple[Tuple[str, Any], ...],
]


def _at_least_tuple(x: Any) -> tuple[Any, ...]:
    """Convert x to a tuple if it is not already a tuple."""
    return x if isinstance(x, tuple) else (x,)


def _default_output_picker(
    output: Any,
    name: str,
    output_name: _OUTPUT_TYPE,
) -> Any:
    """Default output picker function for tuples."""
    return output[output_name.index(name)]


class PipelineFunction(Generic[T]):
    """Function wrapper class for pipeline functions with additional attributes.

    Parameters
    ----------
    func
        The original function to be wrapped.
    output_name
        The identifier for the output of the wrapped function.
    output_picker
        A function that takes the output of the wrapped function as first argument
        and the output_name (str) as second argument, and returns the desired output.
        If None, the output of the wrapped function is returned as is.
    renames
        A dictionary mapping from original argument names to new argument names.
    profile
        Flag indicating whether the wrapped function should be profiled.
    debug
        Flag indicating whether debug information should be printed.
    cache
        Flag indicating whether the wrapped function should be cached.
    save
        Flag indicating whether the output of the wrapped function should be saved.
    save_function
        A function that takes the filename and a dict containing the inputs and output.

    Returns
    -------
        The identifier for the output of the wrapped function.

    Examples
    --------
    >>> def add_one(a, b):
    ...     return a + 1, b + 1
    >>> add_one_func = PipelineFunction(
    ...     add_one,
    ...     output_name="a_plus_one",
    ...     renames={"a": "x", "b": "y"},
    ... )
    >>> add_one_func(x=1, y=2)
    (2, 3)

    """

    def __init__(
        self,
        func: T,
        output_name: _OUTPUT_TYPE,
        *,
        output_picker: Callable[[str, Any], Any] | None = None,
        renames: dict[str, str] | None = None,
        profile: bool = False,
        debug: bool = False,
        cache: bool = False,
        save: bool | None = None,
        save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
    ) -> None:
        """Function wrapper class for pipeline functions with additional attributes."""
        update_wrapper(self, func)
        self.func: Callable[..., Any] = func
        self.output_name: _OUTPUT_TYPE = output_name
        self.debug = debug
        self.cache = cache
        self.save_function = save_function
        self.save = save if save is not None else save_function is not None
        self.output_picker: Callable[[Any, str], Any] | None = output_picker
        if output_picker is None and isinstance(output_name, tuple):
            self.output_picker = partial(
                _default_output_picker,
                output_name=self.output_name,
            )
        self._profile = profile
        self.renames: dict[str, str] = renames or {}
        self._inverse_renames: dict[str, str] = {v: k for k, v in self.renames.items()}
        parameters = inspect.signature(func).parameters
        self.parameters: list[str] = [self.renames.get(k, k) for k in parameters]
        self.defaults: dict[str, Any] = {
            self.renames.get(k, k): v.default
            for k, v in parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        self.profiling_stats: ProfilingStats | None
        self.set_profiling(enable=profile)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function with the given arguments.

        Returns
        -------
        Any
            The return value of the wrapped function.

        """
        kwargs = {self._inverse_renames.get(k, k): v for k, v in kwargs.items()}
        with self._maybe_profiler():
            result = self.func(*args, **kwargs)

        if self.debug and self.profiling_stats is not None:
            dt = self.profiling_stats.time.average
            print(
                f"Function {self.func.__name__} called with args={args},"
                f" kwargs={kwargs}, took {dt:.2e} seconds to execute.",
            )
        return result

    @property
    def profile(self) -> bool:
        """Return whether profiling is enabled for the wrapped function."""
        return self._profile

    @profile.setter
    def profile(self, enable: bool) -> None:
        """Enable or disable profiling for the wrapped function."""
        self.set_profiling(enable=enable)

    def set_profiling(self, *, enable: bool = True) -> None:
        """Enable or disable profiling for the wrapped function."""
        self._profile = enable
        if enable:
            self.profiling_stats = ProfilingStats()
        else:
            self.profiling_stats = None

    @property
    def parameter_annotations(self) -> dict[str, Any]:
        """Return the type annotations of the wrapped function's parameters."""
        type_hints = get_type_hints(self.func)
        return {
            self.renames.get(k, k): v for k, v in type_hints.items() if k != "return"
        }

    @property
    def output_annotation(self) -> Any:
        """Return the type annotation of the wrapped function's output."""
        return get_type_hints(self.func).get("return", Any)

    def _maybe_profiler(self) -> contextlib.AbstractContextManager:
        """Maybe get profiler.

        Get a profiler instance if profiling is enabled, otherwise
        return a dummy context manager.

        Returns
        -------
        AbstractContextManager
            A ResourceProfiler instance if profiling is enabled, or a
            nullcontext if disabled.

        """
        if self.profiling_stats is not None:
            return ResourceProfiler(os.getpid(), self.profiling_stats)
        return contextlib.nullcontext()

    def __getattr__(self, name: str) -> Any:
        """Get attributes of the wrapped function.

        Parameters
        ----------
        name
            The name of the attribute to get.

        Returns
        -------
        Any
            The value of the attribute.

        """
        return getattr(self.func, name)

    def __str__(self) -> str:
        """Return a string representation of the PipelineFunction instance.

        Returns
        -------
        str
            A string representation of the PipelineFunction instance.

        """
        params = ", ".join(self.parameters)
        outputs = ", ".join(_at_least_tuple(self.output_name))
        return f"{self.func.__name__}({params}) → {outputs}"

    def __repr__(self) -> str:
        """Return a string representation of the PipelineFunction instance.

        Returns
        -------
        str
            A string representation of the PipelineFunction instance.

        """
        return f"PipelineFunction({self.func.__name__})"

    def __getstate__(self) -> dict:
        """Prepare the state of the current object for pickling.

        The state includes all picklable instance variables.
        For non-picklable instance variable,  they are transformed
        into a picklable form or ignored.

        Returns
        -------
        state : dict
            A dictionary containing the picklable state of the object.

        """
        state = self.__dict__.copy()
        state["func"] = cloudpickle.dumps(state.pop("func"))
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the current object from the provided state.

        It also handles restoring non-picklable instance variable
        into their original form.

        Parameters
        ----------
        state : dict
            A dictionary containing the picklable state of the object.

        """
        self.__dict__.update(state)
        self.func = cloudpickle.loads(self.func)


def pipefunc(
    output_name: _OUTPUT_TYPE,
    *,
    output_picker: Callable[[Any, str], Any] | None = None,
    renames: dict[str, str] | None = None,
    profile: bool = False,
    debug: bool = False,
    cache: bool = False,
    save: bool | None = None,
    save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
) -> Callable[[Callable[..., Any]], PipelineFunction]:
    """A decorator for tagging pipeline functions with a return identifier.

    Parameters
    ----------
    output_name
        The identifier for the output of the decorated function.
    output_picker
        A function that takes the output of the wrapped function as first argument
        and the output_name (str) as second argument, and returns the desired output.
        If None, the output of the wrapped function is returned as is.
    renames
        A dictionary mapping from original argument names to new argument names.
    profile
        Flag indicating whether the decorated function should be profiled.
    debug
        Flag indicating whether debug information should be printed.
    cache
        Flag indicating whether the decorated function should be cached.
    save
        Flag indicating whether the output of the wrapped function should be saved.
    save_function
        A function that takes the filename and a dict containing the inputs and output.

    Returns
    -------
    Callable[[Callable[..., Any]], PipelineFunction]
        A decorator function that takes the original function and output_name a
        PipelineFunction instance with the specified return identifier.

    """

    def decorator(f: Callable[..., Any]) -> PipelineFunction:
        """Wraps the original function in a PipelineFunction instance.

        Parameters
        ----------
        f
            The original function to be wrapped.

        Returns
        -------
        PipelineFunction
            The wrapped function with the specified return identifier.

        """
        return PipelineFunction(
            f,
            output_name,
            output_picker=output_picker,
            renames=renames,
            profile=profile,
            debug=debug,
            cache=cache,
            save=save,
            save_function=save_function,
        )

    return decorator


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
        self.call_with_root_args = self._create_call_with_root_args_method()

    def __call__(self, **kwargs: Any) -> Any:
        """Call the pipeline function with the given arguments.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
        Any
            The return value of the pipeline function.

        """
        return self.pipeline._run_pipeline(output_name=self.output_name, **kwargs)

    def call_full_output(self, **kwargs: Any) -> dict[str, Any]:
        """Call the pipeline function with the given arguments and return all outputs.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
        Any
            The return value of the pipeline function.

        """
        return self.pipeline._run_pipeline(
            output_name=self.output_name,
            full_output=True,
            **kwargs,
        )

    def call_with_dict(self, kwargs: dict[str, Any]) -> Any:
        """Call the pipeline function with the given arguments.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
        Any
            The return value of the pipeline function.

        """
        return self(**kwargs)

    def __getstate__(self) -> dict:
        """Prepare the state of the current object for pickling."""
        state = self.__dict__.copy()
        state.pop("call_with_root_args", None)  # don't pickle the execute method
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the current object from the provided state."""
        self.__dict__.update(state)
        self.call_with_root_args = self._create_call_with_root_args_method()

    def _create_call_with_parameters_method(
        self,
        parameters: tuple[str, ...],
        output_name: _OUTPUT_TYPE | None = None,
    ) -> Callable[..., Any]:
        sig = inspect.signature(self.__call__)
        new_params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in parameters
        ]
        new_sig = sig.replace(parameters=new_params)

        def call(*args: Any, **kwargs: Any) -> Any:
            """Call the pipeline function with the root arguments."""
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if output_name is None:
                return self(**bound.arguments)
            all_results = self.pipeline._run_pipeline(
                output_name=self.output_name,
                **bound.arguments,
                full_output=True,
            )
            if isinstance(output_name, str):
                return all_results[output_name]
            return tuple(all_results[o] for o in output_name)

        call.__signature__ = new_sig  # type: ignore[attr-defined]
        return call

    def _create_call_with_root_args_method(self) -> Callable[..., Any]:
        return self._create_call_with_parameters_method(self.root_args)


def _next_root_args(
    pipeline: Pipeline,
    arg_set: set[tuple[str, ...]],
) -> tuple[str, ...]:
    """Find the tuple of root arguments."""
    return next(
        args
        for args in arg_set
        if all(isinstance(pipeline.node_mapping[n], str) for n in args)
    )


class Pipeline:
    """Pipeline class for managing and executing a sequence of functions.

    Parameters
    ----------
    functions
        A list of functions that form the pipeline.
    debug
        Flag indicating whether debug information should be printed.
        If None, the value of each PipelineFunction's debug attribute is used.
    profile
        Flag indicating whether profiling information should be collected.
        If None, the value of each PipelineFunction's profile attribute is used.
    cache
        The type of cache to use.

    """

    def __init__(
        self,
        functions: list[PipelineFunction],
        *,
        debug: bool | None = None,
        profile: bool | None = None,
        cache: Literal["shared", "hybrid", "disk"] | None = "hybrid",
    ) -> None:
        """Pipeline class for managing and executing a sequence of functions."""
        # TODO: add support for disk cache
        # TODO: check https://joblib.readthedocs.io/en/latest/memory.html
        # TODO: add caching kwargs

        self.functions: list[PipelineFunction] = []
        self._debug = debug
        self._profile = profile
        self.output_to_func: dict[_OUTPUT_TYPE, PipelineFunction] = {}
        for f in functions:
            self.add(f)
        self._graph = None
        self._arg_combinations: dict[_OUTPUT_TYPE, set[tuple[str, ...]]] = {}
        self.cache: LRUCache | HybridCache

        if cache == "shared":
            self.cache = LRUCache()
        elif cache == "hybrid":
            self.cache = HybridCache()
        elif cache == "disk":
            msg = "Disk cache not yet implemented"
            raise NotImplementedError(msg)
        else:
            msg = f"Unknown cache type {cache}"
            raise ValueError(msg)

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

    def add(self, f: PipelineFunction) -> None:
        """Add a function to the pipeline.

        Parameters
        ----------
        f
            The function to add to the pipeline.
        profile
            Flag indicating whether profiling information should be collected.

        """
        if not isinstance(f, PipelineFunction):
            f = PipelineFunction(f, output_name=f.__name__)
        self.functions.append(f)

        self.output_to_func[f.output_name] = f
        if isinstance(f.output_name, tuple):
            for name in f.output_name:
                self.output_to_func[name] = f

        if self.profile is not None:
            f.set_profiling(enable=self.profile)
        if self.debug is not None:
            f.debug = self.debug
        self._graph = None
        self._arg_combinations = {}

    def _check_consistent_defaults(self) -> None:
        """Check that the default values for shared arguments are consistent."""
        arg_defaults = defaultdict(set)
        for f in self.functions:
            for arg, default_value in f.defaults.items():
                arg_defaults[arg].add(default_value)
                if len(arg_defaults[arg]) > 1:
                    msg = (
                        f"Inconsistent default values for argument '{arg}' in"
                        " functions. Please make sure the shared input arguments have"
                        " the same default value or are set only for one function.",
                    )
                    raise ValueError(msg)

    @property
    def graph(self) -> nx.DiGraph:
        """The directed graph representing the pipeline."""
        if self._graph is None:
            self._graph = self._make_graph()
        return self._graph

    def _make_graph(self) -> nx.DiGraph:
        """Create a directed graph representing the pipeline.

        Returns
        -------
        nx.DiGraph
            A directed graph with nodes representing functions and edges
            representing dependencies between functions.

        """
        self._check_consistent_defaults()
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
                        g.edges[edge]["arg"] = (*_at_least_tuple(current), arg)

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
        Callable[..., Any]
            The composed function that can be called with keyword arguments.

        """
        root_args = self.arg_combinations(output_name, root_args_only=True)
        assert isinstance(root_args, tuple)
        return _Function(self, output_name, root_args=root_args)

    def __call__(self, output_name: _OUTPUT_TYPE, **kwargs: Any) -> Any:
        """Call the pipeline for a specific return value.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline.
        kwargs
            Keyword arguments to be passed to the pipeline functions.

        Returns
        -------
        Any
            The return value of the pipeline.

        """
        return self.func(output_name)(**kwargs)

    def _compute_cache_key(
        self,
        output_name: _OUTPUT_TYPE,
        kwargs: dict[str, Any],
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

        Returns
        -------
        _CACHE_KEY_TYPE | None
            A tuple containing the output name and a tuple of root input keys
            and their corresponding values, or None if the cache key computation
            is skipped.

        """
        root_args = self.arg_combinations(output_name, root_args_only=True)
        assert isinstance(root_args, tuple)
        cache_key_items = []
        for k in sorted(root_args):
            if k not in kwargs:
                # This means the computation was run with non-root inputs
                # i.e., the output of a function was directly provided as an input to
                # another function. In this case, we don't want to cache the result.
                return None
            cache_key_items.append((k, kwargs[k]))

        return output_name, tuple(cache_key_items)

    def _run_pipeline(
        self,
        output_name: _OUTPUT_TYPE,
        *,
        full_output: bool = False,
        **kwargs: Any,
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
        Any
            The return value of the pipeline or a dictionary mapping function
            names to their return values if full_output is True.

        """

        def _update_all_results(
            func: PipelineFunction,
            r: Any,
            output_name: _OUTPUT_TYPE,
            all_results: dict[_OUTPUT_TYPE, Any],
        ) -> None:
            if isinstance(func.output_name, tuple) and not isinstance(
                output_name,
                tuple,
            ):
                for name in func.output_name:
                    assert func.output_picker is not None
                    all_results[name] = func.output_picker(r, name)
            else:
                all_results[func.output_name] = r

        def _execute_pipeline(  # noqa: PLR0912
            output_name: _OUTPUT_TYPE,
            **kwargs: Any,
        ) -> Any:
            if output_name in all_results:
                return all_results[output_name]

            func = self.output_to_func[output_name]

            if func is None:
                msg = (
                    f"Argument {output_name} is not in kwargs and has no default value."
                )
                raise ValueError(msg)

            assert func.parameters is not None
            result_from_cache = False
            if func.cache:
                cache_key = self._compute_cache_key(func.output_name, kwargs)
                if cache_key is not None and cache_key in self.cache:
                    r = self.cache.get(cache_key)
                    assert r is not None
                    _update_all_results(func, r, output_name, all_results)
                    result_from_cache = True
                    if not full_output:
                        return all_results[output_name]

            func_args = {}
            for arg in func.parameters:
                if arg not in kwargs and arg not in func.defaults:
                    func_args[arg] = _execute_pipeline(arg, **kwargs)
                elif arg in kwargs:
                    func_args[arg] = kwargs[arg]
                else:  # arg in func.defaults
                    func_args[arg] = func.defaults[arg]
            if result_from_cache:
                # Can only happen if full_output is True
                return all_results[output_name]
            start_time = time.perf_counter()
            r = func(**func_args)

            if func.cache and cache_key is not None:
                if isinstance(self.cache, HybridCache):
                    duration = time.perf_counter() - start_time
                    self.cache.put(cache_key, r, duration)
                else:
                    self.cache.put(cache_key, r)

            _update_all_results(func, r, output_name, all_results)
            if func.save and not result_from_cache:
                root_args = self.arg_combinations(output_name, root_args_only=True)
                to_save = {k: all_results[k] for k in root_args if k in all_results}
                filename = generate_filename_from_dict(to_save)  # type: ignore[arg-type]
                filename = func.__name__ / filename
                to_save[output_name] = all_results[output_name]
                assert func.save_function is not None
                func.save_function(filename, to_save)  # type: ignore[arg-type]
            return all_results[output_name]

        all_results: dict[_OUTPUT_TYPE, Any] = kwargs.copy()  # type: ignore[assignment]
        _execute_pipeline(output_name, **kwargs)
        return all_results if full_output else all_results[output_name]

    @property
    def node_mapping(self) -> dict[_OUTPUT_TYPE, PipelineFunction | str]:
        """Return a mapping from node names to nodes.

        Returns
        -------
        Dict[_OUTPUT_TYPE, PipelineFunction | str]
            A mapping from node names to nodes.

        """
        mapping: dict[_OUTPUT_TYPE, PipelineFunction | str] = {}
        for node in self.graph.nodes:
            if isinstance(node, PipelineFunction):
                if isinstance(node.output_name, tuple):
                    for name in node.output_name:
                        mapping[name] = node
                mapping[node.output_name] = node
            else:
                assert isinstance(node, str)
                mapping[node] = node
        return mapping

    def arg_combinations(
        self,
        output_name: _OUTPUT_TYPE,
        *,
        root_args_only: bool = False,
    ) -> set[tuple[str, ...]] | tuple[str, ...]:
        """Return the arguments required to compute a specific output.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline.
        root_args_only
            If True, only return the root arguments required to compute the
            output. If False, return all arguments required to compute the
            output.

        Returns
        -------
        Set[Tuple[str, ...]]
            A set of tuples containing possible argument combinations.

        """
        if r := self._arg_combinations.get(output_name):
            if root_args_only:
                return _next_root_args(self, r)
            return r

        def names(nodes: Iterable[PipelineFunction | str]) -> tuple[str, ...]:
            names: list[str] = []
            for n in nodes:
                if isinstance(n, PipelineFunction):
                    names.extend(_at_least_tuple(n.output_name))
                else:
                    assert isinstance(n, str)
                    names.append(n)
            return tuple(names)

        def sort_key(node: PipelineFunction | str) -> str:
            if isinstance(node, PipelineFunction):
                if isinstance(node.output_name, tuple):
                    return ",".join(node.output_name)
                return node.output_name
            return node

        def unique(
            nodes: Iterable[PipelineFunction | str],
        ) -> tuple[PipelineFunction | str, ...]:
            return tuple(sorted(set(nodes), key=sort_key))

        def filter_funcs(
            funcs: Iterable[PipelineFunction | str],
        ) -> list[PipelineFunction]:
            return [f for f in funcs if isinstance(f, PipelineFunction)]

        def compute_arg_mapping(
            node: PipelineFunction,
            head: PipelineFunction,
            args: list[PipelineFunction | str],
            replaced: list[PipelineFunction | str],
        ) -> None:
            preds = [n for n in self.graph.predecessors(node) if n not in replaced]
            deps = unique(args + preds)
            deps_names = names(deps)
            if deps_names in arg_set:
                return
            arg_set.add(deps_names)

            for func in filter_funcs(deps):
                new_args = [dep for dep in deps if dep != func]
                compute_arg_mapping(func, head, new_args, [*replaced, node])

        head = self.node_mapping[output_name]
        arg_set: set[tuple[str, ...]] = set()
        compute_arg_mapping(head, head, [], [])  # type: ignore[arg-type]
        self._arg_combinations[output_name] = arg_set
        if root_args_only:
            return _next_root_args(self, arg_set)
        return arg_set

    def func_dependencies(self, output_name: _OUTPUT_TYPE) -> list[_OUTPUT_TYPE]:
        """Return the functions required to compute a specific output."""

        def _predecessors(x: _OUTPUT_TYPE | PipelineFunction) -> list[_OUTPUT_TYPE]:
            preds = set()
            if isinstance(x, (str, tuple)):
                x = self.node_mapping[x]
            for pred in self.graph.predecessors(x):
                if isinstance(pred, PipelineFunction):
                    preds.add(pred.output_name)
                    for p in _predecessors(pred):
                        preds.add(p)
            return preds  # type: ignore[return-value]

        return sorted(_predecessors(output_name), key=_at_least_tuple)

    def all_arg_combinations(
        self,
        *,
        root_args_only: bool = False,
    ) -> dict[_OUTPUT_TYPE, set[tuple[str, ...]]]:
        """Compute all possible argument mappings for the pipeline.

        Considering only the root input nodes if `root_args_only` is
        set to True.

        Parameters
        ----------
        root_args_only
            If True, the function will only consider the root input nodes
            (i.e., nodes with no predecessor functions) while calculating the
            possible argument combinations. If False, all predecessor nodes,
            including intermediate functions, will be considered.

        Returns
        -------
        Dict[_OUTPUT_TYPE, Set[Tuple[str, ...]]]
            A dictionary mapping function names to sets of tuples containing
            possible argument combinations.

        """
        mapping: dict[_OUTPUT_TYPE, set[tuple[str, ...]]] = defaultdict(set)
        for node in self.graph.nodes:
            if isinstance(node, PipelineFunction):
                arg_combinations = self.arg_combinations(
                    node.output_name,
                    root_args_only=root_args_only,
                )
                if not isinstance(arg_combinations, set):  # root_args_only=True
                    arg_combinations = {arg_combinations}
                mapping[node.output_name] = arg_combinations
        return mapping

    def _unique_leaf_node(self) -> PipelineFunction:
        """Return the unique leaf node of the pipeline graph."""
        leaf_nodes = self.leaf_nodes
        if len(leaf_nodes) != 1:  # pragma: no cover
            msg = (
                "The pipeline has multiple leaf nodes. Please specify the output_name"
                " argument to disambiguate.",
            )
            raise ValueError(msg)
        return leaf_nodes[0]

    def _func_node_colors(
        self,
        *,
        conservatively_combine: bool = False,
        output_name: _OUTPUT_TYPE | None = None,
    ) -> list[str]:
        if output_name is None:
            output_name = self._unique_leaf_node().output_name

        func_node_colors = []
        combinable_nodes = self._identify_combinable_nodes(
            output_name=output_name,
            conservatively_combine=conservatively_combine,
        )
        combinable_nodes = _combine_nodes(combinable_nodes)
        node_sets = [{k, *v} for k, v in combinable_nodes.items()]
        color_index = len(node_sets)  # for non-combinable nodes
        for node in self.graph.nodes:
            if isinstance(node, PipelineFunction):
                i = next(
                    (i for i, nodes in enumerate(node_sets) if node in nodes),
                    None,
                )
                if i is not None:
                    func_node_colors.append(f"C{i}")
                else:
                    func_node_colors.append(f"C{color_index}")
                    color_index += 1
        return func_node_colors

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
        print("Resource Usage Report:")
        for func_name, stats in self.profiling_stats.items():
            print(
                f"{func_name}: average CPU usage: {stats.cpu.average * 100:.2f}%,"
                f" max memory usage: {stats.memory.max / (1024 * 1024):.2f} MB,"
                f" average time: {stats.time.average:.2e} s,"
                f" number of calls: {stats.time.num_executions}",
            )

    def _identify_combinable_nodes(
        self,
        output_name: _OUTPUT_TYPE,
        *,
        conservatively_combine: bool = False,
    ) -> dict[PipelineFunction, set[PipelineFunction]]:
        """Identify which function nodes can be combined into a single function.

        This method identifies the PipelineFunctions in the execution graph that
        can be combined into a single function. The criterion for combinability
        is that the functions share the same root arguments.

        Parameters
        ----------
        output_name
            The name of the output from the pipeline function we are starting
            the search from. It is used to get the starting function in the
            pipeline.
        conservatively_combine
            If True, only combine a function node with its predecessors if all
            of its predecessors have the same root arguments as the function
            node itself. If False, combine a function node with its predecessors
            if any of its predecessors have the same root arguments as the
            function node.

        Returns
        -------
        dict[PipelineFunction, set[PipelineFunction]]
            A dictionary where each key is a PipelineFunction that can be
            combined with others. The value associated with each key is a set of
            PipelineFunctions that can be combined with the key function.

        Notes
        -----
        This function works by performing a depth-first search through the
        pipeline's execution graph. Starting from the PipelineFunction
        corresponding to the `output_name`, it goes through each predecessor in
        the graph (functions that need to be executed before the current one).
        For each predecessor function, it recursively checks if it can be
        combined with others by comparing their root arguments.

        If a function's root arguments are identical to the head function's root
        arguments, it is considered combinable and added to the set of
        combinable functions for the head. If `conservatively_combine=True` and
        all predecessor functions are combinable, the head function and its set
        of combinable functions are added to the `combinable_nodes` dictionary.
        If `conservatively_combine=False` and any predecessor function is
        combinable, the head function and its set of combinable functions are
        added to the `combinable_nodes` dictionary.

        The function 'head' in the nested function `_recurse` represents the
        current function being checked in the execution graph.

        """
        # Nested function _recurse performs the depth-first search and updates the
        # `combinable_nodes` dictionary.

        def _recurse(head: PipelineFunction) -> None:
            head_args = self.arg_combinations(head.output_name, root_args_only=True)
            funcs = set()
            i = 0
            for node in self.graph.predecessors(head):
                if isinstance(node, (tuple, str)):  # node is root_arg
                    continue
                i += 1
                _recurse(node)
                node_args = self.arg_combinations(node.output_name, root_args_only=True)
                if node_args == head_args:
                    funcs.add(node)
            if funcs and (not conservatively_combine or i == len(funcs)):
                combinable_nodes[head] = funcs

        combinable_nodes: dict[PipelineFunction, set[PipelineFunction]] = {}
        func = self.node_mapping[output_name]
        assert isinstance(func, PipelineFunction)
        _recurse(func)
        return combinable_nodes

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
            the simplification from. It is used to get the starting function in the
            pipeline. If None, the unique tip of the pipeline graph is used (if
            there is one).
        conservatively_combine
            If True, only combine a function node with its predecessors if all
            of its predecessors have the same root arguments as the function
            node itself. If False, combine a function node with its predecessors
            if any of its predecessors have the same root arguments as the
            function node.

        Returns
        -------
        Pipeline
            The simplified version of the pipeline.

        Notes
        -----
        The pipeline simplification process works in the following way:

        1.  Identify combinable function nodes in the execution graph by
            checking if they share the same root arguments.
        2.  Simplify the dictionary of combinable nodes by replacing any nodes
            that can be combined with their dependencies.
        3.  Generate the set of nodes to be skipped (those that will be merged).
        4.  Get the input and output signatures for the combined nodes.
        5.  Create new pipeline functions for the combined nodes, and add them
            to the list of new functions.
        6.  Add the remaining (non-combinable) functions to the list of new
            functions.
        7.  Generate a new pipeline with the new functions.

        This process can significantly simplify complex pipelines, making them
        easier to understand and potentially improving performance by simplifying
        function calls.

        """
        if output_name is None:
            output_name = self._unique_leaf_node().output_name
        combinable_nodes = self._identify_combinable_nodes(
            output_name,
            conservatively_combine=conservatively_combine,
        )
        if not combinable_nodes:
            warnings.warn(
                "No combinable nodes found, the pipeline cannot be simplified.",
                UserWarning,
                stacklevel=2,
            )
        # Simplify the combinable_nodes dictionary by replacing any nodes that
        # can be combined with their own dependencies, so that each key in the
        # dictionary only depends on nodes that cannot be further combined.
        combinable_nodes = _combine_nodes(combinable_nodes)
        skip = set.union(*combinable_nodes.values()) if combinable_nodes else set()
        in_sig, out_sig = _get_signature(combinable_nodes, self.graph)
        m = self.node_mapping
        predecessors = [m[o] for o in self.func_dependencies(output_name)]
        head = self.node_mapping[output_name]
        new_functions = []
        for f in self.functions:
            if f != head and f not in predecessors:
                continue
            if f in combinable_nodes:
                inputs = tuple(sorted(in_sig[f]))
                outputs = tuple(sorted(out_sig[f]))
                if len(outputs) == 1:
                    outputs = outputs[0]  # type: ignore[assignment]
                funcs = [f, *combinable_nodes[f]]
                mini_pipeline = Pipeline(funcs, debug=False, profile=False)
                func = mini_pipeline.func(f.output_name).call_full_output
                f_combined = _wrap_dict_to_tuple(func, inputs, outputs)
                f_combined.__name__ = f"combined_{f.__name__}"
                f_pipefunc = PipelineFunction(
                    f_combined,
                    outputs,
                    profile=f.profile,
                    save=f.save,
                    cache=f.cache,
                    save_function=f.save_function,
                )
                # Disable saving for all functions that are being combined
                for f_ in funcs:
                    f_.save = False
                f_pipefunc.parameters = list(inputs)
                new_functions.append(f_pipefunc)
            elif f not in skip:
                new_functions.append(f)
        return Pipeline(new_functions)

    def all_execution_orders(
        self,
        output_name: str,
    ) -> Generator[list[PipelineFunction], None, None]:
        """Generate all possible execution orders for the functions in the pipeline.

        This method generates all possible topological sorts (execution orders)
        of the functions in the pipeline's execution graph. It first simplifies the
        pipeline to a version where combinable function nodes have been merged
        into single function nodes, and then generates all topological sorts of
        the functions in the simplified graph.

        The method only considers the functions in the graph and ignores the
        root arguments.

        Parameters
        ----------
        output_name
            The name of the output from the pipeline function we are starting
            from. It is used to get the starting function in the pipeline and to
            determine the simplified pipeline.

        Returns
        -------
        Generator[list[str], None, None]
            A generator that yields lists of function names, each list
            representing a possible execution order of the functions in the
            pipeline.

        Notes
        -----
        A topological sort of a directed graph is a linear ordering of its
        vertices such that for every directed edge U -> V from vertex U to
        vertex V, U comes before V in the ordering. For a pipeline, this means
        that each function only gets executed after all its dependencies have
        been executed.

        The method uses the NetworkX function `all_topological_sorts` to
        generate all possible topological sorts. If there are cycles in the
        graph (i.e., there's a circular dependency between functions), the
        method will raise a NetworkXUnfeasible exception.

        The function 'head' in the nested function `_recurse` represents the
        current function being checked in the execution graph.

        """
        simplified_pipeline = self.simplified_pipeline(output_name)
        func_only_graph = simplified_pipeline.graph.copy()
        root_args = simplified_pipeline.arg_combinations(
            output_name,
            root_args_only=True,
        )
        for arg in root_args:
            func_only_graph.remove_node(arg)
        return nx.all_topological_sorts(func_only_graph)

    @property
    def leaf_nodes(self) -> list[PipelineFunction]:
        """Return the leaf nodes in the pipeline's execution graph."""
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

    @property
    def root_nodes(self) -> list[PipelineFunction]:
        """Return the root nodes in the pipeline's execution graph."""
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

    def all_transitive_paths(
        self,
        output_name: str,
        *,
        simplify: bool = False,
    ) -> list[list[list[PipelineFunction]]]:
        """Get all possible transitive paths for a specified output.

        This method retrieves all the possible ways the functions in the
        pipeline can be ordered (transitive paths) to produce a specified
        output.

        Parameters
        ----------
        output_name
            The name of the output variable to find paths for.
        simplify
            A flag indicating whether to simplify the pipeline before computing
            the paths. If True, the pipeline is first simplified to a sub-pipeline
            that is necessary for producing the output_name. If False, the paths
            are computed on the full pipeline. Default is False.

        Returns
        -------
        list[list[list[PipelineFunction]]]
            A list of lists of lists of `PipelineFunction`s. Each list of lists
            represents an independent chain of computation in the pipeline that
            can produce the output. Each list of `PipelineFunction` represents a
            possible ordering of functions in that chain.

        """
        pipeline = self.simplified_pipeline(output_name) if simplify else self

        func_only_graph = pipeline.graph.copy()
        root_args = pipeline.arg_combinations(output_name, root_args_only=True)
        for arg in root_args:
            func_only_graph.remove_node(arg)

        leaf = next(
            n
            for n in func_only_graph.nodes
            if output_name in _at_least_tuple(n.output_name)
        )
        roots = [n for n, d in func_only_graph.in_degree() if d == 0]
        graph = nx.transitive_reduction(func_only_graph)
        return [list(nx.all_simple_paths(graph, root, leaf)) for root in roots]

    @property
    def profiling_stats(self) -> dict[str, ProfilingStats]:
        """Return the profiling data for each function in the pipeline."""
        return {
            f.__name__: f.profiling_stats for f in self.functions if f.profiling_stats
        }

    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        pipeline_str = "Pipeline:\n"
        for node in self.graph.nodes:
            if isinstance(node, PipelineFunction):
                fn = node
                input_args = self.all_arg_combinations()[fn.output_name]
                pipeline_str += (
                    f"  {fn.output_name} = {fn.__name__}({', '.join(fn.parameters)})\n"
                )
                pipeline_str += f"    Possible input arguments: {input_args}\n"
        return pipeline_str


def _wrap_dict_to_tuple(
    func: Callable[..., Any],
    inputs: tuple[str, ...],
    output_name: str | tuple[str, ...],
) -> Callable[..., Any]:
    sig = inspect.signature(func)
    new_params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in inputs
    ]
    new_sig = sig.replace(parameters=new_params)

    def call(*args: Any, **kwargs: Any) -> Any:
        """Call the pipeline function with the root arguments."""
        bound = new_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        r = func(**bound.arguments)
        if isinstance(output_name, tuple):
            return tuple(r[k] for k in output_name)
        return r[output_name]

    call.__signature__ = new_sig  # type: ignore[attr-defined]

    return call


def _combine_nodes(
    combinable_nodes: dict[PipelineFunction, set[PipelineFunction]],
) -> dict[PipelineFunction, set[PipelineFunction]]:
    """Reduce the dictionary of combinable nodes to a minimal set.

    The input dictionary `combinable_nodes` indicates which nodes
    (functions in the pipeline) can be combined together. The dictionary
    keys are PipelineFunction objects, and the values are sets of
    PipelineFunction objects that the key depends on and can be
    combined with. For example,
    if `combinable_nodes = {f6: {f5}, f5: {f1, f4}}`, it means that `f6`
    can be combined with `f5`, and `f5` can be combined with `f1` and `f4`.

    This method simplifies the input dictionary by iteratively checking each
    node in the dictionary to see if it is a dependency of any other nodes.
    If it is, the method replaces that dependency with the node's own
    dependencies and removes the node from the dictionary. For example, if
    `f5` is found to be a dependency of `f6`, then `f5` is replaced by its
    own dependencies `{f1, f4}` in the `f6` entry,  and the `f5` entry is
    removed from the dictionary. This results in a new
    dictionary, `{f6: {f1, f4}}`.

    The aim is to get a dictionary where each node only depends on nodes
    that cannot be further combined. This simplified dictionary is useful for
    constructing a simplified graph of the computation.

    Parameters
    ----------
    combinable_nodes
        A dictionary where the keys are PipelineFunction objects, and the
        values are sets of PipelineFunction objects that can be combined
        with the key.

    Returns
    -------
    Dict[PipelineFunction, Set[PipelineFunction]]
        A simplified dictionary where each node only depends on nodes
        that cannot be further combined.

    """
    combinable_nodes = OrderedDict(combinable_nodes)
    for _ in range(len(combinable_nodes)):
        node, deps = combinable_nodes.popitem(last=False)
        added_nodes = []
        for _node, _deps in list(combinable_nodes.items()):
            if node in _deps:
                combinable_nodes[_node] |= deps
                added_nodes.append(_node)
        if not added_nodes:
            combinable_nodes[node] = deps
    return dict(combinable_nodes)


def _get_signature(
    combinable_nodes: dict[PipelineFunction, set[PipelineFunction]],
    graph: nx.DiGraph,
) -> tuple[dict[PipelineFunction, set[str]], dict[PipelineFunction, set[str]]]:
    """Retrieve the inputs and outputs for the signature of the combinable nodes.

    This function generates a mapping of the inputs and outputs required for
    each node in the combinable_nodes dictionary. For each node, it collects
    the outputs of all nodes it depends on and the parameters it and its
    dependent nodes require. In addition, it considers additional outputs
    based on the dependencies in the graph. It then filters these lists to
    ensure that no parameter is considered an output and no output is
    considered a parameter.

    Parameters
    ----------
    combinable_nodes
        Dictionary containing the nodes that can be combined together.
        The keys of the dictionary are the nodes that can be combined,
        and the values are sets of nodes that they depend on.
    graph
        The directed graph of the pipeline functions. Each node represents a
        function, and each edge represents a dependency relationship between
        functions.

    Returns
    -------
    all_inputs : dict[PipelineFunction, set[str]]
        Dictionary where keys are nodes and values are sets of parameter
        names that the node and its dependent nodes require.
    all_outputs : dict[PipelineFunction, set[str]]
        Dictionary where keys are nodes and values are sets of output names
        that the node and its dependent nodes produce, plus additional output
        names based on the dependency relationships in the graph.

    """
    all_inputs = {}
    all_outputs = {}
    for node, to_replace in combinable_nodes.items():
        outputs = set(_at_least_tuple(node.output_name))
        parameters = set(node.parameters)
        additional_outputs = set()  # parameters that are outputs to other functions
        for f in to_replace:
            outputs |= set(_at_least_tuple(f.output_name))
            parameters |= set(f.parameters)
            for successor in graph.successors(f):
                if successor not in to_replace and successor != node:
                    edge = graph.edges[f, successor]
                    additional_outputs |= set(_at_least_tuple(edge["arg"]))
        all_outputs[node] = (outputs - parameters) | additional_outputs
        all_inputs[node] = parameters - outputs
    return all_inputs, all_outputs


def generate_filename_from_dict(obj: dict[str, Any], suffix: str = ".pickle") -> Path:
    """Generate a filename from a dictionary."""
    keys = "_".join(obj.keys())
    obj_string = json.dumps(
        obj,
        sort_keys=True,
    )  # Convert the dictionary to a sorted string
    obj_bytes = obj_string.encode()  # Convert the string to bytes

    sha256_hash = hashlib.sha256()
    sha256_hash.update(obj_bytes)
    # Convert the hash to a hexadecimal string for the filename
    str_hash = sha256_hash.hexdigest()
    return Path(f"{keys}__{str_hash}{suffix}")
