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
import datetime
import functools
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, Union, get_type_hints

import cloudpickle

from pipefunc._perf import ProfilingStats, ResourceProfiler
from pipefunc._utils import at_least_tuple, clear_cached_properties, format_function_call
from pipefunc.lazy import evaluate_lazy
from pipefunc.map._mapspec import MapSpec

if TYPE_CHECKING:
    from pathlib import Path


T = TypeVar("T", bound=Callable[..., Any])
_OUTPUT_TYPE: TypeAlias = Union[str, tuple[str, ...]]
MAX_PARAMS_LEN = 15


def _default_output_picker(
    output: Any,
    name: str,
    output_name: _OUTPUT_TYPE,
) -> Any:
    """Default output picker function for tuples."""
    return output[output_name.index(name)]


class PipeFunc(Generic[T]):
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
    defaults
        Set defaults for parameters. Overwrites any current defaults. Must be in terms
        of the renamed argument names.
    profile
        Flag indicating whether the wrapped function should be profiled.
    debug
        Flag indicating whether debug information should be printed.
    cache
        Flag indicating whether the wrapped function should be cached.
    save_function
        A function that takes the filename and a dict containing the inputs and output.
        If provided, the result will be saved.
    mapspec
        This is a specification for mapping that dictates how input values should
        be merged together. If None, the default behavior is that the input directly
        maps to the output.

    Returns
    -------
        The identifier for the output of the wrapped function.

    Examples
    --------
    >>> def add_one(a, b):
    ...     return a + 1, b + 1
    >>> add_one_func = PipeFunc(
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
        defaults: dict[str, Any] | None = None,
        profile: bool = False,
        debug: bool = False,
        cache: bool = False,
        save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
        mapspec: str | MapSpec | None = None,
    ) -> None:
        """Function wrapper class for pipeline functions with additional attributes."""
        self.func: Callable[..., Any] = func
        self.output_name: _OUTPUT_TYPE = output_name
        self.debug = debug
        self.cache = cache
        self.save_function = save_function
        self.mapspec = MapSpec.from_string(mapspec) if isinstance(mapspec, str) else mapspec
        self.output_picker: Callable[[Any, str], Any] | None = output_picker
        if output_picker is None and isinstance(output_name, tuple):
            self.output_picker = functools.partial(
                _default_output_picker,
                output_name=self.output_name,
            )
        self._profile = profile
        self._renames: dict[str, str] = renames or {}
        self._defaults = defaults or {}
        self.profiling_stats: ProfilingStats | None
        self.set_profiling(enable=profile)
        self._validate_mapspec()

    @property
    def renames(self) -> dict[str, str]:
        """Return the renames for the function arguments.

        See Also
        --------
        update_renames
            Update the `renames` via this method.

        """
        # Is a property to prevent users mutating the renames directly
        return self._renames

    @functools.cached_property
    def parameters(self) -> tuple[str, ...]:
        parameters = inspect.signature(self.func).parameters
        return tuple(self._renames.get(k, k) for k in parameters)

    @functools.cached_property
    def defaults(self) -> dict[str, Any]:
        """Return the defaults for the function arguments.

        Returns
        -------
            A dictionary of default values for the keyword arguments.

        See Also
        --------
        update_defaults
            Update the `defaults` via this method.

        """
        parameters = inspect.signature(self.func).parameters
        if extra := set(self._defaults) - set(self.parameters):
            allowed = ", ".join(parameters)
            msg = (
                f"Unexpected default arguments: `{extra}`."
                f" The allowed arguments are: `{allowed}`."
                " Defaults must be in terms of the renamed argument names."
            )
            raise ValueError(msg)
        defaults = {}
        for original_name, v in parameters.items():
            new_name = self._renames.get(original_name, original_name)
            if new_name in self._defaults:
                defaults[new_name] = self._defaults[new_name]
            elif v.default is not inspect.Parameter.empty:
                defaults[new_name] = v.default
        return defaults

    @functools.cached_property
    def _inverse_renames(self) -> dict[str, str]:
        return {v: k for k, v in self._renames.items()}

    def update_defaults(self, defaults: dict[str, Any], *, overwrite: bool = False) -> None:
        """Update defaults to the provided keyword arguments.

        Parameters
        ----------
        defaults
            A dictionary of default values for the keyword arguments.
        overwrite
            Whether to overwrite the existing defaults. If `False`, the new
            defaults will be added to the existing defaults.

        """
        if overwrite:
            self._defaults = defaults.copy()
        else:
            self._defaults = dict(self._defaults, **defaults)
        clear_cached_properties(self)

    def update_renames(self, renames: dict[str, str], *, overwrite: bool = False) -> None:
        """Update renames to function arguments for the wrapped function.

        Parameters
        ----------
        renames
            A dictionary of renames for the function arguments.
        overwrite
            Whether to overwrite the existing renames. If `False`, the new
            renames will be added to the existing renames.

        """
        old_inverse = self._inverse_renames
        if overwrite:
            self._renames = renames.copy()
        else:
            self._renames = dict(self._renames, **renames)

        # Update defaults with new renames
        new_defaults = {}
        for name, value in self._defaults.items():
            if original_name := old_inverse.get(name):
                name = self._renames.get(original_name, original_name)  # noqa: PLW2901
            new_defaults[name] = value
        self._defaults = new_defaults

        clear_cached_properties(self)

    def copy(self) -> PipeFunc:
        return PipeFunc(
            self.func,
            self.output_name,
            output_picker=self.output_picker,
            renames=self._renames,
            defaults=self.defaults,
            profile=self.profile,
            debug=self.debug,
            cache=self.cache,
            save_function=self.save_function,
            mapspec=self.mapspec,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function with the given arguments.

        Returns
        -------
            The return value of the wrapped function.

        """
        if extra := set(kwargs) - set(self.parameters):
            msg = (
                f"Unexpected keyword arguments: `{extra}`."
                f" The allowed arguments are: `{self.parameters}`."
                f" The provided arguments are: `{kwargs}`."
            )
            raise ValueError(msg)
        defaults = {k: v for k, v in self.defaults.items() if k not in kwargs}
        kwargs.update(defaults)
        kwargs = {self._inverse_renames.get(k, k): v for k, v in kwargs.items()}

        with self._maybe_profiler():
            args = evaluate_lazy(args)
            kwargs = evaluate_lazy(kwargs)
            result = self.func(*args, **kwargs)

        if self.debug:
            func_str = format_function_call(self.func.__name__, (), kwargs)
            now = datetime.datetime.now()  # noqa: DTZ005
            msg = (
                f"{now} - Function returning '{self.output_name}' was invoked"
                f" as `{func_str}` and returned `{result}`."
            )
            if self.profiling_stats is not None:
                dt = self.profiling_stats.time.average
                msg += f" The execution time was {dt:.2e} seconds on average."
            print(msg)
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

    @functools.cached_property
    def parameter_annotations(self) -> dict[str, Any]:
        """Return the type annotations of the wrapped function's parameters."""
        type_hints = get_type_hints(self.func)
        return {self.renames.get(k, k): v for k, v in type_hints.items() if k != "return"}

    @functools.cached_property
    def output_annotation(self) -> Any:
        """Return the type annotation of the wrapped function's output."""
        return get_type_hints(self.func).get("return", Any)

    def _maybe_profiler(self) -> contextlib.AbstractContextManager:
        """Maybe get profiler.

        Get a profiler instance if profiling is enabled, otherwise
        return a dummy context manager.

        Returns
        -------
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
            The value of the attribute.

        """
        return getattr(self.func, name)

    def __str__(self) -> str:
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
            A string representation of the PipeFunc instance.

        """
        outputs = ", ".join(at_least_tuple(self.output_name))
        return f"{self.func.__name__}(...) → {outputs}"

    def __repr__(self) -> str:
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
            A string representation of the PipeFunc instance.

        """
        return f"PipeFunc({self.func.__name__})"

    def __getstate__(self) -> dict:
        """Prepare the state of the current object for pickling.

        The state includes all picklable instance variables.
        For non-picklable instance variable,  they are transformed
        into a picklable form or ignored.

        Returns
        -------
            A dictionary containing the picklable state of the object.

        """
        state = {k: v for k, v in self.__dict__.items() if k != "func"}
        state["func"] = cloudpickle.dumps(self.func)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the current object from the provided state.

        It also handles restoring non-picklable instance variable
        into their original form.

        Parameters
        ----------
        state
            A dictionary containing the picklable state of the object.

        """
        self.__dict__.update(state)
        self.func = cloudpickle.loads(self.func)

    def _validate_mapspec(self) -> None:
        if self.mapspec is None:
            return

        if not isinstance(self.mapspec, MapSpec):  # pragma: no cover
            msg = (
                "The 'mapspec' argument should be an instance of MapSpec,"
                f" not {type(self.mapspec)}."
            )
            raise TypeError(msg)

        mapspec_input_names = set(self.mapspec.input_names)
        if extra := mapspec_input_names - set(self.parameters):
            msg = (
                f"The input of the function `{self.__name__}` should match"
                f" the input of the MapSpec `{self.mapspec}`:"
                f" `{extra} not in {self.parameters}`."
            )
            raise ValueError(msg)

        mapspec_output_names = set(self.mapspec.output_names)
        output_names = set(at_least_tuple(self.output_name))
        if mapspec_output_names != output_names:
            msg = (
                f"The output of the function `{self.__name__}` should match"
                f" the output of the MapSpec `{self.mapspec}`:"
                f" `{mapspec_output_names} != {output_names}`."
            )
            raise ValueError(msg)


def pipefunc(
    output_name: _OUTPUT_TYPE,
    *,
    output_picker: Callable[[Any, str], Any] | None = None,
    renames: dict[str, str] | None = None,
    defaults: dict[str, Any] | None = None,
    profile: bool = False,
    debug: bool = False,
    cache: bool = False,
    save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
    mapspec: str | MapSpec | None = None,
) -> Callable[[Callable[..., Any]], PipeFunc]:
    """A decorator that wraps a function in a PipeFunc instance.

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
    defaults
        Set defaults for parameters. Overwrites any current defaults. Must be in terms
        of the renamed argument names.
    profile
        Flag indicating whether the decorated function should be profiled.
    debug
        Flag indicating whether debug information should be printed.
    cache
        Flag indicating whether the decorated function should be cached.
    save_function
        A function that takes the filename and a dict containing the inputs and output.
        If provided, the result will be saved.
    mapspec
        This is a specification for mapping that dictates how input values should
        be merged together. If None, the default behavior is that the input directly
        maps to the output.

    Returns
    -------
        A decorator function that takes the original function and output_name a
        PipeFunc instance with the specified return identifier.

    """

    def decorator(f: Callable[..., Any]) -> PipeFunc:
        """Wraps the original function in a PipeFunc instance.

        Parameters
        ----------
        f
            The original function to be wrapped.

        Returns
        -------
            The wrapped function with the specified return identifier.

        """
        return PipeFunc(
            f,
            output_name,
            output_picker=output_picker,
            renames=renames,
            defaults=defaults,
            profile=profile,
            debug=debug,
            cache=cache,
            save_function=save_function,
            mapspec=mapspec,
        )

    return decorator
