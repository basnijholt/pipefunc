from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Generator, Hashable, Sequence

import networkx as nx

if TYPE_CHECKING:
    from pipefunc._pipefunc import _OUTPUT_TYPE, Pipeline, PipelineFunction


def _at_least_tuple(x: Any) -> tuple[Any, ...]:
    """Convert x to a tuple if it is not already a tuple."""
    return x if isinstance(x, tuple) else (x,)


def _combined_exclude(
    *func: Callable[[Mapping[str, Any]], bool] | None,
) -> Callable[[Mapping[str, Any]], bool] | None:
    """Combine multiple exclude functions into one."""
    funcs = [f for f in func if f is not None]
    if len(funcs) == 0:
        return None
    if len(funcs) == 1:
        return funcs[0]
    return lambda x: any(func(x) for func in funcs)


def _combine_dicts(
    *maybe_dict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Combine multiple dictionaries into one."""
    dicts = [d for d in maybe_dict if d is not None]
    if len(dicts) == 0:
        return None
    if len(dicts) == 1:
        return dicts[0]
    # make sure no keys are repeated
    assert len(set().union(*dicts)) == sum(len(d) for d in dicts)
    return {k: v for d in dicts for k, v in d.items()}


class Sweep:
    """Create a sweep of a pipeline.

    Considering certain dimensions to be linked together (zipped) rather than
    forming a full Cartesian product. If 'dims' is not provided, a Cartesian
    product is formed.

    Parameters
    ----------
    items
        A dictionary where the key is the name of the dimension and the
        value is a sequence of values for that dimension.
    dims
        A list of tuples. Each tuple contains names of dimensions that are
        linked together. If not provided, a Cartesian product is formed.
    exclude
        A function that takes a dictionary of dimension values and returns
        True if the combination should be excluded from the sweep.
    constants
        A dictionary of constant values to be included in each combination.
    derivers
        A dictionary of functions to be applied to each
        dict. The dictionary keys are attribute names and the
        values are functions that take a dict as input and return
        a new attribute value. The keys might be a subset of the
        items keys, which means the values will be overwritten.

    Returns
    -------
    list
        A list of dictionaries, each representing a specific combination
        of dimension values.

    Examples
    --------
    >>> items = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
    >>> Sweep(items)
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 1, 'b': 3, 'c': 6}, {'a': 1, 'b': 4, 'c': 5}, {'a': 1, 'b': 4, 'c': 6},
     {'a': 2, 'b': 3, 'c': 5}, {'a': 2, 'b': 3, 'c': 6}, {'a': 2, 'b': 4, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]

    Which is equivalent to the following:

    >>> Sweep(items, dims=["a", "b", "c"]).list()
    >>> Sweep(items, dims=[("a",), ("b",), ("c",)]).list()

    Or zip together dimensions:

    >>> Sweep(items, dims=[('a', 'b'), ('c',)]).list()
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}].list()

    """

    def __init__(
        self,
        items: dict[str, Sequence[Any]],
        dims: list[str | tuple[str, ...]] | None = None,
        exclude: Callable[[Mapping[str, Any]], bool] | None = None,
        constants: Mapping[str, Any] | None = None,
        derivers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> None:
        """Initialize the sweep."""
        self.items = items
        self.dims = dims
        self.exclude = exclude
        self.constants = constants
        self.derivers = derivers

    def generate(self) -> Generator[dict[str, Any], None, None]:  # noqa: PLR0912
        if self.dims is None or set(self.dims) == self.items.keys():
            # Create the full Cartesian product if no dimensions are provided.
            names = self.items.keys()
            vals = self.items.values()
            for res in product(*vals):
                combination = dict(zip(names, res))
                if self.constants is not None:
                    for key, value in self.constants.items():
                        combination.setdefault(key, value)
                if self.derivers is not None:
                    for key, func in self.derivers.items():
                        combination[key] = func(combination)
                if self.exclude is None or not self.exclude(combination):
                    yield combination
        else:
            # Otherwise, create a product considering the provided dimensions.
            product_parts = []
            for dim_group in self.dims:
                dims = _at_least_tuple(dim_group)
                dim_seqs = [self.items[dim] for dim in dims]
                product_parts.append([dict(zip(dims, res)) for res in zip(*dim_seqs)])
            for combo in product(*product_parts):
                combination = {k: v for item in combo for k, v in item.items()}
                if self.constants is not None:
                    for key, value in self.constants.items():
                        combination.setdefault(key, value)
                if self.derivers is not None:
                    for key, func in self.derivers.items():
                        combination[key] = func(combination)
                if self.exclude is None or not self.exclude(combination):
                    yield combination

    def list(self) -> list[dict[str, Any]]:
        """Return the sweep as a list."""
        return list(self.generate())

    def filtered_sweep(self, keys: Iterable[str]) -> Sweep:  # noqa: PLR0912
        """Return the sweep as a list, but only include the specified keys in each dictionary, and remove duplicates."""
        if self.derivers is not None:
            ordered_set: dict[tuple[Hashable, ...], None] = {}
            for combo in self.generate():
                filtered_combo = {k: combo[k] for k in keys}
                key = tuple(filtered_combo.values())
                try:
                    ordered_set[key] = None
                except TypeError:
                    msg = "All items must be hashable when using `derivers` and `filtered_sweep`."
                    raise TypeError(msg) from None
            new_items: dict[str, list[Hashable]] = {}
            for item in ordered_set:
                for k, v in zip(keys, item):
                    new_items.setdefault(k, []).append(v)
            return Sweep(
                items=new_items,  # type: ignore[arg-type]
                dims=[tuple(keys)],
            )

        dims: list[str | tuple[str, ...]]
        if self.dims is None or set(self.dims) == self.items.keys():
            dims = [k for k in self.items if k in keys]
        else:
            dims = []
            for dim_group in self.dims:
                if isinstance(dim_group, str) and dim_group in keys:
                    dims.append(dim_group)
                elif isinstance(dim_group, tuple):
                    _dims = tuple(k for k in dim_group if k in keys)
                    if _dims:
                        if len(_dims) == 1:
                            dims.append(_dims[0])
                        else:
                            dims.append(_dims)
        return Sweep(
            self.items,
            dims=dims,
            exclude=self.exclude,
            constants=self.constants,
            derivers=None,
        )

    def __len__(self) -> int:
        """Return the number of unique combinations in the sweep."""
        if self.exclude is not None:
            return len(self.list())
        if self.dims is None or set(self.dims) == self.items.keys():
            # Full Cartesian product; simply multiply together lengths of each dimension
            total_length = 1
            for value in self.items.values():
                total_length *= len(value)
            return total_length
        # Otherwise, calculate lengths considering the provided dimensions.
        total_length = 1
        for dim_group in self.dims:
            dims = _at_least_tuple(dim_group)
            group_length = len(self.items[dims[0]])
            total_length *= group_length
        return total_length

    def __add__(self, other: Sweep) -> MultiSweep:
        """Combine this Sweep with another one, creating a MultiSweep."""
        if not isinstance(other, Sweep):
            msg = "Other object must be a Sweep or a MultiSweep instance."
            raise TypeError(msg)
        return MultiSweep(self, other)

    def combine(self, other: Sweep) -> MultiSweep:
        """Add another sweep to this MultiSweep."""
        return other + self

    def product(self, *others: Sweep) -> Sweep:
        """Create a Cartesian product of this Sweep with other Sweeps.

        Parameters
        ----------
        *others
            One or more Sweep objects to create a Cartesian product with.

        Returns
        -------
        Sweep
            A new Sweep object representing the Cartesian product of the sweeps.

        Examples
        --------
        >>> sweep1 = Sweep({'a': [1, 2], 'b': [3, 4]})
        >>> sweep2 = Sweep({'c': [5, 6]})
        >>> sweep3 = sweep1.product(sweep2)
        >>> sweep3.list()
        [{'a': 1, 'b': 3, 'c': 5}, {'a': 1, 'b': 3, 'c': 6},
         {'a': 1, 'b': 4, 'c': 5}, {'a': 1, 'b': 4, 'c': 6},
         {'a': 2, 'b': 3, 'c': 5}, {'a': 2, 'b': 3, 'c': 6},
         {'a': 2, 'b': 4, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]

        """
        items = self.items.copy()
        dims = self.dims.copy() if self.dims is not None else None

        for other in others:
            if not isinstance(other, Sweep):
                msg = "All arguments must be Sweep instances."
                raise TypeError(msg)
            items.update(other.items)
            if dims is not None:
                if other.dims is not None:
                    dims.extend(other.dims)
                else:
                    dims.extend(list(other.items.keys()))

        return Sweep(
            items,
            dims=dims,
            exclude=_combined_exclude(self.exclude, other.exclude),
            constants=_combine_dicts(self.constants, other.constants),  # type: ignore[arg-type]
            derivers=_combine_dicts(self.derivers, other.derivers),  # type: ignore[arg-type]
        )

    def add_derivers(
        self,
        **derivers: Callable[[dict[str, Any]], Any],
    ) -> Sweep:
        """Add derivers to the sweep, which are functions that modify the sweep items.

        Parameters
        ----------
        derivers
            A dictionary of functions to be applied to each
            dict. The dictionary keys are attribute names and the
            values are functions that take a dict as input and return
            a new attribute value. The keys might be a subset of the
            items keys, which means the values will be overwritten.

        Returns
        -------
        Sweep
            A new Sweep object with the added derivers.

        Examples
        --------
        >>> sweep = Sweep({'a': [1], 'b': [2, 3]})
        >>> sweep = sweep.add_derivers(c=lambda x: x['a'] + x['b'])
        >>> sweep.list()
        [{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 3, 'c': 4}]

        """
        return Sweep(
            self.items,
            dims=self.dims,
            exclude=self.exclude,
            constants=self.constants,
            derivers=derivers,
        )


class MultiSweep(Sweep):
    """A class to concatenate multiple sweeps into one.

    Parameters
    ----------
    *sweeps
        Sweep objects to be concatenated together.

    Returns
    -------
    MultiSweep
        A MultiSweep object containing the concatenated sweeps.

    Examples
    --------
    >>> sweep1 = Sweep({'a': [1, 2], 'b': [3, 4]})
    >>> sweep2 = Sweep({'x': [5, 6], 'y': [7, 8]})
    >>> multi_sweep = MultiSweep(sweep1, sweep2)

    """

    def __init__(self, *sweeps: Sweep) -> None:
        super().__init__(items={})
        self.sweeps = list(sweeps)

    def generate(self) -> Generator[dict[str, Any], None, None]:
        for sweep in self.sweeps:
            yield from sweep.generate()

    def __len__(self) -> int:
        return sum(len(sweep) for sweep in self.sweeps)

    def list(self) -> list[dict[str, Any]]:
        """Return the sweep as a list."""
        return list(self.generate())

    def filtered_sweep(self, keys: Iterable[str]) -> MultiSweep:
        """Return a new MultiSweep, but only include the specified keys in each dictionary, and remove duplicates."""
        filtered_sweeps = [sweep.filtered_sweep(keys) for sweep in self.sweeps]
        return MultiSweep(*filtered_sweeps)

    def __add__(self, other: Sweep) -> MultiSweep:
        """Add another sweep to this MultiSweep."""
        if not isinstance(other, Sweep):
            msg = "Other object must be a Sweep or a MultiSweep instance."
            raise TypeError(msg)
        return self.combine(other)

    def combine(self, other: Sweep) -> MultiSweep:
        """Add another sweep to this MultiSweep."""
        if isinstance(other, MultiSweep):
            self.sweeps.extend(other.sweeps)
        else:
            self.sweeps.append(other)
        return self


def generate_sweep(
    items: dict[str, Sequence[Any]],
    dims: list[str | tuple[str, ...]] | None = None,
    exclude: Callable[[Mapping[str, Any]], bool] | None = None,
    constants: Mapping[str, Any] | None = None,
    derivers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create a sweep of a pipeline.

    Considering certain dimensions to be linked together (zipped) rather than
    forming a full Cartesian product. If 'dims' is not provided, a Cartesian
    product is formed.

    Parameters
    ----------
    items
        A dictionary where the key is the name of the dimension and the
        value is a sequence of values for that dimension.
    dims
        A list of tuples. Each tuple contains names of dimensions that are
        linked together. If not provided, a Cartesian product is formed.
    exclude
        A function that takes a dictionary of dimension values and returns
        True if the combination should be excluded from the sweep.
    constants
        A dictionary with constant values that should be added to each
        combination.
    derivers
        A dictionary of functions to be applied to each
        dict. The dictionary keys are attribute names and the
        values are functions that take a dict as input and return
        a new attribute value. The keys might be a subset of the
        items keys, which means the values will be overwritten.

    Returns
    -------
    list
        A list of dictionaries, each representing a specific combination
        of dimension values.

    Examples
    --------
    >>> items = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
    >>> generate_sweep(items)
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 1, 'b': 3, 'c': 6}, {'a': 1, 'b': 4, 'c': 5}, {'a': 1, 'b': 4, 'c': 6},
     {'a': 2, 'b': 3, 'c': 5}, {'a': 2, 'b': 3, 'c': 6}, {'a': 2, 'b': 4, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]

    Which is equivalent to the following:

    >>> generate_sweep(items, dims=["a", "b", "c"])
    >>> generate_sweep(items, dims=[("a",), ("b",), ("c",)])

    Or zip together dimensions:

    >>> generate_sweep(items, dims=[('a', 'b'), ('c',)])
    [{'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 4, 'c': 6}]

    """
    return Sweep(items, dims, exclude, constants, derivers).list()


def count_sweep(
    output_name: str,
    sweep: list[dict[str, Any]] | Sweep,
    pipeline: Pipeline,
    *,
    use_pandas: bool = False,
) -> dict[str | tuple[str, ...], dict[tuple[Any, ...], int]]:
    """Count the number of times each argument combination is used.

    Useful for determining which functions to execute first or which
    functions to cache.

    Parameters
    ----------
    output_name
        The name of the output to count the argument combinations for.
    sweep
        A list of dictionaries, each representing a values for which
        the function is called. Or a Sweep object.
    pipeline
        The pipeline to count the argument combinations for.
    use_pandas
        Whether to use pandas to create the counts. Note that this is
        slower than the default method for sweeps <≈ 1e6.

    Returns
    -------
    dict
        A dictionary where the keys are the names of the arguments and the
        values are dictionaries where the keys are the argument combinations
        and the values are the number of times that combination is used.

    """
    if isinstance(sweep, Sweep):
        # TODO: we can likely special case this to be faster.
        sweep = sweep.list()  # type: ignore[assignment]
    assert isinstance(sweep, Iterable)
    counts: dict[_OUTPUT_TYPE, dict[tuple[Any, ...], int]] = {}
    deps = pipeline.func_dependencies(output_name)
    for _output_name in deps:
        arg_combination = pipeline.arg_combinations(_output_name, root_args_only=True)
        assert isinstance(arg_combination, tuple)
        if use_pandas:
            import pandas as pd

            df = pd.DataFrame(list(sweep))
            cols = list(arg_combination)
            counts[_output_name] = df[cols].groupby(cols).size().to_dict()  # type: ignore[assignment]
        else:
            _cnt: dict[tuple[Any, ...], int] = {}
            for combo in sweep:
                key = tuple(combo[arg] for arg in arg_combination)
                _cnt[key] = _cnt.get(key, 0) + 1
            counts[_output_name] = _cnt
    return dict(counts)


def set_cache_for_sweep(
    output_name: str,
    pipeline: Pipeline,
    sweep: list[dict[str, Any]],
    min_executions: int = 2,
    *,
    verbose: bool = False,
) -> None:
    """Set the cache for a sweep of a pipeline."""
    # Disable for the output node
    pipeline.node_mapping[output_name].cache = False  # type: ignore[union-attr]
    cnt = count_sweep(output_name, sweep, pipeline)
    max_executions = {k: max(v.values()) for k, v in cnt.items()}
    for _output_name, n in max_executions.items():
        enable_cache = n >= min_executions
        func = pipeline.node_mapping[_output_name]
        if verbose:
            print(f"Setting cache for '{_output_name}' to {enable_cache} (n={n})")
        func.cache = enable_cache  # type: ignore[union-attr]


def get_precalculation_order(
    pipeline: Pipeline,
    counts: dict[str | tuple[str, ...], dict[tuple[Any, ...], int]],
    min_executions: int = 2,
) -> list[PipelineFunction]:
    """Determine the order in which functions in a pipeline should be precalculated and cached.

    The order is determined by the topological dependencies of the functions
    and the count of their executions in the context of a parameter sweep.
    Only functions that are executed multiple times (as specified by `min_executions`)
    are included in the precalculation order.

    Parameters
    ----------
    pipeline
        The pipeline of functions.
    counts
        A dictionary mapping function output names to dictionaries of
        parameter combinations and their counts in the pipeline.
    min_executions
        The minimum number of times a function must be used in the pipeline
        for it to be included in the precalculation order. Defaults to 2.

    Returns
    -------
    list[PipelineFunction]
        The ordered list of functions to be precalculated and cached.

    """

    def key_func(node: PipelineFunction) -> int:
        return -sum(counts[node.output_name].values())

    m = pipeline.node_mapping
    # Get nodes with counts ≥min_executions
    nodes_with_counts = [
        m[node]
        for node, count_dict in counts.items()
        if any(val >= min_executions for val in count_dict.values())
    ]
    # Create a subgraph with only the nodes with sufficient counts
    subgraph = pipeline.graph.subgraph(nodes_with_counts)
    # Return the ordered list of nodes
    return list(nx.lexicographical_topological_sort(subgraph, key=key_func))


def _get_nested_value(d: dict, keys: list) -> Any:
    """Fetches a nested value from a dictionary given a list of keys.

    Parameters
    ----------
    d
        The dictionary from which to fetch the value.
    keys
        The list of keys that lead to the desired value. Each element in the
        list is a key to a subsequent dictionary.

    Returns
    -------
    Any
        The value found at the nested location in the dictionary specified by
        the list of keys.

    Raises
    ------
    KeyError
        If a key in the list does not exist in the dictionary.

    Examples
    --------
    >>> d = {'a': {'b': {'c': 1}}}
    >>> keys = ['a', 'b', 'c']
    >>> _get_nested_value(d, keys)
    1

    """
    for key in keys:
        d = d[key]
    return d


def _nested_defaultdict(n: int, inner_default_factory: type = list) -> defaultdict:
    """Create a nested defaultdict with n levels."""
    if n <= 1:
        return defaultdict(inner_default_factory)
    return defaultdict(lambda: _nested_defaultdict(n - 1, inner_default_factory))


def _count_nested_dict_items(d: dict) -> int:
    """Count the number of items in a nested dict."""
    total = 0
    for v in d.values():
        if isinstance(v, dict):
            total += _count_nested_dict_items(v)
        else:
            total += len(v)
    return total


def _assert_valid_sweep_dict(
    sweep_dict: dict,
    left_overs: list[tuple[str, ...]],
    keys: defaultdict[tuple[str, ...], set] | None = None,
    level: int = 0,
) -> defaultdict[tuple[str, ...], set]:
    """Checks that the keys are unique and that the number of keys matches the number of left_overs."""
    if keys is None:
        keys = defaultdict(set)
    for k, v in sweep_dict.items():
        assert len(k) == len(left_overs[level])
        assert k not in keys[tuple(left_overs[level])]
        keys[left_overs[level]].add(k)
        if isinstance(v, dict):
            _assert_valid_sweep_dict(v, left_overs, keys, level + 1)
    return keys


def get_min_sweep_sets(
    execution_order: list[PipelineFunction],
    pipeline: Pipeline,
    sweep: Sweep,
    output_name: str,
) -> tuple[list[tuple[str, ...]], dict]:
    """Create sweep combinations for each function in the execution order.

    This function iterates through the execution order of the pipeline functions
    and identifies the minimal set of argument combinations for each function to
    satisfy the pipeline sweep requirements. It then creates a nested dictionary
    structure that represents the sweep combinations.

    Parameters
    ----------
    execution_order
        The order in which pipeline functions are to be executed.
    pipeline
        The pipeline object containing the pipeline functions and related
        methods.
    sweep
        The sweep object containing the sweep parameter values and combinations.
    output_name
        The name of the output from the pipeline function we are starting the
        simplification from. It is used to get the starting function in the pipeline.

    Returns
    -------
    Tuple[List[Any], dict]
        A tuple where the first element is a list of the minimal set of argument
        combinations for each function in the execution order and the second
        element is a nested dictionary that represents the sweep combinations.

    """
    root_args = pipeline.arg_combinations(output_name, root_args_only=True)
    assert isinstance(root_args, tuple)
    root_args_set = set(root_args)
    left_over = root_args_set.copy()

    left_overs: list[tuple[str, ...]] = []
    for f in execution_order:
        f_root_args = pipeline.arg_combinations(f.output_name, root_args_only=True)
        assert isinstance(f_root_args, tuple)
        left_over = left_over - set(f_root_args)
        if not left_over:
            break
        args = tuple(sorted(root_args_set - left_over))
        if args not in left_overs:
            left_overs.append(args)

    sweep_dict = _nested_defaultdict(len(left_overs), inner_default_factory=list)
    for combo in sweep.generate():
        keys = [tuple(combo[k] for k in arg) for arg in left_overs]
        _get_nested_value(sweep_dict, keys).append(combo)

    assert _count_nested_dict_items(sweep_dict) == len(sweep)
    _assert_valid_sweep_dict(sweep_dict, left_overs)
    return left_overs, sweep_dict
