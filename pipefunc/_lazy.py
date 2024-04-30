from __future__ import annotations

import contextlib
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Iterable,
    NamedTuple,
)

import networkx as nx

if TYPE_CHECKING:
    import sys

    if sys.version_info < (3, 9):  # pragma: no cover
        from typing import Callable
    else:
        from collections.abc import Callable


class _LazyFunction:
    """Lazy function wrapper for deferred evaluation of a function."""

    __slots__ = [
        "func",
        "args",
        "kwargs",
        "_result",
        "_evaluated",
        "_delayed_callbacks",
        "_id",
    ]

    _counter = 0

    def __init__(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}

        self._result = None
        self._evaluated = False
        self._delayed_callbacks: list[_LazyFunction] = []

        self._id = _LazyFunction._counter
        _LazyFunction._counter += 1

        if _TASK_GRAPH is not None:
            _TASK_GRAPH.graph.add_node(self._id, lazy_func=self)
            _TASK_GRAPH.mapping[self._id] = self

            def add_edge(arg: Any) -> None:
                if isinstance(arg, _LazyFunction):
                    _TASK_GRAPH.graph.add_edge(arg._id, self._id)
                elif isinstance(arg, Iterable):
                    for item in arg:
                        if isinstance(item, _LazyFunction):
                            _TASK_GRAPH.graph.add_edge(item._id, self._id)

            for arg in self.args:
                add_edge(arg)

            if kwargs is not None:
                for arg in kwargs.values():
                    add_edge(arg)

    def add_delayed_callback(self, cb: _LazyFunction) -> None:
        """Add a delayed callback to the lazy function."""
        self._delayed_callbacks.append(cb)

    def evaluate(self) -> Any:
        """Evaluate the lazy function and return the result."""
        if self._evaluated:
            return self._result
        args = evaluate_lazy(self.args)
        kwargs = evaluate_lazy(self.kwargs)
        result = self.func(*args, **kwargs)
        self._result = result
        self._evaluated = True
        for cb in self._delayed_callbacks:
            cb._result = result
            evaluate_lazy(cb)
        return result


class TaskGraph(NamedTuple):
    """A named tuple representing a task graph."""

    graph: nx.DiGraph
    mapping: dict[int, _LazyFunction]


@contextlib.contextmanager
def construct_dag() -> Generator[TaskGraph, None, None]:
    """Create a directed acyclic graph (DAG) for a pipeline."""
    global _TASK_GRAPH
    _TASK_GRAPH = TaskGraph(nx.DiGraph(), {})
    yield _TASK_GRAPH
    _TASK_GRAPH = None


_TASK_GRAPH: nx.DiGraph | None = None


def evaluate_lazy(x: Any) -> Any:
    """Evaluate a lazy object."""
    if isinstance(x, _LazyFunction):
        return x.evaluate()
    if isinstance(x, dict):
        return {k: evaluate_lazy(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(evaluate_lazy(v) for v in x)
    if isinstance(x, list):
        return [evaluate_lazy(v) for v in x]
    if isinstance(x, set):
        return {evaluate_lazy(v) for v in x}
    return x