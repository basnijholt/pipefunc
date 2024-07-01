from __future__ import annotations

from typing import Any, Callable

from pipefunc._pipefunc import PipeFunc
from pipefunc._pipeline import _OUTPUT_TYPE, Pipeline


class Builder:
    def __init__(self):
        self.pipeline = Pipeline([])
        self._nodes = {}

    def add(self, *nodes: Node):
        for node in nodes:
            self.pipeline.add(node.func)


_CNT: dict[str, int] = {}


def _output_name(func: Callable) -> str:
    i = _CNT[func.__name__] = _CNT.get(func.__name__, 0)
    _CNT[func.__name__] = i + 1
    return f"{func.__name__}_{i}"


class Node:
    _id: int = 0

    def __init__(self, func: Callable, output_name: _OUTPUT_TYPE | None = None, **inputs: Any):
        assert not any(k in inputs for k in ["output_name", "func"])
        if output_name is None:
            output_name = _output_name(func)
        self.func = PipeFunc(func, output_name)
        renames = {k: v.func.output_name for k, v in inputs.items()}
        self.func.update_renames(renames)

    def __getattr__(self, name: str):
        return self.output_name[name]
