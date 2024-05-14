from __future__ import annotations

import functools
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import cloudpickle


def at_least_tuple(x: Any) -> tuple[Any, ...]:
    """Convert x to a tuple if it is not already a tuple."""
    return x if isinstance(x, tuple) else (x,)


def generate_filename_from_dict(obj: dict[str, Any], suffix: str = ".pickle") -> Path:
    """Generate a filename from a dictionary."""
    assert all(isinstance(k, str) for k in obj)
    keys = "_".join(obj.keys())
    # Convert the dictionary to a sorted string
    obj_string = json.dumps(obj, sort_keys=True)
    obj_bytes = obj_string.encode()  # Convert the string to bytes

    sha256_hash = hashlib.sha256()
    sha256_hash.update(obj_bytes)
    # Convert the hash to a hexadecimal string for the filename
    str_hash = sha256_hash.hexdigest()
    return Path(f"{keys}__{str_hash}{suffix}")


def load(path: Path, *, cache: bool = False) -> Any:
    """Load a cloudpickled object from a path.

    If `cache` is True, the object will be cached in memory.
    """
    if cache:
        cache_key = _get_cache_key(path)
        return _cached_load(cache_key)

    with path.open("rb") as f:
        return cloudpickle.load(f)


def dump(obj: Any, path: Path) -> None:
    """Dump an object to a path using cloudpickle."""
    with path.open("wb") as f:
        cloudpickle.dump(obj, f)


def _get_cache_key(path: Path) -> tuple:
    """Generate a cache key based on the path, file modification time, and file size."""
    resolved_path = path.resolve()
    stats = resolved_path.stat()
    return (str(resolved_path), stats.st_mtime, stats.st_size)


@functools.lru_cache(maxsize=128)
def _cached_load(cache_key: tuple) -> Any:
    """Load a cloudpickled object using a cache key."""
    path = Path(cache_key[0])
    return load(path, cache=False)


def format_kwargs(kwargs: dict[str, Any]) -> str:
    """Format kwargs as a string."""
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


def handle_error(e: Exception, func: Callable, kwargs: dict[str, Any]) -> None:
    """Handle an error that occurred while executing a function."""
    kwargs_str = format_kwargs(kwargs)
    msg = f"Error occurred while executing function `{func.__name__}({kwargs_str})`."
    if sys.version_info <= (3, 11):  # pragma: no cover
        raise type(e)(e.args[0] + msg) from e
    e.add_note(msg)
    raise
