"""gvec — Python bindings for the gvec SQLite extension.

This module loads `libgvec.so`/`libgvec.dylib` via ctypes and exposes a
high-level Python API that mirrors the Rust binding.

Install the compiled `.so` somewhere on `LD_LIBRARY_PATH` or pass an
explicit path to `load()`.
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "load",
    "Database",
    "Graph",
    "Vector",
    "Distance",
    "VERSION",
]

VERSION = "0.1.0"


def _candidate_paths() -> List[str]:
    paths: List[str] = []
    env = os.environ.get("GVEC_LIB_PATH")
    if env:
        paths.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("libgvec_extension.so", "libgvec.so", "libgvec_extension.dylib", "libgvec.dylib"):
        paths.append(os.path.join(here, name))
        paths.append(os.path.join(here, "..", "..", "target", "release", name))
        paths.append(os.path.join(here, "..", "..", "target", "debug", name))
    return [p for p in paths if os.path.exists(p)]


def load(path: Optional[str] = None) -> ctypes.CDLL:
    """Load the gvec extension shared library."""
    if path is None:
        cands = _candidate_paths()
        if not cands:
            raise FileNotFoundError(
                "gvec shared library not found. Set GVEC_LIB_PATH or pass path=."
            )
        path = cands[0]
    return ctypes.CDLL(path)


class _Lib:
    def __init__(self) -> None:
        self._handle: Optional[ctypes.CDLL] = None

    def __get__(self, instance: Any, owner: Any) -> ctypes.CDLL:
        if self._handle is None:
            self._handle = load()
        return self._handle

    def reset(self) -> None:
        self._handle = None


_lib_singleton = _Lib()


class Distance:
    L2 = "l2"
    COSINE = "cosine"
    INNER_PRODUCT = "ip"


@dataclass
class Vector:
    values: Tuple[float, ...]

    def to_json(self) -> str:
        return json.dumps(list(self.values))


class Database:
    """Opens an in-memory SQLite database, loads the extension, and exposes
    a `Graph` for Cypher-like queries.

    Note: this uses Python's built-in `sqlite3` module which loads the
    extension via `conn.enable_load_extension(True)` + `conn.load_extension(...)`.
    """

    def __init__(self, path: str = ":memory:") -> None:
        import sqlite3

        self.path = path
        self._sqlite3 = sqlite3
        self._conn = sqlite3.connect(path)
        try:
            self._conn.enable_load_extension(True)
        except Exception as exc:  # pragma: no cover - depends on build
            raise RuntimeError(
                "sqlite3 was not built with --enable-load-extension"
            ) from exc
        try:
            self._conn.load_extension(_lib_singleton._handle.name)
        except Exception:
            # Fall back to default discovery if the explicit path didn't work
            self._conn.load_extension()
        # register gvec_version, etc.
        cur = self._conn.execute("SELECT gvec_version()")
        self.version = cur.fetchone()[0]

    def graph(self, name: str = "g") -> "Graph":
        return Graph(self._conn, name)

    def close(self) -> None:
        self._conn.close()


class Graph:
    """A named graph stored in the connected SQLite database."""

    def __init__(self, conn: Any, name: str) -> None:
        self._conn = conn
        self.name = name

    def run(self, cypher: str) -> List[List[Any]]:
        """Run a Cypher-like query via the `gvec_query` SQL function and
        return result rows as a list of lists of JSON-decoded values."""
        cur = self._conn.execute(
            "SELECT gvec_query(?)", (cypher,),
        )
        row = cur.fetchone()
        if not row:
            return []
        data = json.loads(row[0])
        if isinstance(data, list):
            return data
        return [data]

    def knn(
        self,
        query: Vector,
        stored: Sequence[Tuple[int, Vector]],
        k: int = 10,
        metric: str = Distance.L2,
    ) -> List[Tuple[int, float]]:
        # Distance kernels are pure-Rust, accessible via gvec_dist. Compute
        # pairwise and rank.
        import math

        def dot(a: Vector, b: Vector) -> float:
            return sum(x * y for x, y in zip(a.values, b.values))

        def l2(a: Vector, b: Vector) -> float:
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.values, b.values)))

        def cos(a: Vector, b: Vector) -> float:
            return 1.0 - (dot(a, b) / (math.sqrt(dot(a, a)) * math.sqrt(dot(b, b)) or 1.0))

        scores: List[Tuple[int, float]] = []
        for rid, vec in stored:
            if metric == Distance.L2:
                d = l2(query, vec)
            elif metric == Distance.COSINE:
                d = cos(query, vec)
            else:
                d = -dot(query, vec)
            scores.append((rid, d))
        scores.sort(key=lambda x: x[1])
        return scores[:k]
