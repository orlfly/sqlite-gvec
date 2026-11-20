"""gvec Python bindings.

All engine logic is implemented in Rust (PyO3). This package only re-exports
the compiled extension module.
"""

from ._gvec import VERSION, Database, Distance, Graph, Vector

__all__ = ["VERSION", "Database", "Distance", "Graph", "Vector"]
