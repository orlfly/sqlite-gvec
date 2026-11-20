## Context

The existing Python package was a manual binding over the loadable SQLite
extension. It had three layers of duplicated or missing behavior:

1. ctypes path discovery and `sqlite3` connection handling.
2. `SELECT gvec_query(?)`, which the extension serves as a stub.
3. A pure-Python KNN loop that never touched the Rust indexes.

The Rust core already owns storage, graph execution, FTS, and ANN indexes.
PyO3 can expose that API directly, removing the duplicate Python layers.

## Goals / Non-Goals

**Goals**
- Replace all Python engine logic with Rust.
- Preserve the documented Python class surface (`Database`, `Graph`,
  `Vector`, `Distance`, `VERSION`).
- Embed SQLite via `rusqlite`, so no `--enable-load-extension` build is needed.
- Expose vector index and FTS APIs that the old wrapper did not have.
- Keep the main Rust workspace and CI unaffected by extension-module linking.

**Non-Goals**
- A free-threaded Python build or `abi3` wheel in this iteration.
- Removing the small import shim in `bindings/python/gvec/__init__.py`.
- Publishing prebuilt wheels to PyPI.
- Changing the loadable SQLite extension (`gvec-extension`) surface.

## Decisions

### D1. PyO3 module instead of ctypes
`crates/gvec-python` declares `[lib] name = "_gvec"` and
`crate-type = ["cdylib"]`, with `#[pymodule] fn _gvec`. Python imports
`gvec._gvec`; the package `__init__.py` only re-exports its names.

- Rationale: PyO3 handles reference counting, exceptions, and class
  conversion, and the engine is called through the normal Rust API.
- Alternative considered: keep ctypes and expose more C ABI from Rust.
  Rejected because ctypes still needs Python declarations, manual error
  handling, and cannot own a `rusqlite::Connection` cleanly.

### D2. Embed SQLite, do not load the extension
`Database` wraps `gvec::Database` (`rusqlite` with bundled SQLite). This makes
`Graph.run` work immediately and removes the `gvec_query()` stub problem and
the `--enable-load-extension` requirement.

### D3. Exclude the Python crate from the main workspace
PyO3 extension modules use `extension-module`, which changes linker behavior
and can break `cargo test --workspace` on machines without libpython. The
crate is listed under `workspace.exclude` and built with
`cargo --manifest-path` or maturin. The main workspace remains testable with
only a Rust toolchain.

### D4. Preserve the Python API, add native capabilities
`Vector` is a PyO3 class with a `values` list; `Distance` uses `#[classattr]`
constants; `Graph.knn` delegates to the Rust `knn_search`. New methods expose
`create_vector_index`, `vector_search`, `create_text_index`, and `text_search`.

### D5. Serialize results through serde_json::Value
`Graph.run` returns `serde_json::Value` rows from the core. A recursive
converter maps them to Python `None`, `bool`, `int`, `float`, `str`, `list`,
and `dict`. Index options are converted in the opposite direction.

## Risks / Trade-offs

- [PyO3 version churn] -> Pin 0.30 in the Python crate; it is isolated from
  the main workspace lockfile.
- [Python 3.14 support] -> Verified with CPython 3.14.2 and headers present.
- [Extension-module build] -> Documented both maturin and manual `cargo build`
  paths; CI uses a virtualenv plus maturin.
- [No abi3 wheel] -> Wheels are CPython-version-specific for now. `abi3` can
  be enabled later without changing the API.
- [Graph is `unsendable`] -> The `gvec::Graph` contains a non-Send executor
  trait object; PyO3's `unsendable` marker is correct for GIL-bound use.
