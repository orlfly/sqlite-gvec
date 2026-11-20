## Why

`bindings/python/gvec/__init__.py` previously contained a ctypes loader, a
`sqlite3` connection wrapper, and a pure-Python brute-force KNN
implementation. That code duplicated engine logic, did not use the Rust HNSW
or IVF indexes, and called `gvec_query()`, which is a stub in the loadable
extension. As a result, Python graph queries did not work and vector search
was limited to in-memory arrays supplied by the caller.

All of that logic can live in Rust. PyO3 exposes the existing `gvec` crate
directly to Python, embedding SQLite and the full engine in-process.

## What Changes

- Add `crates/gvec-python`, a PyO3 `cdylib` module named `gvec._gvec`.
- Implement the Python API in Rust:
  - `Database(path=":memory:")`, `graph(name)`, `close()`, `version`, `path`
  - `Graph.run(cypher)`, `name`
  - `Graph.knn(...)` retained for API parity, executed by Rust
  - `Graph.create_vector_index`, `drop_vector_index`, `list_vector_indexes`,
    `vector_count`, `vector_search`, `exact_vector_search`
  - `Graph.create_text_index`, `drop_text_index`, `text_search`
  - `Vector`, `Distance` (`L2`, `COSINE`, `INNER_PRODUCT`), `VERSION`
- Replace the Python implementation with a nine-line import shim that
  re-exports `gvec._gvec`.
- Add `pyproject.toml` for maturin packaging and a CI job that builds the
  wheel in a virtualenv and runs the smoke test.
- Update `test_smoke.py` to exercise graph, KNN, ANN indexes, FTS, and hybrid
  search through the Rust extension.

## Capabilities

### Modified Capabilities

- `language-bindings`: the Python binding is now a Rust/PyO3 extension with no
  Python engine logic. It uses the in-process `gvec` crate rather than loading
  a SQLite extension through ctypes.

## Impact

- New crate: `crates/gvec-python` (excluded from the main Cargo workspace so
  PyO3's extension-module linking does not affect `cargo test --workspace`).
- New dependency: `pyo3` 0.30.
- New packaging file: `pyproject.toml`.
- Replaced: `bindings/python/gvec/__init__.py` (172 lines -> 9-line shim).
- Updated: `bindings/python/tests/test_smoke.py`, `bindings/python/README.md`,
  root `README.md`, `.github/workflows/ci.yml`.
- Removed requirement: Python's `sqlite3` must be built with
  `--enable-load-extension`; the Rust extension embeds SQLite directly.
