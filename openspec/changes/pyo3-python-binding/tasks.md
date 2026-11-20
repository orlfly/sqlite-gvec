## 1. PyO3 crate

- [x] 1.1 Create `crates/gvec-python` with `crate-type = ["cdylib"]`, module
  name `_gvec`, and PyO3 0.30
- [x] 1.2 Exclude it from the main Cargo workspace so extension-module
  linking does not affect `cargo test --workspace`
- [x] 1.3 Implement `Database` wrapping `gvec::Database` (open/in-memory,
  graph, close, path, version)
- [x] 1.4 Implement `Graph` wrapping `gvec::Graph` (run, name, knn)
- [x] 1.5 Implement `Vector` and `Distance` classes with the previous Python
  class surface

## 2. Native capabilities in Python

- [x] 2.1 Expose `create_vector_index` returning index metadata
- [x] 2.2 Expose `drop_vector_index`, `list_vector_indexes`, `vector_count`
- [x] 2.3 Expose `vector_search` and `exact_vector_search`
- [x] 2.4 Expose `create_text_index`, `drop_text_index`, `text_search`
- [x] 2.5 Convert `serde_json::Value` rows to Python values recursively
- [x] 2.6 Convert Python index option dictionaries/lists to JSON

## 3. Packaging and tests

- [x] 3.1 Replace the 172-line ctypes/`sqlite3` Python implementation with a
  thin import shim
- [x] 3.2 Add `pyproject.toml` for maturin (`gvec._gvec`)
- [x] 3.3 Update `test_smoke.py` for graph, KNN, ANN, FTS, and hybrid search
- [x] 3.4 Build the extension and run the smoke test on CPython 3.14
- [x] 3.5 Update `bindings/python/README.md` and root README build
  instructions
- [x] 3.6 Add a Python CI job for fmt, clippy, wheel build, and smoke test

## 4. Verification

- [x] 4.1 Main workspace remains green: 61 Rust tests, clippy `-D warnings`,
  `cargo fmt --check`
- [x] 4.2 Python crate is clippy-clean and rustfmt-clean
- [x] 4.3 `python bindings/python/tests/test_smoke.py` prints `python smoke OK`
