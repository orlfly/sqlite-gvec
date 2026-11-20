## ADDED Requirements

### Requirement: Rust-implemented Python extension

The Python binding SHALL be implemented as a PyO3 extension module. Python
source under `bindings/python` MUST NOT implement engine logic such as ctypes
loading, SQLite connection handling, or distance functions; the package
`__init__.py` MAY only re-export the compiled module.

#### Scenario: No Python engine logic

- **WHEN** the Python package is inspected
- **THEN** `bindings/python/gvec/__init__.py` contains only imports and
  `__all__`, and the extension module is `gvec._gvec`

#### Scenario: Engine is embedded

- **WHEN** `Database(":memory:")` is constructed
- **THEN** it opens an in-process Rust database without loading a SQLite
  extension and without requiring `--enable-load-extension`

### Requirement: Preserved Python API

The extension SHALL expose `Database`, `Graph`, `Vector`, `Distance`, and
`VERSION` with the previously documented method names.

#### Scenario: Graph query

- **WHEN** `g.run('MATCH (n:Person) RETURN n.name AS name')` is called
- **THEN** a list of rows (lists of Python values) is returned

#### Scenario: KNN parity method

- **WHEN** `g.knn(Vector((0,0)), [(1, Vector((1,0))), (2, Vector((3,0)))], k=2)` is called
- **THEN** `[(1, 1.0), (2, 3.0)]` is returned and the computation happens in Rust

#### Scenario: Distance constants

- **WHEN** `Distance.L2`, `Distance.COSINE`, and `Distance.INNER_PRODUCT` are read
- **THEN** they return `"l2"`, `"cosine"`, and `"ip"`

### Requirement: Vector index and FTS surface

The Python binding SHALL expose the vector index lifecycle and full-text
search APIs added to the Rust binding.

#### Scenario: Create and query an index

- **WHEN** `g.create_vector_index("Doc", "embedding", "hnsw", "l2", {"m": 8})` is called and `g.vector_search("Doc", "embedding", Vector((0,0)), 2)` follows
- **THEN** index metadata is returned and the two nearest rowids are returned

#### Scenario: Hybrid query

- **WHEN** a query uses `text_match` and `vector_nn` together through `Graph.run`
- **THEN** only rows satisfying both predicates are returned

### Requirement: Packaging and testability

The extension SHALL build with maturin using `pyproject.toml`, and the main
Rust workspace SHALL remain independently testable.

#### Scenario: Excluded workspace member

- **WHEN** `cargo test --workspace` runs without Python development files
- **THEN** it does not build the PyO3 crate and the existing Rust tests pass

#### Scenario: Python smoke test

- **WHEN** the extension is built and `python bindings/python/tests/test_smoke.py` runs
- **THEN** graph, KNN, ANN, FTS, and hybrid checks pass
