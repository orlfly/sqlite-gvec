## ADDED Requirements

### Requirement: Rust binding

The system SHALL provide a Rust crate that loads the extension via `rusqlite` (using `sqlite3_auto_extension`) and offers ergonomic helpers to create graphs and run queries without writing raw SQL for the entry points.

#### Scenario: Auto-load and query from Rust

- **WHEN** a Rust program opens a `rusqlite::Connection`, loads the gvec crate, and runs `SELECT gvec_version()`
- **THEN** a version string starting with `v` is returned

#### Scenario: Graph create and match from Rust

- **WHEN** a Rust program creates a graph, inserts a node and edge, and runs a MATCH query through the binding
- **THEN** the expected node data is returned

### Requirement: Python binding

The system SHALL provide a Python binding (ctypes-based loader plus helper wrappers) that loads `libgvec.so` and supports creating graphs, inserting nodes/edges, running Cypher queries, and running vector KNN queries from Python.

#### Scenario: Python graph operations

- **WHEN** a Python program loads the binding, creates a graph, adds nodes/edges, and runs a `MATCH ... RETURN` query
- **THEN** results are returned as Python data structures

#### Scenario: Python KNN query

- **WHEN** a Python program inserts vectors into a `gvec0` virtual table and runs a KNN query
- **THEN** nearest rowids and distances are returned

### Requirement: Binding parity

The Rust and Python bindings SHALL expose equivalent functionality for the core operations (create graph, add node/edge, run graph query, run vector query) so that documented examples work in both languages.

#### Scenario: Parity smoke test

- **WHEN** the same documented example is run through the Rust binding and the Python binding
- **THEN** both return equivalent results
