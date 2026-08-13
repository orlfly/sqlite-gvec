## 1. Project Scaffolding

- [x] 1.1 Initialize workspace: root `Cargo.toml` with members `gvec-core`, `gvec-extension`, `gvec-bindings` plus a `bindings/python/` package directory
- [x] 1.2 Add dependencies: `rusqlite` (bundled), `serde_json`, `serde`, `bytes`, `sqlite-loadable`; optionally add crates.io libraries for distance kernels (`simsimd`), ANN, or graph algorithms as needed
- [x] 1.3 Wire `libsqlite3-sys`/bundled SQLite so the extension crate links against a known SQLite version
- [x] 1.4 Add CI workflow building and testing on Linux x86_64 (and macOS if available)
- [x] 1.5 Confirm no dependency on `sqlite-vec` or `sqlite-graph` (no `[dependencies]` entry, no vendored code); add a CI deny-check guarding against them

## 2. Core Storage (`gvec-core` capability)

- [x] 2.1 Implement node storage module: shadow tables `gvec0_nodes(id, labels, properties)` with insert/update/delete and JSON label/property helpers
- [x] 2.2 Implement edge storage module: shadow tables `gvec0_edges(id, source, target, type, properties)` enforcing existing-node references
- [x] 2.3 Implement property path access (`n.address.city`) over `serde_json::Value`
- [x] 2.4 Implement transactional node/edge deletion including cascade of incident edges
- [x] 2.5 Write round-trip persistence tests: commit, close, reopen, verify nodes/edges survive
- [x] 2.6 Write unit tests covering gvec-core scenarios from `specs/gvec-core/spec.md`

## 3. Vector Retrieval (`vector-retrieval` capability)

- [x] 3.1 Implement `gvec0` virtual table schema grammar (element types float32/int8/binary, arbitrary dimension, metadata columns)
- [x] 3.2 Implement shadow tables `gvec0_chunks`, `gvec0_rowids`, `gvec0_vector_chunksNN`, `gvec0_auxiliary` with lossless BLOB encoding
- [x] 3.3 Implement virtual table insert/update/delete including dimension checks and metadata/auxiliary columns
- [x] 3.4 Implement exact KNN iterator with L2 (default), cosine, and inner-product distances
- [x] 3.5 Implement `idxStr` query-plan encoding for fullscan (`'1'`), point (`'2'`), and KNN (`'3'`) plans
- [x] 3.6 Implement point lookup by rowid (`WHERE rowid = ?`)
- [x] 3.7 Add `gvec_version()` scalar function returning a `v`-prefixed version
- [x] 3.8 Write tests for KNN ordering, metric selection, round-trips, and rollback across all element types

## 4. Graph Query Language (`graph-query-language` capability)

- [x] 4.1 Implement Cypher-like lexer with token positions for error reporting
- [x] 4.2 Implement recursive-descent parser producing an AST for `CREATE`, `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `SET`, `DELETE`, `ORDER BY`, `SKIP`, `LIMIT`
- [x] 4.3 Implement AST for relationship patterns: directed `->`, undirected `-[r]-`, type filter `:TYPE`, and variable-length `[*1..3]`
- [x] 4.4 Implement logical planner converting AST to physical plan (scan, join, filter, project, aggregate, sort/skip/limit operators)
- [x] 4.5 Implement executor for node match by label, relationship traversal both directions, and variable-length path expansion
- [x] 4.6 Implement `WHERE` evaluation with comparison operators, AND/OR, and property access
- [x] 4.7 Implement `RETURN` projection with aliases, property access, `ORDER BY`, `SKIP`, `LIMIT`
- [x] 4.8 Implement `CREATE`, `SET`, and `DELETE` write operators over the storage module
- [x] 4.9 Implement `OPTIONAL MATCH` producing null-valued variables on no match
- [x] 4.10 Add `gvec_query()` TVF returning rows and scalar variant returning JSON; surface parser/execution errors with positions
- [x] 4.11 Add tests covering all clauses, patterns, errors, and null handling from `specs/graph-query-language/spec.md`

## 5. Graph-Vector Integration (`graph-vector-integration` capability)

- [x] 5.1 Add a `vector_nn(embedding, :query, k)` function usable in `WHERE`
- [x] 5.2 Wire optimizer: recognize `vector_nn` predicates and use the KNN plan, honoring label/other constraints
- [x] 5.3 Expose KNN distance in fused query results and default to distance-based ordering absent explicit `ORDER BY`
- [x] 5.4 Add integration tests: vector-filter on nodes, vector-anchored traversal, distance parity with standalone KNN

## 6. Loadable Extension (`sqlite-extension` capability)

- [x] 6.1 Generate the entry point `sqlite3_gvec_init` with `sqlite-loadable` macros (`#[sqlite_entrypoint]`); no hand-written C or raw FFI
- [x] 6.2 Register `gvec0` virtual table and `gvec_query()`/`gvec_version()` functions on init; make init idempotent
- [x] 6.3 Verify `.load libgvec.so` in the SQLite CLI plus `CREATE VIRTUAL TABLE` and function invocation
- [x] 6.4 Add WAL-mode and concurrency tests (reader during active writer, no corruption)
- [x] 6.5 Confirm zero hand-written C/FFI (audit: entry point + vtab + functions all via `sqlite-loadable` macros), using only stable versions of the crate

## 7. Language Bindings (`language-bindings` capability)

- [x] 7.1 Implement Rust binding crate exposing auto-load via `sqlite3_auto_extension` and graph/query helper APIs
- [x] 7.2 Add Rust binding tests: auto-load, graph create/match, KNN query
- [x] 7.3 Implement Python package: ctypes loader for `libgvec.so` plus node/edge/query/KNN wrappers
- [x] 7.4 Add Python tests mirroring the Rust binding smoke tests
- [x] 7.5 Write shared documented examples (same steps in Rust and Python) and add a parity smoke test

## 8. Docs, Benchmarks, and Release Readiness

- [x] 8.1 Write README with install/build/quick-start and usage examples across SQL, Rust, and Python
- [x] 8.2 Add throughput benchmarks for node/edge creation, MATCH, and KNN (with documented methodology and comparable workloads)
- [x] 8.3 Verify every requirement/scenario in `specs/**/spec.md` maps to at least one automated test
- [x] 8.4 Run full test suite (`cargo test` + Python tests) and lint/clippy clean; finalize version string