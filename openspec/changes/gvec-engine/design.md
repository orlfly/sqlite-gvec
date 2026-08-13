## Context

We are building a new Rust-native graph database that embeds in SQLite, combining vector retrieval and Cypher-like graph query capabilities. `sqlite-vec` and `sqlite-graph` are used ONLY as design/behavioral references — their source is neither linked nor copied, and no runtime/build dependency on them exists. All logic is a from-scratch Rust reimplementation, optionally leaning on third-party crates.

Current state: empty Rust project. Reference architectures studied (inspiration only):
- `sqlite-vec`: `vec0` virtual table, shadow tables (`xyz_chunks`, `xyz_rowids`, `xyz_vector_chunksNN`), an `idxStr` string encoding the query plan (fullscan / point / KNN), KNN with `sqlite3_vtab_in_first/next`.
- `sqlite-graph`: `graph()` virtual table, Cypher lexer/parser/planner/executor pipeline in C, `cypher_execute()` scalar function, TCK-derived tests, node/edge storage with JSON properties.

## Goals / Non-Goals

**Goals:**
- A `gvec0` virtual table that stores nodes with labels, properties (JSON), and one or more vector columns.
- A `gvec0` vector index for KNN search over node embeddings (float32, int8, binary).
- A Cypher-like query language (`CREATE`, `MATCH`, `WHERE`, `RETURN`, relationship patterns) executed via a `gvec_query()` scalar/virtual-table function.
- Query fusion: start from a vector KNN anchor, then traverse the graph; or filter nodes by vector predicates within MATCH.
- Loadable as `libgvec.so` / `gvec.dylib`, with Rust (`rusqlite`) and Python bindings.
- All core logic in safe Rust.

**Non-Goals:**
- Full Neo4j Cypher TCK conformance (initial subset only, as defined by our specs).
- Distributed/clustered graph storage.
- Custom on-disk file format; all persistence goes through SQLite shadow tables.
- Transaction management beyond SQLite's own ACID semantics.
- Any code-level dependency on, or copying from, `sqlite-vec` or `sqlite-graph`.

## Decisions

### D1. Architecture: pure-Rust core + `sqlite-loadable` extension crate
Two-crate layout: an inner `gvec-core` (pure Rust) and an `extension` crate that compiles to a `cdylib` exposing the loadable-extension entry point. The entry point and all vtab/function registration are generated declaratively by the `sqlite-loadable` crate macros (`#[sqlite_entrypoint]`, `#[sqlite_module]`, `#[sqlite_table]`, scalar-function macros) — **no C or raw FFI is written by hand**. The host's `.load` still `dlsym`s the `sqlite3_gvec_init` symbol, but that symbol is produced by the macro, not authored.
- Rationale: keeps the hot path memory-safe, keeps the codebase 100% Rust, and matches the standard SQLite auto-extension loading pattern used with `rusqlite`. sqlite-loadable is the Rust-first way to build SQLite extensions (functions + virtual tables) from macros.
- Alternative considered: full C extension with Rust build hook — rejected, loses safety benefits and requires hand-written C.
- Alternative considered: `rusqlite::vtab` — rejected for the vector table; it only supports simple read-only/eponymous virtual tables and cannot express the KNN query-plan (`idxStr`) machinery the `gvec0` table needs.

### D2. Storage: shadow tables for nodes and edges
`CREATE VIRTUAL TABLE g USING gvec0(...)` creates:
- `gvec0_nodes(id INTEGER PRIMARY KEY, labels TEXT, properties TEXT)` — labels as JSON array, properties as JSON object.
- `gvec0_edges(id INTEGER PRIMARY KEY, source INTEGER, target INTEGER, type TEXT, properties TEXT)`.
- `gvec0_vector_chunksNN` + `gvec0_rowids` + `gvec0_chunks` — the chunked-BLOB vector store for node embeddings (see D4).
- `gvec0_auxiliary` — non-indexed per-node auxiliary metadata.
- Rationale: shadow tables are a well-established SQLite technique that keeps everything in one file with ACID writes; our schema is designed independently from scratch.
- Alternative considered: single `graph` table with separate in-memory adjacency — rejected, must survive restarts via SQLite.

### D3. Query language: parser + planner + executor pipeline
Modules `cypher_lexer`, `cypher_parser` (AST), `cypher_planner` (logical → physical plan), `cypher_executor` (iteration over rows). Initial clause subset: `CREATE`, `MATCH`, `OPTIONAL MATCH`, `WHERE`, `RETURN`, `SET`, `DELETE`, `ORDER BY`, `SKIP`, `LIMIT`. Relationship patterns: `(a)-[:TYPE]->(b)`, undirected `-[r]-`, variable-length `[*1..3]`. Properties as JSON via `serde_json`.
- Rationale: a hand-rolled recursive-descent parser keeps errors Cypher-flavored and avoids heavy grammar-crate overhead; the pipeline is our own design.
- Alternative considered: reuse `pest`/`nom` grammar crates — acceptable if the parser grows; AST/planner remain hand-written regardless, so we start with a hand-rolled recursive-descent parser and swap in a grammar crate only if maintenance demands it.

### D4. Vector index: chunked-BLOB shadow tables + KNN
Vector storage is our own Rust implementation: chunked BLOB vectors, a rowid mapping table, and an encoded query-plan string ('1' fullscan, '2' point, '3' KNN). KNN is a Rust iterator computing distances (L2 default, cosine, inner product selectable) and yielding ordered rowids. Start with brute-force scan; approximate indexing (IVF/HNSW/DiskANN) can slot in behind a `DistanceIndex` trait later.
- Rationale: chunked BLOB storage is compact and works on any SQLite build; brute-force KNN is "fast enough" for the initial release. A `DistanceIndex` trait keeps the door open for crates like `hnsw-rs` without baking in a dependency.
- Alternative considered: faiss/hnswlib FFI — heavy native deps and platform pain; prefer Rust crates (e.g. `hnsw-rs`) only if needed behind the trait.

### D5. Fusion API: `gvec_query()` + virtual table
Two surfaces:
- `SELECT * FROM gvec_query('<cypher>', ...)` — table-valued function returning result rows; params can embed a `:vector` anchor from a KNN index.
- `SELECT gvec_query(...)` scalar returning JSON for simple cases.
- A `MATCH (n) WHERE vector_nn(n.embedding, :query, 10)` predicate hook drives the optimizer to use the KNN plan and feed anchors into traversal.
- Rationale: TVF keeps rows streamable for large results; predicate hook enables fused plans.
- Alternative considered: only a scalar JSON function — rejected, poor ergonomics for large results.

### D6. Bindings
- Rust: crate `gvec` exposing an in-process API over `rusqlite` (embedding SQLite directly), plus an optional auto-load helper (`sqlite3_auto_extension`) for the loadable artifact. No custom C in either path.
- Python: `ctypes`-based loader + thin wrappers for the `cdylib`, our own API design.
- Rationale: Rust-first (pure Rust, no hand-written FFI), Python for parity in demos and quick experimentation.

### D7. Third-party crates policy
- No dependency (build, link, or runtime) on `sqlite-vec` or `sqlite-graph`; their code is never vendored or copied.
- Crates.io libraries MAY be used where they fit, evaluated for license and maintenance: SQLite extension authoring (`sqlite-loadable`), SQLite bindings for in-process use (`rusqlite`, `libsqlite3-sys`), distance kernels (`simsimd`), ANN indexes (`hnsw-rs`, only behind the `DistanceIndex` trait), parsers (`nom`, `pest`, `logos` if the hand-rolled parser grows), graph algorithms (`petgraph`), JSON (`serde_json`).
- Rationale: reuse battle-tested primitives and keep the whole project free of hand-written C/FFI.

## Risks / Trade-offs

- [Cypher grammar creep] → Keep parser hand-written but modular; gate new clauses behind tests; only implement the declared subset.
- [KNN performance without approximate index] → Brute-force is O(N·D); acceptable initially, design the vector module with a `DistanceIndex` trait so crates like `hnsw-rs` can slot in later.
- [Extension ABI fragility across SQLite versions] → Use only the stable, documented `sqlite3` C API; pin CI to multiple SQLite versions.
- [Rust `sqlite-loadable`/`rusqlite` version and symbol conflicts] → Build the extension crate against a bundled/`libsqlite3-sys` SQLite so generated symbols resolve independently of the host's separate sqlite3 build.
- [Distributed lock/multi-process write contention] → Inherit SQLite WAL semantics; document that writes follow SQLite concurrency rules.
- [Third-party crate churn] → Pin known-good crate versions; prefer crates with permissive licenses and active maintenance.

## Migration Plan

- N/A — greenfield project. Release order: core storage → vector index → GQL parser/executor → fusion → bindings → docs/examples.

## Open Questions

- Exact virtual-table DDL surface for `gvec0` (vector column count/type grammar) — to be finalized against specs.
- Whether approximate index (IVF/HNSW) is in scope for the first release or a follow-up capability.
- Which exact Cypher clause subset is required for "support graph query language" in v0.1 — to be defined by our own specs, not bound to any existing implementation.