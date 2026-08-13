# gvec — a Rust graph database with vector retrieval, embedded in SQLite

gvec is a from-scratch Rust implementation of a graph + vector database that
runs on top of any SQLite build. It exposes:

- A `gvec0`-style vector virtual table with KNN search (L2 / cosine / inner-product)
- A Cypher-like graph query language (`CREATE`, `MATCH`, `OPTIONAL MATCH`,
  `WHERE`, `RETURN`, `SET`, `DELETE`, `ORDER BY`, `SKIP`, `LIMIT`) including
  directed, undirected, and variable-length `[r*1..3]` patterns
- Vector-anchored graph traversal via a `vector_nn(...)` predicate
- A pure Rust API (`gvec` crate) that embeds SQLite via `rusqlite`
- A loadable SQLite extension (`gvec_extension` crate) that registers
  `gvec_version()`, `gvec_dist()`, and a `gvec_query()` stub (see "scope")
- Python bindings (ctypes loader)

The engine is **independent of `sqlite-vec` and `sqlite-graph`** — no source
is reused, no dependency on those projects' crates. They are referenced only
as design inspiration.

## Status

| Crate | Description | Status |
| --- | --- | --- |
| `gvec-core` | Storage, vector encoding, Cypher parser/executor, fusion | ✅ implemented (17 tests) |
| `gvec` | In-process Rust binding over `rusqlite` | ✅ implemented (4 tests) |
| `gvec-extension` | Loadable SQLite extension via `sqlite-loadable` | ✅ entry point + scalars (1 test) |

## Quick start (Rust)

```toml
[dependencies]
gvec = { path = "crates/gvec-bindings" }
serde_json = "1"
```

```rust
use gvec::Database;
use gvec::vector::{Distance, VectorValue};

let db = Database::open_in_memory().unwrap();
let g = db.graph("social").unwrap();

g.run(r#"CREATE (a:Person {name: "Alice", age: 30})-[:KNOWS]->(b:Person {name: "Bob", age: 25})"#).unwrap();

let rs = g.run("MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name ORDER BY n.age ASC").unwrap();
assert_eq!(rs.rows.len(), 2);

// KNN over an in-memory vector set
let stored = vec![
    (1, VectorValue::F32(vec![1.0, 0.0])),
    (2, VectorValue::F32(vec![3.0, 0.0])),
    (3, VectorValue::F32(vec![0.0, 0.0])),
];
let q = VectorValue::F32(vec![0.0, 0.0]);
let nearest = g.knn(&q, &stored, 2, Distance::L2);
assert_eq!(nearest[0].0, 3);
```

## Quick start (SQLite extension)

```bash
cargo build --release -p gvec-extension
sqlite3 :memory: ".load target/release/libgvec_extension.so sqlite3_gvec_init" \
  "SELECT gvec_version(), gvec_dist('[1.0,0.0]','[3.0,0.0]','l2');"
```

The extension exposes scalar functions only (`gvec_version`, `gvec_dist`,
`gvec_query` returns a helpful error pointing at the Rust binding). The full
Cypher executor lives in the `gvec` Rust binding and operates against a
`rusqlite::Connection`.

## Quick start (Python)

```bash
cargo build --release -p gvec-extension
PYTHONPATH=bindings/python \
  GVEC_LIB_PATH=$PWD/target/release/libgvec_extension.so \
  python3 bindings/python/test_smoke.py
```

Python requires a SQLite build with `--enable-load-extension` (not the
default in many system Pythons).

## Workspace layout

```
crates/
  gvec-core/         # pure Rust: storage, vector, query, fusion
  gvec-bindings/     # in-process Rust API over rusqlite
  gvec-extension/    # loadable SQLite extension (cdylib)
bindings/python/     # Python ctypes binding
deny.toml            # cargo-deny config forbidding sqlite-vec/sqlite-graph
```

## Independence

`deny.toml` denies these crates:

- `sqlite-vec`, `sqlite-vec-sys`
- `sqlite_graph`, `sqlite-graph`, `sqlitegraph`

CI enforces this with a grep on `Cargo.toml` and `Cargo.lock`.

## Scope & non-goals

- **In scope (v0.1)**: shadow-table node/edge storage, vector encoding
  (f32/i8/binary), KNN, the GQL subset listed above, variable-length
  patterns, Rust binding via `rusqlite`, scalar surface of the loadable
  extension, Python ctypes binding.
- **Out of scope (v0.1)**: full virtual-table implementation in the
  loadable extension (the `gvec0` vtab is exposed via the Rust binding;
  the extension exposes scalars only — see notes in
  `crates/gvec-extension/src/lib.rs`), approximate ANN indexes (HNSW/IVF),
  full Neo4j Cypher conformance, distributed storage.

## Tests

```bash
cargo test --workspace
```

## Reference architecture

Patterns were studied from:

- `sqlite-vec` — shadow-table vector storage, `idxStr` query plans
- `sqlite-graph` — Cypher pipeline (lexer / parser / planner / executor)

These informed the design but no code is reused or vendored; the engine is
a fresh Rust implementation per the project design document.