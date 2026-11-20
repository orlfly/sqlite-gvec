# gvec — graph + vector + full-text storage, embedded in SQLite

gvec is a from-scratch Rust implementation of a graph, vector, and full-text
database that runs on top of any SQLite build. It exposes:

- Persistent vector indexes — **HNSW** (Qdrant-style), **IVF** (Milvus-style),
  and flat exact — with L2 / cosine / inner-product distance over float32,
  int8, and binary vectors
- A Cypher-like graph query language (`CREATE`, `MATCH`, `OPTIONAL MATCH`,
  `WHERE`, `RETURN`, `SET`, `DELETE`, `ORDER BY`, `SKIP`, `LIMIT`) including
  directed, undirected, multi-type `[:A|B]`, and variable-length `[r*1..3]`
  patterns, plus consecutive `MATCH` clauses that reuse bound variables
- Vector-anchored graph traversal and hybrid retrieval via real `vector_nn(...)`
  and `text_match(...)` rowid-set constraints
- Full-text search through SQLite FTS5 with BM25 ranking (`CREATE TEXT INDEX`,
  `text_match`, `text_score`)
- Exact tag indexes with `CREATE TAG INDEX` and `tag_match`, composable with
  graph, text, and vector predicates
- A pure Rust API (`gvec` crate) that embeds SQLite via `rusqlite`
- A loadable SQLite extension (`gvec_extension` crate) that registers
  `gvec_version()`, `gvec_dist()`, and a `gvec_query()` stub (see "scope")
- Python bindings implemented in Rust with PyO3 (no ctypes or Python engine logic)

The engine is **independent of `sqlite-vec` and `sqlite-graph`** — no source
is reused, no dependency on those projects' crates. They are referenced only
as design inspiration.

## Status

| Crate | Description | Status |
| --- | --- | --- |
| `gvec-core` | Storage, HNSW/IVF/Flat indexes, tag indexes, Cypher parser/executor, FTS, fusion | ✅ implemented (29 tests) |
| `gvec` | In-process Rust binding over `rusqlite` | ✅ implemented (61 tests) |
| `gvec-python` | PyO3 Python extension (`gvec._gvec`) | ✅ implemented (smoke tested) |
| `gvec-extension` | Loadable SQLite extension via `sqlite-loadable` | ✅ entry point + scalars (1 test) |

## Quick start (Rust)

```toml
[dependencies]
gvec = { path = "crates/gvec-bindings" }
serde_json = "1"
```

```rust
use gvec::Database;

let db = Database::open_in_memory().unwrap();
let g = db.graph("social").unwrap();

g.run(r#"CREATE (a:Person {name: "Alice", age: 30})-[:KNOWS]->(b:Person {name: "Bob", age: 25})"#).unwrap();

let rs = g.run("MATCH (n:Person) WHERE n.age > 20 RETURN n.name AS name ORDER BY n.age ASC").unwrap();
assert_eq!(rs.rows.len(), 2);

// Persistent ANN index over a node property
g.run(r#"CREATE (a:Doc {title: "origin", body: "the rust programming language", tags: ["database", "rust"], embedding: [0.0, 0.0]})"#).unwrap();
g.run(r#"CREATE (b:Doc {title: "near",   body: "python is dynamic",              tags: ["python"],             embedding: [1.0, 0.0]})"#).unwrap();
g.run(r#"CREATE (c:Doc {title: "far",    body: "unrelated notes",               tags: ["notes"],              embedding: [5.0, 5.0]})"#).unwrap();

g.create_vector_index("Doc", "embedding", "hnsw", Some("l2"), None).unwrap();

let rs = g.run(r#"
    MATCH (n:Doc)
    WHERE vector_nn(n.embedding, [0.0, 0.0], 2)
    RETURN n.title AS title, vector_distance(n) AS distance,
           vector_score(n) AS score
    ORDER BY vector_score(n) DESC
"#).unwrap();
assert_eq!(rs.rows[0][0], "origin");
assert_eq!(rs.rows[1][0], "near");
assert!(rs.rows[0][2].as_f64().unwrap() > rs.rows[1][2].as_f64().unwrap());

// Or create the index with query-language DDL
g.run(r#"CREATE VECTOR INDEX ON :Doc(embedding)
         USING IVF(nlist = 4, nprobe = 2, metric = "cosine")"#).unwrap();

// Full-text search via SQLite FTS5 over the same nodes
g.run(r#"CREATE TEXT INDEX ON :Doc(body)"#).unwrap();
let rs = g.run(r#"MATCH (n:Doc) WHERE text_match(n.body, "rust") RETURN n.title AS title, text_score(n) AS score ORDER BY score DESC"#).unwrap();
assert_eq!(rs.rows.len(), 1);
assert_eq!(rs.rows[0][0], "origin");

// Hybrid: textual relevance AND semantic similarity in one query
let rs = g.run(r#"
    MATCH (n:Doc)
    WHERE text_match(n.body, "rust") AND vector_nn(n.embedding, [1.0, 0.0], 5)
    RETURN n.title, text_score(n) AS text_score, vector_distance(n) AS distance
    ORDER BY distance ASC
"#).unwrap();
assert_eq!(rs.rows.len(), 1);

// Exact tag index, usable alone or with text/vector predicates
g.create_tag_index("Doc", "tags").unwrap();
let rs = g.run(r#"MATCH (n:Doc) WHERE tag_match(n.tags, "database") RETURN n.title AS title"#).unwrap();
assert_eq!(rs.rows[0][0], "origin");
```

## Graph queries

`g.run()` executes a Cypher-like subset directly against the graph shadow tables.
Nodes and relationships can be created in one pattern:

```rust
g.run(r#"CREATE (a:Person {id: 1, name: "Alice"})-[:KNOWS]->(b:Person {id: 2, name: "Bob"})"#)?;

let rs = g.run(r#"
    MATCH (a:Person {name: "Alice"})-[:KNOWS]->(b)
    RETURN b.name AS name
"#)?;
assert_eq!(rs.rows[0][0], "Bob");
```

Fixed patterns can chain arbitrarily and may bind the relationship:

```rust
g.run(r#"
    MATCH (a:Person {id: 1})-[r:KNOWS]->(b)-[:KNOWS]->(c)
    RETURN c.name AS name
"#)?;
```

Variable-length patterns enumerate relationship-unique paths. `*` is `1..inf`,
`*3` is exactly three, `*..3` is `1..3`, `*2..` is `2..inf`, and `*0..2` is
`0..2`. Destination labels and inline properties are checked at every accepted
endpoint, and a relationship variable binds to an array of edge objects:

```rust
let rs = g.run(r#"
    MATCH (a:Person {id: 1})-[path:KNOWS*1..3]->(b:Person)
    RETURN b.name AS name, path
"#)?;
```

Undirected traversal (`-[:KNOWS]-`) and multi-type relationships
(`-[:KNOWS|LIKES]->`) are supported. Consecutive `MATCH` clauses reuse existing
bindings instead of rescanning, so `OPTIONAL MATCH` works as expected:

```rust
let rs = g.run(r#"
    MATCH (a:Person {id: 1})
    OPTIONAL MATCH (a)-[:MANAGES]->(b)
    RETURN a.name AS person, b.name AS report
"#)?;
```

Fixed patterns enumerate paths, so the same endpoint can appear more than once
when two different paths reach it. Variable-length patterns reject reusing the
same relationship within one path, which also makes bare `*` terminate on
cyclic graphs.

An inline scalar anchor such as `MATCH (n:Person {id: 42})` is pushed into
SQLite as a `json_extract` prefilter. For constant-time anchor lookup, create an
exact tag index on that property; the planner then resolves the anchor through
the index:

```rust
g.create_tag_index("Person", "id")?;
let rs = g.run(r#"MATCH (n:Person {id: 42}) RETURN n.name AS name"#)?;
```

Graph limitation summary:

- No `DISTINCT`, `COUNT`, or other aggregation; duplicate path endpoints are
  returned as separate rows.
- No `MATCH ... CREATE` / `MERGE`; `CREATE` always inserts the nodes in its
  pattern.
- No shortest-path operator. Unbounded variable-length traversal is
  relationship-unique but still exponential in the worst case.
- Without a tag index on the anchor property, anchor lookup is a label scan plus
  a SQL JSON prefilter.

### Vector index algorithms

- **HNSW** (default): hierarchical navigable small world. Levels are derived
  deterministically from node ids, so indexes rebuild reproducibly from the
  persisted vectors. Tunables: `m` (16), `m0` (2·m), `ef_construction` (200),
  `ef_search` (64).
- **IVF**: inverted-file clustering (Milvus `IVF_FLAT`). Tunables: `nlist` (16)
  and `nprobe` (4). Float centroids are means; int8/binary centroids are
  medoids.
- **Flat**: exact brute-force scan. Used as the fallback when `vector_nn`
  targets an unindexed property, and as the recall oracle in tests.

Vectors and index definitions are persisted in the graph's shadow tables
(`{graph}_vector_rows`, `{graph}_vector_indexes`, `{graph}_vectors`). ANN
structures are cached and rebuilt deterministically on demand after writes, so
a restart never requires an external sidecar file.

### Persistent vector rows

Every numeric array or typed vector property is also stored as a row in
`{graph}_vector_rows`, keyed by `(node_id, property)`. This makes KNN and
`vector_nn` work before an index is created and keeps embeddings available to
`fusion::node_vector`:

```rust
use gvec::{Distance, VectorValue};

let hits = g.exact_vector_search("Doc", "embedding", &query, 5, Distance::L2)?;
println!("{:?}", g.vector_rows(Some("Doc"), Some("embedding"))?);

// Sidecar embeddings are first-class too.
g.set_vector_row(node_id, "embedding", &VectorValue::F32(vec![0.1, 0.2]))?;
let v = g.node_vector(node_id, "embedding")?;
```

`vector_distance(v)` returns the metric distance (lower is better).
`vector_score(v)` returns a descending similarity: `1/(1+d)` for L2, `1-d`
for cosine, and `-d` for inner product. Use `ORDER BY vector_score(v) DESC`
for "most similar first". `vector_nn` still defaults to ascending distance.

## Benchmarks

```bash
cargo run --release -p gvec --example vector_bench
cargo run --release -p gvec --example vector_rows_bench
cargo run --release -p gvec --example graph_bench
```

The benchmark builds deterministic synthetic datasets, compares against
brute-force ground truth, and reports build time, p50/p95 query latency, and
recall@10. Sample run on one Linux x86-64 machine, single-threaded:

### Graph traversal

`graph_bench` builds a directed social graph with four `FOLLOWS` and one `LIKES`
edge per node, verifies every complete result set against a naive adjacency
model, and then measures the same 20 anchors. At 50k nodes and 250k edges it
inserts nodes at 3.8 µs/node and edges at 16.2 µs/edge (61.8k edges/s), uses
20.69 MB including the `Person.id` tag index, and builds that index in 115 ms.

| Case (50k nodes, 250k edges) | Rows | p50 | p95 |
| --- | ---: | ---: | ---: |
| anchor scan, no index | 1 | 23.57 ms | 25.69 ms |
| anchor via `Person.id` tag index | 1 | 0.035 ms | 0.047 ms |
| 1-hop out | 4 | 0.084 ms | 0.104 ms |
| 1-hop in | 4 | 0.086 ms | 0.100 ms |
| 2-hop fixed | 16 | 0.281 ms | 0.312 ms |
| 3-hop fixed | 64 | 1.299 ms | 1.672 ms |
| variable `*1..2` | 20 | 0.292 ms | 0.307 ms |
| variable `*1..3` | 84 | 1.223 ms | 2.298 ms |
| undirected 1-hop | 8 | 0.129 ms | 0.146 ms |
| multi-type 1-hop | 5 | 0.087 ms | 0.140 ms |
| edge property `r.weight > 0.8` | 0.5 | 0.086 ms | 0.095 ms |
| optional miss | 1 | 0.051 ms | 0.057 ms |

At 10k nodes and 50k edges the anchor scan drops to 4.62 ms and the tag-index
anchor to 0.035 ms; 1/2/3-hop traversal is 0.082/0.289/1.263 ms. Edge endpoint
indexes make each hop an index lookup. The anchor scan remains linear in label
size, which is why the tag index matters for selective entry points.

| Dataset (10k vectors) | Index | Build | p50 | Recall@10 |
| --- | --- | ---: | ---: | ---: |
| 128d f32, clustered (32 clusters) | Flat | n/a | 1.80 ms | 1.000 |
| 128d f32, clustered | HNSW, ef=64 | 2.8 s | 0.16 ms | 0.997 |
| 128d f32, clustered | IVF, nprobe=4 | 0.8 s | 0.30 ms | 1.000 |
| 128d f32, uniform random | Flat | n/a | 1.81 ms | 1.000 |
| 128d f32, uniform random | HNSW, ef=64 | 18.2 s | 0.46 ms | 0.811 |
| 128d f32, uniform random | HNSW, ef=256 | 18.2 s | 1.52 ms | 0.981 |
| 64d int8 | HNSW, ef=64 | 10.6 s | 0.29 ms | 0.889 |
| 256-bit binary (Hamming) | HNSW, ef=64 | 9.9 s | 0.27 ms | 0.555 |

Real embeddings are usually clustered, where ANN indexes deliver roughly
5-11x lower latency than a full scan at near-perfect recall. Uniform random
high-dimensional vectors are the adversarial case: `ef_search` trades recall
for latency at query time without rebuilding (`HnswIndex::search_with_ef`).
Binary Hamming space has many distance ties and is the weakest case for the
current graph heuristic.

### Persistent vector rows

`vector_rows_bench` stores every vector through the node write path, then
compares exact KNN over persisted rows with HNSW over the same data.

| Case (20k rows, 64d f32) | Time |
| --- | ---: |
| node + vector row insert | 21.7 µs/node (0.43 s total) |
| load all rows (`vector_rows`) | 99.8 ms |
| exact KNN, no index p50/p95 | 113.1 / 117.5 ms |
| HNSW build | 0.77 s |
| HNSW KNN p50/p95 | 0.26 / 0.45 ms |
| HNSW recall@10 | 1.000 |

Exact KNN scans and decodes every row on each call, so it is intended for
small graphs or migration fallback. Create an HNSW/IVF index for interactive
retrieval; rows remain the source of truth used to rebuild it.

## Full-text search

gvec indexes text properties with SQLite FTS5 and ranks hits with BM25:

```rust
g.create_text_index("Doc", "body", None)?; // unicode61
let rs = g.run(r#"
    MATCH (n:Doc)
    WHERE text_match(n.body, "rust AND database")
    RETURN n.title, text_score(n) AS score
    ORDER BY score DESC LIMIT 10
"#)?;
```

Tokenizers are `unicode61` (default), `porter`, and `trigram`. Use `trigram`
for substring and CJK matching; `unicode61` treats a run of Han characters as
one token. Queries use FTS5 syntax: `AND`, `OR`, `NOT`, quoted phrases, and
`prefix*`.

`CREATE TEXT INDEX` rebuilds the index and its sync triggers. Inserts,
property updates, label changes, and deletes stay in sync. `text_match`
supports one predicate per `MATCH`, as a top-level `AND` condition; combine
terms in one FTS5 query instead of writing multiple `text_match` calls.

### FTS benchmark

```bash
cargo run --release -p gvec --example fts_bench
```

Deterministic Zipf corpus, 80 tokens/doc, 50k docs, in-memory SQLite,
single-threaded. `direct` is `Graph::text_search(..., k=10)`, `cypher` is
`MATCH ... WHERE text_match(...) RETURN ... LIMIT 10`.

| Query | Matches | Direct p50 | Cypher p50 |
| --- | ---: | ---: | ---: |
| common term | 49,983 | 29.6 ms | 341.7 ms |
| medium term | 3,669 | 2.19 ms | 20.9 ms |
| rare term | 31 | 0.067 ms | 0.223 ms |
| two-term AND | 1,871 | 1.43 ms | 4.80 ms |
| boolean OR | 36,306 | 8.15 ms | 85.6 ms |
| phrase | 3,847 | 3.35 ms | 27.7 ms |
| prefix | 293 | 0.375 ms | 1.98 ms |
| no hit | 0 | 0.028 ms | 0.030 ms |

Index build scales linearly: 19 ms for 1k docs, 0.24 s for 10k, and 1.43 s for
50k (28.6 us/doc), using 40.3 MB at 50k (844 bytes/doc). Incremental inserts
with sync triggers cost 0.27 to 0.52 ms/doc. Selective queries are
sub-millisecond. Broad terms scale with hit count: the Cypher path materializes
every text hit before `LIMIT`, while `text_search` returns only k rows. Use
`text_search` for broad-query top-k and `text_match` when the text filter is
combined with graph or vector predicates. All eight benchmark result sets were
verified against a naive token scan.

## Tag indexes

Tags are exact, case-sensitive scalar strings, scalar numbers, or arrays of
strings/numbers stored in a JSON property:

```rust
g.create_tag_index("Doc", "tags")?;
let rs = g.run(r#"
    MATCH (n:Doc)
    WHERE tag_match(n.tags, "database") AND text_match(n.body, "rust")
    RETURN n.title, text_score(n) AS score
    ORDER BY score DESC
"#)?;
```

`CREATE TAG INDEX ON :Doc(tags)` builds a persisted `(index, node, tag)` table
with trigger-based maintenance. Inserts, updates, deletes, and label changes
stay in sync. Multiple `tag_match` predicates on the same variable under `AND`
intersect their result sets. `OR`, `NOT`, and predicates on different variables
are rejected with an explicit error because rowid pre-filtering would change
semantics.

Tag matching is exact: `data` does not match `database`, and `Database` does not
match `database`. Numeric tag values are compared through their decimal string
form. When no tag index exists, `tag_match` falls back to scanning the property,
so the feature is usable before DDL and after `DROP TAG INDEX`.

A tag index on an integer or string property also accelerates an inline scalar
anchor such as `MATCH (n:Person {id: 42})`. The planner uses the tag index to
fetch the anchor id and then applies the normal exact property filter.

`vector_nn(...) AND tag_match(...)` filters the ANN candidates after retrieval,
so a selective tag can return fewer than k rows. A future iteration can pass an
allowed-id set into the ANN search.

### Tag benchmark

```bash
cargo run --release -p gvec --example tag_bench
```

Deterministic Zipf tags, three tags per document, 50k docs. `direct` is
`Graph::tag_search(..., k=10)`, `cypher` is `tag_match + LIMIT 10`. Fallback
rows run after dropping the index.

| Query | Matches | Indexed direct | Indexed Cypher | Fallback direct | Fallback Cypher |
| --- | ---: | ---: | ---: | ---: | ---: |
| common tag | 22,361 | 0.012 ms | 136.8 ms | 73.4 ms | 201.4 ms |
| medium tag | 1,893 | 0.011 ms | 9.24 ms | 73.6 ms | 77.4 ms |
| rare tag | 75 | 0.013 ms | 0.329 ms | n/a | n/a |
| text + tag | 5,588 | n/a | 71.3 ms | n/a | n/a |

Index build: 241 ms for 50k docs (4.8 us/doc), 8.45 MB (177 bytes/doc). At 10k
docs the build takes 60 ms and 1.59 MB. The indexed rowid lookup is
microseconds; Cypher still materializes every matching node before `LIMIT`, so
broad tags scale with match count. All tag result sets were verified against a
naive scan.

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
python3 -m venv .venv
. .venv/bin/activate
python -m pip install maturin
maturin develop --release
python bindings/python/tests/test_smoke.py
```

The Python package is a thin shim over the Rust extension. The engine,
including SQLite, vector indexes, and FTS, is embedded directly, so no
loadable SQLite extension is required.

## Workspace layout

```
crates/
  gvec-core/         # pure Rust: storage, vector, vector_index (HNSW/IVF), tag, query, fts, fusion
  gvec-bindings/     # in-process Rust API over rusqlite
  gvec-python/       # PyO3 extension module (built with maturin)
  gvec-extension/    # loadable SQLite extension (cdylib)
bindings/python/     # Python package shim that re-exports gvec._gvec
deny.toml            # cargo-deny config forbidding sqlite-vec/sqlite-graph
```

## Independence

`deny.toml` denies these crates:

- `sqlite-vec`, `sqlite-vec-sys`
- `sqlite_graph`, `sqlite-graph`, `sqlitegraph`

CI enforces this with a grep on `Cargo.toml` and `Cargo.lock`.

## Scope & non-goals

- **In scope (v0.2)**: shadow-table node/edge storage, vector encoding
  (f32/i8/binary), persistent HNSW/IVF/Flat indexes, exact tag indexes, KNN
  and hybrid graph + vector + text + tag queries, the GQL subset listed above,
  variable-length patterns, Rust binding via `rusqlite`, scalar surface of the
  loadable extension, Rust/PyO3 Python binding.
- **Out of scope (v0.1)**: full virtual-table implementation in the
  loadable extension (the `gvec0` vtab is exposed via the Rust binding;
  the extension exposes scalars only — see notes in
  `crates/gvec-extension/src/lib.rs`), product/scalar quantization
  (IVF-PQ), exact filtered-vector top-k, case-insensitive tag normalization,
  `DISTINCT`/aggregation, `MATCH ... CREATE`/`MERGE`, relationship-property
  syntax in patterns, shortest-path operators, full Neo4j Cypher conformance,
  distributed storage.

## Tests

```bash
cargo test --workspace
```

The workspace currently has 91 tests: 61 in the Rust binding, 29 in the core,
and 1 in the extension.

## Reference architecture

Patterns were studied from:

- `sqlite-vec` — shadow-table vector storage, `idxStr` query plans
- `sqlite-graph` — Cypher pipeline (lexer / parser / planner / executor)

These informed the design but no code is reused or vendored; the engine is
a fresh Rust implementation per the project design document.
